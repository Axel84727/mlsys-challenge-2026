//! Graph Topology Triage
//!
//! Classifies the input DAG's topology to select the optimal parallelism strategy.
//! This is the "first microseconds" analysis that the Master performs before
//! dispatching work to workers.
//!
//! Classification:
//! - **Linear**: Chains with max fan-out 1. Use deep pipeline (single worker).
//! - **Diamond**: Shared tensors, moderate fan-in/fan-out. Split by branches.
//! - **Monster**: >2000 ops or high fan-out. Apply graph tiling with K-way partition.
//!
//! The triage is purely structural - no magic constants tuned to specific benchmarks.

use crate::models::{OpId, Problem, TensorMeta};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// Topology Classification
// ============================================================================

/// The structural classification of a computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphTopology {
    /// Linear chain: max fan-out/fan-in = 1. Best with single-worker deep fusion.
    Linear,
    /// Diamond pattern: shared tensors with fan-out > 1 but manageable.
    /// Best with branch-level parallelism (2-4 workers).
    Diamond,
    /// Monster graph: very large or complex. Needs graph tiling.
    Monster,
}

/// Detailed triage result with partitioning hints
#[derive(Debug, Clone)]
pub struct TriageResult {
    /// The classified topology
    pub topology: GraphTopology,
    /// Maximum fan-out observed in the graph
    pub max_fan_out: usize,
    /// Maximum fan-in observed
    pub max_fan_in: usize,
    /// Number of "diamond" junctions (tensors with fan-out > 1)
    pub diamond_count: usize,
    /// Number of independent branches (connected components after removing shared tensors)
    pub branch_count: usize,
    /// Topological depth of the graph (longest path from input to output)
    pub depth: usize,
    /// Suggested number of workers based on topology
    pub suggested_workers: usize,
    /// Suggested partition strategy
    pub strategy: PartitionStrategy,
}

/// How to partition the graph for parallel execution
#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    /// Don't partition - run everything in one process
    SingleProcess,
    /// Partition by independent branches (for diamond graphs)
    BranchParallel {
        /// Groups of ops that form independent branches
        branches: Vec<Vec<OpId>>,
    },
    /// Partition by topological levels (for monster graphs)
    TiledPartition {
        /// Tiles of ops, each tile is a self-contained sub-problem
        tiles: Vec<Vec<OpId>>,
        /// Boundary tensors between tiles (need inter-tile communication)
        boundary_tensors: Vec<HashSet<usize>>,
    },
}

// ============================================================================
// Triage Implementation
// ============================================================================

/// Perform topology triage on the computation graph.
///
/// This runs in O(V + E) time and provides the structural classification
/// needed to decide parallelism strategy.
pub fn triage_graph(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    available_workers: usize,
) -> TriageResult {
    let num_ops = problem.ops.len();

    // === Step 1: Compute fan-in/fan-out metrics ===
    let mut max_fan_out: usize = 0;
    let mut max_fan_in: usize = 0;
    let mut diamond_count: usize = 0;

    for meta in tensor_meta.iter() {
        let fan_out = meta.consumers.len();
        if fan_out > max_fan_out {
            max_fan_out = fan_out;
        }
        if fan_out > 1 {
            diamond_count += 1;
        }
    }

    for op in &problem.ops {
        let fan_in = op.inputs.len();
        if fan_in > max_fan_in {
            max_fan_in = fan_in;
        }
    }

    // === Step 2: Compute topological depth ===
    let depth = compute_graph_depth(problem, tensor_meta);

    // === Step 3: Find independent branches ===
    let branches = find_independent_branches(problem, tensor_meta);
    let branch_count = branches.len();

    // === Step 4: Classify topology ===
    let topology = if num_ops > 2000 || (num_ops > 500 && max_fan_out > 4) {
        GraphTopology::Monster
    } else if diamond_count > 0 && max_fan_out > 1 {
        GraphTopology::Diamond
    } else {
        GraphTopology::Linear
    };

    // === Step 5: Determine strategy ===
    let (suggested_workers, strategy) = match topology {
        GraphTopology::Linear => {
            (1, PartitionStrategy::SingleProcess)
        }
        GraphTopology::Diamond => {
            if branch_count > 1 && available_workers > 1 {
                let workers = branch_count.min(available_workers).min(4);
                (workers, PartitionStrategy::BranchParallel { branches: branches.clone() })
            } else {
                (1, PartitionStrategy::SingleProcess)
            }
        }
        GraphTopology::Monster => {
            if available_workers > 1 {
                let (tiles, boundaries) = tile_graph(problem, tensor_meta, available_workers);
                let workers = tiles.len().min(available_workers);
                (workers, PartitionStrategy::TiledPartition {
                    tiles,
                    boundary_tensors: boundaries,
                })
            } else {
                (1, PartitionStrategy::SingleProcess)
            }
        }
    };

    TriageResult {
        topology,
        max_fan_out,
        max_fan_in,
        diamond_count,
        branch_count,
        depth,
        suggested_workers,
        strategy,
    }
}

// ============================================================================
// Graph Analysis Helpers
// ============================================================================

/// Compute the longest path (depth) in the DAG
fn compute_graph_depth(problem: &Problem, tensor_meta: &[TensorMeta]) -> usize {
    let num_ops = problem.ops.len();
    if num_ops == 0 {
        return 0;
    }

    // Build adjacency: op -> downstream ops
    let mut downstream: Vec<Vec<OpId>> = vec![Vec::new(); num_ops];
    let mut in_degree: Vec<usize> = vec![0; num_ops];

    for (op_id, op) in problem.ops.iter().enumerate() {
        for &output_id in &op.outputs {
            if output_id < tensor_meta.len() {
                for &consumer in &tensor_meta[output_id].consumers {
                    if consumer < num_ops {
                        downstream[op_id].push(consumer);
                        in_degree[consumer] += 1;
                    }
                }
            }
        }
    }

    // BFS to find longest path (by levels)
    let mut queue: VecDeque<OpId> = VecDeque::new();
    let mut depth_map: Vec<usize> = vec![0; num_ops];

    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
            depth_map[i] = 1;
        }
    }

    let mut max_depth: usize = 0;

    while let Some(op_id) = queue.pop_front() {
        let current_depth = depth_map[op_id];
        max_depth = max_depth.max(current_depth);

        for &next in &downstream[op_id] {
            in_degree[next] -= 1;
            depth_map[next] = depth_map[next].max(current_depth + 1);
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    max_depth
}

/// Find independent branches in the DAG.
///
/// Two ops are in the same branch if they're connected through non-shared tensors.
/// Shared tensors (fan-out > 1) are treated as "cut points".
fn find_independent_branches(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Vec<Vec<OpId>> {
    let num_ops = problem.ops.len();
    if num_ops == 0 {
        return vec![];
    }

    // Build undirected connectivity (ops connected through non-shared tensors)
    let mut adj: Vec<HashSet<OpId>> = vec![HashSet::new(); num_ops];

    for (_tensor_id, meta) in tensor_meta.iter().enumerate() {
        // Skip shared tensors (these are cut points)
        if meta.consumers.len() > 1 {
            continue;
        }

        // Connect producer to consumers
        if let Some(producer) = meta.producer {
            if producer < num_ops {
                for &consumer in &meta.consumers {
                    if consumer < num_ops {
                        adj[producer].insert(consumer);
                        adj[consumer].insert(producer);
                    }
                }
            }
        }
    }

    // Find connected components using BFS
    let mut visited = vec![false; num_ops];
    let mut branches: Vec<Vec<OpId>> = Vec::new();

    for start in 0..num_ops {
        if visited[start] {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(op_id) = queue.pop_front() {
            component.push(op_id);
            for &neighbor in &adj[op_id] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        component.sort();
        branches.push(component);
    }

    // Sort branches by size (largest first) for load balancing
    branches.sort_by(|a, b| b.len().cmp(&a.len()));
    branches
}

/// Tile a large graph into balanced partitions using BFS from inputs.
///
/// Strategy: Walk the DAG in topological order, filling tiles until they reach
/// the target size. Cut at topological boundaries to minimize inter-tile edges.
///
/// Returns (tiles, boundary_tensors) where boundary_tensors[i] are tensors
/// that cross from tile i to other tiles.
fn tile_graph(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    num_workers: usize,
) -> (Vec<Vec<OpId>>, Vec<HashSet<usize>>) {
    let num_ops = problem.ops.len();
    let target_tiles = (num_workers * 2).max(2); // 2x oversubscription for work stealing
    let target_tile_size = (num_ops / target_tiles).max(50);

    // Get topological order
    let topo_order = topological_sort(problem, tensor_meta);

    // Greedily assign ops to tiles
    let mut tiles: Vec<Vec<OpId>> = Vec::new();
    let mut current_tile: Vec<OpId> = Vec::new();

    for &op_id in &topo_order {
        current_tile.push(op_id);

        if current_tile.len() >= target_tile_size {
            tiles.push(std::mem::take(&mut current_tile));
        }
    }

    // Don't leave orphan ops
    if !current_tile.is_empty() {
        if tiles.is_empty() {
            tiles.push(current_tile);
        } else if current_tile.len() < target_tile_size / 3 {
            // Merge tiny remainder into last tile
            tiles.last_mut().unwrap().extend(current_tile);
        } else {
            tiles.push(current_tile);
        }
    }

    // Compute boundary tensors for each tile
    let mut op_to_tile: HashMap<OpId, usize> = HashMap::new();
    for (tile_idx, tile) in tiles.iter().enumerate() {
        for &op_id in tile {
            op_to_tile.insert(op_id, tile_idx);
        }
    }

    let mut boundary_tensors: Vec<HashSet<usize>> = vec![HashSet::new(); tiles.len()];

    for (tensor_id, meta) in tensor_meta.iter().enumerate() {
        if let Some(producer) = meta.producer {
            if let Some(&prod_tile) = op_to_tile.get(&producer) {
                for &consumer in &meta.consumers {
                    if let Some(&cons_tile) = op_to_tile.get(&consumer) {
                        if prod_tile != cons_tile {
                            boundary_tensors[prod_tile].insert(tensor_id);
                        }
                    }
                }
            }
        }
    }

    (tiles, boundary_tensors)
}

/// Topological sort using Kahn's algorithm
fn topological_sort(problem: &Problem, tensor_meta: &[TensorMeta]) -> Vec<OpId> {
    let num_ops = problem.ops.len();
    let mut in_degree = vec![0usize; num_ops];
    let mut downstream: Vec<Vec<OpId>> = vec![Vec::new(); num_ops];

    for (op_id, op) in problem.ops.iter().enumerate() {
        for &output_id in &op.outputs {
            if output_id < tensor_meta.len() {
                for &consumer in &tensor_meta[output_id].consumers {
                    if consumer < num_ops {
                        downstream[op_id].push(consumer);
                        in_degree[consumer] += 1;
                    }
                }
            }
        }
    }

    let mut queue: VecDeque<OpId> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(num_ops);
    while let Some(op_id) = queue.pop_front() {
        order.push(op_id);
        for &next in &downstream[op_id] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    order
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Problem, Tensor};

    fn make_linear_problem(num_ops: usize) -> Problem {
        let num_tensors = num_ops + 1;
        Problem {
            tensors: (0..num_tensors)
                .map(|_| Tensor { width: 128, height: 128 })
                .collect(),
            ops: (0..num_ops)
                .map(|i| Op {
                    op_type: OpType::Pointwise,
                    inputs: vec![i],
                    outputs: vec![i + 1],
                    base_cost: 100,
                })
                .collect(),
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    fn make_diamond_problem() -> Problem {
        // op0 -> t1 -> op1, op2 -> op3
        Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 }, // t0 input
                Tensor { width: 128, height: 128 }, // t1 shared
                Tensor { width: 128, height: 128 }, // t2 op1 out
                Tensor { width: 128, height: 128 }, // t3 op2 out
                Tensor { width: 128, height: 128 }, // t4 op3 out
            ],
            ops: vec![
                Op { op_type: OpType::Pointwise, inputs: vec![0], outputs: vec![1], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![1], outputs: vec![2], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![1], outputs: vec![3], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![2, 3], outputs: vec![4], base_cost: 100 },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_triage_linear() {
        let problem = make_linear_problem(10);
        let meta = problem.build_tensor_meta();
        let result = triage_graph(&problem, &meta, 4);
        assert_eq!(result.topology, GraphTopology::Linear);
        assert_eq!(result.max_fan_out, 1);
        assert_eq!(result.suggested_workers, 1);
    }

    #[test]
    fn test_triage_diamond() {
        let problem = make_diamond_problem();
        let meta = problem.build_tensor_meta();
        let result = triage_graph(&problem, &meta, 4);
        assert_eq!(result.topology, GraphTopology::Diamond);
        assert!(result.diamond_count >= 1);
    }

    #[test]
    fn test_triage_monster() {
        let problem = make_linear_problem(3000);
        let meta = problem.build_tensor_meta();
        let result = triage_graph(&problem, &meta, 8);
        assert_eq!(result.topology, GraphTopology::Monster);
        assert!(result.suggested_workers > 1);
    }

    #[test]
    fn test_topological_sort_simple() {
        let problem = make_linear_problem(5);
        let meta = problem.build_tensor_meta();
        let order = topological_sort(&problem, &meta);
        assert_eq!(order.len(), 5);
        // Should be in order 0, 1, 2, 3, 4
        for (i, &op) in order.iter().enumerate() {
            assert_eq!(op, i);
        }
    }

    #[test]
    fn test_graph_depth_linear() {
        let problem = make_linear_problem(10);
        let meta = problem.build_tensor_meta();
        let depth = compute_graph_depth(&problem, &meta);
        assert_eq!(depth, 10); // Linear chain has depth == num_ops
    }

    #[test]
    fn test_tile_graph_large() {
        let problem = make_linear_problem(1000);
        let meta = problem.build_tensor_meta();
        let (tiles, boundaries) = tile_graph(&problem, &meta, 4);

        // Should have multiple tiles
        assert!(tiles.len() >= 2);

        // All ops should be covered
        let total_ops: usize = tiles.iter().map(|t| t.len()).sum();
        assert_eq!(total_ops, 1000);

        // Boundary tensors should exist between tiles
        assert_eq!(boundaries.len(), tiles.len());
    }
}

