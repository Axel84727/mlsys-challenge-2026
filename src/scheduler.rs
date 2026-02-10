//! Graph scheduler implementing aggressive fusion with memory-aware tiling.
//!
//! Strategy:
//! 1. Build DAG of operations using petgraph
//! 2. Attempt aggressive fusion of connected ops
//! 3. Validate memory constraints and apply tiling/Split-K if needed
//! 4. Optimize traversal order for data reuse
//! 5. Select tensors to retain for diamond patterns

use crate::cost::{compute_subgraph_latency, generate_snake_traversal, MemoryState};
use crate::memory::{
    analyze_tensor_residency, compute_available_retention_capacity, find_fitting_granularity,
    find_split_k, validate_memory_fit,
};
use crate::models::{
    Granularity, GranularityOutput, OpId, OpType, Problem, Solution, Subgraph, SubgraphOutput,
    TensorId, TensorMeta,
};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Graph Construction
// ============================================================================

/// Node in the operation DAG
#[derive(Debug, Clone)]
pub struct OpNode {
    pub op_id: OpId,
    pub fused: bool, // Already included in a subgraph
}

/// Build a directed acyclic graph of operations
pub fn build_op_dag(problem: &Problem) -> (DiGraph<OpNode, ()>, HashMap<OpId, NodeIndex>) {
    let mut graph = DiGraph::new();
    let mut op_to_node: HashMap<OpId, NodeIndex> = HashMap::new();

    // Add all ops as nodes
    for op_id in 0..problem.ops.len() {
        let node = graph.add_node(OpNode { op_id, fused: false });
        op_to_node.insert(op_id, node);
    }

    // Add edges based on tensor dependencies
    let tensor_meta = problem.build_tensor_meta();
    for (tensor_id, meta) in tensor_meta.iter().enumerate() {
        if let Some(producer) = meta.producer {
            for &consumer in &meta.consumers {
                let from = op_to_node[&producer];
                let to = op_to_node[&consumer];
                // Avoid duplicate edges
                if !graph.contains_edge(from, to) {
                    graph.add_edge(from, to, ());
                }
            }
        }
    }

    (graph, op_to_node)
}

// ============================================================================
// Fusion Engine - Aggressive Data-Flow Based Fusion
// ============================================================================

/// Check if a candidate op can be fused into the current subgraph.
/// A node can be fused if ALL its parent ops (producers of its inputs) are either:
/// 1. Already in the fused set, OR
/// 2. External graph inputs (no producer)
fn can_fuse_op(
    candidate: OpId,
    fused_set: &HashSet<OpId>,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> bool {
    let op = &problem.ops[candidate];

    for &input_id in &op.inputs {
        let meta = &tensor_meta[input_id];
        // If this input has a producer, that producer must be in fused_set
        if let Some(producer) = meta.producer {
            if !fused_set.contains(&producer) {
                return false; // Dependency not satisfied
            }
        }
        // If producer is None, it's an external graph input - that's fine
    }
    true
}

/// Check if fusing a Pointwise after a MatMul should be prioritized.
/// Returns true if candidate is Pointwise and its only parent is a MatMul in the fused set.
fn is_matmul_pointwise_fusion_candidate(
    candidate: OpId,
    fused_set: &HashSet<OpId>,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> bool {
    let op = &problem.ops[candidate];

    // Must be Pointwise
    if op.op_type != OpType::Pointwise {
        return false;
    }

    // Find all parent ops (producers of inputs)
    let mut parent_ops: Vec<OpId> = Vec::new();
    for &input_id in &op.inputs {
        if let Some(producer) = tensor_meta[input_id].producer {
            if !parent_ops.contains(&producer) {
                parent_ops.push(producer);
            }
        }
    }

    // Check if there's exactly one parent and it's a MatMul in the fused set
    if parent_ops.len() == 1 {
        let parent = parent_ops[0];
        if fused_set.contains(&parent) && problem.ops[parent].op_type == OpType::MatMul {
            return true;
        }
    }

    false
}

/// Collect direct successor ops (consumers of outputs) from the fused set.
/// Returns candidates sorted by priority: MatMul→Pointwise fusions first.
fn collect_fusion_candidates(
    fused_set: &HashSet<OpId>,
    available_ops: &HashSet<OpId>,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Vec<OpId> {
    let mut candidates: Vec<OpId> = Vec::new();
    let mut seen: HashSet<OpId> = HashSet::new();

    // Find all direct successors of ops in fused_set
    for &op_id in fused_set {
        let op = &problem.ops[op_id];
        for &output_id in &op.outputs {
            let meta = &tensor_meta[output_id];
            for &consumer in &meta.consumers {
                if available_ops.contains(&consumer)
                    && !fused_set.contains(&consumer)
                    && !seen.contains(&consumer)
                {
                    candidates.push(consumer);
                    seen.insert(consumer);
                }
            }
        }
    }

    // Sort candidates: prioritize MatMul→Pointwise fusion patterns
    candidates.sort_by(|&a, &b| {
        let a_priority = is_matmul_pointwise_fusion_candidate(a, fused_set, problem, tensor_meta);
        let b_priority = is_matmul_pointwise_fusion_candidate(b, fused_set, problem, tensor_meta);
        // true (high priority) comes before false
        b_priority.cmp(&a_priority)
    });

    candidates
}

/// Attempt to fuse operations aggressively using data-flow based expansion.
///
/// Strategy:
/// 1. Start with seed op
/// 2. Repeatedly expand to successors whose ALL parents are already fused
/// 3. Prioritize MatMul→Pointwise fusion patterns
/// 4. Stop when no more candidates fit in memory
fn try_fuse_from_seed(
    seed_op: OpId,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    available_ops: &HashSet<OpId>,
    _graph: &DiGraph<OpNode, ()>,
    _op_to_node: &HashMap<OpId, NodeIndex>,
) -> Vec<OpId> {
    let mut fused_ops = vec![seed_op];
    let mut fused_set: HashSet<OpId> = [seed_op].into_iter().collect();

    // Keep expanding until no more candidates can be added
    loop {
        // Collect all valid fusion candidates (direct successors)
        let candidates = collect_fusion_candidates(&fused_set, available_ops, problem, tensor_meta);

        if candidates.is_empty() {
            break;
        }

        let mut added_any = false;

        // Try to add each candidate in priority order
        for candidate in candidates {
            // Skip if already fused (may have been added in this iteration)
            if fused_set.contains(&candidate) {
                continue;
            }

            // Check data-flow constraint: all parents must be in fused_set or external
            if !can_fuse_op(candidate, &fused_set, problem, tensor_meta) {
                continue;
            }

            // Build test fusion to check memory
            let mut test_fusion: Vec<OpId> = fused_ops.clone();
            test_fusion.push(candidate);

            // Check if MatMul→Pointwise priority fusion
            let is_priority = is_matmul_pointwise_fusion_candidate(
                candidate,
                &fused_set,
                problem,
                tensor_meta,
            );

            // Validate memory fit
            if validate_memory_fit(&test_fusion, problem, &problem.native_granularity, tensor_meta) {
                fused_ops.push(candidate);
                fused_set.insert(candidate);
                added_any = true;

                // For priority fusions, continue immediately to next iteration
                // to pick up any following Pointwise ops
                if is_priority {
                    break;
                }
            } else if is_priority {
                // Even for priority fusions, try with reduced granularity
                let reduced = problem.native_granularity.halve();
                if validate_memory_fit(&test_fusion, problem, &reduced, tensor_meta) {
                    fused_ops.push(candidate);
                    fused_set.insert(candidate);
                    added_any = true;
                    break;
                }
            }
        }

        if !added_any {
            break;
        }
    }

    // Sort ops in topological order within the fused group
    topological_sort_ops(&fused_ops, problem, tensor_meta)
}

/// Sort ops in topological order based on their dependencies
fn topological_sort_ops(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Vec<OpId> {
    let ops_set: HashSet<OpId> = ops.iter().copied().collect();
    let mut sorted = Vec::with_capacity(ops.len());
    let mut visited: HashSet<OpId> = HashSet::new();
    let mut in_progress: HashSet<OpId> = HashSet::new();

    fn visit(
        op_id: OpId,
        ops_set: &HashSet<OpId>,
        problem: &Problem,
        tensor_meta: &[TensorMeta],
        visited: &mut HashSet<OpId>,
        in_progress: &mut HashSet<OpId>,
        sorted: &mut Vec<OpId>,
    ) {
        if visited.contains(&op_id) || !ops_set.contains(&op_id) {
            return;
        }
        if in_progress.contains(&op_id) {
            return; // Cycle detection (shouldn't happen in DAG)
        }

        in_progress.insert(op_id);

        // Visit all dependencies first
        let op = &problem.ops[op_id];
        for &input_id in &op.inputs {
            if let Some(producer) = tensor_meta[input_id].producer {
                if ops_set.contains(&producer) {
                    visit(producer, ops_set, problem, tensor_meta, visited, in_progress, sorted);
                }
            }
        }

        in_progress.remove(&op_id);
        visited.insert(op_id);
        sorted.push(op_id);
    }

    for &op_id in ops {
        visit(op_id, &ops_set, problem, tensor_meta, &mut visited, &mut in_progress, &mut sorted);
    }

    sorted
}

// ============================================================================
// Granularity Selection
// ============================================================================

/// Select optimal granularity for a subgraph, trying Split-K for MatMul chains
fn select_granularity(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Granularity {
    // First, check if native granularity works
    if validate_memory_fit(ops, problem, &problem.native_granularity, tensor_meta) {
        return problem.native_granularity.clone();
    }

    // Check if any op is MatMul - try Split-K first (more efficient than spatial tiling)
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());

    if has_matmul {
        // Try Split-K with native spatial granularity
        if let Some(split_k_gran) = find_split_k(ops, problem, &problem.native_granularity, tensor_meta) {
            return split_k_gran;
        }
    }

    // Fall back to spatial tiling
    find_fitting_granularity(ops, problem, tensor_meta)
}

/// Generate traversal order if beneficial (snake pattern for 2D tiles)
fn generate_traversal_order(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
) -> Option<Vec<i64>> {
    if ops.is_empty() {
        return None;
    }

    // Get output tensor dimensions from first op
    let first_op = &problem.ops[ops[0]];
    let output_tensor = first_op.outputs.first()
        .and_then(|&id| problem.tensors.get(id))?;

    let w_tiles = (output_tensor.width + granularity.width - 1) / granularity.width;
    let h_tiles = (output_tensor.height + granularity.height - 1) / granularity.height;

    // Only generate snake traversal if there are multiple tiles
    if w_tiles * h_tiles > 1 {
        Some(generate_snake_traversal(w_tiles, h_tiles))
    } else {
        None
    }
}

// ============================================================================
// Main Scheduler
// ============================================================================

/// Schedule the problem graph into optimized subgraphs
pub fn schedule(problem: &Problem) -> Solution {
    let tensor_meta = problem.build_tensor_meta();
    let (graph, op_to_node) = build_op_dag(problem);

    let mut subgraphs: Vec<Subgraph> = Vec::new();
    let mut scheduled: HashSet<OpId> = HashSet::new();
    let mut memory_state = MemoryState::new();

    // Process ops in topological order
    let mut topo = Topo::new(&graph);
    let mut topo_order: Vec<OpId> = Vec::new();

    while let Some(node) = topo.next(&graph) {
        topo_order.push(graph[node].op_id);
    }

    for seed_op in topo_order {
        if scheduled.contains(&seed_op) {
            continue;
        }

        // Find available ops (not yet scheduled)
        let available: HashSet<OpId> = (0..problem.ops.len())
            .filter(|op| !scheduled.contains(op))
            .collect();

        // Try aggressive fusion from this seed
        let fused_ops = try_fuse_from_seed(
            seed_op,
            problem,
            &tensor_meta,
            &available,
            &graph,
            &op_to_node,
        );

        // Select optimal granularity
        let granularity = select_granularity(&fused_ops, problem, &tensor_meta);

        // Generate traversal order
        let traversal_order = generate_traversal_order(&fused_ops, problem, &granularity);

        // Determine which output tensors to keep resident
        let remaining_ops: Vec<OpId> = (0..problem.ops.len())
            .filter(|op| !scheduled.contains(op) && !fused_ops.contains(op))
            .collect();

        // Collect output tensors from this subgraph
        let output_tensors: Vec<TensorId> = fused_ops
            .iter()
            .flat_map(|&op_id| problem.ops[op_id].outputs.iter().copied())
            .collect();

        // Calculate available capacity for retention (after working set)
        let available_capacity = compute_available_retention_capacity(
            &fused_ops,
            problem,
            &granularity,
            &tensor_meta,
        );

        // Analyze which tensors to retain (prioritizes immediate consumers)
        let tensors_to_retain = analyze_tensor_residency(
            &output_tensors,
            &remaining_ops,
            problem,
            &tensor_meta,
            available_capacity,
        );

        // Calculate subgraph latency
        let latency = compute_subgraph_latency(
            &fused_ops,
            problem,
            &granularity,
            &tensor_meta,
            &memory_state,
            &tensors_to_retain,
            traversal_order.is_some(),
        );

        // Create subgraph
        let subgraph = Subgraph {
            ops: fused_ops.clone(),
            tensors_to_retain: tensors_to_retain.clone(),
            granularity: GranularityOutput::from(&granularity),
            traversal_order,
            subgraph_latency: latency,
        };

        subgraphs.push(subgraph);

        // Mark ops as scheduled
        for op_id in &fused_ops {
            scheduled.insert(*op_id);
        }

        // Evict tensors that have been fully consumed (no remaining consumers)
        let tensors_to_evict: Vec<TensorId> = memory_state
            .resident_tensors
            .iter()
            .copied()
            .filter(|&tensor_id| {
                let meta = &tensor_meta[tensor_id];
                // Evict if all consumers have been scheduled
                meta.consumers.iter().all(|c| scheduled.contains(c))
            })
            .collect();

        for tensor_id in tensors_to_evict {
            memory_state.evict(tensor_id);
        }

        // Update memory state: mark newly retained tensors as resident
        for tensor_id in tensors_to_retain {
            memory_state.mark_resident(tensor_id);
        }
    }

    Solution {
        subgraphs: subgraphs.iter().map(SubgraphOutput::from).collect(),
    }
}

// ============================================================================
// Advanced Scheduling Strategies
// ============================================================================

/// Try to improve the schedule by exploring alternative fusion patterns
pub fn optimize_schedule(initial: Solution, problem: &Problem) -> Solution {
    // TODO: Implement local search / simulated annealing for better solutions
    // For now, return the greedy solution
    initial
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Tensor};

    fn make_test_problem() -> Problem {
        Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 },
                Tensor { width: 128, height: 128 },
                Tensor { width: 128, height: 128 },
            ],
            ops: vec![
                Op {
                    op_type: OpType::Pointwise,
                    inputs: vec![0],
                    outputs: vec![1],
                    base_cost: 100,
                },
                Op {
                    op_type: OpType::Pointwise,
                    inputs: vec![1],
                    outputs: vec![2],
                    base_cost: 10,
                },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_build_dag() {
        let problem = make_test_problem();
        let (graph, op_to_node) = build_op_dag(&problem);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_schedule_fuses_ops() {
        let problem = make_test_problem();
        let solution = schedule(&problem);

        // Should fuse both ops into one subgraph
        assert_eq!(solution.subgraphs.len(), 1);
        assert_eq!(solution.subgraphs[0].ops.len(), 2);
    }
}

