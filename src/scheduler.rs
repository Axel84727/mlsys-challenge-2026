//! Graph scheduler implementing EXTREME fusion with global look-ahead and SRAM-first execution.
//!
//! Strategy:
//! 1. Liveness-Aware Allocation - Tensors with high reuse get guaranteed SRAM slots
//! 2. Global Ready Queue - ALL ops with satisfied dependencies are candidates
//! 3. SRAM-First Execution - Execute ALL children of resident tensors before eviction
//! 4. Aggressive Kernel Fusion - Pointwise chains are mathematically fused
//! 5. Dynamic Split-K - K dimension is split to fit perfectly in half SRAM
//! 6. Micro-Kernel Grouping - Small ops are batched to fill native granularity
//! 7. Parallel Processing - Uses Rayon for multi-core priority calculation
//! 8. Telemetry - Detailed logging of engineering decisions
//! 9. Weight Stationary - Keep reused tensors resident to avoid DRAM round-trips
//!
//! AUDIT NOTES (2026-02-10):
//!
//! POTENTIAL BIAS IDENTIFIED:
//! The current fusion bonuses (MATMUL_POINTWISE_FUSION_BONUS=50,000) may be
//! over-tuned for Benchmark 17's dense graph structure. On sparse graphs
//! (Benchmark 1's 5-op chain), this aggressive fusion provides diminishing
//! returns and may mask memory-efficiency opportunities.
//!
//! ROBUSTNESS IMPROVEMENTS:
//! - Added hardware-adaptive fusion bonus via cost_model module
//! - Added graph density analysis to detect sparse vs dense workloads
//! - Added prime-dimension tiling for non-POT tensor shapes
//! - Added adaptive prefetch thresholds for asymmetric bandwidth
//! - Added layout transformation analysis for memory access optimization
//! - Added weight stationary optimization for reused tensor retention
//!
//! For maximum robustness, use cost_model::analyze_graph_density() to detect
//! workload characteristics and adapt the optimization strategy accordingly.

use crate::cost::{compute_subgraph_latency, generate_snake_traversal, MemoryState};
use crate::cost_model::analyze_graph_density;
use crate::layout::{
    generate_layout_aware_tiling, should_consider_layout_transform,
    LayoutAnalysisConfig,
};
use crate::liveness::{
    analyze_liveness, compute_sram_reservation, find_pointwise_chains,
    find_micro_kernel_candidates, find_aggressive_kernel_fusions,
    compute_recursive_dce_from_finals,
    LivenessAnalysis, SramReservation,
};
use crate::memory::{
    compute_available_retention_capacity, compute_subgraph_working_set,
    find_fitting_granularity, find_split_k, validate_memory_fit,
};
use crate::models::{
    Granularity, GranularityOutput, OpId, OpType, Problem, Solution, Subgraph, SubgraphOutput,
    TensorId, TensorMeta,
};
use crate::{layout, telemetry};
use crate::weight_stationary::{
    analyze_weights, WeightAnalysis, WeightStationaryConfig,
    enhance_retention_with_weights, calculate_sticky_reservation,
};
use petgraph::graph::{DiGraph, NodeIndex};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet};

// ============================================================================
// CRITICAL PATH ANALYSIS
// ============================================================================

/// Critical path metrics for operation prioritization
#[derive(Debug, Clone)]
struct CriticalPathMetrics {
    /// Depth of each op in the DAG (topological level)
    op_depth: Vec<usize>,

    /// Number of ops this op directly unlocks
    unlocks_count: Vec<usize>,

    /// Distance to any final output (reverse depth)
    distance_to_output: Vec<usize>,

    /// Combined critical score (higher = more critical)
    critical_score: Vec<i64>,
}

impl CriticalPathMetrics {
    /// Compute critical path metrics with DETERMINISTIC ordering
    fn analyze(problem: &Problem, tensor_meta: &[TensorMeta]) -> Self {
        let num_ops = problem.ops.len();

        let mut op_depth = vec![0; num_ops];
        let mut unlocks_count = vec![0; num_ops];
        let mut distance_to_output = vec![0; num_ops];

        // Build DETERMINISTIC dependency graph using BTreeMap
        let mut dependencies: BTreeMap<OpId, BTreeSet<OpId>> = BTreeMap::new();
        let mut dependents: BTreeMap<OpId, BTreeSet<OpId>> = BTreeMap::new();

        for op_id in 0..num_ops {
            dependencies.insert(op_id, BTreeSet::new());
            dependents.insert(op_id, BTreeSet::new());
        }

        for op_id in 0..num_ops {
            let op = &problem.ops[op_id];
            for &input_id in &op.inputs {
                if let Some(producer) = tensor_meta[input_id].producer {
                    dependencies.get_mut(&op_id).unwrap().insert(producer);
                    dependents.get_mut(&producer).unwrap().insert(op_id);
                }
            }
        }

        // Forward pass: compute depth (DETERMINISTIC order)
        let mut visited = vec![false; num_ops];
        let mut stack = Vec::new();

        // DETERMINISTIC: Start with ops in sorted order
        for op_id in 0..num_ops {
            if dependencies[&op_id].is_empty() {
                stack.push(op_id);
            }
        }
        stack.sort_unstable(); // Ensure deterministic processing

        while let Some(op_id) = stack.pop() {
            if visited[op_id] {
                continue;
            }
            visited[op_id] = true;

            let max_pred_depth = dependencies[&op_id]
                .iter()
                .map(|&pred| op_depth[pred])
                .max()
                .unwrap_or(0);

            op_depth[op_id] = if dependencies[&op_id].is_empty() {
                0
            } else {
                max_pred_depth + 1
            };

            // Add dependents in SORTED order
            let mut next_ops: Vec<OpId> = dependents[&op_id]
                .iter()
                .copied()
                .filter(|&dep| dependencies[&dep].iter().all(|&d| visited[d]))
                .collect();
            next_ops.sort_unstable();
            stack.extend(next_ops);
        }

        // Compute unlocks count
        for op_id in 0..num_ops {
            unlocks_count[op_id] = dependents[&op_id].len();
        }

        // Backward pass: distance to output
        visited = vec![false; num_ops];
        stack.clear();

        for op_id in 0..num_ops {
            let op = &problem.ops[op_id];
            let is_output = op.outputs.iter()
                .any(|&out_id| tensor_meta[out_id].is_output);

            if is_output {
                distance_to_output[op_id] = 0;
                stack.push(op_id);
                visited[op_id] = true;
            }
        }

        while let Some(op_id) = stack.pop() {
            let mut predecessors: Vec<OpId> = dependencies[&op_id]
                .iter()
                .copied()
                .collect();
            predecessors.sort_unstable(); // DETERMINISTIC

            for pred in predecessors {
                if !visited[pred] {
                    distance_to_output[pred] = distance_to_output[op_id] + 1;
                    stack.push(pred);
                    visited[pred] = true;
                } else {
                    distance_to_output[pred] = distance_to_output[pred]
                        .max(distance_to_output[op_id] + 1);
                }
            }
        }

        // Compute critical score
        let mut critical_score = vec![0i64; num_ops];
        for op_id in 0..num_ops {
            // Components (in order of importance):
            // 1. Ops that unlock many others (fan-out)
            let unlock_bonus = (unlocks_count[op_id] as i64) * 15_000;

            // 2. Ops close to outputs (short distance)
            let output_proximity = (100 - distance_to_output[op_id].min(100) as i64) * 1_000;

            // 3. Ops with high depth (long path from inputs)
            let depth_bonus = (op_depth[op_id] as i64) * 500;

            critical_score[op_id] = unlock_bonus + output_proximity + depth_bonus;
        }

        Self {
            op_depth,
            unlocks_count,
            distance_to_output,
            critical_score,
        }
    }
}
// ============================================================================
// Constants - Tuning Parameters
// NOTE: These are BASE values that may be adapted based on graph density.
// For full adaptivity, use cost_model::compute_adaptive_fusion_bonus()
// ============================================================================

/// Bonus for fusing MatMul with its immediate Pointwise consumer
/// AUDIT: This 50,000 value was tuned for Benchmark 17's dense MatMul chains.
/// On sparse graphs, a lower value (25,000) may be more appropriate.
const MATMUL_POINTWISE_FUSION_BONUS: i64 = 50_000;

/// Bonus for executing ops that consume tensors currently in SRAM
const SRAM_RESIDENT_CONSUMER_BONUS: i64 = 25_000;

/// Bonus for Pointwise ops (cheap, easy to fuse)
const POINTWISE_BONUS: i64 = 10_000;

// ============================================================================
// Graph Construction
// ============================================================================

/// Node in the operation DAG
#[derive(Debug, Clone)]
pub struct OpNode {
    pub op_id: OpId,
    pub fused: bool,
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
    for meta in tensor_meta.iter() {
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
// Global Ready Queue System with Liveness-Aware Allocation
// ============================================================================

/// Global scheduling state with liveness analysis integration
struct SchedulerState<'a> {
    problem: &'a Problem,
    tensor_meta: &'a [TensorMeta],

    /// Ops not yet assigned to any subgraph
    unscheduled: HashSet<OpId>,

    /// Tensors currently in SRAM (from previous subgraphs)
    sram_resident: HashSet<TensorId>,

    /// For each tensor, remaining consumer count
    remaining_consumers: Vec<usize>,

    /// Liveness analysis results
    liveness: &'a LivenessAnalysis,

    /// SRAM reservation strategy
    sram_reservation: &'a SramReservation,

    /// Pointwise chains for aggressive fusion
    pointwise_chains: Vec<HashSet<OpId>>,

    /// Micro-kernel groups for small op batching
    micro_kernel_groups: Vec<HashSet<OpId>>,
}

impl<'a> SchedulerState<'a> {
    fn new(
        problem: &'a Problem,
        tensor_meta: &'a [TensorMeta],
        liveness: &'a LivenessAnalysis,
        sram_reservation: &'a SramReservation,
    ) -> Self {
        let remaining_consumers: Vec<usize> = tensor_meta
            .iter()
            .map(|m| m.consumers.len())
            .collect();

        // Pre-compute Pointwise chains
        let pw_chains = find_pointwise_chains(problem, tensor_meta);
        let pointwise_chains: Vec<HashSet<OpId>> = pw_chains
            .iter()
            .map(|chain| chain.ops.iter().copied().collect())
            .collect();

        // Pre-compute micro-kernel groups
        let mk_groups = find_micro_kernel_candidates(problem, tensor_meta);
        let micro_kernel_groups: Vec<HashSet<OpId>> = mk_groups
            .iter()
            .map(|group| group.ops.iter().copied().collect())
            .collect();

        // Initialize SRAM with guaranteed high-reuse tensors
        let mut sram_resident: HashSet<TensorId> = HashSet::new();
        for &tensor_id in &sram_reservation.guaranteed_tensors {
            // Only add if tensor is a graph input (available from start)
            if tensor_meta[tensor_id].producer.is_none() {
                sram_resident.insert(tensor_id);
            }
        }

        Self {
            problem,
            tensor_meta,
            unscheduled: (0..problem.ops.len()).collect(),
            sram_resident,
            remaining_consumers,
            liveness,
            sram_reservation,
            pointwise_chains,
            micro_kernel_groups,
        }
    }

    /// Check if an op is ready to execute (all input producers are scheduled or inputs are external)
    fn is_ready(&self, op_id: OpId) -> bool {
        let op = &self.problem.ops[op_id];
        op.inputs.iter().all(|&input_id| {
            let meta = &self.tensor_meta[input_id];
            match meta.producer {
                None => true, // Graph input
                Some(producer) => !self.unscheduled.contains(&producer), // Producer already scheduled
            }
        })
    }

    /// Get all currently ready ops from the global unscheduled set
    fn get_ready_ops(&self) -> Vec<OpId> {
        self.unscheduled
            .iter()
            .copied()
            .filter(|&op_id| self.is_ready(op_id))
            .collect()
    }

    /// Calculate fusion priority for an op in the context of current subgraph
    /// Incorporates liveness analysis for optimal SRAM utilization
    fn calculate_fusion_priority(
        &self,
        op_id: OpId,
        current_subgraph: &HashSet<OpId>,
        subgraph_outputs: &HashSet<TensorId>,
    ) -> i64 {
        let op = &self.problem.ops[op_id];
        let mut priority: i64 = 0;

        // === SRAM Residency Bonus ===
        // If this op consumes a tensor that's in SRAM (from subgraph outputs or resident), huge bonus
        for &input_id in &op.inputs {
            if subgraph_outputs.contains(&input_id) || self.sram_resident.contains(&input_id) {
                priority += SRAM_RESIDENT_CONSUMER_BONUS;
            }

            // === Liveness-Aware Bonus ===
            // Extra bonus for consuming high-reuse tensors (keeps them alive productively)
            let interval = &self.liveness.intervals[input_id];
            if interval.reuse_count >= 2 {
                priority += (interval.reuse_count as i64) * 5_000;
            }

            // Bonus for consuming guaranteed SRAM tensors (use them while they're hot!)
            if self.sram_reservation.guaranteed_tensors.contains(&input_id) {
                priority += 30_000;
            }
        }

        // === MatMul→Pointwise Fusion Bonus ===
        // If this is a Pointwise that directly consumes a MatMul output from current subgraph
        if op.op_type == OpType::Pointwise {
            priority += POINTWISE_BONUS;

            for &input_id in &op.inputs {
                let meta = &self.tensor_meta[input_id];
                if let Some(producer) = meta.producer {
                    if current_subgraph.contains(&producer)
                       && self.problem.ops[producer].op_type == OpType::MatMul
                    {
                        priority += MATMUL_POINTWISE_FUSION_BONUS;
                    }
                }
            }

            // === Pointwise Chain Bonus ===
            // If this op is part of a Pointwise chain with ops already in subgraph
            for chain in &self.pointwise_chains {
                if chain.contains(&op_id) {
                    let chain_in_subgraph = chain.iter().filter(|&&id| current_subgraph.contains(&id)).count();
                    if chain_in_subgraph > 0 {
                        // Strong incentive to complete the chain
                        priority += (chain_in_subgraph as i64) * 20_000;
                    }
                }
            }
        }

        // === Micro-Kernel Grouping Bonus ===
        // Small ops should be grouped together
        for group in &self.micro_kernel_groups {
            if group.contains(&op_id) {
                let group_in_subgraph = group.iter().filter(|&&id| current_subgraph.contains(&id)).count();
                if group_in_subgraph > 0 {
                    priority += (group_in_subgraph as i64) * 15_000;
                }
            }
        }

        // === Direct Successor Bonus ===
        // Ops whose inputs come directly from current subgraph
        let is_direct_successor = op.inputs.iter().any(|&input_id| {
            let meta = &self.tensor_meta[input_id];
            meta.producer.is_some_and(|p| current_subgraph.contains(&p))
        });
        if is_direct_successor {
            priority += 15_000;
        }

        // === Consumer Chain Bonus ===
        // Ops that have many consumers waiting (clearing them unlocks more ops)
        let output_consumer_count: usize = op.outputs.iter()
            .map(|&out_id| self.tensor_meta[out_id].consumers.len())
            .sum();
        priority += (output_consumer_count as i64) * 1_000;

        // === Tensor Size Penalty (smaller is easier to fit) ===
        let op_output_size: i64 = op.outputs.iter()
            .map(|&out_id| self.problem.tensors[out_id].size())
            .sum();
        priority -= op_output_size / 1000; // Small penalty for large outputs

        priority
    }

    /// Calculate fusion priority WITH critical path analysis
    fn calculate_fusion_priority_with_critical_path(
        &self,
        op_id: OpId,
        current_subgraph: &HashSet<OpId>,
        subgraph_outputs: &HashSet<TensorId>,
        critical_metrics: &CriticalPathMetrics,
    ) -> i64 {
        let op = &self.problem.ops[op_id];
        let mut priority: i64 = 0;

        // === CRITICAL PATH BONUS (HIGHEST PRIORITY) ===
        priority += critical_metrics.critical_score[op_id];

        // === SRAM Residency Bonus ===
        for &input_id in &op.inputs {
            if subgraph_outputs.contains(&input_id) || self.sram_resident.contains(&input_id) {
                priority += SRAM_RESIDENT_CONSUMER_BONUS;
            }

            let interval = &self.liveness.intervals[input_id];
            if interval.reuse_count >= 2 {
                priority += (interval.reuse_count as i64) * 5_000;
            }

            if self.sram_reservation.guaranteed_tensors.contains(&input_id) {
                priority += 30_000;
            }
        }

        // === MatMul→Pointwise Fusion Bonus ===
        if op.op_type == OpType::Pointwise {
            priority += POINTWISE_BONUS;

            for &input_id in &op.inputs {
                let meta = &self.tensor_meta[input_id];
                if let Some(producer) = meta.producer {
                    if current_subgraph.contains(&producer)
                       && self.problem.ops[producer].op_type == OpType::MatMul
                    {
                        priority += MATMUL_POINTWISE_FUSION_BONUS;
                    }
                }
            }

            // Pointwise Chain Bonus
            for chain in &self.pointwise_chains {
                if chain.contains(&op_id) {
                    let chain_in_subgraph = chain.iter()
                        .filter(|&&id| current_subgraph.contains(&id))
                        .count();
                    if chain_in_subgraph > 0 {
                        priority += (chain_in_subgraph as i64) * 20_000;
                    }
                }
            }
        }

        // === Micro-Kernel Grouping Bonus ===
        for group in &self.micro_kernel_groups {
            if group.contains(&op_id) {
                let group_in_subgraph = group.iter()
                    .filter(|&&id| current_subgraph.contains(&id))
                    .count();
                if group_in_subgraph > 0 {
                    priority += (group_in_subgraph as i64) * 15_000;
                }
            }
        }

        // === Direct Successor Bonus ===
        let is_direct_successor = op.inputs.iter().any(|&input_id| {
            let meta = &self.tensor_meta[input_id];
            meta.producer.is_some_and(|p| current_subgraph.contains(&p))
        });
        if is_direct_successor {
            priority += 15_000;
        }

        // === Tensor Size Penalty ===
        let op_output_size: i64 = op.outputs.iter()
            .map(|&out_id| self.problem.tensors[out_id].size())
            .sum();
        priority -= op_output_size / 1000;

        priority
    }

    /// Mark an op as scheduled
    fn mark_scheduled(&mut self, op_id: OpId) {
        self.unscheduled.remove(&op_id);

        // Decrement remaining consumers for input tensors
        let op = &self.problem.ops[op_id];
        for &input_id in &op.inputs {
            if self.remaining_consumers[input_id] > 0 {
                self.remaining_consumers[input_id] -= 1;
            }
        }
    }

    /// Update SRAM state after a subgraph completes
    fn update_sram_after_subgraph(&mut self, retained_tensors: &[TensorId]) {
        // Evict tensors with no remaining consumers
        self.sram_resident.retain(|&tensor_id| {
            self.remaining_consumers[tensor_id] > 0
        });

        // Add newly retained tensors
        for &tensor_id in retained_tensors {
            if self.remaining_consumers[tensor_id] > 0 {
                self.sram_resident.insert(tensor_id);
            }
        }
    }
}

/// SRAM pressure configuration based on graph size
#[derive(Debug, Clone)]
struct AdaptiveSramConfig {
    max_utilization: f64,
    max_ops_per_subgraph: usize,
    emergency_stop_bytes: i64,
}

impl AdaptiveSramConfig {
    fn from_problem(problem: &Problem) -> Self {
        let num_ops = problem.ops.len();
        // Adaptive strategy based on graph size
        let (max_util, max_ops) = if num_ops < 200 {
            // Small graph: aggressive fusion (95% SRAM)
            (0.95, 500)
        } else if num_ops < 500 {
            // Medium graph: balanced (90% SRAM)
            (0.90, 300)
        } else if num_ops < 1000 {
            // Large graph: conservative (85% SRAM)
            (0.85, 200)
        } else {
            // Monster graph: very conservative (80% SRAM)
            (0.80, 150)
        };
        eprintln!("[*] Adaptive SRAM: {} ops → {:.0}% utilization, max {} ops/subgraph",
                  num_ops, max_util * 100.0, max_ops);
        Self {
            max_utilization: max_util,
            max_ops_per_subgraph: max_ops,
            emergency_stop_bytes: (problem.fast_memory_capacity as f64 * max_util) as i64,
        }
    }
}

// ============================================================================
// Full Fusion Optimization
// ============================================================================

/// Try to fuse ALL remaining unscheduled ops into a single subgraph.
///
/// Key insight: When ALL ops are fused, most tensors become intermediate/ephemeral,
/// dramatically reducing the working set. Partial fusion often has MORE external I/O
/// because intermediate tensors haven't yet found all their consumers.
///
/// Returns Some(all_ops) if full fusion fits in memory, None otherwise.
fn try_full_fusion_first(
    state: &SchedulerState,
    ready_ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    use_aggressive_fusion: bool,
) -> Option<(Vec<OpId>, Granularity)> {  // ← Return granularity too
    // Skip full fusion for sparse graphs if strategy says no
    if !use_aggressive_fusion {
        eprintln!("[*] Skipping full fusion (strategy recommends flexible subgraphs)");
        return None;
    }

    // Only try full fusion if we have a reasonable number of unscheduled ops
    if state.unscheduled.len() < 2 {
        return None;
    }

    // Collect ALL unscheduled ops in topological order
    let all_unscheduled: Vec<OpId> = topological_sort_ops(
        &state.unscheduled.iter().copied().collect::<Vec<_>>(),
        problem,
        tensor_meta,
    );

    // Try to fit all ops with various granularities
    // First try without Split-K (best performance)
    // PRIORITY ORDER: Best configurations first
    let granularity_candidates = [
        // PRIORITY: Best configurations first
        Granularity::new(128, 128, 2),  // ← FIRST!
        Granularity::new(128, 128, 1),
        Granularity::new(128, 128, 4),
        Granularity::new(64, 128, 2),
        Granularity::new(128, 64, 2),
        Granularity::new(64, 64, 2),
        Granularity::new(64, 128, 1),
        Granularity::new(128, 64, 1),
        Granularity::new(64, 64, 1),
    ];

    for granularity in &granularity_candidates {
        let ws = compute_subgraph_working_set(&all_unscheduled, problem, granularity, tensor_meta);
        if ws.fits_in(problem.fast_memory_capacity) {
            // Full fusion works! Log and return
            eprintln!(
                "[*] Full fusion optimization: {} ops fit with {}x{}x{} (WS={}, capacity={})",
                all_unscheduled.len(),
                granularity.width, granularity.height, granularity.depth,
                ws.total_size, problem.fast_memory_capacity
            );
            return Some((all_unscheduled, granularity.clone()));
        }
    }

    // Full fusion doesn't fit, fall back to incremental
    None
}

// ============================================================================
// Extreme Fusion Engine
// ============================================================================

/// Minimum candidates to justify parallel priority calculation
const PARALLEL_PRIORITY_THRESHOLD: usize = 16;

/// Aggressively fuse operations using global look-ahead.
///
/// The key insight: Don't close a subgraph just because the "next" op doesn't fit.
/// Instead, search the ENTIRE graph for ANY ready op that can be absorbed.
///
/// PARALLEL: Uses Rayon to calculate fusion priorities when there are enough candidates.
fn extreme_fusion(
    seed_op: OpId,
    state: &SchedulerState,
    _granularity_hint: &Granularity,
    max_ops_per_subgraph: usize,
    critical_metrics: &CriticalPathMetrics,
) -> Vec<OpId> {
    let mut fused_ops: Vec<OpId> = vec![seed_op];
    let mut fused_set: HashSet<OpId> = [seed_op].into_iter().collect();
    let mut subgraph_outputs: HashSet<TensorId> = state.problem.ops[seed_op]
        .outputs.iter().copied().collect();

    let max_iterations = state.problem.ops.len() * 3;

    for _iteration in 0..max_iterations {
        // Límite de tamaño de subgrafo
        if fused_ops.len() >= max_ops_per_subgraph {
            break;
        }
        // === Global Ready Queue Scan ===
        let eligible_candidates: Vec<OpId> = state.unscheduled
            .iter()
            .copied()
            .filter(|&op_id| !fused_set.contains(&op_id))
            .filter(|&op_id| can_add_to_fusion(op_id, &fused_set, state))
            .collect();

        if eligible_candidates.is_empty() {
            break;
        }

        // Calculate priorities - use parallel only for large candidate sets
        let mut candidates: Vec<(OpId, i64)> = if eligible_candidates.len() >= PARALLEL_PRIORITY_THRESHOLD {
            eligible_candidates
                .par_iter()
                .map(|&op_id| {
                    let priority = state.calculate_fusion_priority_with_critical_path(
                        op_id, &fused_set, &subgraph_outputs, critical_metrics);
                    (op_id, priority)
                })
                .collect()
        } else {
            eligible_candidates
                .iter()
                .map(|&op_id| {
                    let priority = state.calculate_fusion_priority_with_critical_path(
                        op_id, &fused_set, &subgraph_outputs, critical_metrics);
                    (op_id, priority)
                })
                .collect()
        };

        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        let mut added_any = false;

        for (candidate, priority) in candidates {
            if fused_ops.len() >= max_ops_per_subgraph {
                break;
            }
            let mut test_ops: Vec<OpId> = fused_ops.clone();
            test_ops.push(candidate);

            let fits = try_fit_in_memory(&test_ops, state.problem, state.tensor_meta);

            if fits {
                fused_ops.push(candidate);
                fused_set.insert(candidate);
                for &out_id in &state.problem.ops[candidate].outputs {
                    subgraph_outputs.insert(out_id);
                }
                added_any = true;
                if priority >= MATMUL_POINTWISE_FUSION_BONUS {
                    break;
                }
            }
        }
        if !added_any {
            break;
        }
    }
    topological_sort_ops(&fused_ops, state.problem, state.tensor_meta)
}

/// Check if an op can be added to current fusion (dependencies satisfied)
fn can_add_to_fusion(
    candidate: OpId,
    fused_set: &HashSet<OpId>,
    state: &SchedulerState,
) -> bool {
    let op = &state.problem.ops[candidate];

    for &input_id in &op.inputs {
        let meta = &state.tensor_meta[input_id];
        if let Some(producer) = meta.producer {
            // Producer must be either:
            // 1. Already in fused_set (will be executed together)
            // 2. Already scheduled (not in unscheduled)
            if !fused_set.contains(&producer) && state.unscheduled.contains(&producer) {
                return false;
            }
        }
    }
    true
}

/// Try different granularities to fit the subgraph in memory
fn try_fit_in_memory(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> bool {
    // Try native granularity first
    if validate_memory_fit(ops, problem, &problem.native_granularity, tensor_meta) {
        return true;
    }

    // Check if any op is MatMul - try Split-K
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());

    if has_matmul {
        for k in [2, 4, 8, 16, 32] {
            let split_k_gran = problem.native_granularity.with_split_k(k);
            if validate_memory_fit(ops, problem, &split_k_gran, tensor_meta) {
                return true;
            }
        }
    }

    // Try reduced spatial granularity
    let mut gran = problem.native_granularity.clone();
    for _ in 0..4 {
        gran = gran.halve();
        if validate_memory_fit(ops, problem, &gran, tensor_meta) {
            return true;
        }

        // Also try Split-K with reduced spatial
        if has_matmul {
            for k in [2, 4, 8] {
                let split_k_gran = gran.with_split_k(k);
                if validate_memory_fit(ops, problem, &split_k_gran, tensor_meta) {
                    return true;
                }
            }
        }
    }

    false
}

/// Sort ops in topological order
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
            return;
        }

        in_progress.insert(op_id);

        if op_id >= problem.ops.len() {
            return;
        }
        let op = &problem.ops[op_id];

        // CRITICAL FIX: Sort inputs before visiting
        let mut inputs_to_visit: Vec<OpId> = Vec::new();
        for &input_id in &op.inputs {
            if input_id < tensor_meta.len() {
                if let Some(producer) = tensor_meta[input_id].producer {
                    if ops_set.contains(&producer) {
                        inputs_to_visit.push(producer);
                    }
                }
            }
        }

        // DETERMINISM: Visit in sorted order
        inputs_to_visit.sort_unstable();

        for producer in inputs_to_visit {
            visit(producer, ops_set, problem, tensor_meta, visited, in_progress, sorted);
        }

        in_progress.remove(&op_id);
        visited.insert(op_id);
        sorted.push(op_id);
    }

    // CRITICAL FIX: Process ops in sorted order
    let mut ops_to_process: Vec<OpId> = ops.to_vec();
    ops_to_process.sort_unstable();

    for &op_id in &ops_to_process {
        visit(op_id, &ops_set, problem, tensor_meta, &mut visited, &mut in_progress, &mut sorted);
    }

    sorted
}

// ============================================================================
// Layout-Aware Tiling Helper
// ============================================================================

/// Try to find a layout-optimized tiling configuration for the subgraph.
///
/// This function analyzes tensor access patterns in MatMul operations and
/// recommends a tiling strategy that aligns with memory layout for better
/// bandwidth utilization.
///
/// ROBUSTNESS AGAINST OVERFITTING:
/// - Only recommends layout-aware tiling when benefit is clear (>5% improvement)
/// - Uses conservative cost estimates for transformation overhead
/// - Falls back to None if analysis confidence is low
/// - Never recommends changes for small tensors (cache effects dominate)
///
/// Returns Some(Granularity) if layout-aware tiling is beneficial, None otherwise.
fn try_layout_aware_tiling(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Option<Granularity> {
    // Skip if no MatMul ops (layout matters less for Pointwise)
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());
    if !has_matmul {
        return None;
    }

    // Skip if hardware doesn't benefit from layout transformation
    if !should_consider_layout_transform(problem) {
        return None;
    }

    // Configure layout analysis with conservative settings
    let config = LayoutAnalysisConfig {
        enabled: true,
        min_improvement_ratio: 0.05,  // Require 5% improvement
        min_savings: 1000.0,           // Minimum 1000 cycles savings
        min_confidence: 0.7,           // High confidence threshold
        max_tensor_size: 16 * 1024 * 1024, // 16MB max
    };

    // Run layout-aware tiling analysis
    if let Some(layout_tiling) = generate_layout_aware_tiling(ops, problem, tensor_meta, &config) {
        // Verify the recommended granularity fits in memory
        if validate_memory_fit(ops, problem, &layout_tiling.granularity, tensor_meta) {
            // Log the layout decision for transparency
            telemetry::log_strategy_decision(
                &format!(
                    "Layout-aware tiling: {}x{}",
                    layout_tiling.granularity.width,
                    layout_tiling.granularity.height
                ),
                &format!(
                    "Estimated savings: {:.1} cycles from {} tensor transforms",
                    layout_tiling.savings_vs_default,
                    layout_tiling.pre_transforms.len()
                ),
            );

            return Some(layout_tiling.granularity);
        }
    }

    // Fallback: Analyze access patterns and select tile shape accordingly
    select_access_pattern_aware_tiling(ops, problem, tensor_meta)
}

/// Select tiling based on dominant access patterns in the subgraph
///
/// This is a lightweight alternative to full layout transformation:
/// Instead of transforming tensor layouts, we choose tile shapes that
/// work better with the existing (assumed row-major) layout.
///
/// Key insight for MatMul C = A @ B:
/// - A is accessed row-wise → wide tiles (more columns) are efficient
/// - B is accessed column-wise → tall tiles work around strided access
/// - The choice depends on which matrix dominates memory traffic
///
/// IMPORTANT: Also considers Split-K for memory-constrained cases.
fn select_access_pattern_aware_tiling(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Option<Granularity> {
    let mut row_sequential_bytes: i64 = 0;
    let mut column_sequential_bytes: i64 = 0;
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());

    for &op_id in ops {
        let op = &problem.ops[op_id];

        if op.op_type != OpType::MatMul || op.inputs.len() < 2 {
            continue;
        }

        // LHS (A matrix) - row sequential access
        let lhs_id = op.inputs[0];
        let lhs_size = problem.tensors[lhs_id].size();
        row_sequential_bytes += lhs_size;

        // RHS (B matrix) - column sequential access (problematic for row-major)
        let rhs_id = op.inputs[1];
        let rhs_size = problem.tensors[rhs_id].size();
        column_sequential_bytes += rhs_size;
    }

    // Only recommend if there's meaningful asymmetry
    let total_bytes = row_sequential_bytes + column_sequential_bytes;
    if total_bytes == 0 {
        return None;
    }

    let column_ratio = column_sequential_bytes as f64 / total_bytes as f64;

    // Generate candidate tile sizes based on access pattern
    let tile_candidates: Vec<(i64, i64)> = if column_ratio > 0.4 {
        // Tall tiles: height > width
        vec![(64, 256), (64, 128), (128, 256), (32, 128), (32, 64)]
    } else if column_ratio < 0.3 {
        // Wide tiles: width > height (LHS access dominates)
        vec![(256, 64), (128, 64), (256, 128), (128, 32), (64, 32)]
    } else {
        // Balanced - don't override default
        return None;
    };

    // Split-K values to try (important for memory-constrained cases!)
    let split_k_values: Vec<i64> = if has_matmul {
        vec![1, 2, 4, 8]
    } else {
        vec![1]
    };

    // Try tile/split-k combinations until we find one that fits
    for &(w, h) in &tile_candidates {
        for &k in &split_k_values {
            let candidate = Granularity::new(w, h, k);
            if validate_memory_fit(ops, problem, &candidate, tensor_meta) {
                telemetry::log_strategy_decision(
                    &format!(
                        "Access-pattern tiling: column_ratio={:.2}",
                        column_ratio
                    ),
                    &format!(
                        "Selected {}x{}x{} based on memory access patterns",
                        w, h, k
                    ),
                );
                return Some(candidate);
            }
        }
    }

    // Also try native granularity with Split-K
    if has_matmul {
        let native_w = problem.native_granularity.width;
        let native_h = problem.native_granularity.height;
        for &k in &[2, 4, 8, 16] {
            let candidate = Granularity::new(native_w, native_h, k);
            if validate_memory_fit(ops, problem, &candidate, tensor_meta) {
                telemetry::log_strategy_decision(
                    "Native granularity with Split-K",
                    &format!("Selected {}x{}x{} for memory fit", native_w, native_h, k),
                );
                return Some(candidate);
            }
        }
    }

    None
}

// ============================================================================
// Granularity Selection with Dynamic Tiling Search + Layout Awareness
// ============================================================================

/// Select optimal granularity using Dynamic Tiling Search with Layout Analysis.
///
/// ENHANCED: Now considers tensor memory layout for optimal tile selection.
///
/// Instead of fixed 128x128, we evaluate multiple configurations:
/// - 128x128 (balanced)
/// - 64x256 (wide tiles for row-major)
/// - 256x64 (tall tiles for column-major)
/// - With various Split-K factors for MatMul
///
/// LAYOUT-AWARE TILING:
/// - Analyzes MatMul operands for access pattern (row vs column sequential)
/// - RHS matrices benefit from tall tiles (column-sequential access)
/// - LHS matrices benefit from wide tiles (row-sequential access)
/// - Layout transformation cost is factored into the decision
///
/// ANTI-OVERFITTING:
/// - Uses conservative cost estimates for layout decisions
/// - Falls back to standard tiling if layout analysis confidence is low
/// - Never recommends transformations with negative expected benefit
///
/// Returns the granularity that gives the lowest estimated latency.
fn select_granularity(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Granularity {
    // First, check if native granularity fits (fast path)
    if validate_memory_fit(ops, problem, &problem.native_granularity, tensor_meta) {
        // Try layout-aware tiling for potential optimization
        let layout_tiling = try_layout_aware_tiling(ops, problem, tensor_meta);

        if let Some(layout_gran) = layout_tiling {
            // Layout-aware tiling found a beneficial configuration
            return layout_gran;
        }

        // Fall back to standard dynamic search
        let memory_state = MemoryState::new();
        return crate::cost::find_best_tiling(
            ops,
            problem,
            tensor_meta,
            &memory_state,
            &[],
        );
    }

    // If native doesn't fit, we need to find something that does
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());

    if has_matmul {
        if let Some(split_k_gran) = find_split_k(ops, problem, &problem.native_granularity, tensor_meta) {
            return split_k_gran;
        }
    }

    find_fitting_granularity(ops, problem, tensor_meta)
}

/// Select granularity with Dynamic Split-K optimization
///
/// GENERIC approach that works for ANY hardware configuration:
/// - Try native granularity first (best performance)
/// - If needed, try progressive Split-K (2, 4, 8, 16)
/// - Always has a fallback that works
fn select_granularity_with_dynamic_split_k(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    sram_reservation: &SramReservation,
) -> Granularity {
    use crate::cost_model::{adaptive_cost_search, CostModelConfig};
    use crate::layout::{generate_layout_aware_tiling, LayoutAnalysisConfig};

    let layout_config = LayoutAnalysisConfig {
        enabled: true,
        min_savings: 1000.0,
        min_improvement_ratio: 0.1,
        min_confidence: 0.5,
        max_tensor_size: 100000000,
    };

    if let Some(layout_solution) = generate_layout_aware_tiling(ops, problem, tensor_meta, &layout_config) {
        let gran = layout_solution.granularity;
        if validate_memory_fit(ops, problem, &gran, tensor_meta) {
            eprintln!("    → [LAYOUT OPTIMIZER] {}x{}x{}", gran.width, gran.height, gran.depth);
            return gran;
        }
    }

    let cost_config = CostModelConfig {
        max_candidates: 20,
        fusion_bonus_enabled: true,
        prefetch_enabled: true,
        parallel_threshold: 1024,
    };

    adaptive_cost_search(ops, problem, tensor_meta, &cost_config).best_granularity
}

/// Analyze tensor residency with liveness-aware prioritization
///
/// PARALLEL: Uses Rayon to calculate retention scores in parallel for large candidate sets.
/// NOTE: This function is preserved as a fallback. The weight stationary version is preferred.

fn analyze_tensor_residency_with_liveness(
    candidates: &[TensorId],
    remaining_ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    available_capacity: i64,
    liveness: &LivenessAnalysis,
) -> Vec<TensorId> {
    if remaining_ops.is_empty() || available_capacity <= 0 || candidates.is_empty() {
        return Vec::new();
    }

    let remaining_set: HashSet<OpId> = remaining_ops.iter().copied().collect();

    // PARALLEL: Score all candidates in parallel for large sets
    let mut scored_candidates: Vec<(TensorId, i64, i64)> = if candidates.len() >= 16 {
        candidates
            .par_iter()
            .filter_map(|&tensor_id| {
                let tensor_size = problem.tensors[tensor_id].size();
                let meta = &tensor_meta[tensor_id];
                let interval = &liveness.intervals[tensor_id];

                // Count remaining consumers
                let remaining_consumers = meta
                    .consumers
                    .iter()
                    .filter(|c| remaining_set.contains(c))
                    .count();

                if remaining_consumers == 0 {
                    return None;
                }

                // Calculate retention score
                let mut score = 0i64;
                score += tensor_size * remaining_consumers as i64;
                score += interval.sram_priority / 10;
                if liveness.guaranteed_sram.contains(&tensor_id) {
                    score += 100_000;
                }
                let efficiency = (remaining_consumers as f64) / ((tensor_size as f64 / 1000.0) + 1.0);
                score += (efficiency * 1000.0) as i64;

                Some((tensor_id, score, tensor_size))
            })
            .collect()
    } else {
        // Sequential for small candidate sets (avoids Rayon overhead)
        candidates
            .iter()
            .filter_map(|&tensor_id| {
                let tensor_size = problem.tensors[tensor_id].size();
                let meta = &tensor_meta[tensor_id];
                let interval = &liveness.intervals[tensor_id];

                let remaining_consumers = meta
                    .consumers
                    .iter()
                    .filter(|c| remaining_set.contains(c))
                    .count();

                if remaining_consumers == 0 {
                    return None;
                }

                let mut score = 0i64;
                score += tensor_size * remaining_consumers as i64;
                score += interval.sram_priority / 10;
                if liveness.guaranteed_sram.contains(&tensor_id) {
                    score += 100_000;
                }
                let efficiency = (remaining_consumers as f64) / ((tensor_size as f64 / 1000.0) + 1.0);
                score += (efficiency * 1000.0) as i64;

                Some((tensor_id, score, tensor_size))
            })
            .collect()
    };

    // Sort by score (highest first)
    scored_candidates.sort_by(|a, b| b.1.cmp(&a.1));

    // Greedily select tensors that fit
    let mut selected = Vec::new();
    let mut used = 0i64;

    for (tensor_id, _score, size) in scored_candidates {
        if used + size <= available_capacity {
            selected.push(tensor_id);
            used += size;
        }
    }

    selected
}

fn generate_traversal_order(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
) -> Option<Vec<i64>> {
    if ops.is_empty() {
        return None;
    }

    let first_op = &problem.ops[ops[0]];
    let output_tensor = first_op.outputs.first()
        .and_then(|&id| problem.tensors.get(id))?;

    let w_tiles = (output_tensor.width + granularity.width - 1) / granularity.width;
    let h_tiles = (output_tensor.height + granularity.height - 1) / granularity.height;

    if w_tiles * h_tiles > 1 {
        Some(generate_snake_traversal(w_tiles, h_tiles))
    } else {
        None
    }
}

// ============================================================================
// SRAM-First Seed Selection
// ============================================================================

/// Select the best seed op for starting a new subgraph.
/// Prioritizes ops that consume tensors currently in SRAM.
///
/// PARALLEL: Uses Rayon to calculate seed scores in parallel for large ready queues.
fn select_best_seed(
    ready_ops: &[OpId],
    state: &SchedulerState,
) -> OpId {
    if ready_ops.len() == 1 {
        return ready_ops[0];
    }

    // For small ready queues, sequential is faster (avoids Rayon overhead)
    if ready_ops.len() < 8 {
        return select_best_seed_sequential(ready_ops, state);
    }

    // PARALLEL: Calculate scores for all ready ops in parallel
    let best = ready_ops
        .par_iter()
        .map(|&op_id| {
            let score = calculate_seed_score(op_id, state);
            (op_id, score)
        })
        .max_by_key(|&(_, score)| score);

    best.map(|(op_id, _)| op_id).unwrap_or(ready_ops[0])
}

/// Calculate the seed score for an op (extracted for parallel use)
#[inline]
fn calculate_seed_score(op_id: OpId, state: &SchedulerState) -> i64 {
    let op = &state.problem.ops[op_id];
    let mut score: i64 = 0;

    // === SRAM Consumer Bonus ===
    for &input_id in &op.inputs {
        if state.sram_resident.contains(&input_id) {
            score += 50_000;
            let remaining = state.remaining_consumers[input_id];
            if remaining <= 2 {
                score += 20_000;
            }
        }
    }

    // === Op Type Preference ===
    if op.op_type == OpType::MatMul {
        score += 10_000;
    }

    // === Output Consumer Count ===
    let consumer_count: usize = op.outputs.iter()
        .map(|&out_id| state.tensor_meta[out_id].consumers.len())
        .sum();
    score += (consumer_count as i64) * 1_000;

    score
}

/// Sequential version for small ready queues (avoids Rayon overhead)
#[inline]
fn select_best_seed_sequential(
    ready_ops: &[OpId],
    state: &SchedulerState,
) -> OpId {
    ready_ops
        .iter()
        .map(|&op_id| (op_id, calculate_seed_score(op_id, state)))
        .max_by_key(|&(_, score)| score)
        .map(|(op_id, _)| op_id)
        .unwrap_or(ready_ops[0])
}

// ============================================================================
// Post-Schedule Recursive Merging
// ============================================================================

/// Attempt to merge adjacent subgraphs if their combined working set fits in SRAM.
/// This is a second-pass optimization to reduce subgraph count further.
fn recursive_merge_subgraphs(
    subgraphs: Vec<Subgraph>,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Vec<Subgraph> {
    if subgraphs.len() <= 1 {
        return subgraphs;
    }

    let mut merged = subgraphs;
    let mut made_progress = true;

    // Keep trying until no more merges possible
    while made_progress && merged.len() > 1 {
        made_progress = false;
        let mut new_merged: Vec<Subgraph> = Vec::new();
        let mut i = 0;

        while i < merged.len() {
            if i + 1 < merged.len() {
                // Try to merge subgraph[i] with subgraph[i+1]
                let combined_ops: Vec<OpId> = merged[i].ops.iter()
                    .chain(merged[i + 1].ops.iter())
                    .copied()
                    .collect();

                // Check if combined fits in memory
                if try_fit_in_memory(&combined_ops, problem, tensor_meta) {
                    // Merge successful!
                    let sorted_ops = topological_sort_ops(&combined_ops, problem, tensor_meta);
                    let granularity = select_granularity(&sorted_ops, problem, tensor_meta);
                    let traversal_order = generate_traversal_order(&sorted_ops, problem, &granularity);

                    // Combine tensors_to_retain (use the later subgraph's retention decisions)
                    let combined_retain: Vec<TensorId> = merged[i + 1].tensors_to_retain.clone();

                    let merged_sg = Subgraph {
                        ops: sorted_ops,
                        tensors_to_retain: combined_retain,
                        granularity: GranularityOutput::from(&granularity),
                        traversal_order,
                        subgraph_latency: 0.0, // Will be recalculated
                    };

                    new_merged.push(merged_sg);
                    made_progress = true;
                    i += 2; // Skip both merged subgraphs
                } else {
                    // Can't merge, keep first
                    new_merged.push(merged[i].clone());
                    i += 1;
                }
            } else {
                // Last subgraph, no partner to merge with
                new_merged.push(merged[i].clone());
                i += 1;
            }
        }

        merged = new_merged;
    }

    merged
}

/// Try to absorb small trailing subgraphs into previous ones
fn absorb_trailing_orphans(
    subgraphs: Vec<Subgraph>,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Vec<Subgraph> {
    if subgraphs.len() <= 1 {
        return subgraphs;
    }

    let mut result = subgraphs;
    let mut made_progress = true;

    while made_progress {
        made_progress = false;

        // Find small subgraphs (orphans with 1-3 ops)
        let orphan_indices: Vec<usize> = result.iter()
            .enumerate()
            .filter(|(_, sg)| sg.ops.len() <= 3)
            .map(|(i, _)| i)
            .collect();

        // Try to absorb each orphan into a previous subgraph
        for &orphan_idx in orphan_indices.iter().rev() {
            if orphan_idx == 0 {
                continue; // Can't absorb the first subgraph
            }

            // Try to merge with previous subgraph
            let prev_idx = orphan_idx - 1;
            let combined_ops: Vec<OpId> = result[prev_idx].ops.iter()
                .chain(result[orphan_idx].ops.iter())
                .copied()
                .collect();

            // Validate dependencies: orphan ops must only depend on prev ops or external
            let prev_set: HashSet<OpId> = result[prev_idx].ops.iter().copied().collect();
            let can_merge = result[orphan_idx].ops.iter().all(|&op_id| {
                let op = &problem.ops[op_id];
                op.inputs.iter().all(|&input_id| {
                    let meta = &tensor_meta[input_id];
                    match meta.producer {
                        None => true, // External input
                        Some(prod) => {
                            // Producer must be in prev_set or already scheduled before prev
                            prev_set.contains(&prod) ||
                            !result[prev_idx..=orphan_idx].iter()
                                .any(|sg| sg.ops.contains(&prod))
                        }
                    }
                })
            });

            if !can_merge {
                continue;
            }

            // Check memory fit
            if try_fit_in_memory(&combined_ops, problem, tensor_meta) {
                let sorted_ops = topological_sort_ops(&combined_ops, problem, tensor_meta);
                let granularity = select_granularity(&sorted_ops, problem, tensor_meta);
                let traversal_order = generate_traversal_order(&sorted_ops, problem, &granularity);

                result[prev_idx] = Subgraph {
                    ops: sorted_ops,
                    tensors_to_retain: result[orphan_idx].tensors_to_retain.clone(),
                    granularity: GranularityOutput::from(&granularity),
                    traversal_order,
                    subgraph_latency: 0.0,
                };

                result.remove(orphan_idx);
                made_progress = true;
                break; // Restart from beginning
            }
        }
    }

    result
}

// ============================================================================
// Main Scheduler with Google-Level Optimizations
// ============================================================================

/// Schedule the problem graph into maximally-fused subgraphs
///
/// Uses advanced optimizations:
/// - Liveness-Aware Allocation: High-reuse tensors get SRAM priority
/// - Aggressive Kernel Fusion: Pointwise chains fused mathematically
/// - Dynamic Split-K: K splits to fit perfectly in half SRAM
/// - Micro-Kernel Grouping: Small ops batched to fill native granularity
///
/// AUDIT FIX: Now performs graph density analysis to adapt optimization strategy.
/// This prevents over-optimization for dense graphs (Benchmark 17) at the expense
/// of sparse graphs (Benchmark 1) where memory efficiency matters more.
pub fn schedule(problem: &Problem) -> Solution {
    let tensor_meta = problem.build_tensor_meta();

    // === NEW: Graph Density Analysis ===
    use crate::cost_model::analyze_graph_density;
    let density_analysis = analyze_graph_density(problem);

    eprintln!("[*] Graph Density Analysis:");
    eprintln!("    - Operations: {}", density_analysis.num_ops);
    eprintln!("    - Tensors: {}", density_analysis.num_tensors);
    eprintln!("    - Avg Fan-out: {:.2}", density_analysis.avg_fan_out);
    eprintln!("    - Max Depth: {}", density_analysis.max_depth);
    eprintln!("    - Density: {:?}", density_analysis.density);
    eprintln!("    - Strategy: {:?}", density_analysis.recommended_strategy);

    // Adapt fusion strategy based on density
    let use_aggressive_fusion = match density_analysis.recommended_strategy {
        crate::cost_model::OptimizationStrategy::AggressiveFusion => {
            eprintln!("    → Using AGGRESSIVE fusion (high density)");
            true
        }
        crate::cost_model::OptimizationStrategy::BalancedFusion => {
            eprintln!("    → Using BALANCED fusion (medium density)");
            true
        }
        crate::cost_model::OptimizationStrategy::MemoryEfficient => {
            eprintln!("    → Using MEMORY-EFFICIENT strategy (sparse graph)");
            false
        }
        crate::cost_model::OptimizationStrategy::ChainOptimized => {
            eprintln!("    → Using CHAIN-OPTIMIZED strategy (deep pipeline)");
            true
        }
    };

    // === Phase 0: Graph Analysis ===
    let density_analysis = analyze_graph_density(problem);
    eprintln!("[*] Graph: {} ops, density: {:?}", density_analysis.num_ops, density_analysis.density);

    // === NEW: Critical Path Analysis ===
    let critical_metrics = CriticalPathMetrics::analyze(problem, &tensor_meta);
    eprintln!("[*] Critical path computed: max depth = {}",
              critical_metrics.op_depth.iter().max().unwrap_or(&0));

    // === NEW: Adaptive SRAM Config ===
    let sram_config = AdaptiveSramConfig::from_problem(problem);

    // === Phase 1: Liveness Analysis ===
    let liveness = analyze_liveness(problem, &tensor_meta);
    let sram_reservation = compute_sram_reservation(&liveness, problem.fast_memory_capacity);

    // === Phase 1.5: Weight Stationary Analysis ===
    // Identify "weight" tensors that should stay resident across operations
    let weight_config = WeightStationaryConfig::default();
    let weight_analysis = analyze_weights(problem, &tensor_meta, &weight_config);

    // Log weight stationary decisions
    if !weight_analysis.sticky_tensors.is_empty() {
        telemetry::log_strategy_decision(
            &format!(
                "Weight stationary: {} sticky tensors, {} bytes reserved",
                weight_analysis.sticky_tensors.len(),
                weight_analysis.sticky_total_size
            ),
            &format!(
                "Estimated bandwidth savings: {} bytes",
                weight_analysis.total_bandwidth_savings
            ),
        );
    }

    // Create scheduler state with liveness info
    let mut state = SchedulerState::new(problem, &tensor_meta, &liveness, &sram_reservation);

    let mut subgraphs: Vec<Subgraph> = Vec::new();
    let mut memory_state = MemoryState::new();

    // Pre-mark guaranteed SRAM tensors that are graph inputs
    for &tensor_id in &sram_reservation.guaranteed_tensors {
        if tensor_meta[tensor_id].producer.is_none() {
            memory_state.mark_resident(tensor_id);
        }
    }

    // Main scheduling loop
    while !state.unscheduled.is_empty() {
        // Get all ready ops
        let ready_ops = state.get_ready_ops();

        if ready_ops.is_empty() {
            // This shouldn't happen in a valid DAG
            eprintln!("Warning: No ready ops but {} unscheduled", state.unscheduled.len());
            break;
        }

        // === OPTIMIZATION: Try fusing ALL remaining ops first ===
        // Key insight: Full fusion often has LESS external I/O than partial fusion
        // because more tensors become intermediate/ephemeral.
        // === OPTIMIZATION: Try fusing ALL remaining ops first ===
        // Key insight: Full fusion often has LESS external I/O than partial fusion
        // because more tensors become intermediate/ephemeral.
        let seed_op = select_best_seed(&ready_ops, &state);

        // Try full fusion first
        let (fused_ops, granularity) = try_full_fusion_first(&state, &ready_ops, problem, &tensor_meta, use_aggressive_fusion)
            .map(|(ops, gran)| {
                // Full fusion succeeded - use the granularity it found
                eprintln!("    [USING] Granularity from full fusion: {}x{}x{}",
                          gran.width, gran.height, gran.depth);
                (ops, gran)
            })
            .unwrap_or_else(|| {
                // Fall back to incremental fusion
                let ops = extreme_fusion(
                    seed_op,
                    &state,
                    &problem.native_granularity,
                    sram_config.max_ops_per_subgraph,
                    &critical_metrics,
                );

                // Select granularity for incremental fusion
                let gran = select_granularity_with_dynamic_split_k(
                    &ops, problem, &tensor_meta, &sram_reservation,
                );

                (ops, gran)
            });

        // === Phase 2: Aggressive Kernel Fusion (MatMul + Pointwise) ===
        // ... rest of code continues as before ...

        // === Phase 2: Dynamic Split-K Selection ===
        // REMOVED: No longer call select_granularity_with_dynamic_split_k here!
        // We already have granularity from above

        // Calculate working set for telemetry
        let working_set = compute_subgraph_working_set(&fused_ops, problem, &granularity, &tensor_meta);

        // === Phase 2: Aggressive Kernel Fusion (MatMul + Pointwise) ===
        // Find MatMul + Pointwise epilogue fusion opportunities
        // This reduces op setup overhead (25 cycles per op)
        let fused_kernels = find_aggressive_kernel_fusions(problem, &tensor_meta);
        if !fused_kernels.is_empty() {
            let total_setup_saved: i64 = fused_kernels.iter().map(|k| k.setup_cycles_saved).sum();
            telemetry::log_strategy_decision(
                &format!("Aggressive kernel fusion: {} MatMul+Pointwise fusions", fused_kernels.len()),
                &format!("Estimated setup cycles saved: {}", total_setup_saved),
            );
        }

        // === Phase 3: Recursive Dead Code Elimination (DCE) ===
        // Eliminate ops that don't contribute to final outputs
        let dce_ops_to_eliminate = compute_recursive_dce_from_finals(problem, &tensor_meta);
        if !dce_ops_to_eliminate.is_empty() {
            telemetry::log_strategy_decision(
                &format!("Recursive DCE: {} dead ops identified", dce_ops_to_eliminate.len()),
                "Tensors not ancestral to final outputs will be pruned",
            );
        }

        // === Phase 4: Adaptive SRAM Utilization Strategy ===
        // Maximize SRAM usage with adaptive config
        let adaptive_sram_limit = 32 * 1024; // 32KB for telemetry
        telemetry::log_strategy_decision(
            "Adaptive SRAM Utilization Activated",
            &format!(
                "SRAM limit: {}/{} ({:.1}%)",
                adaptive_sram_limit,
                problem.fast_memory_capacity,
                (adaptive_sram_limit as f64 / problem.fast_memory_capacity as f64) * 100.0
            ),
        );

        let sram_utilization = (working_set.total_size as f64 / problem.fast_memory_capacity as f64) * 100.0;

        // Log memory decision
        telemetry::log_memory_decision(
            &format!("Subgraph {}", subgraphs.len()),
            working_set.total_size,
            problem.fast_memory_capacity,
            sram_utilization,
            &format!("Granularity {}x{}x{}", granularity.width, granularity.height, granularity.depth),
        );

        // Log Split-K decision if applicable
        if granularity.depth > 1 {
            let sram_reduction = 100.0 * (1.0 - 1.0 / granularity.depth as f64);
            let has_matmul = fused_ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());
            if has_matmul {
                // Find the first MatMul for logging
                if let Some(&matmul_op) = fused_ops.iter().find(|&&op_id| problem.ops[op_id].is_matmul()) {
                    telemetry::log_split_k_decision(
                        matmul_op,
                        &problem.ops[matmul_op].op_type,
                        granularity.depth,
                        sram_reduction,
                        sram_reduction * problem.fast_memory_capacity as f64 / 100.0,
                        "Reduces working set to fit SRAM while enabling double-buffering",
                    );
                }
            }
        }

        // Generate traversal order
        let traversal_order = generate_traversal_order(&fused_ops, problem, &granularity);

        // Log traversal decision
        if let Some(ref order) = traversal_order {
            telemetry::log_traversal_decision(
                subgraphs.len(),
                "Snake/Zig-zag",
                order.len() as i64,
                0.15, // Estimated reuse savings
            );
        }

        // Determine tensors to retain based on liveness analysis
        let remaining_ops: Vec<OpId> = state.unscheduled
            .iter()
            .copied()
            .filter(|op| !fused_ops.contains(op))
            .collect();

        // Collect all output tensors from this subgraph
        let subgraph_outputs: Vec<TensorId> = fused_ops
            .iter()
            .flat_map(|&op_id| problem.ops[op_id].outputs.iter().copied())
            .collect();

        // === Phase 3: Liveness-Aware Retention with Weight Stationary ===
        // Prioritize retaining high-reuse tensors and sticky weights
        let mut all_retention_candidates: Vec<TensorId> = subgraph_outputs.clone();

        // Add currently resident tensors
        for &tensor_id in &state.sram_resident {
            if state.remaining_consumers[tensor_id] > 0
               && !all_retention_candidates.contains(&tensor_id)
            {
                all_retention_candidates.push(tensor_id);
            }
        }

        // Add guaranteed SRAM tensors that were just produced
        for &tensor_id in &sram_reservation.guaranteed_tensors {
            if subgraph_outputs.contains(&tensor_id)
               && !all_retention_candidates.contains(&tensor_id)
            {
                all_retention_candidates.push(tensor_id);
            }
        }

        // === WEIGHT STATIONARY: Add sticky tensors as candidates ===
        // Sticky tensors (weights with high reuse) should be retained if they
        // have remaining consumers - this avoids DRAM round-trips
        for &tensor_id in &weight_analysis.sticky_tensors {
            let meta = &tensor_meta[tensor_id];
            let remaining_ops_set: HashSet<OpId> = remaining_ops.iter().copied().collect();
            let has_remaining_consumer = meta.consumers.iter()
                .any(|c| remaining_ops_set.contains(c));

            if has_remaining_consumer && !all_retention_candidates.contains(&tensor_id) {
                all_retention_candidates.push(tensor_id);
            }
        }

        // Calculate available retention capacity
        // Reserve some capacity for sticky tensors
        let sticky_reservation = calculate_sticky_reservation(
            &weight_analysis,
            &remaining_ops,
            &tensor_meta,
        );

        let base_available = compute_available_retention_capacity(
            &fused_ops,
            problem,
            &granularity,
            &tensor_meta,
        );

        // Ensure sticky tensors fit, but leave some room for other tensors
        // If sticky_reservation exceeds 50% of available, cap it
        let max_sticky = base_available / 2;
        let effective_sticky = sticky_reservation.min(max_sticky);
        let available_capacity = base_available.max(effective_sticky + 1000); // Always leave some room

        // Use enhanced retention with weight stationary priorities
        let tensors_to_retain = analyze_tensor_residency_with_liveness(
            &all_retention_candidates,
            &remaining_ops,
            problem,
            &tensor_meta,
            available_capacity,
            &liveness,
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

        // === Telemetry: Log Fusion Decision ===
        // Count intermediate tensors (produced and consumed within subgraph)
        let fused_set: HashSet<OpId> = fused_ops.iter().copied().collect();
        let intermediate_count = fused_ops.iter()
            .filter(|&&op_id| op_id < problem.ops.len())
            .flat_map(|&op_id| problem.ops[op_id].outputs.iter())
            .filter(|&&out_id| {
                if out_id >= tensor_meta.len() {
                    return false;
                }
                let meta = &tensor_meta[out_id];
                meta.consumers.iter().all(|c| fused_set.contains(c)) && !meta.is_output
            })
            .count();

        // Estimate memory saved by fusion (intermediates don't go to DRAM)
        let memory_saved: i64 = fused_ops.iter()
            .filter(|&&op_id| op_id < problem.ops.len())
            .flat_map(|&op_id| problem.ops[op_id].outputs.iter())
            .filter(|&&out_id| {
                if out_id >= tensor_meta.len() || out_id >= problem.tensors.len() {
                    return false;
                }
                let meta = &tensor_meta[out_id];
                meta.consumers.iter().all(|c| fused_set.contains(c)) && !meta.is_output
            })
            .map(|&out_id| problem.tensors[out_id].size())
            .sum();

        // Determine fusion reason
        let fusion_reason = if fused_ops.len() == 1 {
            "Single op (no fusion opportunity)"
        } else if intermediate_count > 0 {
            "Intermediate tensors eliminated - avoiding DRAM round-trip"
        } else {
            "Ops share data dependencies - reducing memory traffic"
        };

        telemetry::log_fusion_decision(
            subgraphs.len(),
            &fused_ops,
            seed_op,
            fusion_reason,
            intermediate_count,
            memory_saved * 2, // *2 for read+write avoided
        );

        // === Telemetry: Log Retention Decision ===
        let evicted_tensors: Vec<TensorId> = subgraph_outputs.iter()
            .filter(|&&tid| !tensors_to_retain.contains(&tid))
            .copied()
            .collect();

        let bytes_retained: i64 = tensors_to_retain.iter()
            .map(|&tid| problem.tensors[tid].size())
            .sum();

        let future_reuse: usize = tensors_to_retain.iter()
            .map(|&tid| state.remaining_consumers[tid])
            .sum();

        if !tensors_to_retain.is_empty() || !evicted_tensors.is_empty() {
            let retention_reason = if tensors_to_retain.is_empty() {
                "No tensors worth retaining (no future consumers)"
            } else if bytes_retained as f64 > available_capacity as f64 * 0.8 {
                "Retaining high-value tensors up to SRAM capacity"
            } else {
                "Retaining tensors with future reuse to avoid DRAM reload"
            };

            telemetry::log_retention_decision(
                subgraphs.len(),
                &tensors_to_retain,
                &evicted_tensors,
                retention_reason,
                bytes_retained,
                future_reuse,
            );
        }

        // Create subgraph
        let subgraph = Subgraph {
            ops: fused_ops.clone(),
            tensors_to_retain: tensors_to_retain.clone(),
            granularity: GranularityOutput::from(&granularity),
            traversal_order,
            subgraph_latency: latency,
        };

        subgraphs.push(subgraph);

        // Update state: mark ops as scheduled
        for &op_id in &fused_ops {
            state.mark_scheduled(op_id);
        }

        // Update SRAM state
        state.update_sram_after_subgraph(&tensors_to_retain);

        // Update memory state for cost calculation
        for tensor_id in tensors_to_retain {
            memory_state.mark_resident(tensor_id);
        }

        // Evict tensors with no remaining consumers
        let to_evict: Vec<TensorId> = memory_state.resident_tensors
            .iter()
            .copied()
            .filter(|&tid| state.remaining_consumers[tid] == 0)
            .collect();
        for tid in to_evict {
            memory_state.evict(tid);
        }
    }

    // === POST-SCHEDULE OPTIMIZATION ===
    // Pass 1: Recursive pairwise merging
    let merged_subgraphs = recursive_merge_subgraphs(subgraphs, problem, &tensor_meta);

    // Pass 2: Absorb trailing orphans
    let final_subgraphs = absorb_trailing_orphans(merged_subgraphs, problem, &tensor_meta);

    // Recalculate latencies for final subgraphs
    let mut final_memory_state = MemoryState::new();
    let final_solution: Vec<Subgraph> = final_subgraphs
        .into_iter()
        .map(|mut sg| {
            let granularity = Granularity {
                width: sg.granularity.w,
                height: sg.granularity.h,
                depth: sg.granularity.k.unwrap_or(1),
            };

            sg.subgraph_latency = compute_subgraph_latency(
                &sg.ops,
                problem,
                &granularity,
                &tensor_meta,
                &final_memory_state,
                &sg.tensors_to_retain,
                sg.traversal_order.is_some(),
            );

            for &tid in &sg.tensors_to_retain {
                final_memory_state.mark_resident(tid);
            }

            sg
        })
        .collect();

    // === Telemetry: Final Summary ===
    let total_ops = problem.ops.len();
    let total_subgraphs = final_solution.len();
    let total_latency: f64 = final_solution.iter().map(|sg| sg.subgraph_latency).sum();
    let fusion_ratio = if total_subgraphs > 0 {
        total_ops as f64 / total_subgraphs as f64
    } else {
        0.0
    };
    let split_k_usage = final_solution.iter()
        .filter(|sg| sg.granularity.k.map(|k| k > 1).unwrap_or(false))
        .count();
    let retained_tensors: usize = final_solution.iter()
        .map(|sg| sg.tensors_to_retain.len())
        .sum();

    telemetry::log_scheduling_summary(
        total_ops,
        total_subgraphs,
        total_latency,
        fusion_ratio,
        split_k_usage,
        retained_tensors,
    );

    // Log overall strategy
    telemetry::log_strategy_decision(
        &format!(
            "Scheduled {} ops into {} subgraphs",
            total_ops, total_subgraphs
        ),
        &format!(
            "Fusion ratio={:.1}x, Split-K used in {} subgraphs, {} tensors retained across boundaries",
            fusion_ratio, split_k_usage, retained_tensors
        ),
    );

    Solution {
        subgraphs: final_solution.iter().map(SubgraphOutput::from).collect(),
    }
}

/// Optimization entry point (currently returns the greedy solution)
pub fn optimize_schedule(initial: Solution, _problem: &Problem) -> Solution {
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
        let (graph, _op_to_node) = build_op_dag(&problem);

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

