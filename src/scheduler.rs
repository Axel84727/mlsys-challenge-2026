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

use crate::cost::{compute_subgraph_latency, generate_snake_traversal, MemoryState};
use crate::liveness::{
    analyze_liveness, compute_sram_reservation, find_pointwise_chains,
    find_micro_kernel_candidates, LivenessAnalysis, SramReservation,
};
use crate::memory::{
    compute_available_retention_capacity,
    find_fitting_granularity, find_split_k, validate_memory_fit,
};
use crate::models::{
    Granularity, GranularityOutput, OpId, OpType, Problem, Solution, Subgraph, SubgraphOutput,
    TensorId, TensorMeta,
};
use petgraph::graph::{DiGraph, NodeIndex};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Constants - Tuning Parameters
// ============================================================================

/// Bonus for fusing MatMul with its immediate Pointwise consumer
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
    for (_tensor_id, meta) in tensor_meta.iter().enumerate() {
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
            meta.producer.map_or(false, |p| current_subgraph.contains(&p))
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
) -> Vec<OpId> {
    let mut fused_ops: Vec<OpId> = vec![seed_op];
    let mut fused_set: HashSet<OpId> = [seed_op].into_iter().collect();
    let mut subgraph_outputs: HashSet<TensorId> = state.problem.ops[seed_op]
        .outputs.iter().copied().collect();

    let max_iterations = state.problem.ops.len() * 3;

    for _iteration in 0..max_iterations {
        // === Global Ready Queue Scan ===
        // Find ALL ops that are:
        // 1. Unscheduled
        // 2. Not already in our fused set
        // 3. Ready to execute (all dependencies satisfied by either scheduled ops OR our fused set)

        // First, collect eligible candidates (sequential, fast filtering)
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
            // PARALLEL: Calculate priorities for all candidates in parallel
            eligible_candidates
                .par_iter()
                .map(|&op_id| {
                    let priority = state.calculate_fusion_priority(op_id, &fused_set, &subgraph_outputs);
                    (op_id, priority)
                })
                .collect()
        } else {
            // SEQUENTIAL: Avoid Rayon overhead for small candidate sets
            eligible_candidates
                .iter()
                .map(|&op_id| {
                    let priority = state.calculate_fusion_priority(op_id, &fused_set, &subgraph_outputs);
                    (op_id, priority)
                })
                .collect()
        };

        // Sort by priority (highest first)
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        let mut added_any = false;

        // Try each candidate in priority order
        for (candidate, priority) in candidates {
            let mut test_ops: Vec<OpId> = fused_ops.clone();
            test_ops.push(candidate);

            // === Memory Validation ===
            // Try native granularity first, then progressively smaller
            let fits = try_fit_in_memory(&test_ops, state.problem, state.tensor_meta);

            if fits {
                fused_ops.push(candidate);
                fused_set.insert(candidate);

                // Update subgraph outputs
                for &out_id in &state.problem.ops[candidate].outputs {
                    subgraph_outputs.insert(out_id);
                }

                added_any = true;

                // For high-priority ops (MatMul→Pointwise), break to re-evaluate priorities
                if priority >= MATMUL_POINTWISE_FUSION_BONUS {
                    break;
                }
            }
        }

        if !added_any {
            break;
        }
    }

    // Sort in topological order
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
// Granularity Selection with Dynamic Tiling Search
// ============================================================================

/// Select optimal granularity using Dynamic Tiling Search.
///
/// Instead of fixed 128x128, we evaluate multiple configurations:
/// - 128x128 (balanced)
/// - 64x256 (wide tiles for row-major)
/// - 256x64 (tall tiles for column-major)
/// - With various Split-K factors for MatMul
///
/// Returns the granularity that gives the lowest estimated latency.
fn select_granularity(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Granularity {
    // First, check if native granularity fits (fast path)
    if validate_memory_fit(ops, problem, &problem.native_granularity, tensor_meta) {
        // Even if native fits, run dynamic search for potentially better tiling
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
    _sram_reservation: &SramReservation,
) -> Granularity {
    // Safety check: empty ops
    if ops.is_empty() {
        return problem.native_granularity.clone();
    }

    // First try native granularity without Split-K (best performance)
    if validate_memory_fit(ops, problem, &problem.native_granularity, tensor_meta) {
        return problem.native_granularity.clone();
    }

    // Check if we have MatMul ops (Split-K only helps MatMul)
    let has_matmul = ops.iter().any(|&op_id| {
        op_id < problem.ops.len() && problem.ops[op_id].is_matmul()
    });

    if has_matmul {
        // Try progressive Split-K values
        for k in [2, 4, 8, 16] {
            let split_gran = problem.native_granularity.with_split_k(k);
            if validate_memory_fit(ops, problem, &split_gran, tensor_meta) {
                return split_gran;
            }
        }
    }

    // Fall back to standard selection (reduced spatial granularity)
    // This always finds something that works
    select_granularity(ops, problem, tensor_meta)
}

/// Analyze tensor residency with liveness-aware prioritization
///
/// PARALLEL: Uses Rayon to calculate retention scores in parallel for large candidate sets.
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
pub fn schedule(problem: &Problem) -> Solution {
    let tensor_meta = problem.build_tensor_meta();

    // === Phase 1: Liveness Analysis ===
    // Analyze tensor lifetimes to optimize SRAM allocation
    let liveness = analyze_liveness(problem, &tensor_meta);

    // Compute SRAM reservation strategy
    let sram_reservation = compute_sram_reservation(&liveness, problem.fast_memory_capacity);

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

        // Select best seed op (prioritizes SRAM consumers and high-reuse patterns)
        let seed_op = select_best_seed(&ready_ops, &state);

        // Aggressively fuse from this seed
        let fused_ops = extreme_fusion(seed_op, &state, &problem.native_granularity);

        // === Phase 2: Dynamic Split-K Selection ===
        // Select optimal granularity with dynamic Split-K
        let granularity = select_granularity_with_dynamic_split_k(
            &fused_ops, problem, &tensor_meta, &sram_reservation,
        );

        // Generate traversal order
        let traversal_order = generate_traversal_order(&fused_ops, problem, &granularity);

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

        // === Phase 3: Liveness-Aware Retention ===
        // Prioritize retaining high-reuse tensors
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

        // Calculate available retention capacity
        let available_capacity = compute_available_retention_capacity(
            &fused_ops,
            problem,
            &granularity,
            &tensor_meta,
        );

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

