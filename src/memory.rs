//! Memory management module for SRAM/Scratchpad capacity tracking.
//!
//! This module implements working set computation and validates that
//! subgraphs fit within the fast memory capacity constraint.

use crate::models::{
    FastMemoryCapacity, Granularity, Op, OpId, OpType, Problem, Tensor, TensorId, TensorMeta,
};
use std::collections::HashSet;

// ============================================================================
// Working Set Computation
// ============================================================================

/// Represents the memory footprint of a subgraph execution step
#[derive(Debug, Clone, Default)]
pub struct WorkingSet {
    /// Total bytes needed in fast memory
    pub total_size: i64,
    /// Input tensor slices needed
    pub input_slices: Vec<(TensorId, i64)>,
    /// Output tensor slices produced
    pub output_slices: Vec<(TensorId, i64)>,
    /// Intermediate tensors (ephemeral, don't count toward external I/O)
    pub intermediate_slices: Vec<(TensorId, i64)>,
}

impl WorkingSet {
    /// Check if this working set fits in the given memory capacity
    #[inline]
    pub fn fits_in(&self, capacity: FastMemoryCapacity) -> bool {
        self.total_size <= capacity
    }
}

/// Compute the working set for a single operation
pub fn compute_op_working_set(
    op: &Op,
    tensors: &[Tensor],
    granularity: &Granularity,
) -> WorkingSet {
    let mut ws = WorkingSet::default();

    // Input slices
    for &input_id in &op.inputs {
        if let Some(tensor) = tensors.get(input_id) {
            let size = tensor.slice_size(granularity);
            ws.input_slices.push((input_id, size));
            ws.total_size += size;
        }
    }

    // Output slices
    for &output_id in &op.outputs {
        if let Some(tensor) = tensors.get(output_id) {
            let size = tensor.slice_size(granularity);
            ws.output_slices.push((output_id, size));
            ws.total_size += size;
        }
    }

    // For MatMul with Split-K, we need accumulator space
    if op.op_type == OpType::MatMul && granularity.depth > 1 {
        // Accumulator is the same size as output but we keep partial sums
        for &output_id in &op.outputs {
            if let Some(tensor) = tensors.get(output_id) {
                let acc_size = tensor.slice_size(granularity);
                ws.total_size += acc_size; // Additional accumulator space
            }
        }
    }

    ws
}

/// Compute the working set for a fused subgraph.
///
/// Key insight: Intermediate tensors (produced and consumed within the subgraph)
/// are EPHEMERAL - they are computed and consumed immediately in registers/local
/// memory during the fused kernel execution. They do NOT occupy Scratchpad space.
///
/// Only external inputs and external outputs count toward the working set.
/// This is the key optimization that enables aggressive fusion.
///
/// For Split-K (depth > 1), the reduction dimension is split, reducing the
/// working set size for the K dimension of MatMul inputs. The accumulator
/// space for partial sums is minimal compared to the working set savings.
pub fn compute_subgraph_working_set(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
) -> WorkingSet {
    if ops.is_empty() {
        return WorkingSet::default();
    }

    let ops_set: HashSet<OpId> = ops.iter().copied().collect();

    // Identify external inputs (produced outside or are graph inputs)
    let mut external_inputs: HashSet<TensorId> = HashSet::new();
    // Identify external outputs (consumed outside or are graph outputs)
    let mut external_outputs: HashSet<TensorId> = HashSet::new();
    // Track intermediates for reporting only (they don't count toward memory)
    let mut intermediates: HashSet<TensorId> = HashSet::new();
    // Track which tensors are MatMul inputs on the K dimension (affected by Split-K)
    let mut matmul_k_inputs: HashSet<TensorId> = HashSet::new();

    for &op_id in ops {
        let op = &problem.ops[op_id];

        // Check inputs
        for &input_id in &op.inputs {
            let meta = &tensor_meta[input_id];
            // External if producer is outside subgraph or it's a graph input
            if meta.producer.map_or(true, |p| !ops_set.contains(&p)) {
                external_inputs.insert(input_id);
            }
        }

        // Check outputs
        for &output_id in &op.outputs {
            let meta = &tensor_meta[output_id];
            // Check if any consumer is outside the subgraph
            let has_external_consumer = meta.consumers.iter().any(|c| !ops_set.contains(c));
            let is_graph_output = meta.is_output;

            if has_external_consumer || is_graph_output {
                external_outputs.insert(output_id);
            } else {
                // All consumers are within subgraph - it's intermediate (ephemeral)
                intermediates.insert(output_id);
            }
        }

        // Track MatMul K-dimension inputs for Split-K reduction
        if op.op_type == OpType::MatMul && granularity.depth > 1 {
            // For MatMul C = A @ B, typically A[M,K] and B[K,N]
            // Both A and B have the K dimension that gets split
            for &input_id in &op.inputs {
                matmul_k_inputs.insert(input_id);
            }
        }
    }

    // Compute total working set - ONLY external I/O counts!
    let mut ws = WorkingSet::default();

    // External inputs must be loaded from DRAM
    for &tensor_id in &external_inputs {
        let tensor = &problem.tensors[tensor_id];
        let size = if matmul_k_inputs.contains(&tensor_id) && granularity.depth > 1 {
            // Split-K reduces the K dimension by depth factor
            // We only need to load 1/depth of the input at a time
            compute_split_k_slice_size(tensor, granularity)
        } else {
            tensor.slice_size(granularity)
        };
        ws.input_slices.push((tensor_id, size));
        ws.total_size += size;
    }

    // External outputs must be stored to DRAM
    for &tensor_id in &external_outputs {
        let size = problem.tensors[tensor_id].slice_size(granularity);
        ws.output_slices.push((tensor_id, size));
        ws.total_size += size;
    }

    // Intermediates are EPHEMERAL - they do NOT add to working set!
    // They are computed and consumed immediately within the fused kernel.
    // We still track them for debugging/analysis purposes.
    for &tensor_id in &intermediates {
        let tensor = &problem.tensors[tensor_id];
        // Intermediates also benefit from Split-K if they're MatMul inputs
        let size = if matmul_k_inputs.contains(&tensor_id) && granularity.depth > 1 {
            compute_split_k_slice_size(tensor, granularity)
        } else {
            tensor.slice_size(granularity)
        };
        ws.intermediate_slices.push((tensor_id, size));
        // NOTE: We intentionally do NOT add to total_size here!
        // Intermediates are ephemeral and don't occupy Scratchpad space.
    }

    // Account for Split-K accumulators in MatMul (small overhead for partial sums)
    // Only external outputs need accumulator space; intermediates are ephemeral
    if granularity.depth > 1 {
        for &op_id in ops {
            let op = &problem.ops[op_id];
            if op.op_type == OpType::MatMul {
                for &output_id in &op.outputs {
                    // Only external outputs need accumulator space
                    if external_outputs.contains(&output_id) {
                        // Accumulator is smaller: just the output tile, independent of K
                        let acc_size = problem.tensors[output_id].slice_size(granularity);
                        ws.total_size += acc_size;
                    }
                    // Intermediate outputs: accumulator is ephemeral too, no extra cost
                }
            }
        }
    }

    ws
}

/// Calculate slice size for Split-K optimization.
/// When using Split-K, we divide the K dimension by depth, reducing memory needed.
#[inline]
fn compute_split_k_slice_size(tensor: &Tensor, granularity: &Granularity) -> i64 {
    let w = tensor.width.min(granularity.width);
    let h = tensor.height.min(granularity.height);
    // The K dimension (typically the larger of w or h for matrix inputs)
    // gets divided by depth. We approximate by dividing the total by depth.
    let base_size = w * h;
    (base_size / granularity.depth).max(1)
}

/// Validate that a subgraph fits within memory constraints
pub fn validate_memory_fit(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
) -> bool {
    let ws = compute_subgraph_working_set(ops, problem, granularity, tensor_meta);
    ws.fits_in(problem.fast_memory_capacity)
}

// ============================================================================
// Granularity Adjustment
// ============================================================================

/// Find the optimal granularity that fits in memory for a subgraph.
///
/// Strategy (in order of preference):
/// 1. Try native granularity first (best performance)
/// 2. If subgraph has MatMul ops, try Split-K with native spatial granularity
///    (Split-K divides the K dimension, reducing memory without hurting spatial locality)
/// 3. Only as a last resort, reduce spatial granularity (w, h)
///
/// This ordering is critical: reducing spatial granularity to 64x64 causes much
/// higher latency than using Split-K at 128x128.
pub fn find_fitting_granularity(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Granularity {
    let native = &problem.native_granularity;

    // Step 1: Try native granularity (best case)
    if validate_memory_fit(ops, problem, native, tensor_meta) {
        return native.clone();
    }

    // Step 2: Check if subgraph has MatMul ops - if so, try Split-K first
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());

    if has_matmul {
        // Try increasing Split-K factors: 2, 4, 8, 16, 32
        // Higher K means more passes but less memory per pass
        for k in [2, 4, 8, 16, 32] {
            let split_k_granularity = native.with_split_k(k);
            if validate_memory_fit(ops, problem, &split_k_granularity, tensor_meta) {
                return split_k_granularity;
            }
        }

        // Also try Split-K with slightly reduced spatial granularity
        // This is still better than just reducing spatial without Split-K
        let reduced_spatial = native.halve();
        for k in [2, 4, 8, 16] {
            let hybrid_granularity = reduced_spatial.with_split_k(k);
            if validate_memory_fit(ops, problem, &hybrid_granularity, tensor_meta) {
                return hybrid_granularity;
            }
        }
    }

    // Step 3: Last resort - reduce spatial granularity without Split-K
    let mut granularity = native.clone();

    // Try up to 8 halvings (256x reduction)
    for _ in 0..8 {
        granularity = granularity.halve();
        if validate_memory_fit(ops, problem, &granularity, tensor_meta) {
            return granularity;
        }
    }

    // Return smallest granularity even if it doesn't fit (scheduler will handle)
    granularity
}

/// Try Split-K optimization for MatMul-heavy subgraphs.
/// Returns optimal k value that fits in memory while minimizing k (fewer passes).
///
/// Split-K divides the K (reduction) dimension of MatMul operations, which:
/// - Reduces the working set because we only need 1/k of the input tiles at a time
/// - Requires accumulator space for partial sums (small overhead)
/// - Results in k passes over the output tiles (increases compute slightly)
///
/// This is preferable to reducing spatial granularity because:
/// - Spatial locality is preserved (128x128 tiles)
/// - Cache behavior remains optimal
/// - Only the reduction dimension is affected
pub fn find_split_k(
    ops: &[OpId],
    problem: &Problem,
    base_granularity: &Granularity,
    tensor_meta: &[TensorMeta],
) -> Option<Granularity> {
    // Only applies if subgraph has MatMul operations
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());
    if !has_matmul {
        return None;
    }

    // First check if base granularity already fits (no Split-K needed)
    if validate_memory_fit(ops, problem, base_granularity, tensor_meta) {
        return Some(base_granularity.clone());
    }

    // Try increasing k values: 2, 4, 8, 16, 32
    // We want the smallest k that fits (minimizes extra passes)
    for k in [2, 4, 8, 16, 32] {
        let split_granularity = base_granularity.with_split_k(k);
        if validate_memory_fit(ops, problem, &split_granularity, tensor_meta) {
            return Some(split_granularity);
        }
    }

    None
}

// ============================================================================
// Tensor Residency Analysis
// ============================================================================

/// Estimate the memory transfer savings from retaining a tensor in fast memory.
/// Returns the bytes saved by not having to re-read from DRAM.
pub fn estimate_retention_savings(
    tensor_id: TensorId,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    remaining_ops: &HashSet<OpId>,
) -> i64 {
    let meta = &tensor_meta[tensor_id];
    let tensor_size = problem.tensors[tensor_id].size();

    // Count how many remaining ops will consume this tensor
    let consumer_count = meta
        .consumers
        .iter()
        .filter(|c| remaining_ops.contains(c))
        .count() as i64;

    // Each consumer saves one full read from DRAM
    tensor_size * consumer_count
}

/// SRAM-Glutton: Aggressively retain ALL tensors that have future consumers and fit in SRAM.
///
/// Philosophy: We have 500k SRAM and tensors are ~16k each. We can hold 30+ tensors!
/// There's no reason to be conservative. If a tensor will be needed later, KEEP IT.
///
/// Strategy:
/// 1. ANY tensor with future consumers should be retained if it fits
/// 2. Prioritize by savings (more consumers = more valuable)
/// 3. Fill SRAM to capacity - don't leave space unused
pub fn analyze_tensor_residency(
    current_subgraph_outputs: &[TensorId],
    remaining_ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    available_capacity: FastMemoryCapacity,
) -> Vec<TensorId> {
    if remaining_ops.is_empty() || available_capacity <= 0 {
        return Vec::new();
    }

    let remaining_set: HashSet<OpId> = remaining_ops.iter().copied().collect();

    // Collect ALL tensors that have future consumers - we want to keep them ALL
    // (tensor_id, size, consumer_count, total_savings)
    let mut candidates: Vec<(TensorId, i64, usize, i64)> = Vec::new();

    for &tensor_id in current_subgraph_outputs {
        let meta = &tensor_meta[tensor_id];
        let tensor_size = problem.tensors[tensor_id].size();

        // Count remaining consumers
        let consumer_count = meta
            .consumers
            .iter()
            .filter(|c| remaining_set.contains(c))
            .count();

        if consumer_count == 0 {
            continue; // No future consumers - don't bother
        }

        // Calculate total bytes saved by retaining this tensor
        let total_savings = tensor_size * consumer_count as i64;

        candidates.push((tensor_id, tensor_size, consumer_count, total_savings));
    }

    // Sort by value: higher savings first, then by size (smaller = fit more)
    candidates.sort_by(|a, b| {
        // Higher savings is better
        b.3.cmp(&a.3)
            // Tie-breaker: more consumers
            .then_with(|| b.2.cmp(&a.2))
            // Tie-breaker: smaller size (can fit more)
            .then_with(|| a.1.cmp(&b.1))
    });

    // GREEDY: Take EVERYTHING that fits!
    let mut selected = Vec::new();
    let mut used_capacity = 0i64;

    for (tensor_id, size, _count, _savings) in candidates {
        if used_capacity + size <= available_capacity {
            selected.push(tensor_id);
            used_capacity += size;
        }
    }

    selected
}

/// Find ops that can be immediately scheduled next (all their inputs are available).
fn find_immediate_consumers(
    available_tensors: &[TensorId],
    remaining_ops: &HashSet<OpId>,
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> HashSet<OpId> {
    let available_set: HashSet<TensorId> = available_tensors.iter().copied().collect();
    let mut immediate = HashSet::new();

    for &op_id in remaining_ops {
        let op = &problem.ops[op_id];

        // Check if ALL inputs of this op are available
        let all_inputs_available = op.inputs.iter().all(|&input_id| {
            let meta = &tensor_meta[input_id];
            // Available if: graph input (no producer) OR tensor is in available set
            meta.producer.is_none() || available_set.contains(&input_id)
        });

        if all_inputs_available {
            immediate.insert(op_id);
        }
    }

    immediate
}

/// Calculate the available capacity for tensor retention after accounting for
/// the working set of the current subgraph execution.
///
/// SRAM-Glutton: Be aggressive! Use as much SRAM as possible.
/// Only keep a minimal 5% margin for safety.
pub fn compute_available_retention_capacity(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
) -> FastMemoryCapacity {
    let ws = compute_subgraph_working_set(ops, problem, granularity, tensor_meta);

    // Available = total capacity - working set
    // Use minimal margin - we want to use as much SRAM as possible!
    let margin = problem.fast_memory_capacity / 20; // Only 5% margin
    let available = problem.fast_memory_capacity - ws.total_size - margin;

    available.max(0)
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
            fast_memory_capacity: 20000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_working_set_single_op() {
        let problem = make_test_problem();
        let ws = compute_op_working_set(
            &problem.ops[0],
            &problem.tensors,
            &problem.native_granularity,
        );
        // Input (128*128) + Output (128*128) = 32768
        assert_eq!(ws.total_size, 32768);
    }

    #[test]
    fn test_subgraph_working_set_fused() {
        let problem = make_test_problem();
        let tensor_meta = problem.build_tensor_meta();
        let ws = compute_subgraph_working_set(
            &[0, 1],
            &problem,
            &problem.native_granularity,
            &tensor_meta,
        );
        // External input: tensor 0 (16384)
        // External output: tensor 2 (16384)
        // Intermediate: tensor 1 is EPHEMERAL - does NOT count!
        // Total = 16384 + 16384 = 32768
        assert_eq!(ws.total_size, 32768);
        assert_eq!(ws.intermediate_slices.len(), 1); // tensor 1 tracked but not counted
    }
}

