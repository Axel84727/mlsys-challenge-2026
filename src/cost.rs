//! Cost model for computing subgraph and total latency.
//!
//! Latency = max(compute_time, memory_transfer_time)
//! This module implements the cost model as specified in PROBLEM.md.

use crate::models::{
    BaseCost, Granularity, Op, OpId, OpType, Problem, SlowMemoryBandwidth,
    SubgraphLatency, Tensor, TensorId, TensorMeta, TotalLatency,
};
use std::collections::HashSet;

// ============================================================================
// Compute Cost Calculation
// ============================================================================

/// Calculate the compute cost for a single operation with given granularity.
///
/// If granularity is smaller than native, there's inefficiency penalty.
/// If using Split-K (depth > 1), there are more compute steps but no penalty.
pub fn compute_op_cost(
    op: &Op,
    native_granularity: &Granularity,
    execution_granularity: &Granularity,
    output_tensor: &Tensor,
) -> f64 {
    let base_cost = op.base_cost as f64;

    // Calculate number of spatial tiles
    let w_tiles = (output_tensor.width as f64 / execution_granularity.width as f64).ceil();
    let h_tiles = (output_tensor.height as f64 / execution_granularity.height as f64).ceil();
    let num_spatial_tiles = w_tiles * h_tiles;

    // Calculate inefficiency due to smaller-than-native granularity
    let native_tile_size = (native_granularity.width * native_granularity.height) as f64;
    let exec_tile_size = (execution_granularity.width * execution_granularity.height) as f64;
    let inefficiency = if exec_tile_size < native_tile_size {
        native_tile_size / exec_tile_size
    } else {
        1.0
    };

    // Split-K factor (for MatMul)
    let split_k_factor = if op.op_type == OpType::MatMul {
        execution_granularity.depth as f64
    } else {
        1.0
    };

    base_cost * num_spatial_tiles * inefficiency * split_k_factor
}

/// Calculate total compute cost for a subgraph
pub fn compute_subgraph_compute_cost(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
) -> f64 {
    let mut total_cost = 0.0;

    for &op_id in ops {
        let op = &problem.ops[op_id];
        // Use first output tensor for tile calculation
        if let Some(&output_id) = op.outputs.first() {
            let output_tensor = &problem.tensors[output_id];
            total_cost += compute_op_cost(
                op,
                &problem.native_granularity,
                granularity,
                output_tensor,
            );
        }
    }

    total_cost
}

// ============================================================================
// Memory Transfer Cost Calculation
// ============================================================================

/// Memory transfer context for tracking what's already in fast memory
#[derive(Debug, Clone, Default)]
pub struct MemoryState {
    /// Tensors currently resident in fast memory
    pub resident_tensors: HashSet<TensorId>,
}

impl MemoryState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_resident(tensors: impl IntoIterator<Item = TensorId>) -> Self {
        Self {
            resident_tensors: tensors.into_iter().collect(),
        }
    }

    pub fn is_resident(&self, tensor_id: TensorId) -> bool {
        self.resident_tensors.contains(&tensor_id)
    }

    pub fn mark_resident(&mut self, tensor_id: TensorId) {
        self.resident_tensors.insert(tensor_id);
    }

    pub fn evict(&mut self, tensor_id: TensorId) {
        self.resident_tensors.remove(&tensor_id);
    }
}

/// Calculate memory transfer cost for a subgraph execution step.
///
/// Cost includes:
/// - Reading external inputs from slow memory (if not already resident)
/// - Writing external outputs to slow memory (if not retained)
pub fn compute_memory_transfer_cost(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
    memory_state: &MemoryState,
    tensors_to_retain: &[TensorId],
) -> f64 {
    if ops.is_empty() {
        return 0.0;
    }

    let ops_set: HashSet<OpId> = ops.iter().copied().collect();
    let retain_set: HashSet<TensorId> = tensors_to_retain.iter().copied().collect();

    let mut read_cost = 0.0;
    let mut write_cost = 0.0;

    // Calculate reads (external inputs not already resident)
    for &op_id in ops {
        let op = &problem.ops[op_id];
        for &input_id in &op.inputs {
            let meta = &tensor_meta[input_id];
            // External if producer is outside subgraph or graph input
            let is_external = meta.producer.map_or(true, |p| !ops_set.contains(&p));

            if is_external && !memory_state.is_resident(input_id) {
                let tensor = &problem.tensors[input_id];
                // Full tensor read (not just a slice) for first access
                read_cost += tensor.size() as f64;
            }
        }
    }

    // Calculate writes (external outputs)
    for &op_id in ops {
        let op = &problem.ops[op_id];
        for &output_id in &op.outputs {
            let meta = &tensor_meta[output_id];
            let has_external_consumer = meta.consumers.iter().any(|c| !ops_set.contains(c));
            let is_graph_output = meta.is_output;

            if (has_external_consumer || is_graph_output) && !retain_set.contains(&output_id) {
                let tensor = &problem.tensors[output_id];
                write_cost += tensor.size() as f64;
            }
        }
    }

    // Apply bandwidth factor
    let total_bytes = read_cost + write_cost;
    total_bytes / problem.slow_memory_bandwidth as f64
}

// ============================================================================
// Traversal Order Optimization
// ============================================================================

/// Generate an optimized traversal order (snake/zig-zag pattern)
/// for maximizing data reuse between adjacent tiles.
///
/// For a 2x2 grid of tiles:
/// Normal:  0 -> 1 -> 2 -> 3  (left-to-right, top-to-bottom)
/// Snake:   0 -> 1 -> 3 -> 2  (zig-zag to maximize locality)
pub fn generate_snake_traversal(w_tiles: i64, h_tiles: i64) -> Vec<i64> {
    let total_tiles = w_tiles * h_tiles;
    if total_tiles <= 1 {
        return vec![0];
    }

    let mut order = Vec::with_capacity(total_tiles as usize);

    for row in 0..h_tiles {
        if row % 2 == 0 {
            // Left to right
            for col in 0..w_tiles {
                order.push(row * w_tiles + col);
            }
        } else {
            // Right to left (snake back)
            for col in (0..w_tiles).rev() {
                order.push(row * w_tiles + col);
            }
        }
    }

    order
}

/// Calculate memory savings from using snake traversal vs linear.
/// Returns the ratio of bytes saved due to tile reuse.
pub fn estimate_snake_savings(
    tensor: &Tensor,
    granularity: &Granularity,
) -> f64 {
    let w_tiles = (tensor.width + granularity.width - 1) / granularity.width;
    let h_tiles = (tensor.height + granularity.height - 1) / granularity.height;
    let total_tiles = w_tiles * h_tiles;

    if total_tiles <= 1 {
        return 1.0;
    }

    // With snake traversal, adjacent tiles share one edge
    // Approximately 50% savings on internal tiles
    let boundary_tiles = 2 * (w_tiles + h_tiles) - 4;
    let internal_tiles = total_tiles - boundary_tiles.max(0);

    if internal_tiles > 0 {
        // Internal tiles get ~50% reuse, boundary tiles get ~25% reuse
        let avg_reuse = (internal_tiles as f64 * 0.5 + boundary_tiles as f64 * 0.25)
            / total_tiles as f64;
        1.0 - avg_reuse
    } else {
        0.9 // Small grids still get some benefit
    }
}

// ============================================================================
// Subgraph Latency Calculation
// ============================================================================

/// Fixed hardware setup cost per subgraph (cycles).
/// This penalty encourages the scheduler to fuse more operations together
/// to amortize the setup overhead across more useful work.
/// Keep this relatively small to not dominate the latency calculation.
pub const SUBGRAPH_SETUP_PENALTY: f64 = 100.0;

/// Calculate the total latency for a subgraph.
/// Latency = max(compute_time, memory_time) + setup_penalty
pub fn compute_subgraph_latency(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
    memory_state: &MemoryState,
    tensors_to_retain: &[TensorId],
    use_snake_traversal: bool,
) -> SubgraphLatency {
    let compute_cost = compute_subgraph_compute_cost(ops, problem, granularity);

    let mut memory_cost = compute_memory_transfer_cost(
        ops,
        problem,
        granularity,
        tensor_meta,
        memory_state,
        tensors_to_retain,
    );

    // Apply snake traversal savings if enabled
    if use_snake_traversal && !ops.is_empty() {
        // Estimate savings based on first output tensor
        let first_op = &problem.ops[ops[0]];
        if let Some(&output_id) = first_op.outputs.first() {
            let savings = estimate_snake_savings(&problem.tensors[output_id], granularity);
            memory_cost *= savings;
        }
    }

    // Latency is the maximum of compute and memory time, PLUS setup penalty
    // The setup penalty discourages creating many small subgraphs
    compute_cost.max(memory_cost) + SUBGRAPH_SETUP_PENALTY
}

/// Calculate total latency for a complete solution
pub fn compute_total_latency(
    subgraphs: &[crate::models::Subgraph],
    problem: &Problem,
) -> TotalLatency {
    let tensor_meta = problem.build_tensor_meta();
    let mut memory_state = MemoryState::new();
    let mut total_latency = 0.0;

    for subgraph in subgraphs {
        let granularity = Granularity {
            width: subgraph.granularity.w,
            height: subgraph.granularity.h,
            depth: subgraph.granularity.k.unwrap_or(1),
        };

        let latency = compute_subgraph_latency(
            &subgraph.ops,
            problem,
            &granularity,
            &tensor_meta,
            &memory_state,
            &subgraph.tensors_to_retain,
            subgraph.traversal_order.is_some(),
        );

        total_latency += latency;

        // Update memory state: mark retained tensors as resident
        for &tensor_id in &subgraph.tensors_to_retain {
            memory_state.mark_resident(tensor_id);
        }
    }

    total_latency
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_traversal_2x2() {
        let order = generate_snake_traversal(2, 2);
        assert_eq!(order, vec![0, 1, 3, 2]);
    }

    #[test]
    fn test_snake_traversal_3x3() {
        let order = generate_snake_traversal(3, 3);
        // Row 0: 0, 1, 2 (left to right)
        // Row 1: 5, 4, 3 (right to left)
        // Row 2: 6, 7, 8 (left to right)
        assert_eq!(order, vec![0, 1, 2, 5, 4, 3, 6, 7, 8]);
    }
}

