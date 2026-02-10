//! Liveness Analysis for Tensor Memory Management
//!
//! This module implements tensor liveness analysis to optimize SRAM allocation:
//! - Tracks when each tensor is first produced and last consumed
//! - Identifies high-priority tensors (reused multiple times)
//! - Enables optimal SRAM reservation strategy
//!
//! Google-level optimization: Tensors with more reuse get guaranteed SRAM slots.

use crate::models::{OpId, Problem, TensorId, TensorMeta};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Liveness Interval
// ============================================================================

/// Represents the lifetime of a tensor in the execution schedule
#[derive(Debug, Clone)]
pub struct LivenessInterval {
    /// Tensor ID
    pub tensor_id: TensorId,
    /// Op that produces this tensor (None for graph inputs)
    pub producer: Option<OpId>,
    /// All ops that consume this tensor
    pub consumers: Vec<OpId>,
    /// First op index where this tensor is needed (producer or first consumer)
    pub start: usize,
    /// Last op index where this tensor is needed (last consumer)
    pub end: usize,
    /// Number of times this tensor is consumed (reuse count)
    pub reuse_count: usize,
    /// Total bytes of this tensor
    pub size: i64,
    /// Priority score for SRAM allocation (higher = more important to keep)
    pub sram_priority: i64,
}

impl LivenessInterval {
    /// Calculate SRAM priority based on reuse and size efficiency
    pub fn calculate_priority(&mut self) {
        // Priority formula:
        // - More reuses = higher priority (saves more DRAM reads)
        // - Smaller tensors with high reuse = highest priority (best ROI for SRAM space)
        // - Longer lifetime = slight penalty (occupies SRAM longer)

        let reuse_value = (self.reuse_count as i64) * self.size; // Total bytes saved
        let lifetime = (self.end - self.start + 1) as i64;
        let efficiency = (self.reuse_count as f64) / (self.size as f64 / 1000.0 + 1.0);

        self.sram_priority = reuse_value + (efficiency * 10000.0) as i64 - lifetime * 100;
    }

    /// Check if this tensor is live at a given op index
    #[inline]
    pub fn is_live_at(&self, op_index: usize) -> bool {
        op_index >= self.start && op_index <= self.end
    }
}

// ============================================================================
// Liveness Analysis Result
// ============================================================================

/// Complete liveness analysis for a computation graph
#[derive(Debug)]
pub struct LivenessAnalysis {
    /// Liveness intervals for all tensors
    pub intervals: Vec<LivenessInterval>,
    /// Tensors sorted by SRAM priority (highest first)
    pub priority_order: Vec<TensorId>,
    /// Maximum number of tensors live simultaneously
    pub max_live_tensors: usize,
    /// Maximum bytes needed simultaneously
    pub max_live_bytes: i64,
    /// High-priority tensors that should be guaranteed SRAM slots
    pub guaranteed_sram: HashSet<TensorId>,
}

impl LivenessAnalysis {
    /// Get tensors that should definitely be in SRAM (high reuse, reasonable size)
    pub fn get_sram_candidates(&self, available_capacity: i64) -> Vec<TensorId> {
        let mut selected = Vec::new();
        let mut used = 0i64;

        for &tensor_id in &self.priority_order {
            let interval = &self.intervals[tensor_id];

            // Only consider tensors with reuse
            if interval.reuse_count <= 1 {
                continue;
            }

            if used + interval.size <= available_capacity {
                selected.push(tensor_id);
                used += interval.size;
            }
        }

        selected
    }

    /// Get the set of live tensors at a specific op
    pub fn live_at(&self, op_index: usize) -> Vec<TensorId> {
        self.intervals
            .iter()
            .filter(|interval| interval.is_live_at(op_index))
            .map(|interval| interval.tensor_id)
            .collect()
    }
}

// ============================================================================
// Analysis Functions
// ============================================================================

/// Perform liveness analysis on the computation graph
pub fn analyze_liveness(problem: &Problem, tensor_meta: &[TensorMeta]) -> LivenessAnalysis {
    let num_tensors = problem.tensors.len();
    let num_ops = problem.ops.len();

    // Build op index mapping (topological order approximation)
    // For now, we use op_id as the index (assuming ops are in reasonable order)
    let op_indices: HashMap<OpId, usize> = (0..num_ops).map(|i| (i, i)).collect();

    // Create liveness intervals for each tensor
    let mut intervals: Vec<LivenessInterval> = Vec::with_capacity(num_tensors);

    for (tensor_id, meta) in tensor_meta.iter().enumerate() {
        let tensor = &problem.tensors[tensor_id];

        // Determine start (producer op index, or 0 for graph inputs)
        let start = meta.producer
            .and_then(|p| op_indices.get(&p).copied())
            .unwrap_or(0);

        // Determine end (last consumer op index)
        let end = meta.consumers
            .iter()
            .filter_map(|c| op_indices.get(c).copied())
            .max()
            .unwrap_or(start);

        let mut interval = LivenessInterval {
            tensor_id,
            producer: meta.producer,
            consumers: meta.consumers.clone(),
            start,
            end,
            reuse_count: meta.consumers.len(),
            size: tensor.size(),
            sram_priority: 0,
        };

        interval.calculate_priority();
        intervals.push(interval);
    }

    // Sort by priority
    let mut priority_order: Vec<TensorId> = (0..num_tensors).collect();
    priority_order.sort_by(|&a, &b| {
        intervals[b].sram_priority.cmp(&intervals[a].sram_priority)
    });

    // Calculate maximum simultaneous live tensors and bytes
    let mut max_live_tensors = 0usize;
    let mut max_live_bytes = 0i64;

    for op_idx in 0..num_ops {
        let live: Vec<&LivenessInterval> = intervals
            .iter()
            .filter(|interval| interval.is_live_at(op_idx))
            .collect();

        let live_count = live.len();
        let live_bytes: i64 = live.iter().map(|i| i.size).sum();

        max_live_tensors = max_live_tensors.max(live_count);
        max_live_bytes = max_live_bytes.max(live_bytes);
    }

    // Identify guaranteed SRAM tensors (reuse >= 3, reasonable size)
    let guaranteed_sram: HashSet<TensorId> = intervals
        .iter()
        .filter(|i| i.reuse_count >= 3)
        .filter(|i| i.size <= problem.fast_memory_capacity / 10) // Max 10% of SRAM per tensor
        .map(|i| i.tensor_id)
        .collect();

    LivenessAnalysis {
        intervals,
        priority_order,
        max_live_tensors,
        max_live_bytes,
        guaranteed_sram,
    }
}

/// Calculate the optimal SRAM reservation based on liveness analysis
pub fn compute_sram_reservation(
    analysis: &LivenessAnalysis,
    available_capacity: i64,
) -> SramReservation {
    // Reserve 50% for guaranteed high-reuse tensors
    let guaranteed_capacity = available_capacity / 2;

    // Rest for working set and double buffering
    let working_capacity = available_capacity - guaranteed_capacity;

    // Select guaranteed tensors
    let guaranteed_tensors = analysis.get_sram_candidates(guaranteed_capacity);
    let guaranteed_bytes: i64 = guaranteed_tensors
        .iter()
        .map(|&tid| analysis.intervals[tid].size)
        .sum();

    SramReservation {
        guaranteed_tensors,
        guaranteed_bytes,
        working_capacity,
        double_buffer_capacity: working_capacity / 2,
    }
}

/// SRAM reservation strategy
#[derive(Debug, Clone)]
pub struct SramReservation {
    /// Tensors with guaranteed SRAM slots
    pub guaranteed_tensors: Vec<TensorId>,
    /// Bytes reserved for guaranteed tensors
    pub guaranteed_bytes: i64,
    /// Remaining capacity for working set
    pub working_capacity: i64,
    /// Capacity reserved for double buffering
    pub double_buffer_capacity: i64,
}

// ============================================================================
// Pointwise Fusion Analysis
// ============================================================================

/// Identifies sequences of Pointwise ops that can be fused aggressively
#[derive(Debug, Clone)]
pub struct PointwiseChain {
    /// Ops in this chain (in execution order)
    pub ops: Vec<OpId>,
    /// Input tensors (from outside the chain)
    pub external_inputs: HashSet<TensorId>,
    /// Output tensors (consumed outside the chain)
    pub external_outputs: HashSet<TensorId>,
    /// Intermediate tensors (purely internal, ephemeral)
    pub intermediates: HashSet<TensorId>,
    /// Total base cost of all ops in the chain
    pub total_base_cost: i64,
}

/// Find all Pointwise chains in the graph for aggressive fusion
pub fn find_pointwise_chains(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Vec<PointwiseChain> {
    let mut chains = Vec::new();
    let mut visited: HashSet<OpId> = HashSet::new();

    // Find all Pointwise ops
    let pointwise_ops: Vec<OpId> = problem.ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == crate::models::OpType::Pointwise)
        .map(|(id, _)| id)
        .collect();

    // Build chains starting from each unvisited Pointwise
    for &start_op in &pointwise_ops {
        if visited.contains(&start_op) {
            continue;
        }

        let mut chain_ops = Vec::new();
        let mut queue = vec![start_op];
        let mut chain_set: HashSet<OpId> = HashSet::new();

        // BFS to find connected Pointwise ops
        while let Some(op_id) = queue.pop() {
            if visited.contains(&op_id) || chain_set.contains(&op_id) {
                continue;
            }

            let op = &problem.ops[op_id];
            if op.op_type != crate::models::OpType::Pointwise {
                continue;
            }

            chain_ops.push(op_id);
            chain_set.insert(op_id);
            visited.insert(op_id);

            // Add Pointwise consumers
            for &output_id in &op.outputs {
                for &consumer in &tensor_meta[output_id].consumers {
                    if problem.ops[consumer].op_type == crate::models::OpType::Pointwise {
                        queue.push(consumer);
                    }
                }
            }

            // Add Pointwise producers
            for &input_id in &op.inputs {
                if let Some(producer) = tensor_meta[input_id].producer {
                    if problem.ops[producer].op_type == crate::models::OpType::Pointwise {
                        queue.push(producer);
                    }
                }
            }
        }

        if chain_ops.len() >= 2 {
            // Sort in topological order
            chain_ops.sort();

            // Analyze I/O
            let mut external_inputs = HashSet::new();
            let mut external_outputs = HashSet::new();
            let mut intermediates = HashSet::new();
            let mut total_base_cost = 0i64;

            for &op_id in &chain_ops {
                let op = &problem.ops[op_id];
                total_base_cost += op.base_cost;

                // Check inputs
                for &input_id in &op.inputs {
                    let meta = &tensor_meta[input_id];
                    if meta.producer.is_none_or(|p| !chain_set.contains(&p)) {
                        external_inputs.insert(input_id);
                    }
                }

                // Check outputs
                for &output_id in &op.outputs {
                    let meta = &tensor_meta[output_id];
                    let has_external_consumer = meta.consumers.iter().any(|c| !chain_set.contains(c));

                    if has_external_consumer || meta.is_output {
                        external_outputs.insert(output_id);
                    } else {
                        intermediates.insert(output_id);
                    }
                }
            }

            chains.push(PointwiseChain {
                ops: chain_ops,
                external_inputs,
                external_outputs,
                intermediates,
                total_base_cost,
            });
        }
    }

    chains
}

// ============================================================================
// Micro-Kernel Grouping
// ============================================================================

/// Identifies small ops that should be grouped to fill native granularity
#[derive(Debug, Clone)]
pub struct MicroKernelGroup {
    /// Small ops grouped together
    pub ops: Vec<OpId>,
    /// Combined output size
    pub total_output_size: i64,
    /// Target granularity for the group
    pub target_granularity: (i64, i64),
}

/// Find small operations that should be grouped into micro-kernels
pub fn find_micro_kernel_candidates(
    problem: &Problem,
    _tensor_meta: &[TensorMeta],
) -> Vec<MicroKernelGroup> {
    let native_tile_size = problem.native_granularity.width * problem.native_granularity.height;
    let threshold = native_tile_size / 4; // Ops smaller than 25% of native tile

    let mut groups = Vec::new();
    let mut small_ops: Vec<(OpId, i64)> = Vec::new();

    // Find small ops
    for (op_id, op) in problem.ops.iter().enumerate() {
        let output_size: i64 = op.outputs
            .iter()
            .map(|&oid| problem.tensors[oid].size())
            .sum();

        if output_size < threshold {
            small_ops.push((op_id, output_size));
        }
    }

    // Group small ops to fill native granularity
    let mut current_group: Vec<OpId> = Vec::new();
    let mut current_size = 0i64;

    for (op_id, size) in small_ops {
        if current_size + size <= native_tile_size {
            current_group.push(op_id);
            current_size += size;
        } else if !current_group.is_empty() {
            groups.push(MicroKernelGroup {
                ops: std::mem::take(&mut current_group),
                total_output_size: current_size,
                target_granularity: (
                    problem.native_granularity.width,
                    problem.native_granularity.height,
                ),
            });
            current_group = vec![op_id];
            current_size = size;
        }
    }

    // Don't forget the last group
    if current_group.len() >= 2 {
        groups.push(MicroKernelGroup {
            ops: current_group,
            total_output_size: current_size,
            target_granularity: (
                problem.native_granularity.width,
                problem.native_granularity.height,
            ),
        });
    }

    groups
}

// ============================================================================
// Dynamic Split-K Calculation
// ============================================================================

/// Maximum Split-K factor - higher values hurt performance more than they help
const MAX_SPLIT_K: i64 = 4;

/// SRAM split ratio: 80% for active working set, 20% for buffering
const SRAM_ACTIVE_RATIO: f64 = 0.80;

/// Calculate optimal Split-K factor based on tensor sizes and SRAM capacity
/// Conservative approach: only split when absolutely necessary, and keep K small
pub fn compute_dynamic_split_k(
    k_dimension: i64,
    bytes_per_element: i64,
    fast_memory_capacity: i64,
) -> i64 {
    // Use 80% of SRAM for active work (more aggressive utilization)
    let available_for_k = (fast_memory_capacity as f64 * SRAM_ACTIVE_RATIO) as i64;
    let bytes_per_k_slice = k_dimension * bytes_per_element;

    if bytes_per_k_slice <= available_for_k {
        // No split needed
        return 1;
    }

    // Calculate minimum split factor - be conservative
    let min_split = (bytes_per_k_slice as f64 / available_for_k as f64).ceil() as i64;

    // Only use power of 2, but cap at MAX_SPLIT_K
    let split_k = if min_split <= 2 { 2 } else { MAX_SPLIT_K };

    split_k.clamp(1, MAX_SPLIT_K)
}

/// Calculate optimal Split-K for a MatMul operation
/// Conservative: prefer K=1 or K=2, only use K=4 when really needed
pub fn compute_matmul_split_k(
    input_a_size: i64,
    input_b_size: i64,
    output_size: i64,
    fast_memory_capacity: i64,
) -> i64 {
    // Working set: A slice + B slice + output + accumulator
    let total_without_split = input_a_size + input_b_size + output_size * 2;

    // Use 80% of SRAM for active work
    let target_size = (fast_memory_capacity as f64 * SRAM_ACTIVE_RATIO) as i64;

    if total_without_split <= target_size {
        return 1;
    }

    // Need to split - but be conservative
    let ratio = total_without_split as f64 / target_size as f64;

    // Simple decision: if ratio <= 2, use k=2; else use k=4
    let split_k = if ratio <= 2.0 { 2 } else { MAX_SPLIT_K };

    split_k.clamp(1, MAX_SPLIT_K)
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
                    base_cost: 100,
                },
                Op {
                    op_type: OpType::Pointwise,
                    inputs: vec![2],
                    outputs: vec![3],
                    base_cost: 100,
                },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_liveness_analysis() {
        let problem = make_test_problem();
        let tensor_meta = problem.build_tensor_meta();
        let analysis = analyze_liveness(&problem, &tensor_meta);

        // Should have intervals for all tensors
        assert_eq!(analysis.intervals.len(), 4);

        // Tensor 1 should have higher reuse priority (consumed by op 1)
        assert!(analysis.intervals[1].reuse_count >= 1);
    }

    #[test]
    fn test_pointwise_chain_detection() {
        let problem = make_test_problem();
        let tensor_meta = problem.build_tensor_meta();
        let chains = find_pointwise_chains(&problem, &tensor_meta);

        // Should detect the Pointwise chain
        assert!(!chains.is_empty());

        // The chain should contain all 3 Pointwise ops
        let total_ops: usize = chains.iter().map(|c| c.ops.len()).sum();
        assert_eq!(total_ops, 3);
    }

    #[test]
    fn test_dynamic_split_k() {
        // Large K that needs splitting: K*bytes > 80% of SRAM
        // With 50000 SRAM, 80% is 40000
        // 15000 * 4 = 60000, which is > 40000, so needs split
        let split = compute_dynamic_split_k(15000, 4, 50000);
        assert!(split >= 2, "Expected split >= 2, got {}", split);

        // Small K that doesn't need splitting
        // 128 * 4 = 512 bytes, easily fits in 40000
        let no_split = compute_dynamic_split_k(128, 4, 50000);
        assert_eq!(no_split, 1, "Expected no split, got {}", no_split);
    }
}



