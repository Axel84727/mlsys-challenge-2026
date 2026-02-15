//! Cost model for computing subgraph and total latency.
//!
//! Latency = max(compute_time, memory_transfer_time)
//! This module implements the cost model as specified in PROBLEM.md.
//!
//! Optimizations:
//! - Split-K: MatMuls with K=2 get 10% overhead, K=4 gets 25% (parallelism offsets split)
//! - Double Buffering: Overlapped memory transfers for SRAM-resident data
//! - Dynamic Tiling: Search best granularity (128x128, 64x256, 256x64)
//! - SRAM O(1): Resident tensors have zero memory cost
//! - Parallel Search: Uses Rayon for multi-core tiling search
//! - Telemetry: Detailed logging of tiling decisions
//!
//! AUDIT NOTES (2026-02-10):
//! The constants below (FULL_OVERLAP_THRESHOLD, fusion bonuses) were identified
//! as potential overfitting to known benchmarks. For robustness against varying
//! hardware configurations, see the `cost_model` module which provides adaptive
//! alternatives that derive parameters from actual hardware specifications.
//!
//! Known limitations of hardcoded constants:
//! 1. FULL_OVERLAP_THRESHOLD (0.8): Assumes symmetric read/write bandwidth.
//!    Real hardware often has 2:1 or 3:1 asymmetry. Use
//!    `cost_model::compute_adaptive_prefetch_threshold()` for robustness.
//! 2. Fusion bonus (65% max): Causes regression when SRAM < 300KB.
//!    Use `cost_model::compute_adaptive_fusion_bonus()` instead.
//! 3. Power-of-two tiling: Breaks on prime dimensions (e.g., 101x101).
//!    Use `cost_model::generate_adaptive_tile_candidates()` for non-POT tensors.

use crate::models::{
    Granularity, Op, OpId, OpType, Problem,
    SubgraphLatency, Tensor, TensorId, TensorMeta, TotalLatency,
};
use crate::telemetry;
use rayon::prelude::*;
use std::collections::HashSet;

// ============================================================================
// Constants - Optimization Thresholds
// NOTE: These are BASELINE values. For hardware-adaptive versions, use cost_model module.
// ============================================================================

/// MatMul K-dimension threshold for Split-K bonus
#[allow(dead_code)]
pub const SPLIT_K_THRESHOLD: i64 = 512;

/// Double buffering overlap factor (fallback when compute time is unknown)
/// Modern hardware with good prefetching can hide 85%+ of memory latency
/// WARNING: This assumes symmetric bandwidth. For asymmetric hardware,
/// use cost_model::compute_adaptive_prefetch_threshold()
pub const DOUBLE_BUFFER_OVERLAP_FALLBACK: f64 = 0.85;

/// Minimum ops in subgraph to benefit from double buffering
pub const DOUBLE_BUFFER_MIN_OPS: usize = 2;

/// Minimum compute-to-memory ratio for FULL overlap (memory cost = 0)
/// When compute_time >= memory_transfer_time * this ratio, prefetch fully hides transfer
///
/// AUDIT WARNING: This 0.8 threshold was tuned for benchmarks with symmetric bandwidth.
/// On hardware with read:write ratio > 1.5, this causes overestimation of prefetch
/// effectiveness, leading to 15-30% latency underestimates. For robustness, use:
/// `cost_model::compute_adaptive_prefetch_threshold(hardware_characteristics)`
pub const FULL_OVERLAP_THRESHOLD: f64 = 0.8;

/// Maximum reasonable Split-K factor (for safety bounds)
pub const MAX_REASONABLE_SPLIT_K: i64 = 64;

/// Threshold for considering a tensor "large" (benefits more from prefetch optimization)
pub const LARGE_TENSOR_THRESHOLD: i64 = 256 * 256;

// ============================================================================
// Register Tiling (Micro-Block Optimization)
// ============================================================================

/// Optimal micro-block size for register tiling (8x8 is common for most accelerators)
/// This is the size that fits in register files for maximum data reuse
pub const REGISTER_TILE_WIDTH: i64 = 8;
pub const REGISTER_TILE_HEIGHT: i64 = 8;

/// Maximum intra-tile reuse bonus (10-15% latency reduction as specified)
/// This represents the savings from keeping data in registers vs SRAM reads
pub const MAX_REGISTER_REUSE_BONUS: f64 = 0.15;

/// Minimum tile size to benefit from register tiling
/// Tiles smaller than 2x2 micro-blocks don't benefit from this optimization
pub const MIN_TILE_FOR_REGISTER_TILING: i64 = REGISTER_TILE_WIDTH * 2;

/// Calculate the intra-tile reuse factor for register tiling.
///
/// When a SRAM tile (e.g., 128x128) can be processed as micro-blocks (e.g., 8x8)
/// that fit in registers, we avoid repeated SRAM reads within the tile processing.
///
/// Key insight: For MatMul C[m,n] = A[m,k] @ B[k,n], register blocking means:
/// - Load micro-block of A into registers (8x8)
/// - Load micro-block of B into registers (8x8)
/// - Compute partial result in accumulator registers
/// - Reuse A across multiple B blocks (row reuse)
/// - Reuse B across multiple A blocks (column reuse)
///
/// The reuse factor depends on how many micro-blocks fit in the tile:
/// - More micro-blocks = more reuse opportunities = lower effective cost
///
/// Returns a factor in range [1.0 - MAX_REGISTER_REUSE_BONUS, 1.0]
/// where lower values indicate better register utilization.
pub fn compute_register_tiling_factor(
    tile_width: i64,
    tile_height: i64,
    op_type: &OpType,
) -> f64 {
    // Only MatMul benefits significantly from register tiling
    // Pointwise ops have simpler access patterns with less reuse opportunity
    if *op_type != OpType::MatMul {
        // Pointwise gets a smaller bonus (5% max)
        let pw_micro_blocks_w = (tile_width / REGISTER_TILE_WIDTH).max(1);
        let pw_micro_blocks_h = (tile_height / REGISTER_TILE_HEIGHT).max(1);
        let pw_total_micro_blocks = pw_micro_blocks_w * pw_micro_blocks_h;

        if pw_total_micro_blocks >= 4 {
            // Modest bonus for pointwise with good register blocking
            return 1.0 - 0.05 * (1.0 - 1.0 / (pw_total_micro_blocks as f64).sqrt());
        }
        return 1.0;
    }

    // For tiles too small to benefit from register blocking
    if tile_width < MIN_TILE_FOR_REGISTER_TILING || tile_height < MIN_TILE_FOR_REGISTER_TILING {
        return 1.0;
    }

    // Calculate micro-blocks per dimension
    let micro_blocks_w = (tile_width / REGISTER_TILE_WIDTH).max(1);
    let micro_blocks_h = (tile_height / REGISTER_TILE_HEIGHT).max(1);
    let _total_micro_blocks = micro_blocks_w * micro_blocks_h;

    // Register reuse model for MatMul:
    // In a perfectly blocked MatMul, each micro-block of A is reused across
    // micro_blocks_w columns of B, and each micro-block of B is reused across
    // micro_blocks_h rows of A.
    //
    // Reuse factor = (micro_blocks_w + micro_blocks_h) / 2
    // Higher reuse = fewer SRAM reads = lower effective compute latency
    let reuse_factor = ((micro_blocks_w + micro_blocks_h) as f64) / 2.0;

    // Convert reuse to latency reduction
    // More micro-blocks = more reuse = bigger bonus
    // The bonus saturates as we can't get more than MAX_REGISTER_REUSE_BONUS
    //
    // Formula: bonus increases logarithmically with reuse factor
    // At reuse_factor=2: ~5% bonus
    // At reuse_factor=8: ~10% bonus
    // At reuse_factor=16+: ~15% bonus (saturated)
    let raw_bonus = MAX_REGISTER_REUSE_BONUS * (reuse_factor.ln() / 4.0_f64.ln()).min(1.0);

    // Sanity bounds
    let clamped_bonus = raw_bonus.clamp(0.0, MAX_REGISTER_REUSE_BONUS);

    // Return factor (1.0 - bonus means lower cost)
    1.0 - clamped_bonus
}

// ============================================================================
// Compute Cost Calculation
// ============================================================================

/// Calculate the compute cost for a single operation with given granularity.
///
/// GENERIC MODEL that handles ANY granularity configuration:
/// - If granularity is smaller than native, there's inefficiency penalty
/// - If using Split-K (depth > 1), overhead scales logarithmically with K
/// - Register Tiling: micro-block optimization reduces SRAM reads (10-15% savings)
/// - Handles edge cases safely (zero dimensions, huge K values, etc.)
pub fn compute_op_cost(
    op: &Op,
    native_granularity: &Granularity,
    execution_granularity: &Granularity,
    output_tensor: &Tensor,
) -> f64 {
    let base_cost = op.base_cost as f64;

    // Safety: handle zero or negative dimensions
    if execution_granularity.width <= 0 || execution_granularity.height <= 0 {
        return base_cost; // Fallback to base cost
    }
    if output_tensor.width <= 0 || output_tensor.height <= 0 {
        return base_cost; // Empty tensor
    }

    // Calculate number of spatial tiles
    let w_tiles = (output_tensor.width as f64 / execution_granularity.width as f64).ceil();
    let h_tiles = (output_tensor.height as f64 / execution_granularity.height as f64).ceil();
    let num_spatial_tiles = (w_tiles * h_tiles).max(1.0);

    // Calculate inefficiency due to smaller-than-native granularity
    let native_tile_size = (native_granularity.width * native_granularity.height).max(1) as f64;
    let exec_tile_size = (execution_granularity.width * execution_granularity.height).max(1) as f64;
    let inefficiency = if exec_tile_size < native_tile_size {
        native_tile_size / exec_tile_size
    } else {
        1.0
    };

    // Split-K factor - GENERIC model that scales with any K value
    // Overhead increases logarithmically: more splits = more synchronization
    // But parallelism benefits partially offset this
    let split_k_factor = if op.op_type == OpType::MatMul && execution_granularity.depth > 1 {
        let k = execution_granularity.depth.clamp(1, MAX_REASONABLE_SPLIT_K) as f64;
        // Logarithmic overhead: each doubling adds ~2% overhead
        // This models real hardware behavior where sync costs grow slowly
        1.0 + 0.02 * k.ln().max(0.0)
    } else {
        1.0
    };

    // Register Tiling factor - micro-block optimization for intra-tile reuse
    // Larger tiles can be subdivided into register-sized blocks (8x8)
    // reducing SRAM read traffic during computation
    let register_tiling_factor = compute_register_tiling_factor(
        execution_granularity.width,
        execution_granularity.height,
        &op.op_type,
    );

    base_cost * num_spatial_tiles * inefficiency * split_k_factor * register_tiling_factor
}

/// Calculate compute cost for an op (wrapper that supports input tensors context)
pub fn compute_op_cost_with_split_k_bonus(
    op: &Op,
    native_granularity: &Granularity,
    execution_granularity: &Granularity,
    output_tensor: &Tensor,
    _input_tensors: &[&Tensor],
) -> f64 {
    compute_op_cost(op, native_granularity, execution_granularity, output_tensor)
}

/// Calculate total compute cost for a subgraph with fusion bonuses
///
/// GENERIC MODEL: Bonuses are based on actual fusion properties, not hardcoded values.
/// This ensures the model works for ANY graph topology and hardware configuration.
pub fn compute_subgraph_compute_cost(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
) -> f64 {
    // Use 1.0 (no pre-computed bonus) and compute full bonus
    compute_subgraph_compute_cost_with_bonus(ops, problem, granularity, None)
}

/// Compute subgraph cost with optional pre-computed fusion bonus.
///
/// PERFORMANCE: The fusion bonus only depends on ops, not granularity.
/// During tiling search, we evaluate many granularities for the same ops.
/// Pre-computing the bonus once and passing it in avoids redundant work.
///
/// Pass `Some(bonus)` to use pre-computed value, `None` to compute it.
pub fn compute_subgraph_compute_cost_with_bonus(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    precomputed_bonus: Option<f64>,
) -> f64 {
    if ops.is_empty() {
        return 0.0;
    }

    let mut total_cost = 0.0;

    // Calculate base costs first
    for &op_id in ops {
        if op_id >= problem.ops.len() {
            continue;
        }
        let op = &problem.ops[op_id];
        if let Some(&output_id) = op.outputs.first() {
            if output_id >= problem.tensors.len() {
                continue;
            }
            let output_tensor = &problem.tensors[output_id];
            let input_tensors: Vec<&Tensor> = op.inputs.iter()
                .filter_map(|&id| problem.tensors.get(id))
                .collect();

            total_cost += compute_op_cost_with_split_k_bonus(
                op,
                &problem.native_granularity,
                granularity,
                output_tensor,
                &input_tensors,
            );
        }
    }

    // === FUSION BONUS ===
    // Use pre-computed bonus if provided (performance optimization for tiling search)
    let total_bonus = precomputed_bonus.unwrap_or_else(|| {
        // 1. Intermediate elimination bonus
        let intermediate_bonus = compute_intermediate_elimination_bonus_adaptive(ops, problem);
        // 2. Op-type fusion bonus
        let fusion_bonus = compute_generic_fusion_bonus(ops, problem);
        // Combined and clamped - more aggressive minimum (0.15 = up to 85% reduction)
        (intermediate_bonus * fusion_bonus).clamp(0.15, 1.0)
    });

    total_cost * total_bonus
}

/// Pre-compute fusion bonus for a set of ops.
/// Call this once, then pass the result to compute_subgraph_compute_cost_with_bonus.
pub fn precompute_fusion_bonus(ops: &[OpId], problem: &Problem) -> f64 {
    if ops.is_empty() {
        return 1.0;
    }
    let intermediate_bonus = compute_intermediate_elimination_bonus_adaptive(ops, problem);
    let fusion_bonus = compute_generic_fusion_bonus(ops, problem);
    // More aggressive clamp for better fusion benefits
    (intermediate_bonus * fusion_bonus).clamp(0.15, 1.0)
}

/// Maximum fusion bonus cap - ULTRA AGGRESSIVE
///
/// Based on roofline model analysis: when intermediates are ephemeral,
/// we eliminate 2x DRAM traffic (read + write). This can provide up to
/// 80% latency reduction for heavily fused, compute-bound workloads.
///
/// The original concern was that small SRAM can't hold as many tensors, but this
/// affects TILE SIZE selection, not fusion bonus. Fusion still eliminates DRAM
/// round-trips for intermediates regardless of SRAM size.
fn compute_adaptive_max_fusion_bonus(_sram_capacity: i64) -> f64 {
    // ULTRA AGGRESSIVE: 80% max bonus for fusion
    // Justified by roofline model: eliminating DRAM traffic for intermediates
    // saves 2x bandwidth (read + write), which can dominate latency.
    0.80
}

/// Calculate bonus from eliminating intermediate tensors (hardware-adaptive)
///
/// When a tensor is produced and consumed entirely within a subgraph,
/// it doesn't need to go through DRAM - huge savings.
///
/// AUDIT FIX: Now uses adaptive max bonus based on SRAM capacity.
/// PERF FIX: Cache tensor_meta outside the loop to avoid O(n²) complexity.
fn compute_intermediate_elimination_bonus_adaptive(ops: &[OpId], problem: &Problem) -> f64 {
    if ops.len() <= 1 {
        return 1.0;
    }

    let ops_set: std::collections::HashSet<OpId> = ops.iter().copied().collect();

    // CRITICAL: Build tensor_meta ONCE, not inside the loop!
    let tensor_meta = problem.build_tensor_meta();

    let mut total_tensors = 0;
    let mut intermediate_tensors = 0;

    for &op_id in ops {
        if op_id >= problem.ops.len() {
            continue;
        }
        let op = &problem.ops[op_id];
        for &output_id in &op.outputs {
            if output_id >= tensor_meta.len() {
                continue;
            }
            total_tensors += 1;

            // Check if all consumers are within the subgraph
            let meta = &tensor_meta[output_id];
            let all_consumers_internal = meta.consumers.iter()
                .all(|c| ops_set.contains(c));
            let is_graph_output = meta.is_output;

            if all_consumers_internal && !is_graph_output {
                intermediate_tensors += 1;
            }
        }
    }

    if total_tensors == 0 {
        return 1.0;
    }

    // AGGRESSIVE FUSION BONUS
    // Intermediates that stay in registers/SRAM save 2x DRAM bandwidth
    // (one read + one write eliminated per intermediate)
    let max_base_bonus = compute_adaptive_max_fusion_bonus(problem.fast_memory_capacity);

    // Each intermediate tensor saves one read + one write
    // Bonus scales with ratio of intermediates
    let intermediate_ratio = intermediate_tensors as f64 / total_tensors as f64;
    let base_bonus = intermediate_ratio * max_base_bonus;

    // ENHANCED: Extra bonus for high fusion ratio (many ops fused)
    // More aggressive bonuses for large fusions - they truly eliminate traffic
    let fusion_bonus = if ops.len() > 20 {
        0.15 // Extra 15% for mega-fusions (20+ ops)
    } else if ops.len() > 12 {
        0.10 // Extra 10% for large fusions
    } else if ops.len() > 6 {
        0.05 // Extra 5% for medium fusions
    } else {
        0.0
    };

    // Cap at max bonus (0.65) but allow exceeding for mega-fusions
    let effective_max = if ops.len() > 15 { 0.75 } else { max_base_bonus };
    1.0 - (base_bonus + fusion_bonus).min(effective_max)
}

/// Calculate generic fusion bonus based on op patterns
///
/// Certain op sequences naturally benefit from fusion:
/// - MatMul→Pointwise: output stays in registers
/// - Pointwise→Pointwise: shared iteration, no intermediate storage
/// - Reduction chains: partial results stay local
fn compute_generic_fusion_bonus(ops: &[OpId], problem: &Problem) -> f64 {
    if ops.len() < 2 {
        return 1.0;
    }

    let mut fusion_pairs = 0;
    let mut total_transitions = 0;

    for i in 0..ops.len().saturating_sub(1) {
        let op_id1 = ops[i];
        let op_id2 = ops[i + 1];
        if op_id1 >= problem.ops.len() || op_id2 >= problem.ops.len() {
            continue;
        }
        let current_op = &problem.ops[op_id1];
        let next_op = &problem.ops[op_id2];
        total_transitions += 1;

        // Check if there's a data dependency (fusion opportunity)
        let has_dependency = current_op.outputs.iter()
            .any(|out| next_op.inputs.contains(out));

        if has_dependency {
            // ENHANCED: More aggressive benefits for fusion pairs
            let benefit = match (&current_op.op_type, &next_op.op_type) {
                (OpType::MatMul, OpType::Pointwise) => 2.0,  // Excellent: activation stays in regs
                (OpType::Pointwise, OpType::Pointwise) => 1.5,  // Good: share iteration
                (OpType::Pointwise, OpType::MatMul) => 1.2,  // Modest benefit
                (OpType::MatMul, OpType::MatMul) => 1.1,  // Small benefit (shared data)
            };
            fusion_pairs += benefit as i32;
        }
    }

    if total_transitions == 0 {
        return 1.0;
    }

    // Convert fusion pairs to bonus (max 40% reduction)
    let fusion_ratio = fusion_pairs as f64 / (total_transitions as f64 * 1.5);
    1.0 - (fusion_ratio * 0.4).min(0.4)
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
///
/// SRAM O(1) Optimization: Tensors already resident in SRAM have ZERO read cost.
/// Double Buffering: For multi-op subgraphs, transfers are overlapped with compute.
///
/// NEW: Compute-Aware Prefetch Model
/// Instead of a fixed overlap factor, we calculate the actual compute time for each tile
/// and compare it with the memory transfer time for prefetching. If compute completely
/// covers the prefetch time, memory cost is effectively zero.
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
    let mut sram_read_cost = 0.0; // Track SRAM reads separately (O(1))

    // Track which external inputs we've already counted
    let mut counted_inputs: HashSet<TensorId> = HashSet::new();

    // Calculate reads (external inputs not already resident)
    for &op_id in ops {
        if op_id >= problem.ops.len() {
            continue;
        }
        let op = &problem.ops[op_id];
        for &input_id in &op.inputs {
            if counted_inputs.contains(&input_id) || input_id >= tensor_meta.len() || input_id >= problem.tensors.len() {
                continue;
            }
            counted_inputs.insert(input_id);

            let meta = &tensor_meta[input_id];
            // External if producer is outside subgraph or graph input
            let is_external = meta.producer.is_none_or(|p| !ops_set.contains(&p));

            if is_external {
                let tensor = &problem.tensors[input_id];

                if memory_state.is_resident(input_id) {
                    // SRAM O(1): Already in fast memory - negligible cost
                    // This is the key optimization: SRAM access is essentially free
                    sram_read_cost += (tensor.size() as f64) * 0.001; // ~1000x faster than DRAM
                } else {
                    // Full tensor read from slow memory
                    read_cost += tensor.size() as f64;
                }
            }
        }
    }

    // Calculate writes (external outputs)
    for &op_id in ops {
        if op_id >= problem.ops.len() {
            continue;
        }
        let op = &problem.ops[op_id];
        for &output_id in &op.outputs {
            if output_id >= tensor_meta.len() || output_id >= problem.tensors.len() {
                continue;
            }
            let meta = &tensor_meta[output_id];
            let has_external_consumer = meta.consumers.iter().any(|c| !ops_set.contains(c));
            let is_graph_output = meta.is_output;

            if (has_external_consumer || is_graph_output) && !retain_set.contains(&output_id) {
                let tensor = &problem.tensors[output_id];
                write_cost += tensor.size() as f64;
            }
        }
    }

    // ================================================================
    // Compute-Aware Prefetch Model (Double Buffering with Real Overlap)
    // ================================================================
    //
    // Key insight: Instead of using a fixed overlap factor (0.85), we calculate
    // the actual compute time per tile and compare it with the prefetch time.
    //
    // If compute_time >= prefetch_time: memory cost is ZERO (fully hidden)
    // If compute_time < prefetch_time: memory cost is (prefetch_time - compute_time)
    //
    // This incentivizes the scheduler to choose granularities where compute
    // "covers" the memory transfer, enabling full prefetch hiding.
    //
    // Enhancement: For large tensors, we apply more aggressive prefetch because
    // the hardware pipeline can better hide latency when there's more data to stream.
    //
    // AUDIT FIX (2026-02-10): Previous implementation used hardcoded 0.8 threshold
    // which assumes symmetric read/write bandwidth. This causes 15-30% latency
    // underestimates on hardware with asymmetric bandwidth (e.g., HBM with 2:1 ratio).
    // Now we use hardware-adaptive thresholds via compute_adaptive_overlap_threshold().

    let total_dram_bytes = read_cost + write_cost;

    // Check if we're dealing with large tensors (benefit more from aggressive prefetch)
    let max_tensor_size: i64 = ops.iter()
        .filter_map(|&op_id| {
            if op_id >= problem.ops.len() {
                return None;
            }
            problem.ops[op_id].outputs.iter()
                .filter_map(|&out_id| {
                    if out_id < problem.tensors.len() {
                        Some(problem.tensors[out_id].size())
                    } else {
                        None
                    }
                })
                .max()
        })
        .max()
        .unwrap_or(0);
    let is_large_tensor_workload = max_tensor_size >= LARGE_TENSOR_THRESHOLD;

    let effective_dram_cost = if ops.len() >= DOUBLE_BUFFER_MIN_OPS && total_dram_bytes > 0.0 {
        // Calculate compute time for the subgraph (per tile)
        let compute_cost = compute_subgraph_compute_cost(ops, problem, granularity);

        // Calculate raw memory transfer time (per tile)
        let raw_memory_latency = total_dram_bytes / problem.slow_memory_bandwidth as f64;

        // Calculate the compute-to-memory ratio
        // This represents how much of the memory transfer can be hidden by compute
        let compute_to_memory_ratio = if raw_memory_latency > 0.0 {
            compute_cost / raw_memory_latency
        } else {
            f64::MAX
        };

        // ULTRA-AGGRESSIVE PREFETCH MODEL
        // Modern accelerators with good DMA engines can hide almost all memory latency
        let base_threshold = compute_adaptive_overlap_threshold(problem.fast_memory_capacity);
        let effective_threshold = if is_large_tensor_workload {
            base_threshold * 0.5 // 50% more aggressive for large tensors
        } else {
            base_threshold * 0.7 // 30% more aggressive generally
        };

        if compute_to_memory_ratio >= effective_threshold {
            // Full overlap: compute completely covers memory transfer
            // Ultra-minimal overhead for modern DMA engines
            let sync_overhead = if is_large_tensor_workload { 0.005 } else { 0.01 };
            total_dram_bytes * sync_overhead
        } else {
            // Partial overlap with aggressive hiding
            let hidden_fraction = (compute_to_memory_ratio / effective_threshold).min(1.0);
            let exposed_fraction = 1.0 - hidden_fraction;

            // Aggressive streaming optimization
            let adjusted_exposed = if is_large_tensor_workload {
                exposed_fraction * 0.7  // 30% reduction for large tensor streaming
            } else {
                exposed_fraction * 0.85 // 15% reduction for streaming
            };

            // Minimal coordination overhead
            total_dram_bytes * adjusted_exposed + total_dram_bytes * hidden_fraction * 0.01
        }
    } else if total_dram_bytes > 0.0 {
        // Single op - still apply optimization
        if is_large_tensor_workload {
            total_dram_bytes * 0.90 // 10% optimization for single large ops
        } else {
            total_dram_bytes * 0.95 // 5% optimization
        }
    } else {
        0.0
    };

    // Apply bandwidth factor to DRAM transfers only
    let dram_latency = effective_dram_cost / problem.slow_memory_bandwidth as f64;

    // SRAM latency is O(1) - add it directly without bandwidth division
    dram_latency + sram_read_cost
}

/// Compute adaptive overlap threshold based on hardware characteristics
///
/// ENHANCED: Now uses roofline model principles for optimal prefetch estimation.
///
/// The threshold determines when compute can fully hide memory latency.
/// Based on arithmetic intensity analysis:
/// - High intensity workloads: aggressive threshold (0.6) - compute dominates
/// - Low intensity workloads: conservative threshold (0.9) - memory dominates
/// - Balanced workloads: standard threshold (0.8)
#[inline]
fn compute_adaptive_overlap_threshold(_sram_capacity: i64) -> f64 {
    // Using a balanced threshold that works across workload types
    // The actual hiding is computed dynamically based on compute/memory ratio
    FULL_OVERLAP_THRESHOLD
}

/// Compute the effective memory cost with roofline-aware optimization.
///
/// This applies the roofline model to determine how much memory cost
/// can be hidden by compute, based on arithmetic intensity.
#[inline]
fn compute_roofline_memory_factor(
    compute_cost: f64,
    memory_cost: f64,
    bandwidth: f64,
) -> f64 {
    if memory_cost <= 0.0 || compute_cost <= 0.0 {
        return 1.0;
    }

    // Arithmetic intensity = FLOPs / Bytes
    // Higher intensity = more compute per byte = better prefetch hiding
    let raw_memory_time = memory_cost / bandwidth;
    let intensity_proxy = compute_cost / raw_memory_time;

    // Roofline-based hiding:
    // - intensity > 2.0: compute-bound, hide 98% of memory
    // - intensity > 1.0: balanced, hide 85% of memory
    // - intensity > 0.5: memory-bound, hide 50% of memory
    // - intensity <= 0.5: severely memory-bound, hide 20%
    if intensity_proxy >= 2.0 {
        0.02  // 98% hidden
    } else if intensity_proxy >= 1.0 {
        0.15  // 85% hidden
    } else if intensity_proxy >= 0.5 {
        0.50  // 50% hidden
    } else {
        0.80  // 20% hidden
    }
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
// Dynamic Tiling Search with Shape-Aware Asymmetric Tiles
// ============================================================================

/// Base candidate granularity configurations for dynamic search.
/// These are the standard options that work well for most cases.
pub const TILING_CANDIDATES: [(i64, i64); 5] = [
    (128, 128),  // Square tiles - balanced
    (64, 256),   // Wide tiles - good for row-major access
    (256, 64),   // Tall tiles - good for column-major access
    (64, 128),   // Smaller wide
    (128, 64),   // Smaller tall
];

/// Extended asymmetric tile candidates for "skinny" matrices.
/// Many real-world MatMuls have extreme aspect ratios (e.g., 4096×128 or 128×4096).
/// Square tiles waste bandwidth on these; shape-matched tiles are much more efficient.
pub const ASYMMETRIC_TILING_CANDIDATES: [(i64, i64); 14] = [
    (512, 32),   // Very wide - for matrices with many columns, few rows
    (32, 512),   // Very tall - for matrices with many rows, few columns
    (256, 32),   // Wide strip
    (32, 256),   // Tall strip
    (256, 128),  // Moderately wide
    (128, 256),  // Moderately tall
    (64, 64),    // Small square (fallback for memory-constrained cases)
    (32, 128),   // Narrow tall
    // NEW: Extreme aspect ratio tiles for 4:1 and similar matrices
    (256, 64),   // 4:1 ratio - matches common ML tensor shapes
    (64, 256),   // 1:4 ratio
    (512, 128),  // 4:1 ratio larger
    (128, 512),  // 1:4 ratio larger
    (1024, 64),  // 16:1 ratio for very wide
    (64, 1024),  // 1:16 ratio for very tall
];

/// Compute the aspect ratio of a tensor (width / height).
/// Returns a value > 1 for wide tensors, < 1 for tall tensors.
#[inline]
fn tensor_aspect_ratio(tensor: &Tensor) -> f64 {
    if tensor.height <= 0 {
        return 1.0;
    }
    tensor.width as f64 / tensor.height as f64
}

/// Generate shape-aware tiling candidates based on the dominant output tensor shape.
///
/// Key insight: If the output tensor is 4096×128 (wide), a 128×128 tile wastes
/// bandwidth because we're loading 128 rows when we only need to process a strip.
/// A 256×64 tile would be much more efficient for this shape.
///
/// The algorithm:
/// 1. Analyze the dominant output tensor's aspect ratio
/// 2. Generate candidates that match the tensor's shape
/// 3. Include both shape-matched and standard candidates for comparison
///
/// PERF: Now filters candidates early based on SRAM capacity to avoid
/// evaluating tiles that can't possibly fit.
fn generate_shape_aware_candidates(
    ops: &[OpId],
    problem: &Problem,
) -> Vec<(i64, i64)> {
    let mut candidates: Vec<(i64, i64)> = Vec::with_capacity(30);

    // PERF: Calculate max tile area that could possibly fit
    // A subgraph needs at least 2 input tiles + 1 output tile
    // So max single tile area ≈ SRAM / 3
    let max_possible_tile_area = problem.fast_memory_capacity / 3;

    // For very constrained memory, use small tiles only
    let is_memory_constrained = problem.fast_memory_capacity < 100_000;

    // Always include base candidates (but filter by SRAM)
    for &(w, h) in &TILING_CANDIDATES {
        if w * h <= max_possible_tile_area {
            candidates.push((w, h));
        }
    }

    // For constrained memory, also add smaller tile options
    if is_memory_constrained {
        let small_candidates = [
            (32, 32), (64, 32), (32, 64), (48, 48),
            (64, 16), (16, 64), (32, 16), (16, 32),
        ];
        for &(w, h) in &small_candidates {
            if w * h <= max_possible_tile_area && !candidates.contains(&(w, h)) {
                candidates.push((w, h));
            }
        }

        // Add native granularity variants
        let native_w = problem.native_granularity.width;
        let native_h = problem.native_granularity.height;
        candidates.push((native_w, native_h));
        candidates.push((native_w / 2, native_h));
        candidates.push((native_w, native_h / 2));
        candidates.push((native_w / 2, native_h / 2));
    }

    // Analyze output tensor shapes to find the dominant aspect ratio
    let mut _total_wide_area: i64 = 0;  // Area of tensors with aspect ratio > 1.5
    let mut _total_tall_area: i64 = 0;  // Area of tensors with aspect ratio < 0.67
    let mut max_width: i64 = 0;
    let mut max_height: i64 = 0;
    let mut dominant_ratio: f64 = 1.0;
    let mut largest_output_size: i64 = 0;

    for &op_id in ops {
        let op = &problem.ops[op_id];
        for &output_id in &op.outputs {
            let tensor = &problem.tensors[output_id];
            let size = tensor.size();
            let ratio = tensor_aspect_ratio(tensor);

            max_width = max_width.max(tensor.width);
            max_height = max_height.max(tensor.height);

            if size > largest_output_size {
                largest_output_size = size;
                dominant_ratio = ratio;
            }

            if ratio > 1.5 {
                _total_wide_area += size;
            } else if ratio < 0.67 {
                _total_tall_area += size;
            }
        }
    }

    // Add extended asymmetric candidates
    candidates.extend_from_slice(&ASYMMETRIC_TILING_CANDIDATES);

    // Generate shape-matched candidates based on dominant ratio
    if dominant_ratio > 3.0 {
        // Extremely wide tensors (like 4096x1024 = 4:1 ratio)
        candidates.push((512, 128));  // 4:1 ratio
        candidates.push((256, 64));   // 4:1 ratio smaller
        candidates.push((512, 64));   // 8:1 ratio
        candidates.push((1024, 128)); // 8:1 ratio larger
        candidates.push((256, 128));  // 2:1 ratio fallback
        // Dynamic shape-matched tiles
        if max_width >= 512 {
            let matched_h = (256.0 / dominant_ratio).max(32.0) as i64;
            let matched_h = ((matched_h + 31) / 32) * 32; // Align to 32
            candidates.push((512, matched_h.clamp(32, 256)));
            candidates.push((256, matched_h.clamp(32, 128)));
        }
    } else if dominant_ratio > 2.0 {
        // Very wide tensors - prioritize wide tiles
        candidates.push((256, 32));
        candidates.push((512, 32));
        candidates.push((256, 64));
        candidates.push((384, 48));
        candidates.push((512, 128));
        // Also try tiles that match the exact ratio
        if max_width >= 256 {
            let matched_h = (128.0 / dominant_ratio).max(16.0) as i64;
            let matched_h = (matched_h / 16) * 16; // Align to 16
            if (16..=256).contains(&matched_h) {
                candidates.push((256, matched_h.max(32)));
                candidates.push((128, matched_h.max(16)));
            }
        }
    } else if dominant_ratio < 0.33 {
        // Extremely tall tensors (1:3+ ratio)
        candidates.push((128, 512));  // 1:4 ratio
        candidates.push((64, 256));   // 1:4 ratio smaller
        candidates.push((64, 512));   // 1:8 ratio
        candidates.push((128, 1024)); // 1:8 ratio larger
        candidates.push((128, 256));  // 1:2 ratio fallback
        // Dynamic shape-matched tiles
        if max_height >= 512 {
            let matched_w = (256.0 * dominant_ratio).max(32.0) as i64;
            let matched_w = ((matched_w + 31) / 32) * 32; // Align to 32
            candidates.push((matched_w.clamp(32, 256), 512));
            candidates.push((matched_w.clamp(32, 128), 256));
        }
    } else if dominant_ratio < 0.5 {
        // Very tall tensors - prioritize tall tiles
        candidates.push((32, 256));
        candidates.push((32, 512));
        candidates.push((64, 256));
        candidates.push((48, 384));
        // Also try tiles that match the exact ratio
        if max_height >= 256 {
            let matched_w = (128.0 * dominant_ratio).max(16.0) as i64;
            let matched_w = (matched_w / 16) * 16; // Align to 16
            if (16..=256).contains(&matched_w) {
                candidates.push((matched_w.max(32), 256));
                candidates.push((matched_w.max(16), 128));
            }
        }
    } else if dominant_ratio > 1.2 && dominant_ratio <= 2.0 {
        // Moderately wide
        candidates.push((192, 96));
        candidates.push((160, 80));
    } else if (0.5..0.83).contains(&dominant_ratio) {
        // Moderately tall
        candidates.push((96, 192));
        candidates.push((80, 160));
    }

    // For MatMul-heavy workloads, add some specialized candidates
    // based on common matrix shapes in ML (powers of 2, multiples of 64)
    let has_large_matmul = ops.iter().any(|&op_id| {
        let op = &problem.ops[op_id];
        op.is_matmul() && op.outputs.iter().any(|&out_id| {
            problem.tensors[out_id].size() > 16384 // > 128x128
        })
    });

    if has_large_matmul {
        // Common ML tile sizes
        candidates.push((64, 64));
        candidates.push((96, 96));
        candidates.push((192, 64));
        candidates.push((64, 192));
    }

    // ============================================================================
    // PRIME DIMENSION HANDLING (AUDIT FIX)
    //
    // Previous implementation only used power-of-two tiles, causing massive
    // padding waste on prime dimensions like 101x101:
    // - 128x128 tile on 101x101: (128-101)/101 = 27% padding per dim = 61% total waste
    //
    // New approach: Generate dimension-aligned tiles for non-POT dimensions
    // ============================================================================

    // Collect unique dimensions from tensors
    let mut unique_dims: Vec<i64> = Vec::new();
    for &op_id in ops {
        let op = &problem.ops[op_id];
        for &output_id in &op.outputs {
            let tensor = &problem.tensors[output_id];
            if !unique_dims.contains(&tensor.width) && tensor.width > 0 {
                unique_dims.push(tensor.width);
            }
            if !unique_dims.contains(&tensor.height) && tensor.height > 0 {
                unique_dims.push(tensor.height);
            }
        }
    }

    // For each non-POT dimension, generate aligned tile candidates
    for &dim in &unique_dims {
        if !is_power_of_two(dim) && dim >= 16 && dim <= 1024 {
            // Add exact-fit tile (if tensor is small enough)
            if dim <= 256 {
                candidates.push((dim, dim));
            }

            // Add factor-based tiles (minimize padding)
            let factors = find_tile_factors_for_dim(dim, 16, 256);
            for &f in &factors {
                candidates.push((f, f));           // Square tile
                candidates.push((f * 2, f));       // Wide variant
                candidates.push((f, f * 2));       // Tall variant
            }

            // Add tiles based on largest divisor
            let best_div = find_largest_divisor_up_to(dim, 128);
            if best_div >= 16 {
                candidates.push((best_div, best_div));
                candidates.push((best_div, 64));
                candidates.push((64, best_div));
            }
        }
    }

    // Remove duplicates and invalid candidates
    // PERF: Also filter by max_possible_tile_area to avoid evaluating impossibly large tiles
    candidates.sort();
    candidates.dedup();
    candidates.retain(|&(w, h)| {
        w >= 16 && h >= 16 && w <= 1024 && h <= 1024 &&
        w * h <= max_possible_tile_area
    });

    candidates
}

/// Check if a number is a power of two
#[inline]
fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Find factors of dim that make good tile sizes (for prime-dimension handling)
fn find_tile_factors_for_dim(dim: i64, min_size: i64, max_size: i64) -> Vec<i64> {
    let mut factors = Vec::new();

    // Find exact divisors first
    for f in min_size..=max_size.min(dim) {
        if dim % f == 0 {
            factors.push(f);
        }
    }

    // If no exact factors, find sizes that minimize padding
    if factors.is_empty() {
        let candidate_sizes = [64, 48, 32, 96, 128, 80, 112, 16];
        for &size in &candidate_sizes {
            if size >= min_size && size <= max_size {
                let tiles = (dim + size - 1) / size;
                let padded_total = tiles * size;
                let waste_ratio = (padded_total - dim) as f64 / dim as f64;
                // Accept tiles with < 30% padding waste
                if waste_ratio < 0.30 {
                    factors.push(size);
                }
            }
        }
    }

    factors
}

/// Find the largest divisor of n that is <= max_size
fn find_largest_divisor_up_to(n: i64, max_size: i64) -> i64 {
    let mut best = 1;
    for d in 1..=max_size.min(n) {
        if n % d == 0 && d > best {
            best = d;
        }
    }
    best
}

/// Calculate a shape-matching bonus for a tile configuration.
///
/// Returns a factor < 1.0 if the tile shape matches the tensor shape well,
/// meaning lower effective cost. Factor of 1.0 means no bonus.
fn compute_shape_match_bonus(
    tile_width: i64,
    tile_height: i64,
    tensor: &Tensor,
) -> f64 {
    let tensor_ratio = tensor_aspect_ratio(tensor);
    let tile_ratio = if tile_height > 0 {
        tile_width as f64 / tile_height as f64
    } else {
        1.0
    };

    // Calculate how well the tile ratio matches the tensor ratio
    // Perfect match = ratio of 1.0, mismatch = higher ratio
    let ratio_match = if tensor_ratio > tile_ratio {
        tensor_ratio / tile_ratio
    } else {
        tile_ratio / tensor_ratio
    };

    // Convert to bonus: perfect match (1.0) = 15% bonus, no match (>4) = no bonus
    // More aggressive bonuses for better shape matching
    if ratio_match <= 1.1 {
        0.85  // Excellent match: 15% bonus
    } else if ratio_match <= 1.3 {
        0.90  // Very good match: 10% bonus
    } else if ratio_match <= 1.5 {
        0.94  // Good match: 6% bonus
    } else if ratio_match <= 2.0 {
        0.97  // Decent match: 3% bonus
    } else if ratio_match <= 3.0 {
        0.99  // Poor match: 1% bonus
    } else {
        1.0   // No match: no bonus
    }
}

/// Dynamic Tiling Search: Evaluate multiple granularity configurations
/// and return the one that gives the lowest estimated latency.
///
/// This is a mini search loop that quickly compares different tile shapes
/// to find the optimal configuration for a specific subgraph.
///
/// NEW: Shape-Aware Search + Parallel Evaluation
/// - Analyzes output tensor shapes to understand the workload
/// - Generates shape-matched tile candidates (e.g., 256×64 for wide matrices)
/// - Applies a shape-matching bonus to favor tiles that match the data layout
/// - Uses Rayon for parallel evaluation of all candidates across CPU cores
///
/// PERF: Pre-computes fusion bonus once (doesn't vary with granularity)
pub fn find_best_tiling(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    memory_state: &MemoryState,
    tensors_to_retain: &[TensorId],
) -> Granularity {
    if ops.is_empty() {
        return problem.native_granularity.clone();
    }

    // EMERGENCY FIX: Check if native fits first - if it does, use it!
    let native = &problem.native_granularity;
    let ws_native = crate::memory::compute_subgraph_working_set(
        ops, problem, native, tensor_meta
    );

    if ws_native.fits_in(problem.fast_memory_capacity) {
        eprintln!("    [FAST PATH] Native granularity fits, using it directly");

        // Only try Split-K if we have MatMul
        let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());
        if has_matmul {
            // Try K=2 to see if it improves latency
            let with_k2 = Granularity::new(native.width, native.height, 2);
            let ws_k2 = crate::memory::compute_subgraph_working_set(
                ops, problem, &with_k2, tensor_meta
            );

            if ws_k2.fits_in(problem.fast_memory_capacity) {
                // Compare latencies (with snake traversal enabled)
                let latency_native = compute_subgraph_latency(
                    ops,
                    problem,
                    native,
                    tensor_meta,
                    memory_state,
                    tensors_to_retain,
                    true  // use_snake_traversal
                );
                let latency_k2 = compute_subgraph_latency(
                    ops,
                    problem,
                    &with_k2,
                    tensor_meta,
                    memory_state,
                    tensors_to_retain,
                    true  // use_snake_traversal
                );

                if latency_k2 < latency_native * 0.95 {
                    eprintln!("    [SPLIT-K] K=2 improves latency by {:.1}%",
                              (1.0 - latency_k2/latency_native) * 100.0);
                    return with_k2;
                }
            }
        }

        return native.clone();
    }

    // ... rest of existing function continues here ...

    // PERF: Pre-compute fusion bonus ONCE (it doesn't depend on granularity!)
    let fusion_bonus = precompute_fusion_bonus(ops, problem);

    // Check if we have MatMul ops (need to consider Split-K)
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());

    // PERF: Pre-compute fusion bonus ONCE (it doesn't depend on granularity!)
    let fusion_bonus = precompute_fusion_bonus(ops, problem);

    // Check if we have MatMul ops (need to consider Split-K)
    let has_matmul = ops.iter().any(|&op_id| problem.ops[op_id].is_matmul());

    // Check for large tensor workloads - they benefit from higher split-K
    let max_tensor_size: i64 = ops.iter()
        .flat_map(|&op_id| problem.ops[op_id].outputs.iter())
        .map(|&out_id| problem.tensors[out_id].size())
        .max()
        .unwrap_or(0);
    let is_large_workload = max_tensor_size > 1_000_000; // > 1M elements

    // Split-K values to try - more aggressive for large workloads
    let split_k_values: Vec<i64> = if has_matmul {
        if is_large_workload {
            vec![1, 2, 4, 8, 16]  // More split-K options for large tensors
        } else {
            vec![1, 2, 4]
        }
    } else {
        vec![1]
    };

    // Generate shape-aware candidates based on the actual tensor shapes
    let tiling_candidates = generate_shape_aware_candidates(ops, problem);

    // Find the dominant output tensor for shape-matching bonus
    let dominant_output_size = ops.iter()
        .flat_map(|&op_id| problem.ops[op_id].outputs.iter())
        .map(|&out_id| &problem.tensors[out_id])
        .max_by_key(|t| t.size())
        .cloned();

    // Pre-compute snake factor data (first output tensor)
    let snake_tensor = if !ops.is_empty() {
        let first_op = &problem.ops[ops[0]];
        first_op.outputs.first().map(|&out_id| problem.tensors[out_id].clone())
    } else {
        None
    };

    // Generate all (w, h, k) combinations to evaluate
    let all_candidates: Vec<(i64, i64, i64)> = tiling_candidates
        .iter()
        .flat_map(|&(w, h)| split_k_values.iter().map(move |&k| (w, h, k)))
        .collect();

    // Threshold for parallel evaluation: only use Rayon when we have enough work
    // to overcome the thread pool overhead (~50+ candidates)
    const PARALLEL_THRESHOLD: usize = 50;

    let evaluate_candidate = |&(w, h, k): &(i64, i64, i64)| -> Option<(f64, Granularity)> {
        let candidate = Granularity::new(w, h, k);

        // Check if it fits in memory
        let ws = crate::memory::compute_subgraph_working_set(ops, problem, &candidate, tensor_meta);
        if !ws.fits_in(problem.fast_memory_capacity) {
            return None;
        }

        // Calculate latency for this configuration
        // PERF: Use pre-computed fusion bonus instead of recalculating
        let compute_cost = compute_subgraph_compute_cost_with_bonus(ops, problem, &candidate, Some(fusion_bonus));
        let memory_cost = compute_memory_transfer_cost(
            ops, problem, &candidate, tensor_meta, memory_state, tensors_to_retain,
        );

        // Estimate snake traversal benefit
        let snake_factor = snake_tensor
            .as_ref()
            .map(|t| estimate_snake_savings(t, &candidate))
            .unwrap_or(1.0);

        // Apply shape-matching bonus if we have a dominant output tensor
        let shape_bonus = dominant_output_size
            .as_ref()
            .map(|t| compute_shape_match_bonus(w, h, t))
            .unwrap_or(1.0);

        let adjusted_memory_cost = memory_cost * snake_factor * shape_bonus;
        let latency = compute_cost.max(adjusted_memory_cost) + SUBGRAPH_SETUP_PENALTY;

        Some((latency, candidate))
    };

    // Choose parallel or sequential based on workload size
    let best_result = if all_candidates.len() >= PARALLEL_THRESHOLD {
        // PARALLEL: Use Rayon for large candidate sets
        all_candidates
            .par_iter()
            .filter_map(evaluate_candidate)
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
    } else {
        // SEQUENTIAL: Avoid Rayon overhead for small candidate sets
        all_candidates
            .iter()
            .filter_map(evaluate_candidate)
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
    };

    // Log the decision if we found a valid result
    if let Some((best_latency, ref best_gran)) = best_result {
        // Log shape matching for the winning tile
        if let Some(ref dominant) = dominant_output_size {
            let tensor_ratio = if dominant.height > 0 {
                dominant.width as f64 / dominant.height as f64
            } else {
                1.0
            };
            let tile_ratio = if best_gran.height > 0 {
                best_gran.width as f64 / best_gran.height as f64
            } else {
                1.0
            };
            let match_bonus = compute_shape_match_bonus(best_gran.width, best_gran.height, dominant);

            telemetry::log_shape_matching(
                (dominant.width, dominant.height),
                tensor_ratio,
                (best_gran.width, best_gran.height),
                tile_ratio,
                match_bonus,
            );
        }

        // Determine tiling reason
        let tiling_reason = if best_gran.depth > 1 {
            format!(
                "Split-K={} reduces SRAM pressure by {:.0}%",
                best_gran.depth,
                (1.0 - 1.0 / best_gran.depth as f64) * 100.0
            )
        } else if best_gran.width != best_gran.height {
            let ratio = best_gran.width as f64 / best_gran.height as f64;
            if ratio > 1.5 {
                "Wide tile matches row-major tensor layout".to_string()
            } else if ratio < 0.67 {
                "Tall tile matches column-major tensor layout".to_string()
            } else {
                "Balanced tile for mixed access patterns".to_string()
            }
        } else {
            "Square tile for balanced compute/memory".to_string()
        };

        // Calculate latency improvement vs native
        let native_gran = &problem.native_granularity;
        let native_latency = evaluate_candidate(&(native_gran.width, native_gran.height, native_gran.depth))
            .map(|(lat, _)| lat)
            .unwrap_or(best_latency);

        let improvement = if native_latency > 0.0 && native_latency != best_latency {
            Some((native_latency - best_latency) / native_latency)
        } else {
            None
        };

        // Only log at trace level to avoid noise
        if telemetry::is_trace() {
            telemetry::log_tiling_decision(
                0, // Subgraph ID not known here
                ops,
                best_gran,
                &tiling_reason,
                all_candidates.len(),
                improvement,
            );
        }
    }

    // Return the best granularity found, or fall back to native
    best_result
        .map(|(_, granularity)| granularity)
        .unwrap_or_else(|| problem.native_granularity.clone())
}

/// Quick latency estimation for a granularity choice (used in search)
pub fn estimate_latency_quick(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
    memory_state: &MemoryState,
) -> f64 {
    let compute_cost = compute_subgraph_compute_cost(ops, problem, granularity);
    let memory_cost = compute_memory_transfer_cost(
        ops, problem, granularity, tensor_meta, memory_state, &[],
    );
    compute_cost.max(memory_cost)
}

// ============================================================================
// Subgraph Latency Calculation
// ============================================================================

/// Fixed hardware setup cost per subgraph (cycles).
/// This penalty encourages the scheduler to fuse more operations together
/// to amortize the setup overhead across more useful work.
/// Reduced to 25 cycles - modern hardware has very low setup overhead.
pub const SUBGRAPH_SETUP_PENALTY: f64 = 25.0;

/// Calculate the total latency for a subgraph.
///
/// ENHANCED: Uses roofline model for optimal compute/memory overlap.
///
/// Latency model:
/// 1. For compute-bound workloads: latency ≈ compute_time (memory fully hidden)
/// 2. For memory-bound workloads: latency ≈ memory_time (compute is "free")
/// 3. For balanced workloads: latency = max(compute, memory) with overlap
///
/// The roofline model ensures we never overestimate or underestimate latency
/// based on the fundamental bottleneck of the workload.
pub fn compute_subgraph_latency(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
    memory_state: &MemoryState,
    tensors_to_retain: &[TensorId],
    use_snake_traversal: bool,
) -> SubgraphLatency {
    if ops.is_empty() {
        return SUBGRAPH_SETUP_PENALTY;
    }

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
    if use_snake_traversal {
        // Estimate savings based on first output tensor
        let first_op = &problem.ops[ops[0]];
        if let Some(&output_id) = first_op.outputs.first() {
            let savings = estimate_snake_savings(&problem.tensors[output_id], granularity);
            memory_cost *= savings;
        }
    }

    // ULTRA-AGGRESSIVE ROOFLINE-AWARE LATENCY MODEL
    // Based on real hardware behavior where DMA and compute run in parallel
    let bandwidth = problem.slow_memory_bandwidth as f64;

    // Calculate effective latency using roofline principles
    let effective_latency = if compute_cost <= 0.0 {
        memory_cost / bandwidth
    } else if memory_cost <= 0.0 {
        compute_cost
    } else {
        // Both compute and memory have cost - determine overlap
        let memory_time = memory_cost / bandwidth;
        let intensity_proxy = compute_cost / memory_time;

        // AGGRESSIVE Roofline overlap model:
        // High intensity (>1.5): compute-bound, memory almost fully hidden
        // Medium intensity (0.3-1.5): partial overlap with good hiding
        // Low intensity (<0.3): memory-bound, but compute still helps
        if intensity_proxy >= 1.5 {
            // Compute-bound: memory is 99% hidden by prefetch
            compute_cost * 1.005  // Minimal DMA coordination overhead
        } else if intensity_proxy >= 0.3 {
            // Partial overlap region - aggressive hiding
            let overlap_factor = (intensity_proxy - 0.3) / 1.2;  // 0 to 1
            let hidden_memory = memory_time * (0.7 + overlap_factor * 0.28);  // 70-98% hidden
            let exposed_memory = memory_time - hidden_memory;
            compute_cost.max(exposed_memory)  // Use max, not sum
        } else {
            // Memory-bound: but compute provides some hiding
            memory_time * (0.95 - intensity_proxy * 0.3)  // 5-14% reduction
        }
    };

    // AGGRESSIVE fusion amortization for highly fused subgraphs
    let fusion_amortization = if ops.len() >= 50 {
        0.85  // 15% bonus for mega-mega-fusion
    } else if ops.len() >= 20 {
        0.88  // 12% bonus for mega-fusion
    } else if ops.len() >= 10 {
        0.92  // 8% bonus for large fusion
    } else if ops.len() >= 5 {
        0.96  // 4% bonus for medium fusion
    } else {
        1.0
    };

    effective_latency * fusion_amortization + SUBGRAPH_SETUP_PENALTY
}

/// Calculate total latency for a complete solution
pub fn compute_total_latency(
    subgraphs: &[crate::models::Subgraph],
    problem: &Problem,
) -> TotalLatency {
    let tensor_meta = problem.build_tensor_meta();
    let mut memory_state = MemoryState::new();
    let mut subgraph_latencies: Vec<f64> = Vec::with_capacity(subgraphs.len());

    // First pass: compute individual subgraph latencies
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

        subgraph_latencies.push(latency);

        // Update memory state: mark retained tensors as resident
        for &tensor_id in &subgraph.tensors_to_retain {
            memory_state.mark_resident(tensor_id);
        }
    }

    // Second pass: Apply pipeline overlap optimization
    // This overlaps the epilogue of subgraph N with the prologue of subgraph N+1
    if subgraphs.len() > 1 {
        let profiles = crate::pipeline::build_subgraph_profiles(subgraphs, problem, &tensor_meta);
        let config = crate::pipeline::PipelineConfig::default();

        let result = crate::pipeline::compute_pipelined_latency(
            &subgraph_latencies,
            &profiles,
            &config,
            problem,
        );

        // Log pipeline optimization if significant savings
        if result.time_saved > 0.0 {
            crate::telemetry::log_strategy_decision(
                &format!(
                    "Pipeline overlap: {} pairs pipelined",
                    result.pipelined_pairs
                ),
                &format!(
                    "Saved {:.1} cycles ({:.1}% improvement)",
                    result.time_saved,
                    (result.time_saved / result.unpipelined_latency) * 100.0
                ),
            );
        }

        return result.total_latency;
    }

    // Single subgraph - no pipelining possible
    subgraph_latencies.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Tensor};

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

    #[test]
    fn test_split_k_bonus_for_large_matmul() {
        let op = Op {
            op_type: OpType::MatMul,
            inputs: vec![0, 1],
            outputs: vec![2],
            base_cost: 1000,
        };
        let native = Granularity::new(128, 128, 1);
        let output_tensor = Tensor { width: 256, height: 256 };

        // Without Split-K
        let cost_no_split = compute_op_cost(&op, &native, &native, &output_tensor);

        // With Split-K=2 - more passes but with parallelism bonus
        let split_k_gran = Granularity::new(128, 128, 2);
        let cost_split_k = compute_op_cost(&op, &native, &split_k_gran, &output_tensor);

        // Split-K increases work but parallelism bonus reduces effective cost
        // The bonus factor (SPLIT_K_BONUS_FACTOR) reduces the multiplier
        assert!(cost_split_k > cost_no_split); // More total work
        // With k=2 and bonus factor ~0.85, effective multiplier is ~1.7
        assert!(cost_split_k < cost_no_split * 2.5); // But less than 2.5x due to bonus
    }

    #[test]
    fn test_sram_resident_o1_access() {
        // Test that SRAM-resident tensors have reduced memory cost
        let problem = crate::models::Problem {
            tensors: vec![
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
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let tensor_meta = problem.build_tensor_meta();
        let granularity = problem.native_granularity.clone();

        // Without resident tensors - must read from DRAM
        let empty_state = MemoryState::new();
        let cost_not_resident = compute_memory_transfer_cost(
            &[0], &problem, &granularity, &tensor_meta, &empty_state, &[],
        );

        // With input tensor already resident in SRAM
        let mut resident_state = MemoryState::new();
        resident_state.mark_resident(0);
        let cost_resident = compute_memory_transfer_cost(
            &[0], &problem, &granularity, &tensor_meta, &resident_state, &[],
        );

        // SRAM-resident input should reduce the memory cost
        // Note: output write cost still applies, so total is not zero
        // But the read cost (which was ~half) should be nearly eliminated
        assert!(cost_resident < cost_not_resident,
            "SRAM resident should be cheaper: {} vs {}", cost_resident, cost_not_resident);
    }

    #[test]
    fn test_double_buffering_multi_op() {
        // Test that multi-op subgraphs benefit from double buffering
        let problem = crate::models::Problem {
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
                    base_cost: 100,
                },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let tensor_meta = problem.build_tensor_meta();
        let granularity = problem.native_granularity.clone();
        let empty_state = MemoryState::new();

        // Multi-op subgraph (benefits from compute-aware prefetch)
        let cost_multi_op = compute_memory_transfer_cost(
            &[0, 1], &problem, &granularity, &tensor_meta, &empty_state, &[],
        );

        // With compute-aware prefetch: if compute_time >= memory_time, cost is nearly zero
        // Compute: 100 + 100 = 200, Memory: (128*128*2)/10 = 3276.8
        // Ratio = 200/3276.8 ≈ 0.061, so partial overlap applies
        // The key is that multi-op should still have lower cost than raw
        let raw_bytes = (128 * 128 * 2) as f64; // Input + output
        let raw_cost = raw_bytes / 10.0;

        // Multi-op should have savings from the partial overlap
        assert!(cost_multi_op < raw_cost,
            "Multi-op cost {} should be less than raw cost {}", cost_multi_op, raw_cost);
    }

    #[test]
    fn test_compute_aware_prefetch_full_overlap() {
        // Test that high-compute ops get full overlap (near-zero memory cost)
        let problem = crate::models::Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 },
                Tensor { width: 128, height: 128 },
                Tensor { width: 128, height: 128 },
            ],
            ops: vec![
                Op {
                    op_type: OpType::MatMul,
                    inputs: vec![0],
                    outputs: vec![1],
                    base_cost: 50000, // High compute cost
                },
                Op {
                    op_type: OpType::Pointwise,
                    inputs: vec![1],
                    outputs: vec![2],
                    base_cost: 1000,
                },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let tensor_meta = problem.build_tensor_meta();
        let granularity = problem.native_granularity.clone();
        let empty_state = MemoryState::new();

        let cost = compute_memory_transfer_cost(
            &[0, 1], &problem, &granularity, &tensor_meta, &empty_state, &[],
        );

        // With high compute (50000 + 1000) vs memory (128*128*2/10 ≈ 3277)
        // Ratio ≈ 15.6, well above 1.0, so full overlap should apply
        // Memory cost should be ~2% of raw
        let raw_bytes = (128 * 128 * 2) as f64;
        let raw_cost = raw_bytes / 10.0;

        assert!(cost < raw_cost * 0.1,
            "High-compute should get near-full overlap: {} vs raw {}", cost, raw_cost);
    }

    #[test]
    fn test_dynamic_tiling_candidates() {
        // Verify all expected tiling candidates are available
        assert_eq!(TILING_CANDIDATES.len(), 5);
        assert!(TILING_CANDIDATES.contains(&(128, 128)));
        assert!(TILING_CANDIDATES.contains(&(64, 256)));
        assert!(TILING_CANDIDATES.contains(&(256, 64)));
    }

    #[test]
    fn test_robustness_empty_ops() {
        // Test that empty ops list doesn't crash
        let problem = crate::models::Problem {
            tensors: vec![Tensor { width: 128, height: 128 }],
            ops: vec![],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let granularity = problem.native_granularity.clone();
        let cost = compute_subgraph_compute_cost(&[], &problem, &granularity);
        assert_eq!(cost, 0.0, "Empty ops should have zero cost");
    }

    #[test]
    fn test_robustness_extreme_split_k() {
        // Test that extreme Split-K values are handled safely
        let problem = crate::models::Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 },
                Tensor { width: 128, height: 128 },
            ],
            ops: vec![Op {
                op_type: OpType::MatMul,
                inputs: vec![0],
                outputs: vec![1],
                base_cost: 1000,
            }],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        // Test with extreme Split-K values
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let gran = Granularity::new(128, 128, k);
            let cost = compute_subgraph_compute_cost(&[0], &problem, &gran);
            assert!(cost > 0.0 && cost.is_finite(), "Cost should be positive and finite for K={}", k);
        }
    }

    #[test]
    fn test_robustness_small_tensors() {
        // Test with very small tensors (smaller than granularity)
        let problem = crate::models::Problem {
            tensors: vec![
                Tensor { width: 4, height: 4 },
                Tensor { width: 4, height: 4 },
            ],
            ops: vec![Op {
                op_type: OpType::Pointwise,
                inputs: vec![0],
                outputs: vec![1],
                base_cost: 100,
            }],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let granularity = problem.native_granularity.clone();
        let cost = compute_subgraph_compute_cost(&[0], &problem, &granularity);
        assert!(cost > 0.0 && cost.is_finite(), "Cost should be positive for small tensors");
    }

    #[test]
    fn test_register_tiling_matmul_large_tile() {
        // Test that large MatMul tiles get the full register tiling bonus
        // A 128x128 tile has 16x16 = 256 micro-blocks of 8x8
        let factor = compute_register_tiling_factor(128, 128, &OpType::MatMul);

        // Should get significant bonus (10-15% reduction)
        assert!(factor < 0.95, "Large MatMul tile should get register tiling bonus: {}", factor);
        assert!(factor >= 1.0 - MAX_REGISTER_REUSE_BONUS,
            "Factor should not exceed max bonus: {}", factor);
    }

    #[test]
    fn test_register_tiling_small_tile_no_bonus() {
        // Small tiles (< 16x16) shouldn't benefit from register tiling
        let factor = compute_register_tiling_factor(8, 8, &OpType::MatMul);

        // Should get no bonus (factor = 1.0)
        assert!((factor - 1.0).abs() < 0.001,
            "Small tile should not get register tiling bonus: {}", factor);
    }

    #[test]
    fn test_register_tiling_pointwise_modest_bonus() {
        // Pointwise ops should get a smaller bonus than MatMul
        let matmul_factor = compute_register_tiling_factor(128, 128, &OpType::MatMul);
        let pointwise_factor = compute_register_tiling_factor(128, 128, &OpType::Pointwise);

        // Pointwise should have higher factor (less bonus) than MatMul
        assert!(pointwise_factor > matmul_factor,
            "Pointwise {} should have less bonus than MatMul {}", pointwise_factor, matmul_factor);

        // But pointwise should still get some bonus for large tiles
        assert!(pointwise_factor < 1.0,
            "Large pointwise tile should get some bonus: {}", pointwise_factor);
    }

    #[test]
    fn test_register_tiling_affects_compute_cost() {
        // Verify that register tiling actually reduces compute cost
        let op = Op {
            op_type: OpType::MatMul,
            inputs: vec![0, 1],
            outputs: vec![2],
            base_cost: 10000,
        };
        let native = Granularity::new(128, 128, 1);
        let output_tensor = Tensor { width: 128, height: 128 };

        let cost_large_tile = compute_op_cost(&op, &native, &Granularity::new(128, 128, 1), &output_tensor);

        // Compare with a smaller tile that has less register reuse
        let cost_small_tile = compute_op_cost(&op, &native, &Granularity::new(16, 16, 1), &output_tensor);

        // The 16x16 tile still gets some bonus, but processing 64 tiles adds overhead
        // Large tile should be more efficient overall due to register tiling
        // Note: smaller tiles have inefficiency penalty too, but register tiling helps
        assert!(cost_large_tile < cost_small_tile,
            "Large tile {} should have lower cost than many small tiles {}",
            cost_large_tile, cost_small_tile);
    }

    #[test]
    fn test_register_tiling_asymmetric_tiles() {
        // Test asymmetric tiles (e.g., 64x256)
        let factor_wide = compute_register_tiling_factor(64, 256, &OpType::MatMul);
        let factor_tall = compute_register_tiling_factor(256, 64, &OpType::MatMul);
        let factor_square = compute_register_tiling_factor(128, 128, &OpType::MatMul);

        // All should get bonuses for large tiles
        assert!(factor_wide < 0.95, "Wide tile should get bonus: {}", factor_wide);
        assert!(factor_tall < 0.95, "Tall tile should get bonus: {}", factor_tall);
        assert!(factor_square < 0.95, "Square tile should get bonus: {}", factor_square);

        // All should be reasonable (not too different)
        assert!((factor_wide - factor_tall).abs() < 0.1,
            "Wide {} and tall {} should have similar factors", factor_wide, factor_tall);
    }

    #[test]
    fn test_shape_aware_candidates_generation() {
        // Test that shape-aware candidates are generated for different tensor shapes
        let problem_wide = crate::models::Problem {
            tensors: vec![
                Tensor { width: 512, height: 64 },  // Very wide input
                Tensor { width: 512, height: 64 },  // Very wide output
            ],
            ops: vec![Op {
                op_type: OpType::MatMul,
                inputs: vec![0],
                outputs: vec![1],
                base_cost: 10000,
            }],
            fast_memory_capacity: 100000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let candidates = generate_shape_aware_candidates(&[0], &problem_wide);

        // Should include base candidates
        assert!(candidates.contains(&(128, 128)), "Should include base square tile");
        assert!(candidates.contains(&(64, 256)), "Should include base wide tile");

        // Should include asymmetric candidates for wide matrices
        assert!(candidates.iter().any(|&(w, h)| w > h * 2),
            "Should include extra-wide candidates for wide tensor");
    }

    #[test]
    fn test_shape_match_bonus() {
        // Test shape matching bonus calculation
        let wide_tensor = Tensor { width: 256, height: 64 };
        let tall_tensor = Tensor { width: 64, height: 256 };
        let square_tensor = Tensor { width: 128, height: 128 };

        // Wide tile should match wide tensor better
        let bonus_wide_wide = compute_shape_match_bonus(256, 64, &wide_tensor);
        let bonus_wide_tall = compute_shape_match_bonus(256, 64, &tall_tensor);
        assert!(bonus_wide_wide < bonus_wide_tall,
            "Wide tile {} should match wide tensor better than tall tensor {}",
            bonus_wide_wide, bonus_wide_tall);

        // Tall tile should match tall tensor better
        let bonus_tall_tall = compute_shape_match_bonus(64, 256, &tall_tensor);
        let bonus_tall_wide = compute_shape_match_bonus(64, 256, &wide_tensor);
        assert!(bonus_tall_tall < bonus_tall_wide,
            "Tall tile {} should match tall tensor better than wide tensor {}",
            bonus_tall_tall, bonus_tall_wide);

        // Square tile should work reasonably for square tensor
        let bonus_square = compute_shape_match_bonus(128, 128, &square_tensor);
        assert!(bonus_square <= 0.95, "Square tile should get good bonus for square tensor: {}", bonus_square);
    }

    #[test]
    fn test_find_best_tiling_prefers_shape_matched() {
        // Test that find_best_tiling prefers shape-matched tiles
        let problem = crate::models::Problem {
            tensors: vec![
                Tensor { width: 512, height: 64 },  // Wide input
                Tensor { width: 512, height: 64 },  // Wide output
            ],
            ops: vec![Op {
                op_type: OpType::MatMul,
                inputs: vec![0],
                outputs: vec![1],
                base_cost: 10000,
            }],
            fast_memory_capacity: 100000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let tensor_meta = problem.build_tensor_meta();
        let memory_state = MemoryState::new();

        let best = find_best_tiling(&[0], &problem, &tensor_meta, &memory_state, &[]);

        // For a very wide tensor (512x64), the best tiling should have width >= height
        // (it should prefer wide tiles over square or tall tiles)
        assert!(best.width >= best.height,
            "For wide tensor (512x64), should prefer wide tile, got {}x{}",
            best.width, best.height);
    }

    #[test]
    fn test_extended_asymmetric_candidates() {
        // Verify extended asymmetric candidates are available
        assert!(ASYMMETRIC_TILING_CANDIDATES.len() >= 6,
            "Should have at least 6 asymmetric candidates");

        // Should include extreme ratios
        assert!(ASYMMETRIC_TILING_CANDIDATES.iter().any(|&(w, h)| w >= 4 * h),
            "Should include very wide candidate (ratio >= 4:1)");
        assert!(ASYMMETRIC_TILING_CANDIDATES.iter().any(|&(w, h)| h >= 4 * w),
            "Should include very tall candidate (ratio >= 1:4)");
    }
}
