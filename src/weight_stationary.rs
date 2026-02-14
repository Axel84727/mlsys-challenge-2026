//! Weight Stationary Optimization - Dynamic Weight Prefetching
//!
//! In neural networks and similar workloads, weight tensors are frequently reused
//! across multiple operations. This module implements a "sticky bit" mechanism
//! to keep high-value reused tensors resident in SRAM across consecutive operations.
//!
//! KEY INSIGHT: When tensor A is used by Op[i] and Op[i+1], evicting it after Op[i]
//! and reloading it for Op[i+1] wastes 2x bandwidth (one write, one read).
//! Keeping it resident costs 0 bandwidth - a 100% savings for that tensor.
//!
//! ROBUSTNESS AGAINST OVERFITTING:
//! 1. Weight detection is based on USAGE PATTERNS, not hardcoded heuristics
//! 2. Sticky decisions are made dynamically based on available SRAM capacity
//! 3. Conservative capacity reservation prevents OOM from over-retention
//! 4. Graceful degradation when memory is constrained
//!
//! The system identifies "weight-like" tensors through:
//! - Multiple consumers (reuse count >= 2)
//! - Consumers span multiple subgraphs (cross-subgraph reuse)
//! - Tensor is read-only (input to ops, never produced within graph)

use crate::models::{OpId, Problem, TensorId, TensorMeta};
use std::collections::HashSet;

// ============================================================================
// Weight Classification
// ============================================================================

/// Classification of a tensor based on its usage pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// Graph input that is never modified (weights, constants)
    Weight,
    /// Intermediate activation produced and consumed within the graph
    Activation,
    /// Graph output (final results)
    Output,
    /// Input that is also an output (in-place operations)
    InOut,
}

/// Information about a weight tensor's reuse pattern
#[derive(Debug, Clone)]
pub struct WeightInfo {
    pub tensor_id: TensorId,
    pub role: TensorRole,
    /// All operations that consume this tensor
    pub consumers: Vec<OpId>,
    /// Size in memory units
    pub size: i64,
    /// Whether this tensor should be "sticky" (never evicted if possible)
    pub sticky: bool,
    /// Priority score for retention (higher = more valuable to keep)
    pub retention_priority: i64,
    /// Estimated bandwidth savings from keeping resident (bytes)
    pub bandwidth_savings: i64,
}

/// Result of weight analysis for the entire graph
#[derive(Debug, Clone)]
pub struct WeightAnalysis {
    /// Information about each tensor
    pub tensor_info: Vec<WeightInfo>,
    /// Tensors marked as sticky (should never be evicted if possible)
    pub sticky_tensors: HashSet<TensorId>,
    /// Total size of all sticky tensors
    pub sticky_total_size: i64,
    /// Estimated total bandwidth savings from sticky retention
    pub total_bandwidth_savings: i64,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for weight stationary optimization
#[derive(Debug, Clone)]
pub struct WeightStationaryConfig {
    /// Enable weight stationary optimization
    pub enabled: bool,
    /// Minimum reuse count to consider a tensor as "weight-like"
    pub min_reuse_count: usize,
    /// Maximum fraction of SRAM that sticky tensors can occupy (0.0 - 1.0)
    pub max_sram_fraction: f64,
    /// Minimum bandwidth savings (bytes) to justify keeping a tensor sticky
    pub min_bandwidth_savings: i64,
    /// Whether to consider graph inputs as potential weights
    pub include_graph_inputs: bool,
}

impl Default for WeightStationaryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_reuse_count: 2,           // At least 2 consumers
            max_sram_fraction: 0.3,       // Max 30% of SRAM for sticky tensors
            min_bandwidth_savings: 1000,  // At least 1000 bytes saved
            include_graph_inputs: true,   // Graph inputs are prime weight candidates
        }
    }
}

// ============================================================================
// Weight Analysis
// ============================================================================

/// Analyze the computation graph to identify weight tensors and their reuse patterns.
///
/// This function classifies tensors based on their usage patterns:
/// - Weights: Graph inputs with multiple consumers
/// - Activations: Produced and consumed within the graph
/// - Outputs: Final results of the computation
pub fn analyze_weights(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    config: &WeightStationaryConfig,
) -> WeightAnalysis {
    let num_tensors = problem.tensors.len();
    let mut tensor_info: Vec<WeightInfo> = Vec::with_capacity(num_tensors);
    let mut sticky_tensors: HashSet<TensorId> = HashSet::new();
    let mut sticky_total_size: i64 = 0;
    let mut total_bandwidth_savings: i64 = 0;

    // Maximum SRAM we can allocate to sticky tensors
    let max_sticky_capacity = (problem.fast_memory_capacity as f64 * config.max_sram_fraction) as i64;

    for (tensor_id, meta) in tensor_meta.iter().enumerate() {
        let tensor = &problem.tensors[tensor_id];
        let size = tensor.size();

        // Classify tensor role
        let role = classify_tensor_role(meta);

        // Count consumers (reuse)
        let consumers = meta.consumers.clone();
        let reuse_count = consumers.len();

        // Calculate bandwidth savings if kept resident
        // Each additional use after the first saves one full read from DRAM
        let bandwidth_savings = if reuse_count > 1 {
            size * (reuse_count as i64 - 1)
        } else {
            0
        };

        // Calculate retention priority
        let retention_priority = calculate_retention_priority(
            size,
            reuse_count,
            &role,
            bandwidth_savings,
        );

        // Determine if this tensor should be sticky
        let should_be_sticky = config.enabled
            && reuse_count >= config.min_reuse_count
            && bandwidth_savings >= config.min_bandwidth_savings
            && (role == TensorRole::Weight || (config.include_graph_inputs && meta.producer.is_none()));

        let info = WeightInfo {
            tensor_id,
            role,
            consumers,
            size,
            sticky: should_be_sticky,
            retention_priority,
            bandwidth_savings,
        };

        tensor_info.push(info);
    }

    // Sort by retention priority and select sticky tensors within capacity
    let mut sticky_candidates: Vec<(TensorId, i64, i64)> = tensor_info
        .iter()
        .filter(|info| info.sticky)
        .map(|info| (info.tensor_id, info.retention_priority, info.size))
        .collect();

    sticky_candidates.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by priority (highest first)

    // Greedily select sticky tensors within capacity
    for (tensor_id, _priority, size) in sticky_candidates {
        if sticky_total_size + size <= max_sticky_capacity {
            sticky_tensors.insert(tensor_id);
            sticky_total_size += size;
            total_bandwidth_savings += tensor_info[tensor_id].bandwidth_savings;
            tensor_info[tensor_id].sticky = true;
        } else {
            // Can't fit this tensor, mark as non-sticky
            tensor_info[tensor_id].sticky = false;
        }
    }

    WeightAnalysis {
        tensor_info,
        sticky_tensors,
        sticky_total_size,
        total_bandwidth_savings,
    }
}

/// Classify a tensor's role based on its metadata
fn classify_tensor_role(meta: &TensorMeta) -> TensorRole {
    let is_input = meta.producer.is_none();
    let is_output = meta.is_output;

    match (is_input, is_output) {
        (true, true) => TensorRole::InOut,
        (true, false) => TensorRole::Weight,
        (false, true) => TensorRole::Output,
        (false, false) => TensorRole::Activation,
    }
}

/// Calculate retention priority for a tensor
///
/// Higher priority = more valuable to keep in SRAM
fn calculate_retention_priority(
    size: i64,
    reuse_count: usize,
    role: &TensorRole,
    bandwidth_savings: i64,
) -> i64 {
    let mut priority: i64 = 0;

    // Base priority from bandwidth savings
    priority += bandwidth_savings;

    // Efficiency bonus: high reuse with small size is most valuable
    // (saves the most bandwidth per SRAM byte)
    let efficiency = if size > 0 {
        (reuse_count as f64 * 1000.0) / (size as f64)
    } else {
        0.0
    };
    priority += (efficiency * 10000.0) as i64;

    // Role-based bonus
    match role {
        TensorRole::Weight => priority += 50000,    // Weights are prime candidates
        TensorRole::Activation => priority += 10000, // Activations can also be valuable
        TensorRole::Output => priority += 0,         // Outputs usually evicted anyway
        TensorRole::InOut => priority += 25000,      // Mixed case
    }

    // Reuse multiplier
    priority += (reuse_count as i64) * 5000;

    priority
}

// ============================================================================
// Integration with Retention Analysis
// ============================================================================

/// Enhance retention candidates with weight stationary information.
///
/// This function modifies the retention analysis to:
/// 1. Always include sticky tensors in retention candidates
/// 2. Boost priority of weight-like tensors
/// 3. Reserve capacity for sticky tensors
pub fn enhance_retention_with_weights(
    base_candidates: &[TensorId],
    weight_analysis: &WeightAnalysis,
    remaining_ops: &[OpId],
    _problem: &Problem,
    tensor_meta: &[TensorMeta],
    available_capacity: i64,
) -> (Vec<TensorId>, i64) {
    let remaining_set: HashSet<OpId> = remaining_ops.iter().copied().collect();

    // Start with sticky tensors that have remaining consumers
    let mut enhanced_candidates: Vec<(TensorId, i64, i64)> = Vec::new();
    let mut sticky_reserved: i64 = 0;

    // First pass: Add sticky tensors with remaining consumers
    for &tensor_id in &weight_analysis.sticky_tensors {
        let info = &weight_analysis.tensor_info[tensor_id];
        let meta = &tensor_meta[tensor_id];

        // Check if any remaining ops need this tensor
        let has_remaining_consumer = meta.consumers.iter().any(|c| remaining_set.contains(c));

        if has_remaining_consumer {
            // Sticky tensors get maximum priority boost
            let boosted_priority = info.retention_priority + 1_000_000;
            enhanced_candidates.push((tensor_id, boosted_priority, info.size));
            sticky_reserved += info.size;
        }
    }

    // Second pass: Add non-sticky candidates
    for &tensor_id in base_candidates {
        // Skip if already added as sticky
        if weight_analysis.sticky_tensors.contains(&tensor_id) {
            continue;
        }

        // Skip if tensor_id is out of bounds
        if tensor_id >= weight_analysis.tensor_info.len() || tensor_id >= tensor_meta.len() {
            continue;
        }

        let info = &weight_analysis.tensor_info[tensor_id];
        let meta = &tensor_meta[tensor_id];

        // Check if any remaining ops need this tensor
        let has_remaining_consumer = meta.consumers.iter().any(|c| remaining_set.contains(c));

        if has_remaining_consumer {
            enhanced_candidates.push((tensor_id, info.retention_priority, info.size));
        }
    }

    // Sort by priority
    enhanced_candidates.sort_by(|a, b| b.1.cmp(&a.1));

    // Select tensors within capacity
    let mut selected: Vec<TensorId> = Vec::new();
    let mut used: i64 = 0;

    for (tensor_id, _priority, size) in enhanced_candidates {
        if used + size <= available_capacity {
            selected.push(tensor_id);
            used += size;
        }
    }

    (selected, sticky_reserved)
}

/// Check if a tensor should be kept resident based on upcoming operations.
///
/// Returns true if the tensor is:
/// 1. Marked as sticky AND
/// 2. Will be used by any of the next N operations
pub fn should_retain_tensor(
    tensor_id: TensorId,
    weight_analysis: &WeightAnalysis,
    upcoming_ops: &[OpId],
    tensor_meta: &[TensorMeta],
    lookahead: usize,
) -> bool {
    // Not sticky = normal retention rules apply
    if !weight_analysis.sticky_tensors.contains(&tensor_id) {
        return false;
    }

    let meta = &tensor_meta[tensor_id];
    let ops_to_check = &upcoming_ops[..upcoming_ops.len().min(lookahead)];

    // Check if any upcoming op needs this tensor
    meta.consumers.iter().any(|c| ops_to_check.contains(c))
}

/// Calculate the minimum SRAM reservation needed for sticky tensors.
///
/// This helps the scheduler reserve space for weights that should stay resident.
pub fn calculate_sticky_reservation(
    weight_analysis: &WeightAnalysis,
    remaining_ops: &[OpId],
    tensor_meta: &[TensorMeta],
) -> i64 {
    let remaining_set: HashSet<OpId> = remaining_ops.iter().copied().collect();

    weight_analysis.sticky_tensors
        .iter()
        .filter(|&&tid| {
            let meta = &tensor_meta[tid];
            meta.consumers.iter().any(|c| remaining_set.contains(c))
        })
        .map(|&tid| weight_analysis.tensor_info[tid].size)
        .sum()
}

// ============================================================================
// Consecutive Operation Analysis
// ============================================================================

/// Identify tensor sharing between consecutive operations.
///
/// This is the core insight of weight stationary: when two consecutive ops
/// share an input tensor, keeping it resident avoids a DRAM round-trip.
pub fn find_consecutive_sharing(
    ops: &[OpId],
    problem: &Problem,
) -> Vec<ConsecutiveShare> {
    let mut shares = Vec::new();

    for i in 0..ops.len().saturating_sub(1) {
        let op_a = ops[i];
        let op_b = ops[i + 1];

        let inputs_a: HashSet<TensorId> = problem.ops[op_a].inputs.iter().copied().collect();
        let inputs_b: HashSet<TensorId> = problem.ops[op_b].inputs.iter().copied().collect();

        // Find shared inputs
        let shared: Vec<TensorId> = inputs_a.intersection(&inputs_b).copied().collect();

        if !shared.is_empty() {
            let savings: i64 = shared.iter()
                .map(|&tid| problem.tensors[tid].size())
                .sum();

            shares.push(ConsecutiveShare {
                op_before: op_a,
                op_after: op_b,
                shared_tensors: shared,
                bandwidth_savings: savings * 2, // Avoid write + read
            });
        }
    }

    shares
}

/// Information about tensor sharing between consecutive operations
#[derive(Debug, Clone)]
pub struct ConsecutiveShare {
    pub op_before: OpId,
    pub op_after: OpId,
    pub shared_tensors: Vec<TensorId>,
    /// Bandwidth saved by keeping shared tensors resident (bytes)
    pub bandwidth_savings: i64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Tensor};

    fn make_test_problem_with_weights() -> Problem {
        // Create a simple graph where tensor 0 is a "weight" used by multiple ops
        // Op0: tensor0 x tensor1 -> tensor3
        // Op1: tensor0 x tensor2 -> tensor4
        // tensor0 is the weight (reused)
        Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 },  // 0: Weight (reused)
                Tensor { width: 128, height: 128 },  // 1: Input A
                Tensor { width: 128, height: 128 },  // 2: Input B
                Tensor { width: 128, height: 128 },  // 3: Output A
                Tensor { width: 128, height: 128 },  // 4: Output B
            ],
            ops: vec![
                Op {
                    op_type: OpType::MatMul,
                    inputs: vec![0, 1],
                    outputs: vec![3],
                    base_cost: 1000,
                },
                Op {
                    op_type: OpType::MatMul,
                    inputs: vec![0, 2],
                    outputs: vec![4],
                    base_cost: 1000,
                },
            ],
            fast_memory_capacity: 100000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_weight_detection() {
        let problem = make_test_problem_with_weights();
        let tensor_meta = problem.build_tensor_meta();
        let config = WeightStationaryConfig::default();

        let analysis = analyze_weights(&problem, &tensor_meta, &config);

        // Tensor 0 should be identified as a weight (2 consumers)
        assert!(analysis.sticky_tensors.contains(&0));
        assert_eq!(analysis.tensor_info[0].role, TensorRole::Weight);
        assert!(analysis.tensor_info[0].bandwidth_savings > 0);
    }

    #[test]
    fn test_consecutive_sharing() {
        let problem = make_test_problem_with_weights();
        let ops = vec![0, 1];

        let shares = find_consecutive_sharing(&ops, &problem);

        assert_eq!(shares.len(), 1);
        assert!(shares[0].shared_tensors.contains(&0));
        assert!(shares[0].bandwidth_savings > 0);
    }

    #[test]
    fn test_retention_priority() {
        // High reuse, small size should have higher priority than low reuse, large size
        let priority_high_reuse = calculate_retention_priority(
            1000,  // small
            5,     // high reuse
            &TensorRole::Weight,
            4000,  // 4x savings
        );

        let priority_low_reuse = calculate_retention_priority(
            10000, // large
            2,     // low reuse
            &TensorRole::Activation,
            10000, // less relative savings
        );

        assert!(priority_high_reuse > priority_low_reuse);
    }
}



