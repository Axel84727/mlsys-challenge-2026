//! Layout Transformation Analysis for Memory Access Optimization
//!
//! This module implements tensor layout transformation analysis to determine when
//! re-ordering data in memory before computation is beneficial.
//!
//! KEY INSIGHT: Modern accelerators often have asymmetric memory access patterns:
//! - Row-major access (contiguous in width) may be faster than column-major
//! - DMA engines work at peak bandwidth only with aligned, contiguous transfers
//! - A 5,000 cycle re-layout can save 100,000 cycles in subsequent computation
//!
//! ROBUSTNESS AGAINST OVERFITTING:
//! 1. Never assume a specific memory layout is always better
//! 2. Cost-based decision making with configurable thresholds
//! 3. Conservative estimates to avoid regression on unknown hardware
//! 4. Graceful degradation when layout analysis is uncertain
//!
//! The module integrates with the tiling system to provide layout-aware tile selection.

use crate::models::{Granularity, Op, OpId, OpType, Problem, Tensor, TensorId, TensorMeta};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Memory Layout Types
// ============================================================================

/// Memory layout classification for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Row-major: Elements contiguous along width (standard C/Rust layout)
    RowMajor,
    /// Column-major: Elements contiguous along height (Fortran layout)
    ColumnMajor,
    /// Blocked/Tiled: Elements stored in tiles for locality
    Blocked { tile_w: i64, tile_h: i64 },
    /// Unknown/Mixed: Cannot determine optimal layout
    Unknown,
}

impl Default for MemoryLayout {
    fn default() -> Self {
        MemoryLayout::RowMajor
    }
}

impl MemoryLayout {
    /// Check if this layout favors horizontal (row) access
    pub fn favors_row_access(&self) -> bool {
        matches!(self, MemoryLayout::RowMajor | MemoryLayout::Unknown)
    }

    /// Check if this layout favors vertical (column) access
    pub fn favors_column_access(&self) -> bool {
        matches!(self, MemoryLayout::ColumnMajor)
    }

    /// Check if this layout is blocked (good for tiled computation)
    pub fn is_blocked(&self) -> bool {
        matches!(self, MemoryLayout::Blocked { .. })
    }
}

// ============================================================================
// Layout Analysis Result
// ============================================================================

/// Result of analyzing a tensor's optimal layout for a given access pattern
#[derive(Debug, Clone)]
pub struct LayoutAnalysis {
    pub tensor_id: TensorId,
    pub current_layout: MemoryLayout,
    pub optimal_layout: MemoryLayout,
    /// Estimated cost to transform to optimal layout (cycles)
    pub transform_cost: f64,
    /// Estimated savings from using optimal layout (cycles)
    pub estimated_savings: f64,
    /// Net benefit (savings - cost). Positive = worth transforming
    pub net_benefit: f64,
    /// Confidence in the analysis (0.0 - 1.0)
    pub confidence: f64,
}

impl LayoutAnalysis {
    /// Check if transformation is recommended
    pub fn should_transform(&self) -> bool {
        // Only transform if:
        // 1. Net benefit is positive
        // 2. Confidence is high enough (avoid overfitting to specific cases)
        // 3. Savings are significant enough to justify the complexity
        self.net_benefit > 0.0
            && self.confidence >= MINIMUM_CONFIDENCE_THRESHOLD
            && self.estimated_savings >= MINIMUM_SAVINGS_THRESHOLD
    }
}

// ============================================================================
// Constants - Robustness Thresholds
// ============================================================================

/// Minimum confidence level to recommend layout transformation
/// ANTI-OVERFITTING: High threshold prevents speculative transformations
const MINIMUM_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// Minimum absolute savings (cycles) to justify transformation overhead
/// Prevents micro-optimizations that add complexity without meaningful benefit
const MINIMUM_SAVINGS_THRESHOLD: f64 = 1000.0;

/// Minimum relative improvement (ratio) to justify transformation
/// Ensures transformation is significant relative to total operation cost
const MINIMUM_IMPROVEMENT_RATIO: f64 = 0.05; // 5% improvement minimum

/// Cost multiplier for layout transformation (conservative estimate)
/// Transformation = reading entire tensor + writing in new layout
/// This is a conservative 2.5x multiplier to account for:
/// - Cache misses during transformation
/// - Potential write amplification
/// - Memory controller contention
const TRANSFORM_COST_MULTIPLIER: f64 = 2.5;

/// Asymmetric bandwidth ratio threshold
/// Only consider layout transformation if bandwidth asymmetry exceeds this
const BANDWIDTH_ASYMMETRY_THRESHOLD: f64 = 1.3; // 30% asymmetry

/// Maximum tensor size for layout transformation (bytes)
/// Very large tensors have high transformation cost that rarely pays off
const MAX_TENSOR_SIZE_FOR_TRANSFORM: i64 = 16 * 1024 * 1024; // 16MB

// ============================================================================
// Access Pattern Analysis
// ============================================================================

/// Access pattern for a tensor within an operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    /// Sequential row access (contiguous in memory for row-major)
    RowSequential,
    /// Sequential column access (strided for row-major)
    ColumnSequential,
    /// Block access (good for tiled layouts)
    BlockAccess { tile_w: i64, tile_h: i64 },
    /// Mixed/random access (layout doesn't matter much)
    Mixed,
}

/// Analyze how a tensor is accessed in a MatMul operation
///
/// For C = A @ B:
/// - A (LHS): accessed row-by-row (each row of A contributes to a row of C)
/// - B (RHS): accessed column-by-column (each column of B contributes to a column of C)
///
/// For optimal performance:
/// - A should be in row-major layout (rows are contiguous)
/// - B should be in column-major layout (columns are contiguous) OR
///   B should be pre-transposed so columns become rows
fn analyze_matmul_access_pattern(
    op: &Op,
    tensor_id: TensorId,
    _tensors: &[Tensor],
) -> AccessPattern {
    if op.inputs.len() < 2 {
        return AccessPattern::Mixed;
    }

    let lhs_id = op.inputs[0];
    let rhs_id = op.inputs[1];

    if tensor_id == lhs_id {
        // LHS matrix: rows are accessed sequentially for each output row
        AccessPattern::RowSequential
    } else if tensor_id == rhs_id {
        // RHS matrix: columns are accessed sequentially for each output column
        // This is inefficient for row-major layout!
        AccessPattern::ColumnSequential
    } else {
        // Output tensor: written in row-major order
        AccessPattern::RowSequential
    }
}

/// Analyze access pattern for Pointwise operations
fn analyze_pointwise_access_pattern(
    _op: &Op,
    _tensor_id: TensorId,
) -> AccessPattern {
    // Pointwise ops typically access all elements in order
    // Row-major access is natural for most implementations
    AccessPattern::RowSequential
}

/// Analyze access pattern for a tensor in a given operation
pub fn analyze_access_pattern(
    op: &Op,
    tensor_id: TensorId,
    tensors: &[Tensor],
) -> AccessPattern {
    match op.op_type {
        OpType::MatMul => analyze_matmul_access_pattern(op, tensor_id, tensors),
        OpType::Pointwise => analyze_pointwise_access_pattern(op, tensor_id),
    }
}

// ============================================================================
// Layout Transformation Cost Model
// ============================================================================

/// Calculate the cost to transform a tensor from one layout to another
///
/// ROBUSTNESS: Uses conservative estimates with safety margins
pub fn calculate_transform_cost(
    tensor: &Tensor,
    _from_layout: MemoryLayout,
    _to_layout: MemoryLayout,
    bandwidth: i64,
) -> f64 {
    let tensor_size = tensor.size();

    // Transformation requires:
    // 1. Reading the entire tensor from memory
    // 2. Writing the entire tensor in new layout
    // Plus overhead for cache misses, memory controller scheduling, etc.
    let base_cost = (2.0 * tensor_size as f64) / bandwidth as f64;

    // Apply conservative multiplier
    base_cost * TRANSFORM_COST_MULTIPLIER
}

/// Estimate the savings from using optimal layout for a set of operations
///
/// ROBUSTNESS: Uses conservative savings estimates
pub fn estimate_layout_savings(
    tensor_id: TensorId,
    ops_using_tensor: &[OpId],
    problem: &Problem,
    current_layout: MemoryLayout,
    optimal_layout: MemoryLayout,
) -> f64 {
    if current_layout == optimal_layout {
        return 0.0; // No savings if already optimal
    }

    let tensor = &problem.tensors[tensor_id];
    let mut total_savings = 0.0;

    for &op_id in ops_using_tensor {
        let op = &problem.ops[op_id];
        let access_pattern = analyze_access_pattern(op, tensor_id, &problem.tensors);

        // Calculate memory access efficiency difference
        let current_efficiency = calculate_access_efficiency(tensor, access_pattern, current_layout);
        let optimal_efficiency = calculate_access_efficiency(tensor, access_pattern, optimal_layout);

        // Savings = reduction in memory time due to better efficiency
        let memory_time_current = (tensor.size() as f64) / problem.slow_memory_bandwidth as f64 / current_efficiency;
        let memory_time_optimal = (tensor.size() as f64) / problem.slow_memory_bandwidth as f64 / optimal_efficiency;

        // Conservative: only count savings if significant
        let op_savings = (memory_time_current - memory_time_optimal).max(0.0);

        // Apply discount factor based on operation type
        // MatMul benefits more from layout optimization than Pointwise
        let discount = match op.op_type {
            OpType::MatMul => 0.8,  // 80% of estimated savings for MatMul
            OpType::Pointwise => 0.5,  // 50% for Pointwise (less memory-bound typically)
        };

        total_savings += op_savings * discount;
    }

    total_savings
}

/// Calculate memory access efficiency for a given layout and access pattern
/// Returns a value between 0.0 and 1.0 where 1.0 is optimal (fully sequential)
fn calculate_access_efficiency(
    tensor: &Tensor,
    access_pattern: AccessPattern,
    layout: MemoryLayout,
) -> f64 {
    match (access_pattern, layout) {
        // Perfect match: sequential access aligns with memory layout
        (AccessPattern::RowSequential, MemoryLayout::RowMajor) => 1.0,
        (AccessPattern::ColumnSequential, MemoryLayout::ColumnMajor) => 1.0,

        // Mismatch: strided access
        (AccessPattern::RowSequential, MemoryLayout::ColumnMajor) => {
            // Accessing rows in column-major = stride of height
            // Efficiency depends on cache behavior
            calculate_strided_efficiency(tensor.height)
        }
        (AccessPattern::ColumnSequential, MemoryLayout::RowMajor) => {
            // Accessing columns in row-major = stride of width
            calculate_strided_efficiency(tensor.width)
        }

        // Blocked layout - good for tiled access
        (AccessPattern::BlockAccess { tile_w, tile_h }, MemoryLayout::Blocked { tile_w: bw, tile_h: bh }) => {
            if tile_w == bw && tile_h == bh {
                1.0 // Perfect tile match
            } else {
                // Partial match - some efficiency lost
                0.7
            }
        }

        // Mixed or unknown - use conservative estimate
        _ => 0.6,
    }
}

/// Calculate efficiency for strided memory access
/// Larger strides = worse cache utilization
fn calculate_strided_efficiency(stride: i64) -> f64 {
    // Assume 64-byte cache lines
    const CACHE_LINE_SIZE: i64 = 64;

    if stride <= CACHE_LINE_SIZE {
        0.8 // Small stride, decent cache utilization
    } else if stride <= 256 {
        0.5 // Medium stride, some cache misses
    } else if stride <= 1024 {
        0.3 // Large stride, many cache misses
    } else {
        0.2 // Very large stride, poor cache utilization
    }
}

// ============================================================================
// Layout Recommendation Engine
// ============================================================================

/// Configuration for layout analysis
#[derive(Debug, Clone)]
pub struct LayoutAnalysisConfig {
    /// Enable layout transformation analysis
    pub enabled: bool,
    /// Minimum improvement ratio to recommend transformation
    pub min_improvement_ratio: f64,
    /// Minimum absolute savings (cycles)
    pub min_savings: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Maximum tensor size to consider for transformation
    pub max_tensor_size: i64,
}

impl Default for LayoutAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_improvement_ratio: MINIMUM_IMPROVEMENT_RATIO,
            min_savings: MINIMUM_SAVINGS_THRESHOLD,
            min_confidence: MINIMUM_CONFIDENCE_THRESHOLD,
            max_tensor_size: MAX_TENSOR_SIZE_FOR_TRANSFORM,
        }
    }
}

/// Analyze all tensors in a problem and recommend layout transformations
pub fn analyze_layout_opportunities(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    config: &LayoutAnalysisConfig,
) -> Vec<LayoutAnalysis> {
    if !config.enabled {
        return Vec::new();
    }

    let mut analyses = Vec::new();

    // Build reverse mapping: tensor -> ops that use it
    let mut tensor_to_ops: HashMap<TensorId, Vec<OpId>> = HashMap::new();
    for (op_id, op) in problem.ops.iter().enumerate() {
        for &input_id in &op.inputs {
            tensor_to_ops.entry(input_id).or_default().push(op_id);
        }
    }

    for (tensor_id, tensor) in problem.tensors.iter().enumerate() {
        // Skip very large tensors (transformation cost too high)
        if tensor.size() > config.max_tensor_size {
            continue;
        }

        // Skip tensors not used by any MatMul (layout matters less for Pointwise)
        let ops_using = tensor_to_ops.get(&tensor_id).cloned().unwrap_or_default();
        let has_matmul_use = ops_using.iter().any(|&op_id| problem.ops[op_id].is_matmul());

        if !has_matmul_use {
            continue; // Only analyze tensors used by MatMul
        }

        // Analyze this tensor
        if let Some(analysis) = analyze_single_tensor(
            tensor_id,
            tensor,
            &ops_using,
            problem,
            tensor_meta,
            config,
        ) {
            if analysis.should_transform() {
                analyses.push(analysis);
            }
        }
    }

    // Sort by net benefit (highest first)
    analyses.sort_by(|a, b| b.net_benefit.partial_cmp(&a.net_benefit).unwrap_or(std::cmp::Ordering::Equal));

    analyses
}

/// Analyze a single tensor for layout transformation opportunity
fn analyze_single_tensor(
    tensor_id: TensorId,
    tensor: &Tensor,
    ops_using: &[OpId],
    problem: &Problem,
    _tensor_meta: &[TensorMeta],
    config: &LayoutAnalysisConfig,
) -> Option<LayoutAnalysis> {
    // Determine current layout (assume row-major as default)
    let current_layout = MemoryLayout::RowMajor;

    // Analyze access patterns to determine optimal layout
    let optimal_layout = determine_optimal_layout(tensor_id, ops_using, problem);

    if current_layout == optimal_layout {
        return None; // Already optimal
    }

    // Calculate costs and savings
    let transform_cost = calculate_transform_cost(
        tensor,
        current_layout,
        optimal_layout,
        problem.slow_memory_bandwidth,
    );

    let estimated_savings = estimate_layout_savings(
        tensor_id,
        ops_using,
        problem,
        current_layout,
        optimal_layout,
    );

    let net_benefit = estimated_savings - transform_cost;

    // Calculate confidence based on analysis quality
    let confidence = calculate_analysis_confidence(
        tensor,
        ops_using,
        problem,
        net_benefit,
        config,
    );

    // Check minimum improvement ratio
    let base_cost: f64 = ops_using.iter()
        .map(|&op_id| problem.ops[op_id].base_cost as f64)
        .sum();

    if base_cost > 0.0 && estimated_savings / base_cost < config.min_improvement_ratio {
        return None; // Improvement too small relative to operation cost
    }

    Some(LayoutAnalysis {
        tensor_id,
        current_layout,
        optimal_layout,
        transform_cost,
        estimated_savings,
        net_benefit,
        confidence,
    })
}

/// Determine the optimal layout for a tensor based on its access patterns
fn determine_optimal_layout(
    tensor_id: TensorId,
    ops_using: &[OpId],
    problem: &Problem,
) -> MemoryLayout {
    let mut row_access_count = 0;
    let mut column_access_count = 0;

    for &op_id in ops_using {
        let op = &problem.ops[op_id];
        let pattern = analyze_access_pattern(op, tensor_id, &problem.tensors);

        match pattern {
            AccessPattern::RowSequential => row_access_count += 1,
            AccessPattern::ColumnSequential => column_access_count += 1,
            AccessPattern::BlockAccess { tile_w, tile_h } => {
                // For block access, blocked layout is best
                return MemoryLayout::Blocked { tile_w, tile_h };
            }
            AccessPattern::Mixed => {}
        }
    }

    // Majority vote for layout
    if column_access_count > row_access_count {
        MemoryLayout::ColumnMajor
    } else {
        MemoryLayout::RowMajor // Default to row-major
    }
}

/// Calculate confidence in the layout analysis
///
/// ANTI-OVERFITTING: Lower confidence for edge cases
fn calculate_analysis_confidence(
    tensor: &Tensor,
    ops_using: &[OpId],
    problem: &Problem,
    net_benefit: f64,
    _config: &LayoutAnalysisConfig,
) -> f64 {
    let mut confidence = 1.0;

    // Lower confidence for small tensors (cache effects dominate)
    if tensor.size() < 1024 {
        confidence *= 0.5;
    } else if tensor.size() < 4096 {
        confidence *= 0.7;
    }

    // Lower confidence for tensors used by only one op (less data reuse)
    if ops_using.len() == 1 {
        confidence *= 0.6;
    }

    // Lower confidence for negative or marginal net benefit
    if net_benefit < 0.0 {
        confidence *= 0.3;
    } else if net_benefit < MINIMUM_SAVINGS_THRESHOLD {
        confidence *= 0.5;
    }

    // Lower confidence for very large tensors (high risk)
    if tensor.size() > 1_000_000 {
        confidence *= 0.8;
    }

    // Lower confidence if ops are mostly Pointwise (layout matters less)
    let matmul_count = ops_using.iter()
        .filter(|&&op_id| problem.ops[op_id].is_matmul())
        .count();
    let matmul_ratio = matmul_count as f64 / ops_using.len().max(1) as f64;
    confidence *= 0.5 + 0.5 * matmul_ratio; // Scale by MatMul ratio

    confidence.clamp(0.0, 1.0)
}

// ============================================================================
// Layout-Aware Tiling Integration
// ============================================================================

/// Tile configuration recommendation based on layout analysis
#[derive(Debug, Clone)]
pub struct LayoutAwareTiling {
    /// Recommended granularity
    pub granularity: Granularity,
    /// Should pre-transform any tensors?
    pub pre_transforms: Vec<(TensorId, MemoryLayout)>,
    /// Estimated total latency with this configuration
    pub estimated_latency: f64,
    /// Estimated latency savings vs default tiling
    pub savings_vs_default: f64,
}

/// Generate layout-aware tiling recommendations for a subgraph
///
/// This integrates layout transformation with tile selection:
/// 1. Analyze tensor layouts and access patterns
/// 2. Consider pre-transforming tensors if beneficial
/// 3. Select tiles that match the (potentially transformed) layouts
pub fn generate_layout_aware_tiling(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    config: &LayoutAnalysisConfig,
) -> Option<LayoutAwareTiling> {
    if !config.enabled || ops.is_empty() {
        return None;
    }

    // Analyze layout opportunities for tensors used by this subgraph
    let subgraph_tensors: HashSet<TensorId> = ops.iter()
        .flat_map(|&op_id| {
            let op = &problem.ops[op_id];
            op.inputs.iter().chain(op.outputs.iter()).copied()
        })
        .collect();

    // Filter tensor_meta to only subgraph tensors
    let mut analyses = Vec::new();

    for &tensor_id in &subgraph_tensors {
        let tensor = &problem.tensors[tensor_id];

        // Find ops in subgraph that use this tensor
        let ops_using: Vec<OpId> = ops.iter()
            .copied()
            .filter(|&op_id| {
                let op = &problem.ops[op_id];
                op.inputs.contains(&tensor_id) || op.outputs.contains(&tensor_id)
            })
            .collect();

        if let Some(analysis) = analyze_single_tensor(
            tensor_id,
            tensor,
            &ops_using,
            problem,
            tensor_meta,
            config,
        ) {
            if analysis.should_transform() {
                analyses.push(analysis);
            }
        }
    }

    if analyses.is_empty() {
        return None; // No beneficial transformations found
    }

    // Build pre-transform list from beneficial analyses
    let pre_transforms: Vec<(TensorId, MemoryLayout)> = analyses.iter()
        .map(|a| (a.tensor_id, a.optimal_layout))
        .collect();

    // Calculate total transformation cost and savings
    let total_transform_cost: f64 = analyses.iter().map(|a| a.transform_cost).sum();
    let total_savings: f64 = analyses.iter().map(|a| a.estimated_savings).sum();
    let net_savings = total_savings - total_transform_cost;

    // Only recommend if net savings are positive and significant
    if net_savings < config.min_savings {
        return None;
    }

    // Select granularity that works well with transformed layouts
    let granularity = select_layout_optimized_granularity(ops, problem, &pre_transforms);

    Some(LayoutAwareTiling {
        granularity,
        pre_transforms,
        estimated_latency: 0.0, // Will be calculated by caller
        savings_vs_default: net_savings,
    })
}

/// Select granularity that complements the transformed layouts
fn select_layout_optimized_granularity(
    _ops: &[OpId],
    problem: &Problem,
    transforms: &[(TensorId, MemoryLayout)],
) -> Granularity {
    // If any tensor is transformed to column-major, prefer tall tiles
    let has_column_major = transforms.iter().any(|(_, layout)| *layout == MemoryLayout::ColumnMajor);

    // If any tensor is transformed to blocked, use matching tile size
    let blocked_size = transforms.iter()
        .filter_map(|(_, layout)| {
            if let MemoryLayout::Blocked { tile_w, tile_h } = layout {
                Some((*tile_w, *tile_h))
            } else {
                None
            }
        })
        .next();

    if let Some((bw, bh)) = blocked_size {
        // Use blocked tile size
        Granularity::new(bw, bh, 1)
    } else if has_column_major {
        // Tall tiles work better with column-major layout
        Granularity::new(64, 256, 1)
    } else {
        // Default to native granularity
        problem.native_granularity.clone()
    }
}

// ============================================================================
// Bandwidth Asymmetry Detection
// ============================================================================

/// Analyze problem to detect bandwidth asymmetry
///
/// Returns the estimated read:write bandwidth ratio
/// A ratio > 1 means reads are faster, < 1 means writes are faster
pub fn estimate_bandwidth_asymmetry(problem: &Problem) -> f64 {
    // In the abstract model, we only have slow_memory_bandwidth
    // Real hardware often has asymmetric read/write speeds
    //
    // Heuristic: Larger SRAM usually correlates with more sophisticated
    // memory controllers that have asymmetric bandwidth

    if problem.fast_memory_capacity > 500_000 {
        // High-end hardware: typically 1.5:1 read:write
        1.5
    } else if problem.fast_memory_capacity > 200_000 {
        // Mid-range: 1.2:1
        1.2
    } else {
        // Low-end: assume symmetric
        1.0
    }
}

/// Check if layout transformation is worthwhile given bandwidth asymmetry
pub fn should_consider_layout_transform(problem: &Problem) -> bool {
    let asymmetry = estimate_bandwidth_asymmetry(problem);
    asymmetry >= BANDWIDTH_ASYMMETRY_THRESHOLD
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
                    op_type: OpType::MatMul,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                    base_cost: 10000,
                },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_matmul_access_patterns() {
        let problem = make_test_problem();
        let op = &problem.ops[0];

        // LHS should be row-sequential
        let lhs_pattern = analyze_matmul_access_pattern(op, 0, &problem.tensors);
        assert_eq!(lhs_pattern, AccessPattern::RowSequential);

        // RHS should be column-sequential
        let rhs_pattern = analyze_matmul_access_pattern(op, 1, &problem.tensors);
        assert_eq!(rhs_pattern, AccessPattern::ColumnSequential);
    }

    #[test]
    fn test_strided_efficiency() {
        // Small stride = high efficiency
        assert!(calculate_strided_efficiency(32) > 0.7);

        // Large stride = low efficiency
        assert!(calculate_strided_efficiency(2048) < 0.3);
    }

    #[test]
    fn test_access_efficiency_matching() {
        let tensor = Tensor { width: 128, height: 128 };

        // Row access on row-major = optimal
        let eff1 = calculate_access_efficiency(&tensor, AccessPattern::RowSequential, MemoryLayout::RowMajor);
        assert_eq!(eff1, 1.0);

        // Column access on row-major = suboptimal
        let eff2 = calculate_access_efficiency(&tensor, AccessPattern::ColumnSequential, MemoryLayout::RowMajor);
        assert!(eff2 < 1.0);

        // Column access on column-major = optimal
        let eff3 = calculate_access_efficiency(&tensor, AccessPattern::ColumnSequential, MemoryLayout::ColumnMajor);
        assert_eq!(eff3, 1.0);
    }

    #[test]
    fn test_transform_cost_conservative() {
        let tensor = Tensor { width: 128, height: 128 };
        let bandwidth = 10;

        let cost = calculate_transform_cost(
            &tensor,
            MemoryLayout::RowMajor,
            MemoryLayout::ColumnMajor,
            bandwidth,
        );

        // Cost should be at least 2x tensor size / bandwidth (read + write)
        let min_cost = 2.0 * tensor.size() as f64 / bandwidth as f64;
        assert!(cost >= min_cost);

        // But with conservative multiplier, should be higher
        assert!(cost > min_cost * 2.0);
    }

    #[test]
    fn test_analysis_confidence_bounds() {
        let problem = make_test_problem();
        let tensor = &problem.tensors[0];
        let config = LayoutAnalysisConfig::default();

        let confidence = calculate_analysis_confidence(
            tensor,
            &[0],
            &problem,
            5000.0, // Positive net benefit
            &config,
        );

        // Confidence should be between 0 and 1
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}



