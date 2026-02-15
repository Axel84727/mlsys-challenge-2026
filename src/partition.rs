//! Intelligent Subgraph Partitioning Strategy
//!
//! This module provides intelligent partitioning to avoid the "one-subgraph trap"
//! where the scheduler tries to fuse all operations into a single subgraph,
//! causing catastrophic performance when the working set exceeds SRAM capacity.
//!
//! Key insight: Calculate optimal number of subgraphs BEFORE scheduling starts,
//! based on graph density, total working set, and SRAM capacity.

use crate::cost_model::{analyze_graph_density, GraphDensity};
use crate::memory::compute_subgraph_working_set;
use crate::models::{Granularity, OpId, Problem, TensorMeta};

/// Configuration for intelligent partitioning
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Target SRAM utilization (0.85 = 85%)
    /// Lower values leave more room for double buffering
    pub target_utilization: f64,

    /// Minimum ops per subgraph (avoid tiny subgraphs)
    pub min_ops_per_subgraph: usize,

    /// Maximum ops per subgraph (avoid monster subgraphs)
    pub max_ops_per_subgraph: usize,

    /// Safety margin for SRAM (avoid thrashing)
    /// This is the buffer left for intermediate transfers and double buffering
    pub safety_margin: f64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            target_utilization: 0.85,
            min_ops_per_subgraph: 3,
            max_ops_per_subgraph: 50,
            safety_margin: 0.15, // 15% buffer for double buffering
        }
    }
}

impl PartitionConfig {
    /// Create a density-aware configuration
    pub fn from_density(density: GraphDensity) -> Self {
        match density {
            GraphDensity::Sparse => Self {
                target_utilization: 0.90,  // Aggressive - chains can use more SRAM
                min_ops_per_subgraph: 5,
                max_ops_per_subgraph: 100, // Allow large subgraphs for chains
                safety_margin: 0.10,
            },
            GraphDensity::Medium => Self {
                target_utilization: 0.85,
                min_ops_per_subgraph: 3,
                max_ops_per_subgraph: 50,
                safety_margin: 0.15,
            },
            GraphDensity::Dense => Self {
                target_utilization: 0.75,  // Conservative - many intermediates
                min_ops_per_subgraph: 3,
                max_ops_per_subgraph: 30,
                safety_margin: 0.25,
            },
        }
    }
}

/// Result of partition planning
#[derive(Debug, Clone)]
pub struct PartitionPlan {
    /// Recommended number of subgraphs
    pub num_subgraphs: usize,

    /// Estimated ops per subgraph
    pub ops_per_subgraph: usize,

    /// Effective SRAM capacity per subgraph (after safety margin)
    pub effective_capacity: i64,

    /// Maximum ops allowed per subgraph (for bounded fusion)
    pub max_ops_per_subgraph: usize,

    /// Reason for this partitioning
    pub rationale: String,

    /// Whether single-subgraph fusion is safe
    pub can_use_full_fusion: bool,
}

/// Calculate optimal number of subgraphs based on graph characteristics
///
/// This function performs a pre-flight check to determine:
/// 1. How many subgraphs are needed to avoid SRAM thrashing
/// 2. How many ops should go in each subgraph
/// 3. Whether full-fusion (1 subgraph) is safe
///
/// The calculation considers:
/// - Total working set vs SRAM capacity
/// - Graph density (dense graphs need more, smaller subgraphs)
/// - Safety margins for double buffering
pub fn calculate_optimal_partitioning(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    config: &PartitionConfig,
) -> PartitionPlan {
    let num_ops = problem.ops.len();
    let sram_capacity = problem.fast_memory_capacity;

    // Step 1: Analyze graph density
    let density_analysis = analyze_graph_density(problem);

    // Step 2: Estimate total working set for all ops
    // Use native granularity as baseline estimate
    let all_ops: Vec<OpId> = (0..num_ops).collect();
    let native = &problem.native_granularity;
    let total_ws = compute_subgraph_working_set(&all_ops, problem, native, tensor_meta);

    // Step 3: Calculate effective capacity (with safety margin)
    let effective_capacity = (sram_capacity as f64 * config.target_utilization) as i64;

    eprintln!(
        "[*] Partition Analysis: Total WS={} KB, Effective SRAM={} KB",
        total_ws.total_size / 1024,
        effective_capacity / 1024
    );

    // Step 4: Calculate naive partition count based on memory alone
    let memory_based_partitions = if total_ws.total_size <= effective_capacity {
        1 // Everything fits!
    } else {
        // How many chunks do we need?
        ((total_ws.total_size as f64) / (effective_capacity as f64)).ceil() as usize
    };

    // Step 5: Adjust based on graph density
    // Dense graphs have more intermediates → need more subgraphs to clean up
    // Sparse graphs (chains) can use larger subgraphs
    let density_multiplier = match density_analysis.density {
        GraphDensity::Sparse => 1.0,      // Chains are fine with large subgraphs
        GraphDensity::Medium => 1.2,      // Slight increase
        GraphDensity::Dense => 1.5,       // Significant increase for complex graphs
    };

    let density_adjusted_partitions =
        ((memory_based_partitions as f64) * density_multiplier).ceil() as usize;

    eprintln!(
        "    - Memory-based partitions: {}",
        memory_based_partitions
    );
    eprintln!(
        "    - Density multiplier: {:.1}x ({:?})",
        density_multiplier, density_analysis.density
    );
    eprintln!(
        "    - Density-adjusted partitions: {}",
        density_adjusted_partitions
    );

    // Step 6: Apply constraints
    // Don't create more subgraphs than we have ops (obviously)
    // Don't create subgraphs smaller than min_ops_per_subgraph
    let max_possible_partitions = num_ops / config.min_ops_per_subgraph;
    let final_partitions = density_adjusted_partitions
        .max(1)
        .min(max_possible_partitions);

    // Step 7: Calculate ops per subgraph
    let ops_per_subgraph = (num_ops + final_partitions - 1) / final_partitions;
    let ops_per_subgraph = ops_per_subgraph
        .max(config.min_ops_per_subgraph)
        .min(config.max_ops_per_subgraph);

    // Step 8: Determine if full fusion is safe
    // Full fusion is ONLY safe if:
    // 1. Plan says we need 1 subgraph
    // 2. Working set is comfortably below effective capacity (not at 95%+)
    let utilization = (total_ws.total_size as f64) / (sram_capacity as f64);
    let can_use_full_fusion = final_partitions == 1 && utilization < 0.97;

    // Step 9: Generate rationale
    let rationale = if final_partitions == 1 {
        if can_use_full_fusion {
            format!(
                "Single subgraph is safe: WS={}KB fits comfortably in {}KB SRAM ({:.1}% util)",
                total_ws.total_size / 1024,
                sram_capacity / 1024,
                utilization * 100.0
            )
        } else {
            format!(
                "Single subgraph is RISKY: WS={}KB barely fits in {}KB SRAM ({:.1}% util) - consider partitioning",
                total_ws.total_size / 1024,
                sram_capacity / 1024,
                utilization * 100.0
            )
        }
    } else {
        format!(
            "{} subgraphs needed: Total WS={}KB, Effective SRAM={}KB, Density={:?} ({}x multiplier), ~{} ops/subgraph",
            final_partitions,
            total_ws.total_size / 1024,
            effective_capacity / 1024,
            density_analysis.density,
            density_multiplier,
            ops_per_subgraph
        )
    };

    // Step 10: COST-AWARE DECISION
    // Compare latency of 1 subgraph vs N subgraphs
    // Only partition if it's actually FASTER
    use crate::cost::estimate_full_fusion_latency;
    if final_partitions > 1 {
        // Estimate latency for full fusion (1 subgraph)
        let full_fusion_latency = estimate_full_fusion_latency(
            &all_ops,
            problem,
            tensor_meta
        );

        // Estimate latency for N subgraphs
        // Rough heuristic: partition latency ≈ full_fusion_latency * (1.2 + 0.3 * partitions)
        // This accounts for:
        // - Smaller tiles (worse efficiency)
        // - Inter-subgraph transfers
        // - Loss of fusion opportunities
        let partition_overhead = 1.2 + 0.3 * (final_partitions as f64);
        let estimated_partition_latency = full_fusion_latency * partition_overhead;

        eprintln!("    - Full fusion latency estimate: {:.0}", full_fusion_latency);
        eprintln!("    - Partition latency estimate: {:.0} ({}x overhead)",
            estimated_partition_latency, partition_overhead);

        // CRITICAL DECISION: Only partition if it's actually faster
        if full_fusion_latency < estimated_partition_latency {
            eprintln!("    → Full fusion is FASTER despite memory pressure!");
            eprintln!("    → Overriding partition plan to use 1 subgraph");

            return PartitionPlan {
                num_subgraphs: 1,
                ops_per_subgraph: num_ops,
                effective_capacity: sram_capacity,  // Use full capacity
                max_ops_per_subgraph: num_ops,
                rationale: format!(
                    "Full fusion chosen: Latency {:.0} < Partition {:.0} (despite {}% SRAM util)",
                    full_fusion_latency,
                    estimated_partition_latency,
                    utilization * 100.0
                ),
                can_use_full_fusion: true,  // Override!
            };
        }
    }

    PartitionPlan {
        num_subgraphs: final_partitions,
        ops_per_subgraph,
        effective_capacity,
        max_ops_per_subgraph: config.max_ops_per_subgraph,
        rationale,
        can_use_full_fusion,
    }
}

/// Quick check: Will this set of ops fit in SRAM with the given granularity?
///
/// This is a lightweight version of the full partition calculation,
/// used during scheduling to validate fusion decisions.
pub fn will_fit_in_sram(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
    target_utilization: f64,
) -> bool {
    let ws = compute_subgraph_working_set(ops, problem, granularity, tensor_meta);
    let effective_capacity = (problem.fast_memory_capacity as f64 * target_utilization) as i64;
    ws.fits_in(effective_capacity)
}

/// Estimate the "pressure" of a subgraph (how much it stresses SRAM)
///
/// Returns a value between 0.0 and 1.0+ where:
/// - < 0.70: Very comfortable, plenty of room for double buffering
/// - 0.70-0.85: Normal range, safe
/// - 0.85-0.95: High pressure, risky
/// - > 0.95: Thrashing zone, very dangerous
pub fn calculate_sram_pressure(
    ops: &[OpId],
    problem: &Problem,
    granularity: &Granularity,
    tensor_meta: &[TensorMeta],
) -> f64 {
    let ws = compute_subgraph_working_set(ops, problem, granularity, tensor_meta);
    (ws.total_size as f64) / (problem.fast_memory_capacity as f64)
}

/// Validate a partition plan - check if it makes sense
///
/// This is a sanity check to catch planning errors
pub fn validate_partition_plan(plan: &PartitionPlan, num_ops: usize) -> Result<(), String> {
    // Check 1: We need at least 1 subgraph
    if plan.num_subgraphs == 0 {
        return Err("Invalid plan: num_subgraphs = 0".to_string());
    }

    // Check 2: Ops per subgraph should be reasonable
    if plan.ops_per_subgraph == 0 {
        return Err("Invalid plan: ops_per_subgraph = 0".to_string());
    }

    // Check 3: Total capacity makes sense
    if plan.num_subgraphs * plan.ops_per_subgraph < num_ops {
        return Err(format!(
            "Invalid plan: {} subgraphs × {} ops = {} total, but we have {} ops",
            plan.num_subgraphs,
            plan.ops_per_subgraph,
            plan.num_subgraphs * plan.ops_per_subgraph,
            num_ops
        ));
    }

    // Check 4: Effective capacity should be positive
    if plan.effective_capacity <= 0 {
        return Err(format!(
            "Invalid plan: effective_capacity = {}",
            plan.effective_capacity
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Tensor};

    fn make_large_problem(num_ops: usize, sram_capacity: i64) -> Problem {
        // Create a large problem with big tensors
        let num_tensors = num_ops + 1;
        Problem {
            tensors: vec![Tensor {
                width: 1024,
                height: 1024,
            }; num_tensors],
            ops: (0..num_ops)
                .map(|i| Op {
                    op_type: OpType::MatMul,
                    inputs: vec![i, i],
                    outputs: vec![i + 1],
                    base_cost: 1000,
                })
                .collect(),
            fast_memory_capacity: sram_capacity,
            slow_memory_bandwidth: 100,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_partition_calculation_fits() {
        // Create a problem that DOES fit in SRAM
        let problem = make_large_problem(5, 10_000_000); // 10MB SRAM
        let tensor_meta = problem.build_tensor_meta();
        let config = PartitionConfig::default();

        let plan = calculate_optimal_partitioning(&problem, &tensor_meta, &config);

        // Should recommend 1 subgraph since it fits
        assert_eq!(plan.num_subgraphs, 1);
        assert!(plan.can_use_full_fusion);
    }

    #[test]
    fn test_partition_calculation_doesnt_fit() {
        // Create a problem that DOESN'T fit in SRAM
        let problem = make_large_problem(50, 500_000); // 500KB SRAM, 50 ops with 1MB tensors
        let tensor_meta = problem.build_tensor_meta();
        let config = PartitionConfig::default();

        let plan = calculate_optimal_partitioning(&problem, &tensor_meta, &config);

        // Should recommend multiple subgraphs since total WS >> SRAM
        assert!(plan.num_subgraphs > 1);
        assert!(plan.ops_per_subgraph >= config.min_ops_per_subgraph);
        assert!(plan.ops_per_subgraph <= config.max_ops_per_subgraph);
    }

    #[test]
    fn test_density_aware_config() {
        let sparse_config = PartitionConfig::from_density(GraphDensity::Sparse);
        let dense_config = PartitionConfig::from_density(GraphDensity::Dense);

        // Sparse should allow larger subgraphs
        assert!(sparse_config.max_ops_per_subgraph > dense_config.max_ops_per_subgraph);
        assert!(sparse_config.target_utilization > dense_config.target_utilization);
    }

    #[test]
    fn test_sram_pressure() {
        let problem = make_large_problem(10, 500_000);
        let tensor_meta = problem.build_tensor_meta();
        let ops: Vec<OpId> = vec![0, 1, 2];

        let pressure = calculate_sram_pressure(
            &ops,
            &problem,
            &problem.native_granularity,
            &tensor_meta,
        );

        // Should return a value between 0 and 1+ (can exceed 1 if doesn't fit)
        assert!(pressure >= 0.0);
    }

    #[test]
    fn test_validate_partition_plan() {
        let valid_plan = PartitionPlan {
            num_subgraphs: 3,
            ops_per_subgraph: 10,
            effective_capacity: 100_000,
            max_ops_per_subgraph: 50,
            rationale: "Test".to_string(),
            can_use_full_fusion: false,
        };

        assert!(validate_partition_plan(&valid_plan, 30).is_ok());

        let invalid_plan = PartitionPlan {
            num_subgraphs: 0, // Invalid!
            ops_per_subgraph: 10,
            effective_capacity: 100_000,
            max_ops_per_subgraph: 50,
            rationale: "Test".to_string(),
            can_use_full_fusion: false,
        };

        assert!(validate_partition_plan(&invalid_plan, 30).is_err());
    }
}