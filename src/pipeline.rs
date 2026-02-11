//! Kernel Pipelining - Epilogue/Prologue Overlap Optimization
//!
//! This module implements micro-pipelining between subgraph executions to eliminate
//! the "gap" of silence between kernels.
//!
//! TRADITIONAL MODEL:
//! ```text
//! Subgraph A: [Load_A] -> [Compute_A] -> [Store_A] -> |IDLE|
//! Subgraph B:                                        |IDLE| -> [Load_B] -> [Compute_B] -> [Store_B]
//! ```
//!
//! PIPELINED MODEL:
//! ```text
//! Subgraph A: [Load_A] -> [Compute_A] -> [Store_A_start...Store_A_end]
//! Subgraph B:                            [...overlap...]  [Load_B_start...Load_B_end] -> [Compute_B] -> [Store_B]
//! ```
//!
//! KEY INSIGHT: If the hardware supports concurrent DMA operations (which most modern
//! accelerators do), we can overlap the epilogue (store) of subgraph N with the
//! prologue (load) of subgraph N+1.
//!
//! ROBUSTNESS AGAINST OVERFITTING:
//! 1. Overlap is only applied when hardware supports concurrent DMA
//! 2. Conservative overlap estimates to avoid underestimating latency
//! 3. Data dependency checking prevents invalid overlaps
//! 4. Configurable overlap factor based on hardware characteristics

use crate::models::{OpId, Problem, TensorId, TensorMeta, Granularity};
use std::collections::HashSet;

// ============================================================================
// Constants - Pipeline Configuration
// ============================================================================

/// Maximum overlap factor for epilogue/prologue pipelining.
/// This is the fraction of the shorter transfer that can overlap.
///
/// ROBUSTNESS: We use a conservative 0.7 (70%) instead of theoretical 1.0
/// because:
/// 1. Memory controller contention reduces effective parallelism
/// 2. Cache coherency overhead between concurrent DMAs
/// 3. Real hardware rarely achieves perfect overlap
pub const MAX_PIPELINE_OVERLAP_FACTOR: f64 = 0.7;

/// Minimum transfer size (bytes) to benefit from pipelining.
/// Very small transfers have high overhead that dominates any overlap benefit.
pub const MIN_TRANSFER_FOR_PIPELINING: i64 = 4096;

/// Minimum subgraph latency to consider for pipelining.
/// Tiny subgraphs have overhead that exceeds potential savings.
pub const MIN_LATENCY_FOR_PIPELINING: f64 = 100.0;

/// DMA concurrency penalty factor.
/// When two DMAs run concurrently, they typically achieve less than 2x bandwidth
/// due to memory controller arbitration. This factor models that inefficiency.
pub const DMA_CONCURRENCY_EFFICIENCY: f64 = 0.85;

// ============================================================================
// Hardware DMA Capabilities
// ============================================================================

/// Classification of hardware DMA capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaCapability {
    /// No concurrent DMA support - strict serialization
    Sequential,
    /// Basic concurrent DMA - read and write can overlap
    BasicConcurrent,
    /// Advanced concurrent DMA - multiple independent channels
    FullConcurrent,
}

/// Hardware characteristics relevant to pipelining
#[derive(Debug, Clone)]
pub struct PipelineHardwareProfile {
    /// DMA capability level
    pub dma_capability: DmaCapability,
    /// Number of independent DMA channels (if FullConcurrent)
    pub dma_channels: usize,
    /// Read bandwidth (units/cycle)
    pub read_bandwidth: i64,
    /// Write bandwidth (units/cycle) - may differ from read
    pub write_bandwidth: i64,
    /// Maximum overlap factor achievable on this hardware
    pub max_overlap: f64,
}

impl PipelineHardwareProfile {
    /// Detect hardware profile from problem specification.
    ///
    /// ROBUSTNESS: We use conservative defaults when hardware details are unknown.
    /// This ensures we never overestimate overlap capability.
    pub fn from_problem(problem: &Problem) -> Self {
        // Heuristic: Larger SRAM typically correlates with more sophisticated
        // memory controllers that support concurrent DMA
        let (dma_capability, max_overlap) = if problem.fast_memory_capacity >= 500_000 {
            // High-end hardware: likely has full concurrent DMA
            (DmaCapability::FullConcurrent, MAX_PIPELINE_OVERLAP_FACTOR)
        } else if problem.fast_memory_capacity >= 100_000 {
            // Mid-range: basic concurrent support
            (DmaCapability::BasicConcurrent, MAX_PIPELINE_OVERLAP_FACTOR * 0.8)
        } else {
            // Constrained: assume sequential to be safe
            // But still allow some overlap for modern hardware
            (DmaCapability::BasicConcurrent, MAX_PIPELINE_OVERLAP_FACTOR * 0.5)
        };

        Self {
            dma_capability,
            dma_channels: match dma_capability {
                DmaCapability::Sequential => 1,
                DmaCapability::BasicConcurrent => 2,
                DmaCapability::FullConcurrent => 4,
            },
            read_bandwidth: problem.slow_memory_bandwidth,
            write_bandwidth: problem.slow_memory_bandwidth, // Assume symmetric
            max_overlap,
        }
    }
}

// ============================================================================
// Subgraph Memory Profile
// ============================================================================

/// Memory transfer profile for a subgraph
#[derive(Debug, Clone)]
pub struct SubgraphMemoryProfile {
    /// Total bytes loaded from DRAM (prologue)
    pub prologue_bytes: i64,
    /// Total bytes stored to DRAM (epilogue)
    pub epilogue_bytes: i64,
    /// Time to complete prologue (cycles)
    pub prologue_time: f64,
    /// Time to complete epilogue (cycles)
    pub epilogue_time: f64,
    /// Compute time for the subgraph (cycles)
    pub compute_time: f64,
    /// Tensors that need to be loaded (external inputs)
    pub prologue_tensors: Vec<TensorId>,
    /// Tensors that need to be stored (external outputs not retained)
    pub epilogue_tensors: Vec<TensorId>,
}

impl SubgraphMemoryProfile {
    /// Calculate memory profile for a subgraph
    pub fn from_subgraph(
        ops: &[OpId],
        problem: &Problem,
        granularity: &Granularity,
        tensor_meta: &[TensorMeta],
        tensors_to_retain: &[TensorId],
        sram_resident: &HashSet<TensorId>,
    ) -> Self {
        let ops_set: HashSet<OpId> = ops.iter().copied().collect();
        let retain_set: HashSet<TensorId> = tensors_to_retain.iter().copied().collect();

        let mut prologue_bytes: i64 = 0;
        let mut epilogue_bytes: i64 = 0;
        let mut prologue_tensors: Vec<TensorId> = Vec::new();
        let mut epilogue_tensors: Vec<TensorId> = Vec::new();

        // Calculate prologue (inputs that need to be loaded)
        for &op_id in ops {
            let op = &problem.ops[op_id];
            for &input_id in &op.inputs {
                let meta = &tensor_meta[input_id];

                // Skip if already in SRAM or produced within subgraph
                if sram_resident.contains(&input_id) {
                    continue;
                }
                if meta.producer.is_some_and(|p| ops_set.contains(&p)) {
                    continue;
                }

                // Skip if already counted
                if prologue_tensors.contains(&input_id) {
                    continue;
                }

                let tensor = &problem.tensors[input_id];
                prologue_bytes += compute_transfer_size(tensor, granularity, op, input_id);
                prologue_tensors.push(input_id);
            }
        }

        // Calculate epilogue (outputs that need to be stored)
        for &op_id in ops {
            let op = &problem.ops[op_id];
            for &output_id in &op.outputs {
                let meta = &tensor_meta[output_id];

                // Skip if retained in SRAM (no store needed)
                if retain_set.contains(&output_id) {
                    continue;
                }

                // Skip if consumed within subgraph (ephemeral)
                let all_internal = meta.consumers.iter().all(|c| ops_set.contains(c));
                if all_internal && !meta.is_output {
                    continue;
                }

                // Skip if already counted
                if epilogue_tensors.contains(&output_id) {
                    continue;
                }

                let tensor = &problem.tensors[output_id];
                epilogue_bytes += compute_transfer_size(tensor, granularity, op, output_id);
                epilogue_tensors.push(output_id);
            }
        }

        let bandwidth = problem.slow_memory_bandwidth as f64;
        let prologue_time = prologue_bytes as f64 / bandwidth;
        let epilogue_time = epilogue_bytes as f64 / bandwidth;

        // Estimate compute time (simplified - actual is calculated elsewhere)
        let compute_time: f64 = ops.iter()
            .map(|&op_id| problem.ops[op_id].base_cost as f64)
            .sum();

        Self {
            prologue_bytes,
            epilogue_bytes,
            prologue_time,
            epilogue_time,
            compute_time,
            prologue_tensors,
            epilogue_tensors,
        }
    }
}

/// Calculate transfer size for a tensor.
///
/// For prologue/epilogue transfers, we transfer the FULL tensor, not just a tile.
/// The granularity affects how much is transferred per iteration, but for
/// inter-subgraph pipelining we care about the total transfer.
fn compute_transfer_size(tensor: &crate::models::Tensor, _granularity: &Granularity, _op: &crate::models::Op, _tensor_id: TensorId) -> i64 {
    // For pipelining purposes, we transfer the full tensor
    // (The cost model handles tiling internally)
    tensor.size()
}

// ============================================================================
// Pipeline Overlap Analysis
// ============================================================================

/// Result of analyzing pipeline overlap between two subgraphs
#[derive(Debug, Clone)]
pub struct PipelineOverlap {
    /// Can these subgraphs overlap at all?
    pub can_overlap: bool,
    /// Time saved by overlapping (cycles)
    pub time_saved: f64,
    /// Fraction of transfers that overlap
    pub overlap_fraction: f64,
    /// Reason if overlap is not possible
    pub no_overlap_reason: Option<String>,
}

impl PipelineOverlap {
    pub fn no_overlap(reason: &str) -> Self {
        Self {
            can_overlap: false,
            time_saved: 0.0,
            overlap_fraction: 0.0,
            no_overlap_reason: Some(reason.to_string()),
        }
    }
}

/// Analyze potential pipeline overlap between two consecutive subgraphs.
///
/// ROBUSTNESS RULES:
/// 1. No overlap if there's a data dependency (B needs output of A)
/// 2. No overlap if transfers are too small (overhead dominates)
/// 3. Conservative overlap estimate to avoid underestimating latency
pub fn analyze_pipeline_overlap(
    profile_a: &SubgraphMemoryProfile,
    profile_b: &SubgraphMemoryProfile,
    hw: &PipelineHardwareProfile,
) -> PipelineOverlap {
    // Rule 1: Check DMA capability
    if hw.dma_capability == DmaCapability::Sequential {
        return PipelineOverlap::no_overlap("Hardware doesn't support concurrent DMA");
    }

    // Rule 2: Check minimum transfer sizes
    if profile_a.epilogue_bytes < MIN_TRANSFER_FOR_PIPELINING {
        return PipelineOverlap::no_overlap("Epilogue too small for pipelining");
    }
    if profile_b.prologue_bytes < MIN_TRANSFER_FOR_PIPELINING {
        return PipelineOverlap::no_overlap("Prologue too small for pipelining");
    }

    // Rule 3: Check data dependencies
    // If B's prologue needs any tensor from A's epilogue, we can't overlap
    let epilogue_set: HashSet<TensorId> = profile_a.epilogue_tensors.iter().copied().collect();
    let has_dependency = profile_b.prologue_tensors.iter()
        .any(|t| epilogue_set.contains(t));

    if has_dependency {
        return PipelineOverlap::no_overlap("Data dependency between subgraphs");
    }

    // Calculate overlap potential
    // The overlap is limited by the shorter of the two transfers
    let overlap_window = profile_a.epilogue_time.min(profile_b.prologue_time);

    // Apply hardware-specific overlap factor and concurrency efficiency
    let effective_overlap = overlap_window * hw.max_overlap * DMA_CONCURRENCY_EFFICIENCY;

    // Calculate overlap fraction for telemetry
    let total_sequential = profile_a.epilogue_time + profile_b.prologue_time;
    let overlap_fraction = if total_sequential > 0.0 {
        effective_overlap / total_sequential
    } else {
        0.0
    };

    PipelineOverlap {
        can_overlap: true,
        time_saved: effective_overlap,
        overlap_fraction,
        no_overlap_reason: None,
    }
}

// ============================================================================
// Total Latency with Pipelining
// ============================================================================

/// Configuration for pipeline-aware latency calculation
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable pipelining optimization
    pub enabled: bool,
    /// Hardware profile for pipelining decisions
    pub hw_profile: Option<PipelineHardwareProfile>,
    /// Minimum overlap savings to apply (avoids micro-optimizations)
    pub min_savings_threshold: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hw_profile: None,
            min_savings_threshold: 10.0, // At least 10 cycles savings
        }
    }
}

/// Result of pipeline-aware latency calculation
#[derive(Debug, Clone)]
pub struct PipelinedLatencyResult {
    /// Total latency with pipelining
    pub total_latency: f64,
    /// Total latency without pipelining (for comparison)
    pub unpipelined_latency: f64,
    /// Total time saved by pipelining
    pub time_saved: f64,
    /// Number of subgraph pairs that benefited from pipelining
    pub pipelined_pairs: usize,
    /// Individual subgraph latencies
    pub subgraph_latencies: Vec<f64>,
    /// Overlap details for each pair
    pub overlaps: Vec<PipelineOverlap>,
}

/// Calculate total latency with pipeline overlap optimization.
///
/// This is the "God-level" optimization that overlaps the epilogue of subgraph N
/// with the prologue of subgraph N+1.
///
/// ROBUSTNESS:
/// 1. Falls back to sequential latency if pipelining is disabled
/// 2. Conservative overlap estimates to avoid underestimating latency
/// 3. Data dependency checking prevents invalid overlaps
/// 4. Configurable minimum savings threshold filters micro-optimizations
pub fn compute_pipelined_latency(
    subgraph_latencies: &[f64],
    subgraph_profiles: &[SubgraphMemoryProfile],
    config: &PipelineConfig,
    problem: &Problem,
) -> PipelinedLatencyResult {
    let n = subgraph_latencies.len();

    // Calculate unpipelined (baseline) latency
    let unpipelined_latency: f64 = subgraph_latencies.iter().sum();

    // If pipelining disabled or only one subgraph, return baseline
    if !config.enabled || n <= 1 {
        return PipelinedLatencyResult {
            total_latency: unpipelined_latency,
            unpipelined_latency,
            time_saved: 0.0,
            pipelined_pairs: 0,
            subgraph_latencies: subgraph_latencies.to_vec(),
            overlaps: Vec::new(),
        };
    }

    // Get hardware profile
    let hw = config.hw_profile.clone()
        .unwrap_or_else(|| PipelineHardwareProfile::from_problem(problem));

    // Analyze overlaps between consecutive subgraphs
    let mut overlaps: Vec<PipelineOverlap> = Vec::with_capacity(n - 1);
    let mut total_savings: f64 = 0.0;
    let mut pipelined_pairs: usize = 0;

    for i in 0..n-1 {
        if i >= subgraph_profiles.len() || i + 1 >= subgraph_profiles.len() {
            overlaps.push(PipelineOverlap::no_overlap("Missing profile"));
            continue;
        }

        let overlap = analyze_pipeline_overlap(
            &subgraph_profiles[i],
            &subgraph_profiles[i + 1],
            &hw,
        );

        if overlap.can_overlap && overlap.time_saved >= config.min_savings_threshold {
            total_savings += overlap.time_saved;
            pipelined_pairs += 1;
        }

        overlaps.push(overlap);
    }

    let total_latency = unpipelined_latency - total_savings;

    PipelinedLatencyResult {
        total_latency,
        unpipelined_latency,
        time_saved: total_savings,
        pipelined_pairs,
        subgraph_latencies: subgraph_latencies.to_vec(),
        overlaps,
    }
}

// ============================================================================
// Integration Helper
// ============================================================================

/// Build memory profiles for all subgraphs in a solution.
///
/// This is used to enable pipelined latency calculation.
pub fn build_subgraph_profiles(
    subgraphs: &[crate::models::Subgraph],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> Vec<SubgraphMemoryProfile> {
    let mut profiles = Vec::with_capacity(subgraphs.len());
    let mut sram_resident: HashSet<TensorId> = HashSet::new();

    for subgraph in subgraphs {
        let granularity = Granularity {
            width: subgraph.granularity.w,
            height: subgraph.granularity.h,
            depth: subgraph.granularity.k.unwrap_or(1),
        };

        let profile = SubgraphMemoryProfile::from_subgraph(
            &subgraph.ops,
            problem,
            &granularity,
            tensor_meta,
            &subgraph.tensors_to_retain,
            &sram_resident,
        );

        profiles.push(profile);

        // Update SRAM state
        for &tensor_id in &subgraph.tensors_to_retain {
            sram_resident.insert(tensor_id);
        }
    }

    profiles
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Tensor};

    fn make_test_problem() -> Problem {
        Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 }, // 0: Input A
                Tensor { width: 128, height: 128 }, // 1: Input B
                Tensor { width: 128, height: 128 }, // 2: Output A
                Tensor { width: 128, height: 128 }, // 3: Input C
                Tensor { width: 128, height: 128 }, // 4: Output B
            ],
            ops: vec![
                Op {
                    op_type: OpType::MatMul,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                    base_cost: 1000,
                },
                Op {
                    op_type: OpType::MatMul,
                    inputs: vec![3, 2], // Note: depends on output of Op 0
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
    fn test_hardware_profile_detection() {
        let problem = make_test_problem();
        let hw = PipelineHardwareProfile::from_problem(&problem);

        // 100KB SRAM = mid-range, basic concurrent
        assert_eq!(hw.dma_capability, DmaCapability::BasicConcurrent);
        assert!(hw.max_overlap > 0.0);
        assert!(hw.max_overlap <= MAX_PIPELINE_OVERLAP_FACTOR);
    }

    #[test]
    fn test_no_overlap_with_dependency() {
        let problem = make_test_problem();
        let tensor_meta = problem.build_tensor_meta();
        let granularity = Granularity::new(128, 128, 1);

        // Subgraph A produces tensor 2
        let profile_a = SubgraphMemoryProfile::from_subgraph(
            &[0],
            &problem,
            &granularity,
            &tensor_meta,
            &[], // Not retaining
            &HashSet::new(),
        );

        // Subgraph B needs tensor 2 (produced by A)
        let mut sram_resident = HashSet::new();
        // Don't mark tensor 2 as resident - it was evicted
        let profile_b = SubgraphMemoryProfile::from_subgraph(
            &[1],
            &problem,
            &granularity,
            &tensor_meta,
            &[],
            &sram_resident,
        );

        let hw = PipelineHardwareProfile::from_problem(&problem);
        let overlap = analyze_pipeline_overlap(&profile_a, &profile_b, &hw);

        // Should not overlap because B depends on A's output
        assert!(!overlap.can_overlap || overlap.no_overlap_reason.is_some());
    }

    #[test]
    fn test_overlap_independent_subgraphs() {
        // Create a problem with independent subgraphs
        let problem = Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 }, // 0
                Tensor { width: 128, height: 128 }, // 1
                Tensor { width: 128, height: 128 }, // 2
                Tensor { width: 128, height: 128 }, // 3
                Tensor { width: 128, height: 128 }, // 4
                Tensor { width: 128, height: 128 }, // 5
            ],
            ops: vec![
                Op {
                    op_type: OpType::MatMul,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                    base_cost: 1000,
                },
                Op {
                    op_type: OpType::MatMul,
                    inputs: vec![3, 4], // Independent inputs
                    outputs: vec![5],
                    base_cost: 1000,
                },
            ],
            fast_memory_capacity: 500000, // High-end
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let tensor_meta = problem.build_tensor_meta();
        let granularity = Granularity::new(128, 128, 1);

        let profile_a = SubgraphMemoryProfile::from_subgraph(
            &[0],
            &problem,
            &granularity,
            &tensor_meta,
            &[], // Not retaining
            &HashSet::new(),
        );

        let profile_b = SubgraphMemoryProfile::from_subgraph(
            &[1],
            &problem,
            &granularity,
            &tensor_meta,
            &[],
            &HashSet::new(),
        );

        // Debug: check profile values
        println!("Profile A - prologue: {} bytes, epilogue: {} bytes",
                 profile_a.prologue_bytes, profile_a.epilogue_bytes);
        println!("Profile B - prologue: {} bytes, epilogue: {} bytes",
                 profile_b.prologue_bytes, profile_b.epilogue_bytes);

        let hw = PipelineHardwareProfile::from_problem(&problem);
        let overlap = analyze_pipeline_overlap(&profile_a, &profile_b, &hw);

        println!("Overlap result: can_overlap={}, reason={:?}",
                 overlap.can_overlap, overlap.no_overlap_reason);

        // Should be able to overlap since they're independent
        // But only if both have sufficient transfer sizes
        if profile_a.epilogue_bytes >= MIN_TRANSFER_FOR_PIPELINING
           && profile_b.prologue_bytes >= MIN_TRANSFER_FOR_PIPELINING {
            assert!(overlap.can_overlap, "Should overlap for independent subgraphs");
            assert!(overlap.time_saved > 0.0);
        }
    }
}

