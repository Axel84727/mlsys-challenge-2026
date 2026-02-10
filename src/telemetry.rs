//! Engineering Decision Telemetry System
//!
//! This module provides detailed logging of scheduler decisions to help
//! explain WHY certain choices were made, not just WHAT was chosen.
//!
//! Enable verbose logging with: RUST_LOG=mlsys=debug cargo run ...
//! For trace-level details: RUST_LOG=mlsys=trace cargo run ...

use crate::models::{Granularity, OpId, OpType, TensorId};
use log::{debug, info, trace, warn};
use std::fmt::Write;

// ============================================================================
// Decision Categories
// ============================================================================

/// Categories of engineering decisions for structured logging
#[derive(Debug, Clone, Copy)]
pub enum DecisionCategory {
    /// Granularity/tiling selection
    Tiling,
    /// Split-K optimization for MatMul
    SplitK,
    /// Memory management and SRAM allocation
    Memory,
    /// Operation fusion decisions
    Fusion,
    /// Traversal order optimization
    Traversal,
    /// Tensor retention in SRAM
    Retention,
    /// Overall scheduling strategy
    Strategy,
}

impl DecisionCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            DecisionCategory::Tiling => "TILING",
            DecisionCategory::SplitK => "SPLIT-K",
            DecisionCategory::Memory => "MEMORY",
            DecisionCategory::Fusion => "FUSION",
            DecisionCategory::Traversal => "TRAVERSAL",
            DecisionCategory::Retention => "RETENTION",
            DecisionCategory::Strategy => "STRATEGY",
        }
    }
}

// ============================================================================
// Telemetry Logger
// ============================================================================

/// Log a granularity/tiling decision with detailed reasoning
pub fn log_tiling_decision(
    subgraph_id: usize,
    ops: &[OpId],
    chosen: &Granularity,
    reason: &str,
    alternatives_tried: usize,
    latency_improvement: Option<f64>,
) {
    let ops_str = if ops.len() <= 5 {
        format!("{:?}", ops)
    } else {
        format!("[{}, {}, ... {} more]", ops[0], ops[1], ops.len() - 2)
    };

    let improvement_str = latency_improvement
        .map(|imp| format!(", {:.1}% latency reduction", imp * 100.0))
        .unwrap_or_default();

    info!(
        "[{}] Subgraph {}: Chose {}x{}x{} for ops {} | {} | Tried {} alternatives{}",
        DecisionCategory::Tiling.as_str(),
        subgraph_id,
        chosen.width,
        chosen.height,
        chosen.depth,
        ops_str,
        reason,
        alternatives_tried,
        improvement_str
    );
}

/// Log a Split-K decision with SRAM pressure analysis
pub fn log_split_k_decision(
    op_id: OpId,
    op_type: &OpType,
    chosen_k: i64,
    sram_reduction_percent: f64,
    cycles_hidden: f64,
    reason: &str,
) {
    if chosen_k > 1 {
        info!(
            "[{}] Op {}: Split-K={} for {:?} | SRAM pressure reduced by {:.1}% | \
             Hiding {:.0} DRAM cycles | {}",
            DecisionCategory::SplitK.as_str(),
            op_id,
            chosen_k,
            op_type,
            sram_reduction_percent,
            cycles_hidden,
            reason
        );
    } else {
        debug!(
            "[{}] Op {}: No Split-K needed for {:?} | {}",
            DecisionCategory::SplitK.as_str(),
            op_id,
            op_type,
            reason
        );
    }
}

/// Log a fusion decision explaining why ops were grouped together
pub fn log_fusion_decision(
    subgraph_id: usize,
    fused_ops: &[OpId],
    seed_op: OpId,
    fusion_reason: &str,
    intermediate_tensors_eliminated: usize,
    memory_saved: i64,
) {
    let ops_preview: String = if fused_ops.len() <= 6 {
        format!("{:?}", fused_ops)
    } else {
        format!(
            "[{}, {}, {}, ... {} more]",
            fused_ops[0],
            fused_ops[1],
            fused_ops[2],
            fused_ops.len() - 3
        )
    };

    info!(
        "[{}] Subgraph {}: Fused {} ops {} (seed Op {}) | {} | \
         Eliminated {} intermediates | Saved {} bytes DRAM traffic",
        DecisionCategory::Fusion.as_str(),
        subgraph_id,
        fused_ops.len(),
        ops_preview,
        seed_op,
        fusion_reason,
        intermediate_tensors_eliminated,
        memory_saved
    );
}

/// Log memory/SRAM allocation decisions
pub fn log_memory_decision(
    context: &str,
    working_set_size: i64,
    sram_capacity: i64,
    utilization_percent: f64,
    decision: &str,
) {
    let status = if utilization_percent > 90.0 {
        "HIGH PRESSURE"
    } else if utilization_percent > 70.0 {
        "MODERATE"
    } else {
        "COMFORTABLE"
    };

    debug!(
        "[{}] {}: Working set {} / {} SRAM ({:.1}% - {}) | {}",
        DecisionCategory::Memory.as_str(),
        context,
        working_set_size,
        sram_capacity,
        utilization_percent,
        status,
        decision
    );
}

/// Log tensor retention decisions
pub fn log_retention_decision(
    subgraph_id: usize,
    retained_tensors: &[TensorId],
    evicted_tensors: &[TensorId],
    reason: &str,
    bytes_retained: i64,
    future_reuse_count: usize,
) {
    if !retained_tensors.is_empty() {
        info!(
            "[{}] Subgraph {}: Retaining {} tensors ({} bytes) for {} future uses | \
             Evicting {} tensors | {}",
            DecisionCategory::Retention.as_str(),
            subgraph_id,
            retained_tensors.len(),
            bytes_retained,
            future_reuse_count,
            evicted_tensors.len(),
            reason
        );

        trace!(
            "[{}] Retained tensor IDs: {:?}",
            DecisionCategory::Retention.as_str(),
            retained_tensors
        );
    }
}

/// Log traversal order optimization
pub fn log_traversal_decision(
    subgraph_id: usize,
    pattern: &str,
    tiles_count: i64,
    estimated_reuse_savings: f64,
) {
    if tiles_count > 1 {
        debug!(
            "[{}] Subgraph {}: Using {} traversal for {} tiles | \
             Estimated {:.1}% memory reuse savings",
            DecisionCategory::Traversal.as_str(),
            subgraph_id,
            pattern,
            tiles_count,
            estimated_reuse_savings * 100.0
        );
    }
}

/// Log overall strategy decisions
pub fn log_strategy_decision(decision: &str, metrics: &str) {
    info!(
        "[{}] {} | {}",
        DecisionCategory::Strategy.as_str(),
        decision,
        metrics
    );
}

// ============================================================================
// Detailed Analysis Logging
// ============================================================================

/// Log detailed comparison of tiling candidates (trace level)
pub fn log_tiling_comparison(
    candidates: &[(i64, i64, i64, f64, bool)], // (w, h, k, latency, fits)
    winner: (i64, i64, i64),
) {
    if !log::log_enabled!(log::Level::Trace) {
        return;
    }

    let mut report = String::from("\n┌─────────────────────────────────────────────────────────────┐\n");
    writeln!(report, "│ TILING CANDIDATE COMPARISON                                 │").unwrap();
    writeln!(report, "├──────────┬──────────┬──────────┬───────────────┬───────────┤").unwrap();
    writeln!(report, "│  Width   │  Height  │  Split-K │    Latency    │   Fits?   │").unwrap();
    writeln!(report, "├──────────┼──────────┼──────────┼───────────────┼───────────┤").unwrap();

    for (w, h, k, latency, fits) in candidates {
        let is_winner = (*w, *h, *k) == winner;
        let marker = if is_winner { "→" } else { " " };
        let fits_str = if *fits { "✓" } else { "✗" };
        writeln!(
            report,
            "│{} {:>6}  │  {:>6}  │    {:>2}    │ {:>11.2}  │    {}      │",
            marker, w, h, k, latency, fits_str
        )
        .unwrap();
    }

    write!(report, "└──────────┴──────────┴──────────┴───────────────┴───────────┘").unwrap();
    trace!("{}", report);
}

/// Log compute vs memory analysis for prefetch decisions
pub fn log_prefetch_analysis(
    subgraph_id: usize,
    compute_time: f64,
    memory_time: f64,
    overlap_ratio: f64,
    effective_memory_cost: f64,
) {
    let overlap_status = if overlap_ratio >= 1.0 {
        "FULL OVERLAP - Memory fully hidden by compute"
    } else if overlap_ratio >= 0.5 {
        "PARTIAL OVERLAP - Some memory exposed"
    } else {
        "MEMORY BOUND - Compute cannot hide transfer"
    };

    debug!(
        "[{}] Subgraph {}: Compute={:.0} cycles, Memory={:.0} cycles | \
         Overlap ratio={:.2} | {} | Effective memory cost={:.0}",
        DecisionCategory::Memory.as_str(),
        subgraph_id,
        compute_time,
        memory_time,
        overlap_ratio,
        overlap_status,
        effective_memory_cost
    );
}

/// Log register tiling analysis
pub fn log_register_tiling_analysis(
    tile_size: (i64, i64),
    micro_blocks: (i64, i64),
    reuse_factor: f64,
    latency_reduction: f64,
) {
    trace!(
        "[{}] Tile {}x{} → {}x{} micro-blocks | \
         Reuse factor={:.1} | Latency reduction={:.1}%",
        DecisionCategory::Tiling.as_str(),
        tile_size.0,
        tile_size.1,
        micro_blocks.0,
        micro_blocks.1,
        reuse_factor,
        latency_reduction * 100.0
    );
}

/// Log shape-aware tiling decision
pub fn log_shape_matching(
    tensor_shape: (i64, i64),
    tensor_ratio: f64,
    tile_shape: (i64, i64),
    tile_ratio: f64,
    match_bonus: f64,
) {
    let match_quality = if match_bonus <= 0.92 {
        "EXCELLENT"
    } else if match_bonus <= 0.96 {
        "GOOD"
    } else if match_bonus <= 0.99 {
        "FAIR"
    } else {
        "POOR"
    };

    debug!(
        "[{}] Tensor {}x{} (ratio={:.2}) matched with tile {}x{} (ratio={:.2}) | \
         {} match | Bonus={:.1}%",
        DecisionCategory::Tiling.as_str(),
        tensor_shape.0,
        tensor_shape.1,
        tensor_ratio,
        tile_shape.0,
        tile_shape.1,
        tile_ratio,
        match_quality,
        (1.0 - match_bonus) * 100.0
    );
}

// ============================================================================
// Summary Reports
// ============================================================================

/// Generate a summary of all scheduling decisions
pub fn log_scheduling_summary(
    total_ops: usize,
    total_subgraphs: usize,
    total_latency: f64,
    fusion_ratio: f64,
    split_k_usage: usize,
    retained_tensors: usize,
) {
    info!("╔═══════════════════════════════════════════════════════════════╗");
    info!("║              SCHEDULING DECISION SUMMARY                      ║");
    info!("╠═══════════════════════════════════════════════════════════════╣");
    info!(
        "║  Total Operations:     {:>6}                                 ║",
        total_ops
    );
    info!(
        "║  Subgraphs Created:    {:>6}                                 ║",
        total_subgraphs
    );
    info!(
        "║  Fusion Ratio:         {:>5.1}x (ops/subgraph)                 ║",
        fusion_ratio
    );
    info!(
        "║  Split-K Optimizations:{:>6}                                 ║",
        split_k_usage
    );
    info!(
        "║  Tensors Retained:     {:>6}                                 ║",
        retained_tensors
    );
    info!(
        "║  Estimated Latency:    {:>12.2}                        ║",
        total_latency
    );
    info!("╚═══════════════════════════════════════════════════════════════╝");
}

/// Log a warning when a suboptimal decision was forced
pub fn log_fallback_decision(context: &str, reason: &str, impact: &str) {
    warn!(
        "[FALLBACK] {}: {} | Impact: {}",
        context, reason, impact
    );
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the telemetry system
/// Call this at the start of main() to enable logging
pub fn init() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn")
    )
    .format_timestamp(None)
    .format_module_path(false)
    .format_target(false)
    .init();
}

/// Check if detailed logging is enabled
pub fn is_verbose() -> bool {
    log::log_enabled!(log::Level::Debug)
}

/// Check if trace-level logging is enabled
pub fn is_trace() -> bool {
    log::log_enabled!(log::Level::Trace)
}


