//! # MLSys Challenge 2026 - Graph Scheduler
//!
//! Optimized graph scheduler for minimizing execution latency while respecting
//! Scratchpad (SRAM) memory constraints.
//!
//! ## Architecture
//! - `models`: Data structures matching mlsys.h (Problem, Solution, Tensor, Op, etc.)
//! - `graph_rewrite`: Graph canonicalization and rewriting transformations
//! - `scheduler`: Greedy fusion engine with aggressive operator fusion
//! - `memory`: SRAM capacity management and working set computation
//! - `cost`: Latency model for subgraph evaluation
//! - `liveness`: Tensor liveness analysis for optimal SRAM allocation
//! - `telemetry`: Engineering decision logging for transparency
//! - `cost_model`: Adaptive cost model for hardware-aware optimization (robustness)
//! - `layout`: Memory layout transformation for bandwidth optimization
//! - `weight_stationary`: Dynamic weight prefetching to keep reused tensors resident
//! - `pipeline`: Kernel pipelining for epilogue/prologue overlap
//! - `optimizer`: Advanced optimization engine (Roofline model, polyhedral optimization)
//! - `hw_profile`: Hardware auto-detection (CPU cores, cache, memory)
//! - `triage`: Graph topology classification (Linear/Diamond/Monster)
//! - `bitset_liveness`: O(1) bitset-based liveness collision detection
//! - `parallel`: Multi-process parallel scheduler (fork + pipe IPC)

pub mod cost;
pub mod cost_model;
pub mod graph_rewrite;
pub mod layout;
pub mod liveness;
pub mod memory;
pub mod models;
pub mod optimizer;
pub mod pipeline;
pub mod scheduler;
pub mod telemetry;
pub mod weight_stationary;

pub mod partition;
pub mod hw_profile;
pub mod triage;
pub mod bitset_liveness;
pub mod parallel;
