//! # MLSys Challenge 2026 - Graph Scheduler
//!
//! Optimized graph scheduler for minimizing execution latency while respecting
//! Scratchpad (SRAM) memory constraints.
//!
//! ## Architecture
//! - `models`: Data structures matching mlsys.h (Problem, Solution, Tensor, Op, etc.)
//! - `scheduler`: Greedy fusion engine with aggressive operator fusion
//! - `memory`: SRAM capacity management and working set computation
//! - `cost`: Latency model for subgraph evaluation

pub mod cost;
pub mod memory;
pub mod models;
pub mod scheduler;

