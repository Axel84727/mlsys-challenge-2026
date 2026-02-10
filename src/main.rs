//! MLSys Challenge 2026 - Graph Scheduler
//!
//! Usage: mlsys <input.json> <output.json>
//!
//! Optimized execution scheduler that minimizes latency while respecting
//! Scratchpad (SRAM) memory constraints.
//!
//! ## Telemetry / Decision Logging
//! 
//! Enable verbose logging to see engineering decisions:
//!   RUST_LOG=mlsys=info cargo run --release -- input.json output.json
//!
//! For detailed trace output:
//!   RUST_LOG=mlsys=debug cargo run --release -- input.json output.json
//!   RUST_LOG=mlsys=trace cargo run --release -- input.json output.json

use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::io::Write;
use std::time::Instant;

use mlsys::cost::compute_total_latency;
use mlsys::models::{Problem, ProblemJson};
use mlsys::scheduler::{optimize_schedule, schedule};
use mlsys::telemetry;

fn main() -> Result<()> {
    // Initialize telemetry system (respects RUST_LOG environment variable)
    telemetry::init();
    
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("MLSys Challenge 2026 - Graph Scheduler");
        eprintln!("=====================================");
        eprintln!();
        eprintln!("Usage: {} <input.json> <output.json>", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  <input.json>   Problem definition file");
        eprintln!("  <output.json>  Output solution file");
        eprintln!();
        eprintln!("Telemetry (Engineering Decision Logs):");
        eprintln!("  RUST_LOG=mlsys=info  - Show key decisions");
        eprintln!("  RUST_LOG=mlsys=debug - Show detailed analysis");
        eprintln!("  RUST_LOG=mlsys=trace - Show all comparisons");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} example_problem.json solution.json", args[0]);
        eprintln!("  RUST_LOG=mlsys=info {} input.json output.json", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Read and parse problem
    eprintln!("[*] Reading problem from: {}", input_path);
    let start = Instant::now();

    let input_data = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read input file: {}", input_path))?;

    let problem_json: ProblemJson = serde_json::from_str(&input_data)
        .with_context(|| "Failed to parse input JSON")?;

    let problem: Problem = problem_json.into();

    eprintln!("[*] Problem loaded in {:?}", start.elapsed());
    eprintln!("    - Tensors: {}", problem.tensors.len());
    eprintln!("    - Operations: {}", problem.ops.len());
    eprintln!("    - Fast Memory: {} units", problem.fast_memory_capacity);
    eprintln!("    - Slow Memory BW: {} units/cycle", problem.slow_memory_bandwidth);
    eprintln!(
        "    - Native Granularity: {}x{}x{}",
        problem.native_granularity.width,
        problem.native_granularity.height,
        problem.native_granularity.depth
    );

    // Run scheduler
    eprintln!("[*] Running scheduler...");
    let schedule_start = Instant::now();

    let initial_solution = schedule(&problem);
    let solution = optimize_schedule(initial_solution, &problem);

    eprintln!("[*] Scheduling completed in {:?}", schedule_start.elapsed());
    eprintln!("    - Subgraphs: {}", solution.subgraphs.len());

    // Calculate and report latency
    let subgraphs_for_latency: Vec<mlsys::models::Subgraph> = solution.subgraphs
        .iter()
        .map(|sg| mlsys::models::Subgraph {
            ops: sg.ops.clone(),
            tensors_to_retain: sg.tensors_to_retain.clone(),
            granularity: mlsys::models::GranularityOutput {
                w: sg.granularity.w,
                h: sg.granularity.h,
                k: sg.granularity.k,
            },
            traversal_order: sg.traversal_order.clone(),
            subgraph_latency: 0.0,
        })
        .collect();

    let total_latency = compute_total_latency(&subgraphs_for_latency, &problem);
    eprintln!("[*] Estimated total latency: {:.2}", total_latency);

    // Print subgraph summary
    for (i, sg) in solution.subgraphs.iter().enumerate() {
        let op_types: Vec<&str> = sg.ops
            .iter()
            .map(|&op_id| problem.ops[op_id].op_type.as_str())
            .collect();

        eprintln!(
            "    Subgraph {}: {} ops ({:?}), granularity={}x{}{}, retain={} tensors{}",
            i,
            sg.ops.len(),
            op_types,
            sg.granularity.w,
            sg.granularity.h,
            sg.granularity.k.map_or(String::new(), |k| format!("x{}", k)),
            sg.tensors_to_retain.len(),
            if sg.traversal_order.is_some() { ", snake" } else { "" }
        );
    }

    // Write solution
    eprintln!("[*] Writing solution to: {}", output_path);

    let output_json = serde_json::to_string_pretty(&solution)
        .with_context(|| "Failed to serialize solution")?;

    let mut output_file = fs::File::create(output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path))?;

    output_file
        .write_all(output_json.as_bytes())
        .with_context(|| "Failed to write solution")?;

    eprintln!("[✓] Done! Total time: {:?}", start.elapsed());

    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use mlsys::models::{Granularity, Op, OpType, Tensor};

    #[test]
    fn test_end_to_end_simple() {
        let problem = Problem {
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
                    base_cost: 10,
                },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };

        let initial = schedule(&problem);
        let solution = optimize_schedule(initial, &problem);

        // Verify solution is valid
        assert!(!solution.subgraphs.is_empty());

        // All ops should be scheduled exactly once
        let mut scheduled_ops: Vec<usize> = solution.subgraphs
            .iter()
            .flat_map(|sg| sg.ops.iter().copied())
            .collect();
        scheduled_ops.sort();
        assert_eq!(scheduled_ops, vec![0, 1]);
    }

    #[test]
    fn test_diamond_pattern() {
        // Diamond: op0 -> (op1, op2) -> op3
        let problem = Problem {
            tensors: vec![
                Tensor { width: 64, height: 64 },  // input
                Tensor { width: 64, height: 64 },  // op0 output, op1 & op2 input
                Tensor { width: 64, height: 64 },  // op1 output
                Tensor { width: 64, height: 64 },  // op2 output
                Tensor { width: 64, height: 64 },  // op3 output (final)
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
                    inputs: vec![1],
                    outputs: vec![3],
                    base_cost: 100,
                },
                Op {
                    op_type: OpType::Pointwise,
                    inputs: vec![2, 3],
                    outputs: vec![4],
                    base_cost: 100,
                },
            ],
            fast_memory_capacity: 20000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(64, 64, 1),
        };

        let solution = schedule(&problem);

        // All 4 ops should be scheduled
        let total_ops: usize = solution.subgraphs.iter().map(|sg| sg.ops.len()).sum();
        assert_eq!(total_ops, 4);
    }
}

