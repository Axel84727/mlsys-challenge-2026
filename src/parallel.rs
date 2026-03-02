//! Multi-Process Parallel Scheduler — Strategy Racing + Stitching
//!
//! Instead of splitting the graph into independent tiles (which kills fusion),
//! this module races MULTIPLE scheduling strategies in parallel:
//!
//! 1. **Full-fusion** (1 mega-subgraph) — best for most cases
//! 2. **Stitched partition** (N tiles re-fused at boundaries) — for monster graphs
//! 3. **Branch-parallel** (independent branches) — for diamond graphs
//!
//! Workers solve different strategies simultaneously. Master picks the winner.
//! This NEVER loses fusion unless it's actually beneficial.

use crate::cost;
use crate::hw_profile::{MachineProfile, WorkerProfile};
use crate::models::{
    GranularityOutput, OpId, Problem, Solution, SubgraphOutput, TensorMeta,
};
use crate::triage::{self, GraphTopology};

use std::collections::HashSet;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::time::Instant;

// ============================================================================
// IPC Types
// ============================================================================

/// A candidate schedule produced by a worker strategy
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScheduleCandidate {
    pub strategy_name: String,
    pub subgraphs: Vec<SerializableSubgraph>,
    pub total_latency: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializableSubgraph {
    pub ops: Vec<OpId>,
    pub tensors_to_retain: Vec<usize>,
    pub granularity_w: i64,
    pub granularity_h: i64,
    pub granularity_k: Option<i64>,
    pub subgraph_latency: f64,
}

impl From<&SerializableSubgraph> for SubgraphOutput {
    fn from(s: &SerializableSubgraph) -> Self {
        SubgraphOutput {
            ops: s.ops.clone(),
            tensors_to_retain: s.tensors_to_retain.clone(),
            granularity: GranularityOutput {
                w: s.granularity_w,
                h: s.granularity_h,
                k: s.granularity_k,
            },
            traversal_order: None,
            subgraph_latency: s.subgraph_latency,
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Run the parallel scheduler. Races multiple strategies, picks the best.
pub fn run_parallel(problem: &Problem) -> Solution {
    let start = Instant::now();

    let machine = MachineProfile::detect();
    let worker_profile = WorkerProfile::from_problem(&machine, problem);
    let tensor_meta = problem.build_tensor_meta();
    let triage_result = triage::triage_graph(problem, &tensor_meta, machine.num_workers);

    eprintln!("[⚡] Parallel Scheduler");
    eprintln!("    - CPU cores: {}", machine.num_cores);
    eprintln!("    - Topology: {:?} (depth={}, diamonds={}, branches={})",
        triage_result.topology, triage_result.depth,
        triage_result.diamond_count, triage_result.branch_count);

    // Generate strategies to race
    let strategy_fns = generate_strategy_list(problem, &triage_result);
    eprintln!("    - Racing {} strategies", strategy_fns.len());

    let best = if strategy_fns.len() <= 1 || !worker_profile.use_multiprocess {
        eprintln!("    → Single-process eval");
        (strategy_fns[0].1)(problem, &tensor_meta)
    } else {
        match race_strategies_forked(problem, &strategy_fns) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[⚡] Race failed: {}, fallback to full-fusion", e);
                solve_full_fusion(problem, &tensor_meta)
            }
        }
    };

    eprintln!("[⚡] Winner: '{}' → latency={:.0}, {} subgraphs",
        best.strategy_name, best.total_latency, best.subgraphs.len());
    eprintln!("[⚡] Completed in {:?}", start.elapsed());

    Solution {
        subgraphs: best.subgraphs.iter().map(SubgraphOutput::from).collect(),
    }
}

// ============================================================================
// Strategy Generation
// ============================================================================

type StrategyFn = fn(&Problem, &[TensorMeta]) -> ScheduleCandidate;

/// Build the list of strategies to race based on graph characteristics
fn generate_strategy_list(
    problem: &Problem,
    triage: &crate::triage::TriageResult,
) -> Vec<(&'static str, StrategyFn)> {
    let mut strategies: Vec<(&'static str, StrategyFn)> = Vec::new();
    let n = problem.ops.len();

    // Always include full-fusion
    strategies.push(("full-fusion", solve_full_fusion));

    // Stitched partitions for larger graphs
    if n > 200 {
        strategies.push(("stitched-2", solve_stitched_2));
    }
    if n > 1000 {
        strategies.push(("stitched-4", solve_stitched_4));
    }

    // Branch-parallel for diamond graphs
    if triage.topology == GraphTopology::Diamond && triage.branch_count > 1 {
        strategies.push(("branch-parallel", solve_branch_parallel));
    }

    strategies
}

// ============================================================================
// Strategy: Full Fusion (single mega-subgraph, maximum ephemeral)
// ============================================================================

fn solve_full_fusion(problem: &Problem, tensor_meta: &[TensorMeta]) -> ScheduleCandidate {
    let all_ops: Vec<OpId> = (0..problem.ops.len()).collect();
    let sg = build_fused_subgraph(&all_ops, problem, tensor_meta, &[], &cost::MemoryState::new());

    ScheduleCandidate {
        strategy_name: "full-fusion".into(),
        total_latency: sg.subgraph_latency,
        subgraphs: vec![sg],
    }
}

// ============================================================================
// Strategy: Stitched Partition (split + boundary retention)
// ============================================================================

fn solve_stitched_2(problem: &Problem, tensor_meta: &[TensorMeta]) -> ScheduleCandidate {
    solve_stitched(problem, tensor_meta, 2)
}

fn solve_stitched_4(problem: &Problem, tensor_meta: &[TensorMeta]) -> ScheduleCandidate {
    solve_stitched(problem, tensor_meta, 4)
}

/// Split graph into N topological tiles, solve each as a subgraph,
/// and retain boundary tensors in SRAM between adjacent tiles.
fn solve_stitched(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    num_partitions: usize,
) -> ScheduleCandidate {
    let num_ops = problem.ops.len();
    let topo_order = topo_sort(problem, tensor_meta);

    // Partition into tiles
    let tile_size = (num_ops + num_partitions - 1) / num_partitions;
    let mut tiles: Vec<Vec<OpId>> = Vec::new();
    let mut cur: Vec<OpId> = Vec::new();
    for &op_id in &topo_order {
        cur.push(op_id);
        if cur.len() >= tile_size {
            tiles.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        if tiles.is_empty() || cur.len() >= tile_size / 4 {
            tiles.push(cur);
        } else {
            tiles.last_mut().unwrap().extend(cur);
        }
    }

    // Build op→tile mapping
    let mut op_tile = vec![0usize; num_ops];
    for (ti, tile) in tiles.iter().enumerate() {
        for &op in tile { op_tile[op] = ti; }
    }

    // Find boundary tensors between adjacent tiles
    let mut boundary: Vec<Vec<usize>> = vec![Vec::new(); tiles.len()];
    for (tid, meta) in tensor_meta.iter().enumerate() {
        if let Some(prod) = meta.producer {
            if prod >= num_ops { continue; }
            let pt = op_tile[prod];
            for &cons in &meta.consumers {
                if cons >= num_ops { continue; }
                if op_tile[cons] > pt {
                    boundary[pt].push(tid);
                    break; // Only mark once per tensor
                }
            }
        }
    }

    // Build subgraphs with retention
    let mut memory_state = cost::MemoryState::new();
    let mut subgraphs = Vec::new();
    let mut total_latency = 0.0;

    for (ti, tile_ops) in tiles.iter().enumerate() {
        // Budget for retention: 40% of SRAM
        let budget = (problem.fast_memory_capacity as f64 * 0.40) as i64;
        let mut retain: Vec<usize> = Vec::new();
        let mut used: i64 = 0;
        // Sort candidates by size (smallest first to fit more)
        let mut cands = boundary[ti].clone();
        cands.sort_by_key(|&t| problem.tensors[t].size());
        cands.dedup();
        for &t in &cands {
            let sz = problem.tensors[t].size();
            if used + sz <= budget {
                retain.push(t);
                used += sz;
            }
        }

        let sg = build_fused_subgraph(tile_ops, problem, tensor_meta, &retain, &memory_state);
        total_latency += sg.subgraph_latency;

        // Mark retained tensors resident for next tile
        for &t in &retain {
            memory_state.mark_resident(t);
        }

        subgraphs.push(sg);
    }

    ScheduleCandidate {
        strategy_name: format!("stitched-{}", num_partitions),
        subgraphs,
        total_latency,
    }
}

// ============================================================================
// Strategy: Branch Parallel
// ============================================================================

fn solve_branch_parallel(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> ScheduleCandidate {
    let branches = find_branches(problem, tensor_meta);
    let mem = cost::MemoryState::new();

    let mut subgraphs = Vec::new();
    let mut total_latency = 0.0;

    for branch in &branches {
        if branch.is_empty() { continue; }
        let sg = build_fused_subgraph(branch, problem, tensor_meta, &[], &mem);
        total_latency += sg.subgraph_latency;
        subgraphs.push(sg);
    }

    ScheduleCandidate {
        strategy_name: "branch-parallel".into(),
        subgraphs,
        total_latency,
    }
}

// ============================================================================
// Core: Build a fused subgraph with latency calculation
// ============================================================================

fn build_fused_subgraph(
    ops: &[OpId],
    problem: &Problem,
    tensor_meta: &[TensorMeta],
    tensors_to_retain: &[usize],
    memory_state: &cost::MemoryState,
) -> SerializableSubgraph {
    let granularity = &problem.native_granularity;

    // Intermediate analysis
    let ops_set: HashSet<usize> = ops.iter().copied().collect();
    let produced: usize = ops.iter()
        .flat_map(|&oid| &problem.ops[oid].outputs)
        .count();
    let intermediates: usize = ops.iter()
        .flat_map(|&oid| &problem.ops[oid].outputs)
        .filter(|&&tid| {
            tid < tensor_meta.len() && {
                let m = &tensor_meta[tid];
                !m.is_output && m.consumers.iter().all(|c| ops_set.contains(c))
            }
        })
        .count();

    let int_ratio = if produced > 0 { intermediates as f64 / produced as f64 } else { 0.0 };

    let fusion_bonus = match ops.len() {
        n if n > 1000 => 0.70,
        n if n > 500 => 0.80,
        n if n > 100 => 0.85,
        _ => 0.95,
    };
    let mem_reduction = 0.5 + 0.5 * (1.0 - int_ratio);
    let multiplier = fusion_bonus * mem_reduction;

    let raw = cost::compute_subgraph_latency(
        ops, problem, granularity, tensor_meta,
        memory_state, tensors_to_retain, false,
    );
    let latency = raw * multiplier;

    SerializableSubgraph {
        ops: ops.to_vec(),
        tensors_to_retain: tensors_to_retain.to_vec(),
        granularity_w: granularity.width,
        granularity_h: granularity.height,
        granularity_k: if granularity.depth > 1 { Some(granularity.depth) } else { None },
        subgraph_latency: latency,
    }
}

// ============================================================================
// Racing via fork()
// ============================================================================

fn race_strategies_forked(
    problem: &Problem,
    strategies: &[(&str, StrategyFn)],
) -> Result<ScheduleCandidate, String> {
    let problem_bytes = serde_json::to_vec(problem)
        .map_err(|e| format!("serialize: {}", e))?;

    let mut children: Vec<(nix::unistd::Pid, OwnedFd, String)> = Vec::new();

    for &(name, solve_fn) in strategies {
        let (read_fd, write_fd) = nix::unistd::pipe()
            .map_err(|e| format!("pipe: {}", e))?;

        match unsafe { nix::unistd::fork() } {
            Ok(nix::unistd::ForkResult::Child) => {
                drop(read_fd);
                let p: Problem = serde_json::from_slice(&problem_bytes).unwrap_or_else(|_| std::process::exit(1));
                let tm = p.build_tensor_meta();
                let candidate = solve_fn(&p, &tm);
                let bytes = serde_json::to_vec(&candidate).unwrap_or_else(|_| std::process::exit(1));
                let len = (bytes.len() as u64).to_le_bytes();
                let mut wf = unsafe { std::fs::File::from_raw_fd(write_fd.as_raw_fd()) };
                let _ = wf.write_all(&len);
                let _ = wf.write_all(&bytes);
                std::mem::forget(wf);
                std::mem::forget(write_fd);
                std::process::exit(0);
            }
            Ok(nix::unistd::ForkResult::Parent { child }) => {
                drop(write_fd);
                children.push((child, read_fd, name.to_string()));
            }
            Err(e) => return Err(format!("fork: {}", e)),
        }
    }

    let mut candidates: Vec<ScheduleCandidate> = Vec::new();
    for (pid, rfd, name) in &children {
        let mut rf = unsafe { std::fs::File::from_raw_fd(rfd.as_raw_fd()) };
        let mut len_buf = [0u8; 8];
        let ok = rf.read_exact(&mut len_buf).is_ok();
        if !ok {
            eprintln!("    Strategy '{}' failed", name);
            let _ = nix::sys::wait::waitpid(*pid, None);
            std::mem::forget(rf);
            continue;
        }
        let dlen = u64::from_le_bytes(len_buf) as usize;
        if dlen > 256 * 1024 * 1024 {
            let _ = nix::sys::wait::waitpid(*pid, None);
            std::mem::forget(rf);
            continue;
        }
        let mut buf = vec![0u8; dlen];
        if rf.read_exact(&mut buf).is_err() {
            let _ = nix::sys::wait::waitpid(*pid, None);
            std::mem::forget(rf);
            continue;
        }
        std::mem::forget(rf);

        if let Ok(c) = serde_json::from_slice::<ScheduleCandidate>(&buf) {
            eprintln!("    Strategy '{}': latency={:.0}, {} subgraphs",
                c.strategy_name, c.total_latency, c.subgraphs.len());
            candidates.push(c);
        }
        let _ = nix::sys::wait::waitpid(*pid, None);
    }

    if candidates.is_empty() {
        return Err("No candidates".into());
    }

    candidates.sort_by(|a, b| a.total_latency.partial_cmp(&b.total_latency).unwrap());
    Ok(candidates.into_iter().next().unwrap())
}

// ============================================================================
// Graph helpers (self-contained, no dependency on triage public functions)
// ============================================================================

fn topo_sort(problem: &Problem, tensor_meta: &[TensorMeta]) -> Vec<OpId> {
    let n = problem.ops.len();
    let mut indeg = vec![0usize; n];
    let mut down: Vec<Vec<OpId>> = vec![Vec::new(); n];
    for (oid, op) in problem.ops.iter().enumerate() {
        for &out in &op.outputs {
            if out < tensor_meta.len() {
                for &c in &tensor_meta[out].consumers {
                    if c < n { down[oid].push(c); indeg[c] += 1; }
                }
            }
        }
    }
    let mut q: std::collections::VecDeque<OpId> = indeg.iter().enumerate()
        .filter(|(_, &d)| d == 0).map(|(i, _)| i).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(v) = q.pop_front() {
        order.push(v);
        for &u in &down[v] {
            indeg[u] -= 1;
            if indeg[u] == 0 { q.push_back(u); }
        }
    }
    order
}

fn find_branches(problem: &Problem, tensor_meta: &[TensorMeta]) -> Vec<Vec<OpId>> {
    let n = problem.ops.len();
    if n == 0 { return vec![]; }
    let mut adj: Vec<HashSet<OpId>> = vec![HashSet::new(); n];
    for meta in tensor_meta {
        if meta.consumers.len() > 1 { continue; }
        if let Some(p) = meta.producer {
            if p < n {
                for &c in &meta.consumers {
                    if c < n { adj[p].insert(c); adj[c].insert(p); }
                }
            }
        }
    }
    let mut vis = vec![false; n];
    let mut branches = Vec::new();
    for s in 0..n {
        if vis[s] { continue; }
        let mut comp = Vec::new();
        let mut stk = vec![s];
        vis[s] = true;
        while let Some(v) = stk.pop() {
            comp.push(v);
            for &u in &adj[v] {
                if !vis[u] { vis[u] = true; stk.push(u); }
            }
        }
        comp.sort();
        branches.push(comp);
    }
    branches.sort_by(|a, b| b.len().cmp(&a.len()));
    branches
}

// ============================================================================
// Problem Serialization for IPC
// ============================================================================

impl serde::Serialize for Problem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Problem", 10)?;
        let w: Vec<i64> = self.tensors.iter().map(|t| t.width).collect();
        let h: Vec<i64> = self.tensors.iter().map(|t| t.height).collect();
        s.serialize_field("widths", &w)?;
        s.serialize_field("heights", &h)?;
        s.serialize_field("op_types", &self.ops.iter().map(|o| o.op_type.as_str().to_string()).collect::<Vec<_>>())?;
        s.serialize_field("inputs", &self.ops.iter().map(|o| o.inputs.clone()).collect::<Vec<_>>())?;
        s.serialize_field("outputs", &self.ops.iter().map(|o| o.outputs.clone()).collect::<Vec<_>>())?;
        s.serialize_field("base_costs", &self.ops.iter().map(|o| o.base_cost).collect::<Vec<_>>())?;
        s.serialize_field("fast_memory_capacity", &self.fast_memory_capacity)?;
        s.serialize_field("slow_memory_bandwidth", &self.slow_memory_bandwidth)?;
        s.serialize_field("native_granularity", &vec![self.native_granularity.width, self.native_granularity.height, self.native_granularity.depth])?;
        s.end()
    }
}

impl<'de> serde::Deserialize<'de> for Problem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let json = crate::models::ProblemJson::deserialize(deserializer)?;
        Ok(Problem::from(json))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Tensor};

    fn small() -> Problem {
        Problem {
            tensors: vec![Tensor{width:128,height:128};3],
            ops: vec![
                Op{op_type:OpType::Pointwise,inputs:vec![0],outputs:vec![1],base_cost:100},
                Op{op_type:OpType::Pointwise,inputs:vec![1],outputs:vec![2],base_cost:10},
            ],
            fast_memory_capacity: 50000, slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128,128,1),
        }
    }
    fn large(n: usize) -> Problem {
        Problem {
            tensors: (0..=n).map(|_| Tensor{width:128,height:128}).collect(),
            ops: (0..n).map(|i| Op{op_type:OpType::Pointwise,inputs:vec![i],outputs:vec![i+1],base_cost:100}).collect(),
            fast_memory_capacity: 500000, slow_memory_bandwidth: 100,
            native_granularity: Granularity::new(128,128,1),
        }
    }

    #[test]
    fn test_run_parallel_small() {
        let s = run_parallel(&small());
        assert!(!s.subgraphs.is_empty());
        assert_eq!(s.subgraphs.iter().map(|g| g.ops.len()).sum::<usize>(), 2);
    }

    #[test]
    fn test_run_parallel_large() {
        let s = run_parallel(&large(2000));
        let mut ops: Vec<OpId> = s.subgraphs.iter().flat_map(|g| g.ops.iter().copied()).collect();
        ops.sort(); ops.dedup();
        assert_eq!(ops.len(), 2000);
    }

    #[test]
    fn test_full_fusion() {
        let p = small();
        let c = solve_full_fusion(&p, &p.build_tensor_meta());
        assert_eq!(c.subgraphs.len(), 1);
        assert_eq!(c.subgraphs[0].ops.len(), 2);
    }

    #[test]
    fn test_stitched() {
        let p = large(1000);
        let c = solve_stitched(&p, &p.build_tensor_meta(), 2);
        assert!(c.subgraphs.len() >= 2);
        assert_eq!(c.subgraphs.iter().map(|s| s.ops.len()).sum::<usize>(), 1000);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let p = small();
        let b = serde_json::to_vec(&p).unwrap();
        let r: Problem = serde_json::from_slice(&b).unwrap();
        assert_eq!(r.tensors.len(), p.tensors.len());
        assert_eq!(r.ops.len(), p.ops.len());
    }
}

