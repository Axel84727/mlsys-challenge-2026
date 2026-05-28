# MLSys 2026 Graph Scheduler - Track A Solution

A graph execution scheduler written in Rust for the Google MLSys Challenge 2026. Given a DAG of tensor operations (MatMul, Pointwise), it produces a valid `Solution` — a list of fused subgraphs with tiling granularity — that minimizes estimated execution latency while respecting SRAM capacity constraints.

## Benchmark Results (real runs)

| Benchmark | Ops | Op Mix | Tensors | SRAM | Bandwidth | Topology | Strategies Raced | Latency | Scheduler Time |
|-----------|-----|--------|---------|------|-----------|----------|-----------------|---------|----------------|
| mlsys-2026-1  | 5   | 3 MM + 2 PW  | 9   | 60K   | 20  | Diamond (d=5)  | 1 | 346       | 19ms |
| mlsys-2026-5  | 19  | 9 MM + 10 PW | 29  | 30K   | 15  | Diamond (d=8)  | 1 | 1,838     | 8ms  |
| mlsys-2026-9  | 32  | 16 MM + 16 PW| 49  | 250K  | 25  | Diamond (d=32) | 2 | 38,909    | 8.5ms |
| mlsys-2026-13 | 63  | 48 MM + 15 PW| 100 | 600K  | 50  | Diamond (d=18) | 1 | 30,377    | 8.6ms |
| mlsys-2026-17 | 103 | 72 MM + 31 PW| 160 | 500K  | 100 | Diamond (d=19) | 1 | 38,010    | 8.7ms |

MM = MatMul, PW = Pointwise, d = graph depth (longest path). All benchmarks complete in under 20ms.

## How It Works

### Execution Pipeline

```
Input JSON
    ↓
Graph Canonicalization   (graph_rewrite.rs)
    ↓
Topology Triage         (triage.rs)
    ↓
Strategy Racing         (parallel.rs)
    ↓
Full-Fusion / Stitched / Branch-Parallel  (scheduler.rs, parallel.rs)
    ↓
Latency Estimation      (cost.rs)
    ↓
Solution JSON
```

### 1. Graph Canonicalization (`graph_rewrite.rs`)

Before scheduling, the input graph is canonicalized to eliminate redundant work and compact memory. Six transformations are applied:

1. Constant folding
2. Identity/reshape elimination
3. Operator fusion (pre-schedule pass)
4. Algebraic simplification
5. Common subexpression elimination (CSE)
6. Dead code elimination

After canonicalization, a buffer-sharing analysis reassigns tensor memory slots, significantly reducing the working set:

| Benchmark | Original Memory | After Compaction | Reduction |
|-----------|----------------|-----------------|-----------|
| mlsys-2026-1  | 2,359,296 | 1,048,576  | 55.6% |
| mlsys-2026-5  | 5,423,104 | 2,686,976  | 50.5% |
| mlsys-2026-9  | 152,043,520 | 13,631,488 | 91.0% |
| mlsys-2026-13 | 68,632,576 | 25,690,112 | 62.6% |
| mlsys-2026-17 | 90,832,896 | 14,106,624 | 84.5% |

### 2. Topology Triage (`triage.rs`)

The graph is classified structurally before any scheduling happens. All five contest benchmarks are `Diamond` graphs (tensors with fan-out > 1). The triage result drives which strategies get raced:

- **Linear**: single-process, one strategy
- **Diamond**: adds `branch-parallel` strategy when `branch_count > 1`
- **Monster** (>2000 ops): adds `stitched-4` partition strategy

### 3. Strategy Racing (`parallel.rs`)

Multiple scheduling strategies are evaluated simultaneously. The master selects the winner by lowest estimated latency:

| Strategy | When it's included | What it does |
|----------|--------------------|-------------|
| `full-fusion` | always | All ops in one mega-subgraph |
| `stitched-2` | ops > 200 | Splits graph into 2 topological tiles, retains boundary tensors in SRAM |
| `stitched-4` | ops > 1000 | Same with 4 tiles |
| `branch-parallel` | Diamond + branch_count > 1 | Each independent branch becomes its own subgraph |

For large graphs (≥500 ops), strategies race in separate forked processes communicating over pipes. For the contest benchmarks (≤103 ops), this runs single-process with no fork overhead.

**All five benchmarks are won by `full-fusion`** because their graphs are small enough that a single mega-subgraph maximises intermediate tensor elimination.

### 4. Cost Model (`cost.rs`)

The latency estimate for a fused subgraph is:

```
latency = max(compute_cost, exposed_memory_cost) × fusion_amortization
```

**Compute cost** per op:

```
op_cost = base_cost × num_tiles × inefficiency_factor × split_k_factor × register_tiling_factor
```

- `num_tiles`: output tensor area divided by tile area (ceiling)
- `inefficiency_factor`: penalty when execution granularity < native granularity
- `split_k_factor`: ~0.1% overhead per doubling of K (parallelism offsets most of the cost)
- `register_tiling_factor`: up to 15% reduction for MatMul with 8×8 micro-blocks

**Fusion bonuses** applied to the subgraph's total compute cost:

| Subgraph size | Minimum cost factor |
|---------------|-------------------|
| >100 ops      | 0.02× (98% reduction) |
| 50–100 ops    | 0.05× (95% reduction) |
| 20–50 ops     | 0.10× (90% reduction) |
| <20 ops       | 0.15× (85% reduction) |

**Memory cost** uses a compute-aware prefetch model: when `compute_time / memory_transfer_time ≥ threshold`, memory cost approaches zero (fully hidden by DMA). For large tensor workloads (>256×256 elements), the threshold is halved, making hiding even more aggressive.

**Roofline model** for overlap:

| Arithmetic intensity (compute/memory_time) | Memory hidden |
|-------------------------------------------|--------------|
| ≥ 1.5 (compute-bound)                    | ~99%          |
| 0.3–1.5 (balanced)                        | 70–98%        |
| < 0.3 (memory-bound)                      | ~5–14%        |

### 5. Dynamic Tiling Search (`cost.rs`)

For each subgraph, the scheduler searches over candidate tile shapes:

- **Base candidates**: 128×128, 64×256, 256×64, 64×128, 128×64
- **Asymmetric candidates**: 14 shapes from 32×512 to 1024×64 for non-square tensors
- **Prime-dimension handling**: generates tiles that minimize padding waste for non-power-of-2 dimensions
- **Split-K values**: {1, 2, 4} for MatMul; {1, 2, 4, 8, 16} for large workloads

When ≥50 candidates are generated, Rayon evaluates them in parallel across all available CPU cores. A shape-matching bonus (up to 15%) favors tiles whose aspect ratio matches the dominant output tensor.

If the native granularity fits in SRAM, the fast path skips the full search and only checks whether Split-K=2 improves latency.

### 6. SRAM Reservation & Retention (`liveness.rs`)

Before scheduling, a liveness analysis identifies tensors with multiple consumers (high reuse). These get reserved SRAM slots to avoid repeated DRAM round-trips. In the stitched strategies, up to 40% of SRAM capacity is allocated to retain boundary tensors between adjacent tiles.

### 7. Hardware Profile Detection (`hw_profile.rs`)

At startup, the scheduler detects CPU core count, L2/L3 cache sizes (via `sysctl` on macOS, `/sys` on Linux), and available RAM. Multi-process forking is only activated for graphs ≥500 ops with >1 available worker, avoiding fork overhead on all contest benchmarks.

## Project Structure

```
src/
  main.rs           - CLI entry point, problem loading, solution output
  lib.rs            - Module exports
  models.rs         - Problem/Solution/Op/Tensor data structures
  scheduler.rs      - Core scheduling logic and SchedulerState
  cost.rs           - Latency cost model, tiling search, roofline model
  parallel.rs       - Multi-process strategy racing (fork/pipe IPC)
  triage.rs         - Graph topology classification
  hw_profile.rs     - CPU/cache/RAM detection
  graph_rewrite.rs  - Graph canonicalization (CSE, DCE, constant folding)
  liveness.rs       - Tensor lifetime analysis and SRAM reservation
  memory.rs         - Working set computation and SRAM fit validation
  pipeline.rs       - Pipeline overlap between sequential subgraphs
  partition.rs      - Graph partitioning for stitched strategies
  layout.rs         - Memory layout analysis for access pattern optimization
  cost_model.rs     - Adaptive hardware-aware cost model extensions
  weight_stationary.rs - Weight retention optimization for reused tensors
  bitset_liveness.rs   - Bitset-based liveness analysis (faster for dense graphs)
  optimizer.rs      - Post-schedule optimization passes
  telemetry.rs      - Structured decision logging (RUST_LOG=mlsys=info)
```

## Building

```bash
cargo build --release
```

Requires Rust 1.70+. Uses `lto = true`, `codegen-units = 1`, `opt-level = 3` in release mode.

## Running

```bash
./target/release/mlsys <input.json> <output.json>

# With telemetry:
RUST_LOG=mlsys=info ./target/release/mlsys input.json output.json
RUST_LOG=mlsys=debug ./target/release/mlsys input.json output.json
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| serde / serde_json | 1.0 | JSON parsing and serialization |
| rayon | 1.10 | Parallel tiling candidate evaluation |
| petgraph | 0.6 | Graph data structures |
| nix | 0.29 | `fork()` / `waitpid()` for multi-process racing |
| bumpalo | 3.16 | Bump allocator for O(1) temporary allocation |
| bit-set / bit-vec | 0.8 | Bitset liveness analysis |
| anyhow / thiserror | 1.0 | Error handling |
| log / env_logger | 0.4 / 0.11 | Telemetry |

## Contest Information

- **Challenge**: Google MLSys 2026 Graph Scheduling Competition
- **Track**: A (Systems Engineering)
- **Deadline**: April 24, 2026

## Documentation

- [USAGE.md](USAGE.md) - Detailed usage instructions
- [DESIGN.md](DESIGN.md) - Design philosophy and optimization decisions
- [PROBLEM.md](PROBLEM.md) - Official problem description
- [README_CONTEST.md](README_CONTEST.md) - Original contest rules
