# MLSys 2026 Graph Scheduler - Track A Solution

A high-performance graph scheduler for the Google MLSys Challenge 2026, implementing advanced optimization techniques for minimizing execution latency while respecting SRAM memory constraints.

## Performance Results

| Benchmark | Ops | Tensors | Scheduling Time | Estimated Latency | Subgraphs |
|-----------|-----|---------|-----------------|-------------------|-----------|
| mlsys-2026-1 | 5 | 9 | ~100us | 43,211 | 1 |
| mlsys-2026-5 | 19 | 29 | ~220us | 260,889 | 1 |
| mlsys-2026-9 | 32 | 49 | ~320us | 4,411,014 | 1 |
| mlsys-2026-13 | 63 | 100 | ~2.2ms | 1,958,738 | 1 |
| mlsys-2026-17 | 103 | 160 | ~8.1ms | 2,057,311 | 1 |

All benchmarks complete well under their timeout limits (2s to 60s).

## Key Optimizations

### 1. Compute-Aware Prefetch (Double Buffering)

Instead of using a fixed overlap factor, the scheduler calculates the actual compute-to-memory ratio for each tile. When compute time exceeds memory transfer time, the memory cost becomes effectively zero (fully hidden by prefetch).

- Full overlap when compute >= memory transfer time
- Partial overlap proportional to compute/memory ratio
- Results in 10-15% latency improvement

### 2. Register Tiling (Micro-Block Optimization)

Tiles are subdivided into 8x8 micro-blocks that fit in registers, minimizing SRAM reads during computation.

- MatMul operations get up to 15% latency reduction
- Pointwise operations get up to 5% reduction
- Larger tiles benefit more from register reuse

### 3. Shape-Aware Asymmetric Tiling

The tiling search dynamically generates candidates based on tensor shapes, not just fixed options.

- Analyzes dominant tensor aspect ratios
- Generates shape-matched tiles (e.g., 256x64 for wide matrices)
- Applies shape-matching bonus to favor aligned tiles
- Includes extreme ratios: 512x32, 32x512

### 4. Parallel Processing with Rayon

Multi-core parallelization for the computationally intensive parts of scheduling.

- Parallel tiling candidate evaluation
- Parallel fusion priority calculation
- Intelligent thresholds to avoid overhead on small inputs

### 5. Engineering Decision Telemetry

Detailed logging system that explains WHY decisions were made.

```bash
# Enable telemetry
RUST_LOG=mlsys=info ./mlsys input.json output.json
```

Example output:
```
[SPLIT-K] Op 0: Split-K=2 for MatMul | SRAM pressure reduced by 50.0% | Hiding 30000 DRAM cycles
[FUSION] Subgraph 0: Fused 5 ops | Eliminated 4 intermediates | Saved 2097152 bytes DRAM traffic
```

## Architecture

```
src/
  main.rs       - Entry point and CLI
  lib.rs        - Module exports
  models.rs     - Problem/Solution data structures
  scheduler.rs  - Main scheduling algorithm
  cost.rs       - Latency cost model
  memory.rs     - SRAM working set computation
  liveness.rs   - Tensor lifetime analysis
  telemetry.rs  - Decision logging system
```

### Scheduling Strategy

1. **Liveness Analysis**: Compute tensor lifetimes and identify high-reuse candidates
2. **SRAM Reservation**: Reserve slots for tensors with multiple consumers
3. **Aggressive Fusion**: Fuse all possible ops into minimal subgraphs
4. **Dynamic Tiling**: Search for optimal tile shape and Split-K factor
5. **Retention Planning**: Decide which tensors to keep in SRAM across subgraphs

## Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test
```

## Usage

```bash
./target/release/mlsys <input.json> <output.json>
```

See [USAGE.md](USAGE.md) for detailed usage instructions.

## Dependencies

- Rust 1.70+
- serde / serde_json - JSON parsing
- petgraph - Graph data structures
- rayon - Parallel processing
- log / env_logger - Telemetry

## License

MIT License

## Documentation

- [USAGE.md](USAGE.md) - How to build, run, and use the scheduler
- [DESIGN.md](DESIGN.md) - Design philosophy and optimization decisions explained
- [PROBLEM.md](PROBLEM.md) - Official problem description from Google
- [README_CONTEST.md](README_CONTEST.md) - Original contest rules and information

## Contest Information

- **Challenge**: Google MLSys 2026 Graph Scheduling Competition
- **Track**: A (Systems Engineering)
- **Deadline**: April 24, 2026



