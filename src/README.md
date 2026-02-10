# MLSys Scheduler - Source Code Documentation

## Overview

This is a high-performance graph scheduler for ML workloads that implements aggressive operator fusion to minimize memory transfers between fast memory (SRAM/Scratchpad) and slow memory (DRAM). The scheduler processes computation graphs and outputs an optimized execution plan.

The core strategy is **Monolith Fusion**: fuse ALL operators into a single subgraph whenever possible, eliminating intermediate tensor materialization to DRAM.

## Architecture

```
src/
  lib.rs          - Library exports and module declarations
  main.rs         - CLI entry point and solution serialization
  models.rs       - Data structures (Problem, Solution, Tensor, Op, Granularity)
  scheduler.rs    - Main scheduling algorithm (monolith fusion strategy)
  cost.rs         - Cost model for latency estimation
  memory.rs       - Working set computation and memory validation
  liveness.rs     - Tensor lifetime analysis and SRAM prioritization
```

## Module Details

### models.rs

Defines the core data structures:

- **Problem**: Input specification with tensors, operations, memory capacity, bandwidth
- **Solution**: Output with subgraphs, granularity, traversal order
- **Tensor**: Width x Height dimensions
- **Op**: Operation with type (MatMul/Pointwise), inputs, outputs, base_cost
- **Granularity**: Tile dimensions (width, height, depth/Split-K)
- **TensorMeta**: Metadata tracking producer/consumer relationships

### scheduler.rs

Implements the scheduling algorithm with these key strategies:

1. **Monolith-First Fusion**: Attempts to fuse all operations into a single subgraph
2. **SRAM-First Execution**: Prioritizes operations whose inputs are already in fast memory
3. **Dynamic Split-K**: Splits MatMul reduction dimension to fit in SRAM
4. **Snake Traversal**: Zig-zag tile ordering for maximum data reuse

Key functions:
- `schedule()`: Main entry point that orchestrates the scheduling
- `select_granularity_with_dynamic_split_k()`: Chooses optimal tile size and Split-K
- `calculate_fusion_priority()`: Scores operations for scheduling order

### cost.rs

Generic cost model that works with any hardware configuration:

**Compute Cost:**
```
cost = base_cost * num_tiles * inefficiency * split_k_factor
```

Where:
- `num_tiles = ceil(W/tile_w) * ceil(H/tile_h)`
- `inefficiency = native_tile_size / exec_tile_size` (penalty for small tiles)
- `split_k_factor = 1.0 + 0.02 * ln(k)` (logarithmic overhead for Split-K)

**Fusion Bonuses:**
- Intermediate elimination: Up to 50% reduction when tensors stay in registers
- Op-pattern fusion: MatMul->Pointwise (1.5x), Pointwise->Pointwise (1.3x)

**Memory Transfer Cost:**
- DRAM reads/writes divided by bandwidth
- Double buffering hides 85% of transfer latency
- SRAM-resident tensors have near-zero access cost

### memory.rs

Handles working set computation and memory validation:

- **Working Set**: Sum of input slices + output slices + accumulator (for Split-K)
- **Intermediate Tensors**: Produced and consumed within subgraph = ephemeral (no SRAM cost)
- **Split-K**: Divides K dimension, reducing per-tile memory footprint

Key functions:
- `compute_subgraph_working_set()`: Calculates memory needed for fused execution
- `validate_memory_fit()`: Checks if subgraph fits in fast memory
- `find_fitting_granularity()`: Searches for valid tile configuration

### liveness.rs

Tensor lifetime analysis for optimal SRAM allocation:

- **LivenessInterval**: Tracks when each tensor is produced and last consumed
- **SRAM Priority**: Tensors with high reuse get priority for SRAM residence
- **Pointwise Chain Detection**: Identifies fusible sequences
- **Dynamic Split-K Calculation**: Computes optimal K based on SRAM capacity

## Algorithm Flow

```
1. Parse Problem (tensors, ops, hardware params)
2. Build tensor metadata (producer/consumer graph)
3. Perform liveness analysis
4. Attempt monolith fusion (all ops in one subgraph)
5. Select granularity:
   a. Try native granularity (K=1)
   b. Try Split-K (K=2, 4, 8, 16) if needed
   c. Reduce spatial granularity as fallback
6. Generate snake traversal order
7. Compute estimated latency
8. Output solution JSON
```

## Key Optimizations

### Split-K

When MatMul working set exceeds SRAM, split the reduction dimension:
- K=2: Process half the K dimension at a time, accumulate partial sums
- Overhead is logarithmic: `1 + 0.02 * ln(K)`
- Maximum K=16 for reasonable performance

### Intermediate Elimination

Tensors produced and consumed within a fused subgraph never touch DRAM:
```
MatMul -> [intermediate] -> Pointwise
         ^^^^^^^^^^^^^^
         stays in registers, no DRAM write
```

### Double Buffering

Overlap memory transfers with computation:
- While computing tile N, prefetch data for tile N+1
- Hides 85% of memory transfer latency

### Snake Traversal

Zig-zag pattern maximizes tile-to-tile data reuse:
```
Row 0: [0] -> [1] -> [2] -> [3]
                            |
Row 1: [7] <- [6] <- [5] <- [4]
```

## Benchmark Results

All benchmarks processed in **1 subgraph** (perfect monolith fusion):

| Benchmark | Tensors | Ops | Granularity | Estimated Latency | Time |
|-----------|---------|-----|-------------|-------------------|------|
| mlsys-2026-1 | 9 | 5 | 128x128x2 | 49,966 | 0.5ms |
| mlsys-2026-5 | 29 | 19 | 128x32x2 | 315,558 | 0.4ms |
| mlsys-2026-9 | 49 | 32 | 128x128x2 | 5,158,877 | 0.6ms |
| mlsys-2026-13 | 100 | 63 | 128x128 | 2,303,967 | 1.7ms |
| mlsys-2026-17 | 160 | 103 | 128x128x2 | 2,415,320 | 5.6ms |

Notes:
- Benchmark 13 uses K=1 (no Split-K needed) because SRAM is larger (600k)
- Benchmark 9 has high latency due to large tensors (4096x1024) with 16 MatMul ops
- All execution times are under 10ms

## Robustness

The scheduler handles edge cases safely:

- Empty operation lists (returns zero cost)
- Extreme Split-K values (clamped to 1-64)
- Tensors smaller than tile granularity
- Any hardware configuration (SRAM size, bandwidth)

## Testing

Run all tests:
```bash
cargo test
```

20 tests covering:
- Unit tests for each module
- Integration tests for end-to-end scheduling
- Robustness tests for edge cases

## Usage

```bash
# Build
cargo build --release

# Run scheduler
./target/release/mlsys input.json output.json

# Run with verbose output
./target/release/mlsys input.json output.json --verbose
```

## Output Format

```json
{
  "subgraphs": [
    {
      "ops": [0, 1, 2, ...],
      "granularity": {"w": 128, "h": 128, "k": 2},
      "traversal_order": [0, 1, 3, 2, ...]
    }
  ]
}
```

