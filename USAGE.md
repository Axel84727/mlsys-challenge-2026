# Usage Guide

## Quick Start

```bash
# Build the scheduler
cargo build --release

# Run on a benchmark
./target/release/mlsys benchmarks/mlsys-2026-1.json solution.json
```

## Command Line Interface

```
mlsys <input.json> <output.json>

Arguments:
  input.json   - Problem definition file (JSON format)
  output.json  - Output solution file (will be created/overwritten)
```

## Input Format

The input JSON file describes the computation graph:

```json
{
  "tensors": [
    {"w": 512, "h": 512},
    {"w": 512, "h": 512}
  ],
  "op_types": ["MatMul", "Pointwise"],
  "op_inputs": [[0], [1]],
  "op_outputs": [[1], [2]],
  "op_base_costs": [1000, 100],
  "fast_mem": 60000,
  "slow_bw": 20,
  "native_granularity": [128, 128, 1]
}
```

## Output Format

The output JSON contains the scheduling solution:

```json
{
  "subgraphs": [
    {
      "ops": [0, 1],
      "granularity": {"w": 128, "h": 128, "k": 2},
      "tensors_to_retain": [1],
      "traversal_order": [0, 1, 3, 2]
    }
  ]
}
```

## Telemetry (Decision Logging)

Enable detailed logging to understand scheduling decisions:

```bash
# Key decisions only
RUST_LOG=mlsys=info ./target/release/mlsys input.json output.json

# Detailed analysis
RUST_LOG=mlsys=debug ./target/release/mlsys input.json output.json

# Full trace (all comparisons)
RUST_LOG=mlsys=trace ./target/release/mlsys input.json output.json
```

### Telemetry Output Examples

**INFO level** - Key decisions:
```
[SPLIT-K] Op 0: Split-K=2 for MatMul | SRAM pressure reduced by 50.0% | Hiding 30000 DRAM cycles
[FUSION] Subgraph 0: Fused 5 ops [0, 1, 2, 3, 4] | Eliminated 4 intermediates | Saved 2097152 bytes
[STRATEGY] Scheduled 5 ops into 1 subgraphs | Fusion ratio=5.0x, Split-K used in 1 subgraphs
```

**DEBUG level** - Adds memory and traversal analysis:
```
[MEMORY] Subgraph 0: Working set 49152 / 60000 SRAM (81.9% - MODERATE) | Granularity 128x128x2
[TRAVERSAL] Subgraph 0: Using Snake/Zig-zag traversal for 16 tiles | Estimated 15.0% memory reuse
```

**TRACE level** - Adds tiling candidate comparisons and shape matching details.

## Running Benchmarks

```bash
# Run all available benchmarks
for f in benchmarks/*.json; do
  echo "=== $f ==="
  ./target/release/mlsys "$f" /tmp/solution.json
done

# Run with telemetry
RUST_LOG=mlsys=info ./target/release/mlsys benchmarks/mlsys-2026-17.json solution.json
```

## Building for Submission

```bash
# Build optimized release binary
cargo build --release

# The binary is at:
# ./target/release/mlsys

# For static linking (Ubuntu submission):
RUSTFLAGS='-C target-feature=+crt-static' cargo build --release --target x86_64-unknown-linux-gnu
```

## Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_register_tiling
```

## Development

```bash
# Check for errors without building
cargo check

# Build with warnings
cargo build 2>&1 | head -50

# Format code
cargo fmt

# Lint
cargo clippy
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| RUST_LOG | Logging level | `mlsys=info`, `mlsys=debug`, `mlsys=trace` |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments or file not found |
| Other | Internal error |

## Performance Tips

1. Always use release builds for benchmarking: `cargo build --release`
2. The scheduler is already parallelized with Rayon (auto-detects CPU cores)
3. Telemetry has minimal overhead at INFO level, more at DEBUG/TRACE
4. Large benchmarks (100+ ops) benefit most from parallel processing

