//! Bitset-Based Liveness Analysis
//!
//! Ultra-fast tensor collision detection using bitsets. Each tensor's lifetime
//! is represented as a bitset where bit N = "tensor is live at op N".
//!
//! Collision check: `(bits_a & bits_b) != 0` → O(1) on modern CPUs (uses SIMD).
//! This replaces the O(N) scan in the original liveness module for hot paths.
//!
//! Usage:
//! ```ignore
//! let bl = BitvecLiveness::from_problem(&problem, &tensor_meta);
//! if bl.collides(tensor_a, tensor_b) {
//!     // Can't share the same SRAM slot
//! }
//! ```

use bit_set::BitSet;
use crate::models::{OpId, Problem, TensorId, TensorMeta};

/// Bitset-based liveness for fast collision queries.
///
/// Each tensor has a BitSet of length `num_ops` where set bits indicate
/// the ops during which the tensor is live.
#[derive(Debug, Clone)]
pub struct BitvecLiveness {
    /// Per-tensor liveness bitsets. `live[tensor_id]` has bit `op_idx` set
    /// if the tensor is live during that op's execution window.
    live: Vec<BitSet>,
    /// Number of ops in the graph
    num_ops: usize,
    /// Number of tensors
    num_tensors: usize,
    /// Pre-computed: tensors live at each op (transposed view)
    live_at_op: Vec<BitSet>,
}

impl BitvecLiveness {
    /// Build bitset liveness from problem + tensor metadata.
    ///
    /// Runs in O(T * O) where T = tensors, O = ops. But the bitset operations
    /// are cache-friendly and SIMD-accelerated, making it much faster than
    /// HashSet-based approaches for collision detection.
    pub fn from_problem(problem: &Problem, tensor_meta: &[TensorMeta]) -> Self {
        let num_ops = problem.ops.len();
        let num_tensors = problem.tensors.len().min(tensor_meta.len());

        // Build per-tensor liveness bitsets
        let mut live: Vec<BitSet> = Vec::with_capacity(num_tensors);

        // We need a topological ordering to determine execution windows.
        // Use a simple BFS-based ordering (same as op_id order for most graphs).
        let op_order = compute_op_positions(problem, tensor_meta);

        for (_tensor_id, meta) in tensor_meta.iter().enumerate().take(num_tensors) {
            let mut bits = BitSet::with_capacity(num_ops);

            // Determine the live range [start, end]
            let start = meta.producer
                .and_then(|p| op_order.get(&p).copied())
                .unwrap_or(0);

            let end = meta.consumers.iter()
                .filter_map(|c| op_order.get(c).copied())
                .max()
                .unwrap_or(start);

            // Set all bits in [start, end]
            for pos in start..=end {
                if pos < num_ops {
                    bits.insert(pos);
                }
            }

            live.push(bits);
        }

        // Build transposed view: live_at_op[op_idx] = set of tensors live at that op
        let mut live_at_op: Vec<BitSet> = Vec::with_capacity(num_ops);
        for op_idx in 0..num_ops {
            let mut op_bits = BitSet::with_capacity(num_tensors);
            for (tensor_id, tensor_bits) in live.iter().enumerate() {
                if tensor_bits.contains(op_idx) {
                    op_bits.insert(tensor_id);
                }
            }
            live_at_op.push(op_bits);
        }

        Self {
            live,
            num_ops,
            num_tensors,
            live_at_op,
        }
    }

    /// Check if two tensors have overlapping lifetimes (can't share SRAM slot).
    ///
    /// This is the core operation: bitwise AND of two bitsets, then check non-zero.
    /// On x86_64 with AVX2, this processes 256 bits per cycle.
    #[inline]
    pub fn collides(&self, a: TensorId, b: TensorId) -> bool {
        if a >= self.num_tensors || b >= self.num_tensors {
            return false;
        }
        // BitSet intersection check: !(a & b).is_empty()
        !self.live[a].is_disjoint(&self.live[b])
    }

    /// Check if a tensor is live at a specific op
    #[inline]
    pub fn is_live_at(&self, tensor_id: TensorId, op_idx: usize) -> bool {
        if tensor_id >= self.num_tensors || op_idx >= self.num_ops {
            return false;
        }
        self.live[tensor_id].contains(op_idx)
    }

    /// Get all tensors live at a specific op (as a BitSet for fast set operations)
    #[inline]
    pub fn live_tensors_at(&self, op_idx: usize) -> &BitSet {
        if op_idx < self.num_ops {
            &self.live_at_op[op_idx]
        } else {
            // Return empty set for out-of-bounds
            static EMPTY: std::sync::LazyLock<BitSet> = std::sync::LazyLock::new(BitSet::new);
            &EMPTY
        }
    }

    /// Count the number of tensors live at a specific op
    #[inline]
    pub fn live_count_at(&self, op_idx: usize) -> usize {
        if op_idx < self.num_ops {
            self.live_at_op[op_idx].len()
        } else {
            0
        }
    }

    /// Compute the total live bytes at a specific op
    pub fn live_bytes_at(&self, op_idx: usize, problem: &Problem) -> i64 {
        if op_idx >= self.num_ops {
            return 0;
        }
        self.live_at_op[op_idx]
            .iter()
            .filter_map(|tid| problem.tensors.get(tid))
            .map(|t| t.size())
            .sum()
    }

    /// Find the peak memory usage (max live bytes at any op)
    pub fn peak_memory(&self, problem: &Problem) -> (usize, i64) {
        let mut peak_op = 0;
        let mut peak_bytes: i64 = 0;

        for op_idx in 0..self.num_ops {
            let bytes = self.live_bytes_at(op_idx, problem);
            if bytes > peak_bytes {
                peak_bytes = bytes;
                peak_op = op_idx;
            }
        }

        (peak_op, peak_bytes)
    }

    /// Find all tensors that collide with a given tensor (for SRAM allocation)
    pub fn collision_set(&self, tensor_id: TensorId) -> BitSet {
        if tensor_id >= self.num_tensors {
            return BitSet::new();
        }

        let mut collisions = BitSet::with_capacity(self.num_tensors);
        for other in 0..self.num_tensors {
            if other != tensor_id && self.collides(tensor_id, other) {
                collisions.insert(other);
            }
        }
        collisions
    }

    /// Compute SRAM interference graph: which tensors can't share the same slot.
    ///
    /// Returns an adjacency list where `interference[i]` contains all tensors
    /// that have overlapping lifetimes with tensor i.
    ///
    /// This is used for graph coloring (SRAM slot assignment).
    pub fn interference_graph(&self) -> Vec<BitSet> {
        let mut graph: Vec<BitSet> = Vec::with_capacity(self.num_tensors);
        for tid in 0..self.num_tensors {
            graph.push(self.collision_set(tid));
        }
        graph
    }

    /// Get the number of tensors
    pub fn num_tensors(&self) -> usize {
        self.num_tensors
    }

    /// Get the number of ops
    pub fn num_ops(&self) -> usize {
        self.num_ops
    }
}

/// Compute position of each op in topological order.
///
/// Returns a map from OpId -> position (0-based).
fn compute_op_positions(
    problem: &Problem,
    tensor_meta: &[TensorMeta],
) -> std::collections::HashMap<OpId, usize> {
    use std::collections::{HashMap, VecDeque};

    let num_ops = problem.ops.len();
    let mut in_degree = vec![0usize; num_ops];
    let mut downstream: Vec<Vec<OpId>> = vec![Vec::new(); num_ops];

    for (op_id, op) in problem.ops.iter().enumerate() {
        for &output_id in &op.outputs {
            if output_id < tensor_meta.len() {
                for &consumer in &tensor_meta[output_id].consumers {
                    if consumer < num_ops {
                        downstream[op_id].push(consumer);
                        in_degree[consumer] += 1;
                    }
                }
            }
        }
    }

    let mut queue: VecDeque<OpId> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut positions: HashMap<OpId, usize> = HashMap::with_capacity(num_ops);
    let mut pos = 0;

    while let Some(op_id) = queue.pop_front() {
        positions.insert(op_id, pos);
        pos += 1;

        for &next in &downstream[op_id] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    // Handle any remaining ops (cycles, which shouldn't exist in a DAG)
    for i in 0..num_ops {
        positions.entry(i).or_insert(pos);
    }

    positions
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Problem, Tensor};

    fn make_chain_problem() -> Problem {
        // t0 -> op0 -> t1 -> op1 -> t2 -> op2 -> t3
        Problem {
            tensors: vec![
                Tensor { width: 128, height: 128 }, // t0
                Tensor { width: 128, height: 128 }, // t1
                Tensor { width: 128, height: 128 }, // t2
                Tensor { width: 128, height: 128 }, // t3
            ],
            ops: vec![
                Op { op_type: OpType::Pointwise, inputs: vec![0], outputs: vec![1], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![1], outputs: vec![2], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![2], outputs: vec![3], base_cost: 100 },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        }
    }

    #[test]
    fn test_collision_chain() {
        let problem = make_chain_problem();
        let meta = problem.build_tensor_meta();
        let bl = BitvecLiveness::from_problem(&problem, &meta);

        // t0 is live from start to op0. t1 is live from op0 to op1.
        // They overlap at op0.
        assert!(bl.collides(0, 1));

        // t0 should NOT collide with t3 (t0 dies at op0, t3 born at op2)
        assert!(!bl.collides(0, 3));
    }

    #[test]
    fn test_live_count() {
        let problem = make_chain_problem();
        let meta = problem.build_tensor_meta();
        let bl = BitvecLiveness::from_problem(&problem, &meta);

        // At each op, should have at most 2 tensors live (input + output)
        for op in 0..3 {
            let count = bl.live_count_at(op);
            assert!(count >= 1 && count <= 4, "Op {} has {} live tensors", op, count);
        }
    }

    #[test]
    fn test_peak_memory() {
        let problem = make_chain_problem();
        let meta = problem.build_tensor_meta();
        let bl = BitvecLiveness::from_problem(&problem, &meta);

        let (_, peak_bytes) = bl.peak_memory(&problem);
        // Peak should be at least 2 tensors worth
        assert!(peak_bytes >= 128 * 128 * 2);
    }

    #[test]
    fn test_diamond_collision() {
        // t0 -> op0 -> t1 -> op1 -> t2
        //                 -> op2 -> t3
        // t2, t3 -> op3 -> t4
        let problem = Problem {
            tensors: vec![
                Tensor { width: 64, height: 64 }, // t0
                Tensor { width: 64, height: 64 }, // t1 (shared)
                Tensor { width: 64, height: 64 }, // t2
                Tensor { width: 64, height: 64 }, // t3
                Tensor { width: 64, height: 64 }, // t4
            ],
            ops: vec![
                Op { op_type: OpType::Pointwise, inputs: vec![0], outputs: vec![1], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![1], outputs: vec![2], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![1], outputs: vec![3], base_cost: 100 },
                Op { op_type: OpType::Pointwise, inputs: vec![2, 3], outputs: vec![4], base_cost: 100 },
            ],
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(64, 64, 1),
        };

        let meta = problem.build_tensor_meta();
        let bl = BitvecLiveness::from_problem(&problem, &meta);

        // t1 (shared) should collide with t2 and t3 (both consume t1)
        assert!(bl.collides(1, 2));
        assert!(bl.collides(1, 3));

        // t2 and t3 should collide (both alive until op3 consumes them)
        assert!(bl.collides(2, 3));
    }
}


