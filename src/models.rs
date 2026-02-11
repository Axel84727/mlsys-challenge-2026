//! Data models mapping the C++ structures from mlsys.h
//!
//! These structures faithfully represent the Problem and Solution types
//! used by the MLSys Challenge evaluator.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ============================================================================
// Type Aliases (matching mlsys.h)
// ============================================================================

pub type BaseCost = i64;
pub type Depth = i64;
pub type FastMemoryCapacity = i64;
pub type Height = i64;
pub type Width = i64;
pub type SlowMemoryBandwidth = i64;
pub type SubgraphLatency = f64;
pub type TotalLatency = f64;
pub type TraversalOrder = Vec<i64>;
pub type TensorId = usize;
pub type OpId = usize;

// ============================================================================
// Core Structures
// ============================================================================

/// Represents a tensor with spatial dimensions.
/// Tensors are the data flowing between operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tensor {
    pub width: Width,
    pub height: Height,
}

impl Tensor {
    /// Calculate the total size of this tensor in memory units
    #[inline]
    pub fn size(&self) -> i64 {
        self.width * self.height
    }

    /// Calculate the size of a tile/slice of this tensor
    #[inline]
    pub fn slice_size(&self, granularity: &Granularity) -> i64 {
        let w = self.width.min(granularity.width);
        let h = self.height.min(granularity.height);
        w * h
    }
}

/// Operation types supported by the system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpType {
    Pointwise,
    MatMul,
}

impl std::str::FromStr for OpType {
    type Err = ();
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "MatMul" => Ok(OpType::MatMul),
            _ => Ok(OpType::Pointwise),
        }
    }
}

impl OpType {
    pub fn parse(s: &str) -> Self {
        s.parse().unwrap_or(OpType::Pointwise)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            OpType::Pointwise => "Pointwise",
            OpType::MatMul => "MatMul",
        }
    }
}

/// Represents a computational operation in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Op {
    pub op_type: OpType,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub base_cost: BaseCost,
}

impl Op {
    /// Check if this operation is fusible with another (Pointwise ops are always fusible)
    #[inline]
    pub fn is_fusible(&self) -> bool {
        matches!(self.op_type, OpType::Pointwise)
    }

    /// Check if this is a MatMul operation (candidate for Split-K)
    #[inline]
    pub fn is_matmul(&self) -> bool {
        matches!(self.op_type, OpType::MatMul)
    }
}

/// Granularity for tiling operations.
/// Controls how operations are split across spatial and reduction dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Granularity {
    pub width: Width,
    pub height: Height,
    #[serde(default = "default_depth")]
    pub depth: Depth,
}

fn default_depth() -> Depth {
    1
}

impl Granularity {
    pub fn new(width: Width, height: Height, depth: Depth) -> Self {
        Self { width, height, depth }
    }

    /// Create from native granularity array [w, h] or [w, h, d]
    pub fn from_array(arr: &[i64]) -> Self {
        Self {
            width: arr.first().copied().unwrap_or(1),
            height: arr.get(1).copied().unwrap_or(1),
            depth: arr.get(2).copied().unwrap_or(1),
        }
    }

    /// Number of spatial tiles needed to cover a tensor
    pub fn num_tiles(&self, tensor: &Tensor) -> i64 {
        let w_tiles = (tensor.width + self.width - 1) / self.width;
        let h_tiles = (tensor.height + self.height - 1) / self.height;
        w_tiles * h_tiles
    }

    /// Reduce granularity by half (for memory pressure)
    pub fn halve(&self) -> Self {
        Self {
            width: (self.width / 2).max(1),
            height: (self.height / 2).max(1),
            depth: self.depth,
        }
    }

    /// Increase depth for Split-K optimization
    pub fn with_split_k(&self, k: Depth) -> Self {
        Self {
            width: self.width,
            height: self.height,
            depth: k,
        }
    }
}

impl Default for Granularity {
    fn default() -> Self {
        Self { width: 128, height: 128, depth: 1 }
    }
}

// ============================================================================
// Problem Definition (Input)
// ============================================================================

/// The complete problem specification read from input JSON.
#[derive(Debug, Clone)]
pub struct Problem {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    pub fast_memory_capacity: FastMemoryCapacity,
    pub slow_memory_bandwidth: SlowMemoryBandwidth,
    pub native_granularity: Granularity,
}

/// JSON format for problem input (matches example_problem.json)
#[derive(Debug, Deserialize)]
pub struct ProblemJson {
    pub widths: Vec<i64>,
    pub heights: Vec<i64>,
    pub op_types: Vec<String>,
    pub inputs: Vec<Vec<usize>>,
    pub outputs: Vec<Vec<usize>>,
    pub base_costs: Vec<i64>,
    pub fast_memory_capacity: i64,
    pub slow_memory_bandwidth: i64,
    pub native_granularity: Vec<i64>,
}

impl From<ProblemJson> for Problem {
    fn from(json: ProblemJson) -> Self {
        let tensors: Vec<Tensor> = json.widths
            .iter()
            .zip(json.heights.iter())
            .map(|(&w, &h)| Tensor { width: w, height: h })
            .collect();

        let ops: Vec<Op> = json.op_types
            .iter()
            .zip(json.inputs.iter())
            .zip(json.outputs.iter())
            .zip(json.base_costs.iter())
            .map(|(((op_type, inputs), outputs), &base_cost)| Op {
                op_type: OpType::parse(op_type),
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                base_cost,
            })
            .collect();

        Problem {
            tensors,
            ops,
            fast_memory_capacity: json.fast_memory_capacity,
            slow_memory_bandwidth: json.slow_memory_bandwidth,
            native_granularity: Granularity::from_array(&json.native_granularity),
        }
    }
}

// ============================================================================
// Solution Definition (Output)
// ============================================================================

/// A subgraph represents a fused group of operations executed together.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Subgraph {
    /// Indices of ops included in this subgraph
    pub ops: Vec<OpId>,
    /// Tensors to keep resident in fast memory after this subgraph completes
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tensors_to_retain: Vec<TensorId>,
    /// Tiling granularity for this subgraph
    pub granularity: GranularityOutput,
    /// Optional traversal order for tile processing (snake/zig-zag pattern)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traversal_order: Option<TraversalOrder>,
    /// Computed latency for this subgraph (internal use)
    #[serde(skip)]
    pub subgraph_latency: SubgraphLatency,
}

/// Granularity output format for JSON
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GranularityOutput {
    pub w: Width,
    pub h: Height,
    #[serde(skip_serializing_if = "is_one")]
    pub k: Option<Depth>,
}

fn is_one(k: &Option<Depth>) -> bool {
    k.is_none_or(|v| v == 1)
}

impl From<&Granularity> for GranularityOutput {
    fn from(g: &Granularity) -> Self {
        Self {
            w: g.width,
            h: g.height,
            k: if g.depth > 1 { Some(g.depth) } else { None },
        }
    }
}

/// The complete solution to be written to output JSON.
#[derive(Debug, Clone, Serialize)]
pub struct Solution {
    pub subgraphs: Vec<SubgraphOutput>,
}

/// Output format for a single subgraph in the solution
#[derive(Debug, Clone, Serialize)]
pub struct SubgraphOutput {
    pub ops: Vec<OpId>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tensors_to_retain: Vec<TensorId>,
    pub granularity: GranularityOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traversal_order: Option<TraversalOrder>,
}

impl From<&Subgraph> for SubgraphOutput {
    fn from(sg: &Subgraph) -> Self {
        Self {
            ops: sg.ops.clone(),
            tensors_to_retain: sg.tensors_to_retain.clone(),
            granularity: sg.granularity.clone(),
            traversal_order: sg.traversal_order.clone(),
        }
    }
}

// ============================================================================
// Graph Analysis Helpers
// ============================================================================

/// Metadata about tensors in the computation graph
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Which op produces this tensor (None if it's an input tensor)
    pub producer: Option<OpId>,
    /// Which ops consume this tensor
    pub consumers: Vec<OpId>,
    /// Is this a graph input (no producer)?
    pub is_input: bool,
    /// Is this a graph output (no consumers or final output)?
    pub is_output: bool,
}

impl Problem {
    /// Build metadata about tensor producers and consumers
    pub fn build_tensor_meta(&self) -> Vec<TensorMeta> {
        let mut meta: Vec<TensorMeta> = self.tensors
            .iter()
            .map(|_| TensorMeta {
                producer: None,
                consumers: Vec::new(),
                is_input: true,
                is_output: true,
            })
            .collect();

        for (op_id, op) in self.ops.iter().enumerate() {
            // Mark outputs as produced by this op
            for &out_id in &op.outputs {
                if out_id < meta.len() {
                    meta[out_id].producer = Some(op_id);
                    meta[out_id].is_input = false;
                }
            }
            // Mark inputs as consumed by this op
            for &in_id in &op.inputs {
                if in_id < meta.len() {
                    meta[in_id].consumers.push(op_id);
                    meta[in_id].is_output = false;
                }
            }
        }

        // Final tensors with no consumers are outputs
        for m in &mut meta {
            if m.consumers.is_empty() && m.producer.is_some() {
                m.is_output = true;
            } else if !m.consumers.is_empty() {
                m.is_output = false;
            }
        }

        meta
    }

    /// Find tensors that are used by multiple ops (diamond pattern candidates)
    pub fn find_shared_tensors(&self) -> HashSet<TensorId> {
        let meta = self.build_tensor_meta();
        meta.iter()
            .enumerate()
            .filter(|(_, m)| m.consumers.len() > 1)
            .map(|(id, _)| id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_size() {
        let t = Tensor { width: 128, height: 128 };
        assert_eq!(t.size(), 16384);
    }

    #[test]
    fn test_granularity_halve() {
        let g = Granularity::new(128, 128, 1);
        let h = g.halve();
        assert_eq!(h.width, 64);
        assert_eq!(h.height, 64);
    }
}

