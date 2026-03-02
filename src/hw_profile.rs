//! Hardware Profile Auto-Detection
//!
//! Detects machine capabilities (CPU cores, cache sizes, available memory)
//! and derives optimal parallelism parameters for the multi-process scheduler.
//!
//! All downstream modules consume this profile instead of hardcoded constants,
//! ensuring robustness against any benchmark or hardware configuration.

use crate::models::Problem;

/// Machine-level hardware profile detected at startup
#[derive(Debug, Clone)]
pub struct MachineProfile {
    /// Number of physical CPU cores available
    pub num_cores: usize,
    /// Number of worker processes to spawn (cores - 1 for master)
    pub num_workers: usize,
    /// Estimated L2 cache size per core (bytes), 0 if unknown
    pub l2_cache_per_core: usize,
    /// Estimated L3 cache size (bytes), 0 if unknown
    pub l3_cache_total: usize,
    /// Available system RAM (bytes), 0 if unknown
    pub available_ram: usize,
}

/// Problem-aware profile that combines machine + problem characteristics
#[derive(Debug, Clone)]
pub struct WorkerProfile {
    /// Machine-level profile
    pub machine: MachineProfile,
    /// Target tile size (ops) for graph tiling, derived from problem
    pub tile_size_hint: usize,
    /// SRAM budget per memory zone (for zone-based allocation)
    pub sram_budget_per_zone: i64,
    /// Whether multi-process is beneficial for this problem
    pub use_multiprocess: bool,
    /// Maximum shared memory segment size to allocate (bytes)
    pub shm_segment_size: usize,
}

impl MachineProfile {
    /// Auto-detect machine capabilities
    pub fn detect() -> Self {
        let num_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // On macOS, try to get cache sizes via sysctl
        let (l2_cache_per_core, l3_cache_total) = detect_cache_sizes();

        // Conservative RAM estimate (we don't need much for shm)
        let available_ram = detect_available_ram();

        // Workers = cores - 1 (master keeps one), min 1
        let num_workers = if num_cores > 2 {
            num_cores - 1
        } else {
            1
        };

        Self {
            num_cores,
            num_workers,
            l2_cache_per_core,
            l3_cache_total,
            available_ram,
        }
    }

    /// Create a profile with specific values (for testing)
    pub fn with_cores(num_cores: usize) -> Self {
        Self {
            num_cores,
            num_workers: if num_cores > 2 { num_cores - 1 } else { 1 },
            l2_cache_per_core: 256 * 1024,  // 256KB default
            l3_cache_total: 8 * 1024 * 1024, // 8MB default
            available_ram: 8 * 1024 * 1024 * 1024, // 8GB default
        }
    }
}

impl WorkerProfile {
    /// Create a worker profile from machine profile + problem characteristics.
    ///
    /// This is the key function that decides parallelism strategy based on
    /// the ACTUAL problem, not hardcoded thresholds.
    pub fn from_problem(machine: &MachineProfile, problem: &Problem) -> Self {
        let num_ops = problem.ops.len();
        let _num_tensors = problem.tensors.len();
        let sram_capacity = problem.fast_memory_capacity;

        // === Decision: Should we use multi-process? ===
        // Multi-process has overhead (fork, shm setup, result merge).
        // Only beneficial when:
        // 1. Graph is large enough to amortize overhead (>= 500 ops)
        // 2. We have multiple cores available
        // 3. The graph has enough parallelism (not purely linear)
        let min_ops_for_multiprocess = 500;
        let use_multiprocess = num_ops >= min_ops_for_multiprocess
            && machine.num_workers > 1;

        // === Tile size hint ===
        // How many ops per tile? Balance between:
        // - Too small: overhead from inter-tile communication dominates
        // - Too large: poor load balancing, can't distribute evenly
        let tile_size_hint = if !use_multiprocess {
            num_ops // Single process, one tile
        } else {
            // Target: enough tiles for 2x oversubscription (for work stealing)
            let target_tiles = machine.num_workers * 2;
            let raw_tile_size = num_ops / target_tiles;
            // Clamp to reasonable range
            raw_tile_size.clamp(100, 2000)
        };

        // === SRAM budget per zone ===
        // Divide SRAM capacity into zones for parallel allocation
        // Keep 10% as shared overflow reserve
        let num_zones = if use_multiprocess { machine.num_workers } else { 1 };
        let usable_sram = (sram_capacity as f64 * 0.90) as i64;
        let sram_budget_per_zone = usable_sram / num_zones as i64;

        // === Shared memory segment size ===
        // Need enough for: tile descriptors + result buffers + coordination header
        // Each tile descriptor: ~256 bytes
        // Each result: ~(num_ops * 64) bytes for the solution
        // Header: 4KB for atomics and sync
        let descriptor_size = num_ops * 256;
        let result_size = num_ops * 128;
        let header_size = 4096;
        let shm_segment_size = (descriptor_size + result_size + header_size)
            .max(64 * 1024) // At least 64KB
            .min(64 * 1024 * 1024); // At most 64MB

        Self {
            machine: machine.clone(),
            tile_size_hint,
            sram_budget_per_zone,
            use_multiprocess,
            shm_segment_size,
        }
    }
}

/// Detect CPU cache sizes (best-effort, platform-specific)
fn detect_cache_sizes() -> (usize, usize) {
    // Try macOS sysctl
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.l2cachesize"])
            .output()
        {
            let l2 = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<usize>()
                .unwrap_or(256 * 1024);

            let l3 = std::process::Command::new("sysctl")
                .args(["-n", "hw.l3cachesize"])
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<usize>()
                        .ok()
                })
                .unwrap_or(8 * 1024 * 1024);

            return (l2, l3);
        }
    }

    // Try Linux sysfs
    #[cfg(target_os = "linux")]
    {
        let l2 = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
            .ok()
            .and_then(|s| {
                let s = s.trim().trim_end_matches('K');
                s.parse::<usize>().ok().map(|v| v * 1024)
            })
            .unwrap_or(256 * 1024);

        let l3 = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size")
            .ok()
            .and_then(|s| {
                let s = s.trim().trim_end_matches('K').trim_end_matches('M');
                s.parse::<usize>().ok().map(|v| v * 1024)
            })
            .unwrap_or(8 * 1024 * 1024);

        return (l2, l3);
    }

    // Fallback defaults
    (256 * 1024, 8 * 1024 * 1024)
}

/// Detect available RAM (best-effort)
fn detect_available_ram() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            return String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<usize>()
                .unwrap_or(8 * 1024 * 1024 * 1024);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") || line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) {
                        return kb * 1024;
                    }
                }
            }
        }
    }

    // Fallback default
    8 * 1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Granularity, Op, OpType, Problem, Tensor};

    #[test]
    fn test_machine_profile_detect() {
        let profile = MachineProfile::detect();
        assert!(profile.num_cores >= 1);
        assert!(profile.num_workers >= 1);
        eprintln!("Detected: {:?}", profile);
    }

    #[test]
    fn test_worker_profile_small_graph() {
        let machine = MachineProfile::with_cores(8);
        let problem = Problem {
            tensors: vec![Tensor { width: 128, height: 128 }; 10],
            ops: (0..5).map(|i| Op {
                op_type: OpType::Pointwise,
                inputs: vec![i],
                outputs: vec![i + 1],
                base_cost: 100,
            }).collect(),
            fast_memory_capacity: 50000,
            slow_memory_bandwidth: 10,
            native_granularity: Granularity::new(128, 128, 1),
        };
        let wp = WorkerProfile::from_problem(&machine, &problem);
        // Small graph should NOT use multiprocess
        assert!(!wp.use_multiprocess);
    }

    #[test]
    fn test_worker_profile_large_graph() {
        let machine = MachineProfile::with_cores(8);
        let num_ops = 2000;
        let problem = Problem {
            tensors: vec![Tensor { width: 128, height: 128 }; num_ops + 1],
            ops: (0..num_ops).map(|i| Op {
                op_type: OpType::Pointwise,
                inputs: vec![i],
                outputs: vec![i + 1],
                base_cost: 100,
            }).collect(),
            fast_memory_capacity: 500000,
            slow_memory_bandwidth: 100,
            native_granularity: Granularity::new(128, 128, 1),
        };
        let wp = WorkerProfile::from_problem(&machine, &problem);
        // Large graph should use multiprocess
        assert!(wp.use_multiprocess);
        assert!(wp.tile_size_hint >= 100);
        assert!(wp.tile_size_hint <= 2000);
    }
}




