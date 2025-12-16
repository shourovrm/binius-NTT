# Additive NTT Optimization Project

This directory contains the optimized implementation of the Additive NTT for binary tower fields, targeting GPU acceleration.

## Phase 1 Optimizations (Implemented)

### 1. Pre-computed Twiddle Factors
- **Original:** Twiddle factors calculated on-the-fly in each kernel invocation
- **Optimized:** All twiddle factors pre-computed during setup and stored in GPU texture memory
- **Benefit:** Reduces computation and leverages hardware texture cache

### 2. Texture Memory Caching
- **Implementation:** `cudaTextureObject_t` for twiddle factor lookups
- **Benefit:** Automatic caching via GPU texture units, reduced memory latency

### 3. Optimized Memory Access
- **Implementation:** `__ldg()` intrinsic for read-only global memory loads
- **Benefit:** Cached in read-only data cache, better memory throughput

## Files

- **`modified_antt.cuh`** - Optimized Additive NTT implementation
- **`tests/benchmark_antt.cu`** - Benchmark comparing original vs modified
- **`run_antt_benchmark.sh`** - Quick build & run script
- **`profile_antt.sh`** - Nsight Compute profiling script

## Quick Start

### Build and Run Benchmark

```bash
# From repo root
./run_antt_benchmark.sh
```

This will:
1. Configure CMake (if needed)
2. Build only the `antt_benchmark` target (fast!)
3. Run benchmark comparing both versions
4. Display speedup results

### Profile with Nsight Compute

```bash
./profile_antt.sh
```

Requirements:
- CUDA Toolkit with Nsight Compute installed
- `ncu` command in PATH

### Manual Build

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target antt_benchmark
./antt_benchmark
```

## Expected Results

Based on Phase 1 optimizations, expected improvements:
- **Speedup:** 1.15x - 1.35x (15-35% faster)
- **Memory Bandwidth:** Increased by 10-20%
- **Occupancy:** Maintained or improved

## Benchmark Output Format

```
========================================
  Additive NTT Benchmark Comparison
  Original vs Modified (Phase 1 Opts)
========================================

Testing LOG_RATE = 0

   log_h   Original (ms)   Modified (ms)     Speedup      Status
--------------------------------------------------------------
      10           X.XXX           Y.YYY       Z.ZZx        PASS
      11           X.XXX           Y.YYY       Z.ZZx        PASS
      ...
--------------------------------------------------------------
Average Speedup: Z.ZZx
Performance Improvement: XX.X%
```

## Next Steps (Phase 2)

Planned optimizations:
1. **Merge Kernel Launches** - Single persistent kernel for all stages
2. **Reduce Bit-Reversal Overhead** - Fused with NTT stages
3. **Multi-Stream Execution** - Parallel coset processing

## Kaggle Usage

### Push to GitHub
```bash
git add src/ulvt/ntt/modified_antt.cuh \
        src/ulvt/ntt/tests/benchmark_antt.cu \
        CMakeLists.txt \
        run_antt_benchmark.sh \
        profile_antt.sh
git commit -m "Add Phase 1 ANTT optimizations"
git push origin main
```

### Pull in Kaggle Notebook
```python
!git clone https://github.com/YOUR_USERNAME/binius-NTT.git
%cd binius-NTT
!./run_antt_benchmark.sh
```

## Correctness Verification

The benchmark verifies correctness by:
1. Computing MD5 hash of output
2. Comparing against reference hashes from original implementation
3. Both original and modified must produce identical results

## Performance Metrics

Key metrics to monitor:
- **Execution Time** - Wall clock time for NTT computation
- **Memory Bandwidth** - GB/s achieved (target: >80% peak)
- **Occupancy** - Active warps per SM (target: >60%)
- **Cache Hit Rate** - Texture cache effectiveness

## Troubleshooting

### Build Errors
```bash
# Clean build
rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
```

### Runtime Errors
```bash
# Check CUDA setup
nvidia-smi
nvcc --version

# Verify GPU compute capability
deviceQuery  # From CUDA samples
```

### Hash Mismatches
- Indicates incorrect computation
- Check kernel logic modifications
- Verify twiddle precomputation matches original

## Code Structure

### Original Implementation
```cuda
// src/ulvt/ntt/additive_ntt.cuh
template <typename T, typename P>
class AdditiveNTT {
    // On-the-fly twiddle calculation
    T twiddle = calculate_twiddle<T, P>(...);
};
```

### Modified Implementation
```cuda
// src/ulvt/ntt/modified_antt.cuh
template <typename T, typename P>
class ModifiedAdditiveNTT {
    // Pre-computed twiddles
    cudaTextureObject_t twiddle_texture;
    T twiddle = lookup_twiddle_texture<T>(...);
};
```

## References

- Original NTT paper: [Citation needed]
- CUDA Texture Memory: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)
- Nsight Compute: [Profiling Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

## License

Same as parent project (MIT/Apache dual license)

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
