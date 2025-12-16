# Benchmark Implementation Details

## Overview
The benchmark compares **original Additive NTT** vs **modified (optimized) Additive NTT** using the exact test structure from `test_ntt.cu`.

## Test Structure

### Based on `test_ntt.cu`
The benchmark follows the same pattern as the original tests:
- Uses same MD5 hash validation table
- Identical input generation (RNG seed: `0xdeadbeef + log_h + log_rate`)
- Same test range: log_h from 1 to 28
- Same field: `FanPaarTowerField<5>`
- Same log_rate: 0 (for initial testing)

### Key Differences from Original Test
1. **Timing Added:** Measures execution time for both versions
2. **Side-by-Side:** Runs both implementations with identical input
3. **Separate Validation:** Each implementation has its own correctness check
4. **Performance Metrics:** Calculates speedup and improvement percentage

## Benchmark Flow

```
For each log_h from 1 to 28:
  ┌─────────────────────────────────────────┐
  │ 1. Generate Input Data                   │
  │    std::mt19937 gen(0xdeadbeef + ...)   │
  │    NTTData<uint32_t> ntt_inp(...)       │
  └─────────────────────────────────────────┘
                    │
    ┌───────────────┴───────────────┐
    │                               │
    v                               v
┌───────────────────────┐   ┌───────────────────────┐
│ ORIGINAL NTT          │   │ MODIFIED NTT          │
├───────────────────────┤   ├───────────────────────┤
│ - Warmup run          │   │ - Warmup run          │
│ - Timed execution     │   │ - Timed execution     │
│ - MD5 hash calc       │   │ - MD5 hash calc       │
│ - Validate vs table   │   │ - Validate vs table   │
└───────────────────────┘   └───────────────────────┘
            │                           │
            └───────────┬───────────────┘
                        v
            ┌───────────────────────────┐
            │  Compare & Display        │
            │  - Timing comparison      │
            │  - Speedup calculation    │
            │  - Correctness status     │
            └───────────────────────────┘
```

## Output Format

### Per-Test Output
```
log_h    Original (ms)   Modified (ms)   Speedup     Orig.Valid  Mod.Valid
------------------------------------------------------------------------------------
  1      0.12            0.10            1.200x      ✓ PASS      ✓ PASS
  2      0.15            0.12            1.250x      ✓ PASS      ✓ PASS
  ...
  20     1234.56         890.12          1.387x      ✓ PASS      ✓ PASS
  ...
  28     67890.12        49876.54        1.361x      ✓ PASS      ✓ PASS
```

### Summary Output
```
SUMMARY:
  Original tests passed: 28/28
  Modified tests passed: 28/28
  Total original time: 45678.90 ms
  Total modified time: 33456.78 ms
  Average speedup: 1.365x
  Performance improvement: 36.5%
  ✓ OPTIMIZATION SUCCESSFUL!
```

## Validation Method

### MD5 Hash Comparison
Both implementations compute MD5 hash of output:
```cpp
MD5Context md5;
md5Init(&md5);
for (size_t i = 0; i < ntt_out.size; i++) {
    uint32_t d = ntt_out.data[i];
    md5Update(&md5, (uint8_t*)&d, 4);
}
md5Finalize(&md5);
```

Then compare against known-good hashes from `test_ntt.cu`:
```cpp
bool correct = (memcmp(md5.digest, additive_ntt_hashes[log_h], 16) == 0);
```

## Timing Methodology

### Warmup
Each implementation runs once before timing to:
- Allocate GPU memory
- Initialize CUDA kernels
- Prime caches

### Measurement
```cpp
auto start = high_resolution_clock::now();
add_ntt.apply(ntt_inp, ntt_out);
cudaDeviceSynchronize();  // Wait for GPU to finish
auto end = high_resolution_clock::now();

double time_ms = duration_cast<milliseconds>(end - start).count();
```

## Expected Results

### Phase 1 Optimizations Target
- **Speedup:** 1.15x - 1.35x (15-35% improvement)
- **All Tests Pass:** Both original and modified should validate correctly
- **Larger Sizes:** Bigger improvements expected at log_h ≥ 20

### Typical Performance Pattern
```
Small sizes (log_h < 15):  Speedup ~1.10-1.20x (overhead dominates)
Medium sizes (log_h 15-20): Speedup ~1.25-1.35x (sweet spot)
Large sizes (log_h > 20):   Speedup ~1.30-1.40x (memory bottleneck)
```

## How to Interpret Results

### Success Indicators
✓ All tests pass validation  
✓ Speedup > 1.0 consistently  
✓ Larger log_h shows bigger improvement  
✓ No correctness failures  

### Warning Signs
✗ Modified version slower (speedup < 1.0)  
✗ Correctness failures (hash mismatch)  
✗ Speedup decreases with size  
✗ Inconsistent timing (may need multiple runs)  

## Next Steps Based on Results

### If Speedup 1.15x - 1.35x ✓
- Phase 1 successful!
- Proceed to Phase 2 (kernel merging)
- Profile with Nsight Compute

### If Speedup < 1.15x
- Run profiling to identify bottlenecks
- Check texture memory is being used
- Verify vectorized loads are working
- May need Phase 2 earlier than planned

### If Speedup > 1.35x 🎉
- Excellent result!
- Document optimization details
- Consider publishing findings
- Proceed confidently to Phase 2 & 3

## Running the Benchmark

### Quick Run
```bash
./run_antt_benchmark.sh
```

### With Profiling
```bash
./run_antt_benchmark.sh
./profile_antt.sh
```

### Custom Test Range
Edit `benchmark_antt.cu`:
```cpp
constexpr int LOG_H_START = 15;  // Start at log_h=15
constexpr int LOG_H_END = 25;    // End at log_h=25
```

Then rebuild:
```bash
cd build
cmake --build . --target antt_benchmark
./antt_benchmark
```
