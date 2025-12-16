# Quick Start Guide - Phase 1 ANTT Optimization

## 🚀 Fast Track (In Kaggle)

```bash
# 1. Clone/Update repo
git clone https://github.com/shourovrm/binius-NTT.git
cd binius-NTT

# 2. Run benchmark (builds + tests)
./run_antt_benchmark.sh

# 3. (Optional) Profile
./profile_antt.sh
```

That's it! Results will show if optimization worked.

---

## 📊 What the Benchmark Tests

- **Original NTT:** Baseline Additive NTT from `additive_ntt.cuh`
- **Modified NTT:** Optimized version from `modified_antt.cuh`
- **Test Range:** log_h = 1 to 28 (268 million elements max)
- **Validation:** MD5 hash comparison against known-good values

---

## ✅ Success Criteria

**Good Result:**
```
Average speedup: 1.25x
Performance improvement: 25.0%
✓ OPTIMIZATION SUCCESSFUL!
```

**Great Result:**
```
Average speedup: 1.35x
Performance improvement: 35.0%
✓ OPTIMIZATION SUCCESSFUL!
```

---

## 🔍 Understanding Output

### Per-Test Line
```
log_h  Original(ms)  Modified(ms)  Speedup   Orig.Valid  Mod.Valid
  20     1234.56       890.12      1.387x    ✓ PASS     ✓ PASS
```

- **log_h:** Problem size (2^log_h elements)
- **Original(ms):** Baseline execution time
- **Modified(ms):** Optimized execution time
- **Speedup:** How much faster (>1.0 is good!)
- **Orig.Valid/Mod.Valid:** Correctness check

### Summary
```
SUMMARY:
  Original tests passed: 28/28      ← Should be 28/28
  Modified tests passed: 28/28      ← Should be 28/28
  Average speedup: 1.365x           ← Target: 1.15-1.35x
  Performance improvement: 36.5%    ← Target: 15-35%
```

---

## 🛠️ Phase 1 Optimizations

What's in `modified_antt.cuh`:

1. **Pre-computed Twiddle Factors**
   - Calculated once during setup
   - Stored in texture memory
   - Eliminates on-the-fly computation

2. **Texture Memory Caching**
   - `cudaTextureObject_t` for twiddle lookups
   - Hardware-accelerated caching
   - Better memory latency

3. **Vectorized Loads**
   - `__ldg()` intrinsics for read-only data
   - Improved L1 cache utilization
   - Reduced global memory traffic

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/ulvt/ntt/modified_antt.cuh` | Optimized implementation |
| `src/ulvt/ntt/tests/benchmark_antt.cu` | Comparison benchmark |
| `run_antt_benchmark.sh` | Quick build + run |
| `profile_antt.sh` | Nsight profiling |
| `BENCHMARK_DETAILS.md` | How benchmark works |
| `ANTT_OPTIMIZATION_README.md` | Full documentation |

---

## 🐛 Troubleshooting

### Compilation Fails
```bash
# Check CUDA is available
nvcc --version

# Rebuild from scratch
rm -rf build && mkdir build && cd build
cmake .. && cmake --build . --target antt_benchmark
```

### Wrong Results (✗ FAIL)
- Check you pulled latest code
- Verify CUDA version compatibility
- Check GPU has enough memory

### Slowdown (Speedup < 1.0)
- Something went wrong with optimization
- Run profiler to investigate: `./profile_antt.sh`
- Check GPU utilization

---

## 📈 Next Steps After Success

### If Speedup 1.15x - 1.35x ✓
1. Document exact speedup achieved
2. Run profiler to understand bottlenecks
3. Proceed to Phase 2 (kernel merging)
4. Consider Phase 3 (tensor cores)

### If Speedup < 1.15x
1. Run `./profile_antt.sh` to identify issues
2. Check texture memory is being used
3. Verify vectorized loads are working
4. May need to adjust optimization strategy

### If Speedup > 1.35x 🎉
1. Excellent! Document your setup
2. Consider writing up findings
3. Proceed confidently to advanced optimizations
4. Share results with community

---

## 💡 Pro Tips

- **Larger is better:** Speedup typically increases with log_h
- **Warm GPU:** First run might be slower (cache cold)
- **Multiple runs:** Run 2-3 times to verify consistency
- **Profile early:** Use `profile_antt.sh` to understand why

---

## 📖 Full Documentation

For detailed information, see:

- [BENCHMARK_DETAILS.md](BENCHMARK_DETAILS.md) - Test methodology
- [BENCHMARK_UPDATE.md](BENCHMARK_UPDATE.md) - What changed
- [ANTT_OPTIMIZATION_README.md](ANTT_OPTIMIZATION_README.md) - Complete guide
- [GIT_COMMIT_CHECKLIST.md](GIT_COMMIT_CHECKLIST.md) - Git workflow

---

**Questions?** Check the documentation files or review the benchmark output carefully.

**Ready?** Run `./run_antt_benchmark.sh` and see the results! 🚀
