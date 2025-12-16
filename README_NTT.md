# NTT Optimization Testing Guide

## Quick Start

### 1. Clone and Setup
```bash
# On GitHub
git clone https://github.com/shourovrm/binius-NTT.git
cd binius-NTT

# Or in Kaggle
!git clone https://github.com/shourovrm/binius-NTT.git
%cd binius-NTT
```

### 2. Build
```bash
# Make scripts executable
chmod +x build_ntt.sh compare_ntt.sh

# Build NTT tests
./build_ntt.sh
```

This will:
- Auto-detect CUDA path (/usr/local/cuda or `which nvcc`)
- Download Catch2 if needed
- Build only NTT tests (fast)

### 3. Run Tests
```bash
./build/ntt_tests -d yes
```

Output shows:
- Original additive_ntt.cuh tests (Additive NTT r 0, r 2)
- Pass/Fail for each test case (log_h 1-28)

### 4. Compare Implementations
```bash
./compare_ntt.sh
```

## Manual Commands

If scripts fail, use these direct commands:

**Build:**
```bash
rm -rf build
cmake -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++ \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DULVT_ENABLE_NVBench=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

**Run:**
```bash
./build/ntt_tests -d yes              # Full test suite
./build/ntt_tests -d yes "[additive]" # Just additive NTT tests
```

## Implementation Details

- **Original**: `src/ulvt/ntt/additive_ntt.cuh` - Working baseline
- **Modified**: `src/ulvt/ntt/modified_antt.cuh` - Phase 1 optimizations
- **Tests**: `src/ulvt/ntt/tests/test_ntt.cu` - Validation with MD5 hashes
- **CMakeLists**: Configured to build both, test runs original

## Kaggle Environment

In Kaggle notebook:
```python
!git clone https://github.com/shourovrm/binius-NTT.git
%cd binius-NTT
!chmod +x build_ntt.sh
!./build_ntt.sh
```

Then:
```bash
!./build/ntt_tests -d yes
```

## Troubleshooting

**CUDA not found:**
- Check: `which nvcc`
- Or install: CUDA Toolkit to `/usr/local/cuda`

**Catch2 download fails:**
- Manual: `git clone --depth 1 --branch v3.4.0 https://github.com/catchorg/Catch2.git third-party/Catch2`

**CMake not found:**
- Install: `sudo apt-get install cmake`

## Files Modified/Created

- `build_ntt.sh` - Build script with auto-detection
- `compare_ntt.sh` - Comparison script
- `CMakeLists.txt` - Original config (unchanged)
- `src/ulvt/ntt/modified_antt.cuh` - Phase 1 optimizations (new)
