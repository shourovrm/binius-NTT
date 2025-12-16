#!/bin/bash

# ANTT Profiling with Nsight Compute
# Generates detailed performance metrics for optimization analysis

set -e

echo "========================================="
echo "  ANTT Profiling with Nsight Compute"
echo "========================================="
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "ERROR: Nsight Compute (ncu) not found in PATH"
    echo "Please install CUDA Toolkit with Nsight Compute"
    echo "Or run without profiling: ./run_antt_benchmark.sh"
    exit 1
fi

# Navigate to repo root
cd "$(dirname "$0")"

# Build first
echo "Building antt_benchmark..."
./run_antt_benchmark.sh > /dev/null 2>&1 || {
    echo "Build failed. Running build script..."
    ./run_antt_benchmark.sh
}

cd build

echo ""
echo "========================================="
echo "Profiling Original Additive NTT..."
echo "========================================="

# Profile with key metrics
ncu --set full \
    --target-processes all \
    --kernel-name additive_ntt_kernel \
    --launch-skip 0 \
    --launch-count 1 \
    -o antt_profile_original \
    ./antt_benchmark 2>&1 | head -50

echo ""
echo "========================================="
echo "Profiling Modified Additive NTT..."
echo "========================================="

ncu --set full \
    --target-processes all \
    --kernel-name modified_additive_ntt_kernel \
    --launch-skip 0 \
    --launch-count 1 \
    -o antt_profile_modified \
    ./antt_benchmark 2>&1 | head -50

echo ""
echo "========================================="
echo "Profiling Complete!"
echo "========================================="
echo ""
echo "Profile reports saved:"
echo "  - antt_profile_original.ncu-rep"
echo "  - antt_profile_modified.ncu-rep"
echo ""
echo "Open with: ncu-ui antt_profile_original.ncu-rep"
echo ""
echo "Key Metrics to Compare:"
echo "  - Memory Bandwidth Utilization"
echo "  - Compute Throughput"
echo "  - Occupancy"
echo "  - Warp Execution Efficiency"
echo "========================================="
