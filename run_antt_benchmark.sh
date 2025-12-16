#!/bin/bash

# Quick ANTT Benchmark Build & Run Script
# This script builds only the ANTT benchmark target (not all tests)

set -e  # Exit on error

echo "========================================="
echo "  ANTT Benchmark Build & Run"
echo "  Phase 1 Optimizations"
echo "========================================="
echo ""

# Navigate to repo root
cd "$(dirname "$0")"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

cd build

# Configure CMake (only once or when CMakeLists changes)
if [ ! -f "CMakeCache.txt" ]; then
    echo "Configuring CMake..."
    cmake -DCMAKE_CUDA_HOST_COMPILER="g++" -DCMAKE_CXX_COMPILER="g++" -DCMAKE_BUILD_TYPE=Release ..
    echo ""
fi

# Build only the antt_benchmark target (fast!)
echo "Building antt_benchmark target..."
cmake --build . --target antt_benchmark -- -j$(nproc)
echo ""

# Run the benchmark
echo "Running ANTT Benchmark..."
echo "========================================="
./antt_benchmark

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
