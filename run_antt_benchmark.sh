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

# Check/initialize Catch2 if empty
echo "Checking dependencies..."
if [ ! -f "third-party/Catch2/CMakeLists.txt" ]; then
    echo "Catch2 not found. Downloading..."
    if command -v git &> /dev/null; then
        # Try to clone minimal Catch2
        rm -rf third-party/Catch2
        git clone --depth 1 --branch v3.4.0 https://github.com/catchorg/Catch2.git third-party/Catch2 2>/dev/null || {
            echo "WARNING: Could not download Catch2"
            echo "You may need to initialize submodules manually:"
            echo "  git submodule update --init third-party/Catch2"
            exit 1
        }
    else
        echo "ERROR: git not found and Catch2 is missing"
        exit 1
    fi
    echo "Catch2 initialized."
    echo ""
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

cd build

# Configure CMake (only once or when CMakeLists changes)
if [ ! -f "CMakeCache.txt" ]; then
    echo "Configuring CMake..."
    
    # Detect CUDA toolkit path
    CUDA_PATH="/usr/local/cuda"
    if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
        if command -v nvcc &> /dev/null; then
            CUDA_PATH=$(dirname $(dirname $(which nvcc)))
        fi
    fi
    
    cmake \
        -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
        -DCMAKE_CUDA_HOST_COMPILER=g++ \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCUDAToolkit_ROOT=$CUDA_PATH \
        -DULVT_ENABLE_NVBench=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        ..
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
