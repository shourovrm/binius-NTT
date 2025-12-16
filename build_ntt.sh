#!/bin/bash

# Simple NTT build script
# Builds only the NTT tests for validation

set -e

echo "🔨 Building NTT tests..."
echo ""

# Ensure Catch2 is available
if [ ! -f "third-party/Catch2/CMakeLists.txt" ]; then
    echo "📥 Downloading Catch2 v3.4.0..."
    mkdir -p third-party
    git clone --depth 1 --branch v3.4.0 \
        https://github.com/catchorg/Catch2.git \
        third-party/Catch2 || {
        echo "❌ Failed to download Catch2"
        exit 1
    }
    echo "✓ Catch2 ready"
fi

# Detect CUDA path (try multiple locations)
CUDA_PATH=""
if [ -f "/usr/local/cuda/bin/nvcc" ]; then
    CUDA_PATH="/usr/local/cuda"
elif [ -f "/opt/cuda/bin/nvcc" ]; then
    CUDA_PATH="/opt/cuda"
elif command -v nvcc &> /dev/null; then
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
else
    echo "❌ CUDA not found. Install CUDA Toolkit or set it in /usr/local/cuda"
    exit 1
fi

echo "✓ CUDA detected at: $CUDA_PATH"
echo ""

# Clean and build
echo "Configuring CMake..."
rm -rf build
cmake -B build \
    -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCUDAToolkit_ROOT=$CUDA_PATH \
    -DULVT_ENABLE_NVBench=OFF \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "^--|^CMAKE|Error" || true

echo "Building..."
cmake --build build -j$(nproc) 2>&1 | tail -5

echo ""
echo "✅ Build complete!"
echo ""
echo "Run tests with:"
echo "  ./build/ntt_tests -d yes"
echo ""
echo "Run specific test:"
echo "  ./build/ntt_tests -d yes \"[additive]\""
echo ""
