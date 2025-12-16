#!/bin/bash
# Kaggle Setup Script for Binius-NTT
# Run this in Kaggle Code console after enabling GPU

set -e

echo "=== Binius-NTT Kaggle Setup ==="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Check CUDA version
echo ""
echo "CUDA Version:"
nvcc --version

# Install newer CMake (Kaggle usually has old version)
echo ""
echo "Installing CMake 3.27..."
cd /tmp
wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh
chmod +x cmake-3.27.0-linux-x86_64.sh
./cmake-3.27.0-linux-x86_64.sh --skip-license --prefix=/usr/local
rm cmake-3.27.0-linux-x86_64.sh

# Verify CMake
echo ""
echo "CMake version:"
cmake --version

# Clone repository (or use uploaded code)
echo ""
echo "Cloning repository..."
cd /kaggle/working
if [ ! -d "binius-NTT" ]; then
    # Option 1: Clone from your fork/repo if you have one
    # git clone https://github.com/YOUR_USERNAME/binius-NTT.git
    
    # Option 2: User should upload as dataset or copy files manually
    echo "NOTE: You need to upload the binius-NTT code as a Kaggle dataset"
    echo "Or manually copy files to /kaggle/working/binius-NTT/"
    echo ""
    echo "For now, creating placeholder directory..."
    mkdir -p binius-NTT
fi

cd binius-NTT

# Initialize submodules if needed
if [ -d ".git" ]; then
    echo ""
    echo "Initializing submodules..."
    git submodule update --init --recursive
fi

# Build
echo ""
echo "Building project..."
cmake -B./build -DCMAKE_CUDA_HOST_COMPILER="g++" -DCMAKE_CXX_COMPILER="g++"
cmake --build ./build -j$(nproc)

echo ""
echo "=== Build Complete! ==="
echo ""
echo "Run tests with:"
echo "  ./build/ntt_tests"
echo "  ./build/finite_field_tests"
echo "  ./build/sumcheck_test"
echo ""
echo "Run benchmarks with:"
echo "  ./build/sumcheck_bench"
echo "  ./build/gpu-benchmarks"
