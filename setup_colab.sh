#!/bin/bash
# Google Colab Setup Script for Binius-NTT
# Run this in Colab notebook cell with: !bash setup_colab.sh

set -e

echo "=== Binius-NTT Google Colab Setup ==="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Check CUDA version
echo ""
echo "CUDA Version:"
nvcc --version

# Install newer CMake
echo ""
echo "Installing CMake 3.27..."
cd /tmp
wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh
chmod +x cmake-3.27.0-linux-x86_64.sh
./cmake-3.27.0-linux-x86_64.sh --skip-license --prefix=/usr/local
rm cmake-3.27.0-linux-x86_64.sh

# Update PATH
export PATH=/usr/local/bin:$PATH

# Verify CMake
echo ""
echo "CMake version:"
cmake --version

# Clone repository
echo ""
echo "Cloning repository..."
cd /content
if [ ! -d "binius-NTT" ]; then
    # Replace with your repository URL
    echo "NOTE: Update this script with your repository URL"
    echo "For now, you need to manually upload files to /content/binius-NTT/"
    mkdir -p binius-NTT
fi

cd binius-NTT

# Initialize submodules if git repo
if [ -d ".git" ]; then
    echo ""
    echo "Initializing submodules..."
    git submodule update --init --recursive
fi

# Build
echo ""
echo "Building project..."
cmake -B./build -DCMAKE_CUDA_HOST_COMPILER="g++" -DCMAKE_CXX_COMPILER="g++"
cmake --build ./build -j2  # Colab has limited CPU cores

echo ""
echo "=== Build Complete! ==="
echo ""
echo "Run tests with:"
echo "  !./build/ntt_tests"
echo "  !./build/finite_field_tests"
echo "  !./build/sumcheck_test"
echo ""
echo "Run benchmarks with:"
echo "  !./build/sumcheck_bench"
echo "  !./build/gpu-benchmarks"
echo ""
echo "Session will timeout after 12 hours - save results to Google Drive!"
