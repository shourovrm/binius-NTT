#!/bin/bash

# NTT Implementation Comparison Script
# Validates both original and modified NTT implementations

set -e

echo ""
echo "📊 NTT Implementation Comparison"
echo "================================"
echo ""

# Build if needed
if [ ! -f "build/ntt_tests" ]; then
    echo "Building NTT tests (first run)..."
    ./build_ntt.sh
fi

echo "Running Additive NTT validation tests..."
echo ""

# Run tests with filtering for additive NTT
./build/ntt_tests -d yes "[additive]" 2>&1

echo ""
echo "================================"
echo "Comparison Results:"
echo ""
echo "Implementation:    src/ulvt/ntt/additive_ntt.cuh (ORIGINAL)"
echo "Optimization:      None - Baseline"
echo "Status:            ✓ Validated with test suite"
echo ""
echo "Implementation:    src/ulvt/ntt/modified_antt.cuh (Phase 1)"
echo "Optimization:      Pre-computed twiddle cache"
echo "Status:            Ready for benchmarking"
echo ""
echo "Next: Run ./build/ntt_tests -d yes to see all test results"
echo ""
