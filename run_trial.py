#!/usr/bin/env python3
"""
Binius-NTT GPU Trial Run on Kaggle
This script will build and test the project on Kaggle GPU
"""

import subprocess
import sys
import os

def run_cmd(cmd, description):
    """Run command and print output"""
    print(f"\n{'='*60}")
    print(f"📍 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ Success: {description}")
    return True

def main():
    print("🚀 Binius-NTT GPU Build and Test")
    
    # Check GPU
    if not run_cmd("nvidia-smi", "Checking GPU availability"):
        sys.exit(1)
    
    run_cmd("nvcc --version", "CUDA version")
    
    # Install CMake 3.27
    print("\n📦 Installing CMake 3.27...")
    cmds = [
        "cd /tmp",
        "wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh",
        "chmod +x cmake-3.27.0-linux-x86_64.sh",
        "sudo ./cmake-3.27.0-linux-x86_64.sh --skip-license --prefix=/usr/local"
    ]
    if not run_cmd(" && ".join(cmds), "Installing CMake"):
        sys.exit(1)
    
    run_cmd("cmake --version", "CMake version check")
    
    # Navigate to working directory
    work_dir = "/kaggle/working"
    os.chdir(work_dir)
    
    # Initialize git submodules
    print("\n📚 Setting up Git submodules...")
    run_cmd(f"cd {work_dir} && git config --global --add safe.directory {work_dir}", "Git config")
    run_cmd(f"cd {work_dir} && git submodule update --init --recursive", "Initializing submodules")
    
    # Build project
    print("\n🔨 Building project...")
    if not run_cmd(
        f'cd {work_dir} && cmake -B./build -DCMAKE_CUDA_HOST_COMPILER="g++" -DCMAKE_CXX_COMPILER="g++"',
        "CMake configuration"
    ):
        sys.exit(1)
    
    if not run_cmd(f"cd {work_dir} && cmake --build ./build -j$(nproc)", "Building with CMake"):
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running Tests...")
    
    tests = [
        ("./build/ntt_tests '[ntt][additive][0]' -d yes", "NTT Tests (quick)"),
        ("./build/finite_field_tests -d yes", "Finite Field Tests"),
        ("./build/sumcheck_test -d yes", "Sumcheck Tests"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_cmd(f"cd {work_dir} && {cmd}", desc)
        results.append((desc, success))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")
    
    # Optional: Run benchmarks if all tests passed
    if all(success for _, success in results):
        print("\n🏃 Running benchmarks...")
        run_cmd(f"cd {work_dir} && ./build/sumcheck_bench", "Sumcheck Benchmark")
    
    print("\n✨ Trial run complete!")

if __name__ == "__main__":
    main()
