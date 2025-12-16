# Kaggle Compilation Status & Setup

## Last Updated
December 15, 2025

## Project Overview
This document tracks the Kaggle cloud GPU compilation setup and execution status for the binius-NTT project.

## Kaggle Setup Code

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

## Build & Test Commands

### 1. Clone Repository
```bash
git clone https://github.com/rshourov/binius-NTT.git
cd binius-NTT
git submodule update --init --recursive
```

### 2. Verify CUDA Environment
```bash
nvidia-smi
nvcc --version  # should exist; if not, `apt-get` is not available on Kaggle, so use preinstalled toolchain
```

### 3. Build Project
```bash
rm -rf build
cmake -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++ \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DULVT_ENABLE_NVBench=OFF
cmake --build build -j$(nproc)
```

### 4. Run Tests
```bash
./build/ntt_tests -d yes
./build/finite_field_tests -d yes
./build/sumcheck_test -d yes
```

## Current Status

### Build Status
- **Status:** Not yet executed
- **Last Build:** N/A
- **Build Time:** N/A
- **Warnings/Errors:** N/A

### Test Results

#### NTT Tests
- **Status:** Pending
- **Last Run:** N/A
- **Result:** N/A
- **Notes:** N/A

#### Finite Field Tests
- **Status:** Pending
- **Last Run:** N/A
- **Result:** N/A
- **Notes:** N/A

#### Sumcheck Tests
- **Status:** Pending
- **Last Run:** N/A
- **Result:** N/A
- **Notes:** N/A

## Kaggle Environment Details

### GPU Information
- **GPU Model:** TBD
- **CUDA Version:** TBD
- **Driver Version:** TBD
- **Memory:** TBD

### Compiler Information
- **NVCC Version:** TBD
- **GCC Version:** TBD
- **CMake Version:** TBD

## Performance Metrics

### Compilation Time
- **Total Build Time:** TBD
- **Parallel Jobs:** $(nproc)

### Test Execution Time
- **NTT Tests:** TBD
- **Finite Field Tests:** TBD
- **Sumcheck Tests:** TBD

## Known Issues
- None reported yet

## Optimization Notes
- NVBench is disabled for faster compilation
- Using parallel build with all available CPU cores

## Notes
- Kaggle provides up to 20GB in `/kaggle/working/` directory
- Temporary files can be written to `/kaggle/temp/` but won't persist
- Repository cloned fresh for each run to ensure clean state

## Update History
- **December 15, 2025:** Initial document creation
