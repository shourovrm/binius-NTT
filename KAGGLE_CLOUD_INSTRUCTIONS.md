# Running Binius-NTT on Kaggle Cloud (with GPU)

## Important Clarification

The Kaggle CLI on your local Ubuntu PC **does NOT provide GPU access**. It's only for:
- Downloading datasets
- Uploading datasets  
- Managing competitions
- Submitting to competitions

To actually **run GPU code**, you must use **Kaggle Notebooks** in the cloud.

---

## Step-by-Step Guide to Run on Kaggle Cloud

### Step 1: Upload This Project as a Kaggle Dataset

From your local Ubuntu PC with Kaggle CLI:

```bash
cd /home/rms/repos/binius-NTT

# Create a dataset metadata file
cat > dataset-metadata.json << 'EOF'
{
  "title": "binius-ntt-source",
  "id": "YOUR_KAGGLE_USERNAME/binius-ntt-source",
  "licenses": [{"name": "MIT"}]
}
EOF

# Create the dataset (first time)
kaggle datasets create -p . -r zip

# OR update if already exists
# kaggle datasets version -p . -m "Updated source code" -r zip
```

**Note**: Replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username.

---

### Step 2: Create a Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. In the notebook settings (right panel):
   - **Accelerator**: Select **"GPU T4 x2"** or **"GPU P100"** (if available)
   - **Internet**: Turn ON (to install CMake)
   - **Add Data**: Click and search for your dataset `binius-ntt-source`

---

### Step 3: Run Setup in Kaggle Notebook

In the first cell of your Kaggle notebook, run:

```python
# Cell 1: Extract and setup
import os
import subprocess

# The dataset is mounted at /kaggle/input/binius-ntt-source/
# Copy to working directory (which is writable)
!cp -r /kaggle/input/binius-ntt-source/* /kaggle/working/
!cd /kaggle/working && unzip -q '*.zip' 2>/dev/null || echo "No zip to extract"

# List what we have
!ls -la /kaggle/working/

# Check GPU
!nvidia-smi
```

---

### Step 4: Install Dependencies

```python
# Cell 2: Install CMake 3.27
!cd /tmp && wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh
!cd /tmp && chmod +x cmake-3.27.0-linux-x86_64.sh
!cd /tmp && ./cmake-3.27.0-linux-x86_64.sh --skip-license --prefix=/usr/local
!cmake --version
!nvcc --version
```

---

### Step 5: Initialize Submodules

```python
# Cell 3: Setup git submodules
!cd /kaggle/working && git config --global --add safe.directory /kaggle/working
!cd /kaggle/working && git submodule update --init --recursive
```

---

### Step 6: Build the Project

```python
# Cell 4: Build with CMake
!cd /kaggle/working && cmake -B./build -DCMAKE_CUDA_HOST_COMPILER="g++" -DCMAKE_CXX_COMPILER="g++"
!cd /kaggle/working && cmake --build ./build -j$(nproc)
```

This will take 2-5 minutes on first build.

---

### Step 7: Run Tests (Trial Run)

```python
# Cell 5: Run NTT tests
!cd /kaggle/working && ./build/ntt_tests
```

```python
# Cell 6: Run Finite Field tests  
!cd /kaggle/working && ./build/finite_field_tests
```

```python
# Cell 7: Run Sumcheck tests
!cd /kaggle/working && ./build/sumcheck_test
```

---

### Step 8: Run Benchmarks (Optional)

```python
# Cell 8: Run sumcheck benchmark
!cd /kaggle/working && ./build/sumcheck_bench
```

```python
# Cell 9: Run GPU benchmarks
!cd /kaggle/working && ./build/gpu-benchmarks
```

---

## Alternative: Use Kaggle's Code Editor Directly

Instead of uploading as dataset, you can:

1. Go to https://www.kaggle.com/code
2. Create **New Notebook**
3. Enable **GPU** in settings
4. Use **Add Utility Script** to upload files directly
5. Or clone from GitHub if you push this code there

---

## Troubleshooting

### "kaggle: command not found" on local PC
```bash
pip install kaggle
# Or
pip3 install kaggle
```

### "Forbidden" when creating dataset
Make sure you've set up Kaggle API credentials:
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Submodule initialization fails
If git submodules don't work, you can manually download dependencies:
- Catch2: https://github.com/catchorg/Catch2
- nvbench: https://github.com/NVIDIA/nvbench

Place them in `third-party/` directory.

### Build fails with CUDA errors
Check that GPU is enabled in notebook settings (right panel).

---

## Quick Start Summary

**Fastest way to trial run:**

1. Upload code as Kaggle dataset (from local PC)
2. Create Kaggle notebook with GPU enabled
3. Add your dataset to the notebook
4. Copy code to /kaggle/working/
5. Install CMake 3.27
6. Build with cmake
7. Run `./build/ntt_tests`

**Estimated time:**
- Setup: 2 minutes
- Build: 3-5 minutes  
- Tests: 5-10 minutes
- **Total: ~15 minutes for first run**

---

## GPU Quota

- **Free tier**: 30 GPU hours per week
- **Session limit**: 9 hours per session
- **Idle timeout**: ~60 minutes

Save your results before session expires!
