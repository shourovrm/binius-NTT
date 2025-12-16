# Quick Trial Run Script for Kaggle Cloud

This is a ready-to-paste notebook for Kaggle cloud with GPU.

## Copy-Paste into Kaggle Notebook Cells

### Prerequisites in Kaggle:
1. Create new notebook at https://www.kaggle.com/code
2. Settings → Accelerator → GPU T4 x2 or P100
3. Settings → Internet → ON

---

### Cell 1: Clone Repository & Check GPU

```python
!nvidia-smi
!nvcc --version

# Clone the repository (if it's public on GitHub)
# Or upload as a dataset and mount it
!cd /kaggle/working && git clone https://github.com/IrreducibleOSS/binius-NTT.git
!cd /kaggle/working/binius-NTT && git submodule update --init --recursive

!ls -la /kaggle/working/binius-NTT/
```

---

### Cell 2: Install CMake 3.27

```python
!wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh -O /tmp/cmake.sh
!chmod +x /tmp/cmake.sh
!sudo /tmp/cmake.sh --skip-license --prefix=/usr/local
!cmake --version
```

---

### Cell 3: Build Project

```python
!cd /kaggle/working/binius-NTT && cmake -B./build -DCMAKE_CUDA_HOST_COMPILER="g++" -DCMAKE_CXX_COMPILER="g++"
!cd /kaggle/working/binius-NTT && cmake --build ./build -j$(nproc)
```

---

### Cell 4: Run NTT Tests (Quick)

```python
# Run quick NTT tests
!cd /kaggle/working/binius-NTT && ./build/ntt_tests "[ntt][additive][0]" -d yes
```

---

### Cell 5: Run Finite Field Tests

```python
!cd /kaggle/working/binius-NTT && ./build/finite_field_tests -d yes
```

---

### Cell 6: Run Sumcheck Test

```python
!cd /kaggle/working/binius-NTT && ./build/sumcheck_test -d yes
```

---

### Cell 7: Run Benchmark (Optional)

```python
# This might take a few minutes
!cd /kaggle/working/binius-NTT && ./build/sumcheck_bench
```

---

## Alternative: If Repository is Private

If you can't clone directly, create a dataset from your local PC:

### On Your Local Ubuntu PC:

```bash
# Install kaggle CLI if not already
pip3 install kaggle

# Setup credentials from https://www.kaggle.com/settings (download kaggle.json)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Navigate to project
cd /home/rms/repos/binius-NTT

# Create dataset metadata
cat > dataset-metadata.json << 'EOF'
{
  "title": "binius-ntt-gpu",
  "id": "YOUR_USERNAME/binius-ntt-gpu",
  "licenses": [{"name": "MIT"}]
}
EOF

# Create zip of the project
tar czf binius-ntt.tar.gz --exclude='.git' --exclude='build' .

# Create dataset
kaggle datasets create -p . -r zip
```

Then in Kaggle notebook:
1. Add Data → Search "binius-ntt-gpu" → Add
2. Use this cell:

```python
!mkdir -p /kaggle/working/binius-NTT
!cd /kaggle/working && tar xzf /kaggle/input/binius-ntt-gpu/binius-ntt.tar.gz -C binius-NTT/
# Then continue with Cell 2 above (install CMake)
```

---

## Expected Output

### NTT Tests:
```
All tests passed (X assertions in Y test cases)
```

### Finite Field Tests:
```
All tests passed
```

### Sumcheck Tests:
```
All tests passed
```

### Benchmark Results:
You'll see timing results for different problem sizes (2^20, 2^24, 2^28).

---

## Troubleshooting

**GPU not detected**: Check notebook settings → Accelerator → Select GPU

**CMake too old**: Make sure Cell 2 completed successfully

**Submodule errors**: Internet must be ON in settings

**Build fails**: Check CUDA version compatibility with `nvcc --version`

**Out of memory**: Reduce test sizes or use tests with smaller inputs
