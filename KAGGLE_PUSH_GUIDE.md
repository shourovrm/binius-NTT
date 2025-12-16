# Step-by-Step Guide: Running Binius-NTT on Kaggle GPU using CLI

## What You Have Setup ✅
- Conda env `cuda` with Kaggle CLI installed
- Kaggle API credentials in `~/.kaggle/kaggle.json`
- Repository ready to push

## How It Works

When you run `kaggle kernels push`, it:
1. **Uploads** all files in the directory to Kaggle (except `.kaggleignore` patterns)
2. **Starts a GPU instance** (T4 or P100) in the cloud
3. **Executes** your `run_trial.py` script on that GPU
4. **Shows output** in real-time or you can check status/output later
5. **Shuts down** automatically when done

---

## Step 1: Push the Kernel to Kaggle

From the binius-NTT directory:

```bash
# Activate your conda environment
conda activate cuda

# Navigate to project
cd /home/rms/repos/binius-NTT

# Push to Kaggle (this uploads and starts execution)
kaggle kernels push
```

**What happens:**
- Uploads ~1.4MB of source code
- Creates/updates kernel "binius-ntt-gpu-trial"
- Starts GPU instance and runs `run_trial.py`

---

## Step 2: Check Status

```bash
# Check if kernel is running or completed
kaggle kernels status riadmashrubshourov/binius-ntt-gpu-trial
```

**Possible statuses:**
- `queued`: Waiting for GPU
- `running`: Currently executing (build + tests)
- `complete`: Finished successfully
- `error`: Failed (check output)

---

## Step 3: View Output

```bash
# See the execution logs and results
kaggle kernels output riadmashrubshourov/binius-ntt-gpu-trial

# Or view in browser
# https://www.kaggle.com/code/riadmashrubshourov/binius-ntt-gpu-trial
```

---

## Step 4: Download Results (if needed)

```bash
# Download any output files the kernel generated
kaggle kernels output riadmashrubshourov/binius-ntt-gpu-trial -p ./kaggle_output/
```

---

## Expected Timeline

1. **Upload**: 10-30 seconds (depending on connection)
2. **Queue**: 0-60 seconds (usually instant for free tier)
3. **CMake Install**: ~30 seconds
4. **Build**: 3-5 minutes
5. **Tests**: 2-5 minutes
6. **Total**: ~10 minutes for first run

---

## Monitoring the Run

### Option 1: Poll Status
```bash
# Keep checking status
watch -n 10 'kaggle kernels status riadmashrubshourov/binius-ntt-gpu-trial'
```

### Option 2: Web Interface
Open in browser: https://www.kaggle.com/code/riadmashrubshourov/binius-ntt-gpu-trial
- See real-time logs
- View GPU utilization
- Download outputs

---

## Updating and Re-running

If you make changes to the code:

```bash
# Make your changes locally
vim src/ulvt/ntt/additive_ntt.cuh

# Push updated version (creates new version)
kaggle kernels push

# Check the new run
kaggle kernels status riadmashrubshourov/binius-ntt-gpu-trial
```

Each push creates a new version and triggers a new run.

---

## Troubleshooting

### "Kernel not found"
The kernel doesn't exist yet. First push will create it.

### "403 Forbidden"
Check your `~/.kaggle/kaggle.json` has correct credentials.

### "Kernel failed" or "error" status
```bash
# View the error logs
kaggle kernels output riadmashrubshourov/binius-ntt-gpu-trial
```

Common issues:
- CMake installation failed (internet setting)
- Submodule init failed (internet setting)
- CUDA version mismatch (unlikely)

### Build takes too long
The first build is slowest. Subsequent runs are faster if you push updates.

### GPU quota exhausted
Free tier: 30 GPU hours/week. Check your quota at https://www.kaggle.com/settings

---

## Advanced: Custom Configurations

### Change GPU Type
Edit `kernel-metadata.json`:
```json
"enable_gpu": "true",    // Keep this
```
(Kaggle assigns GPU automatically, usually T4 or P100)

### Run Specific Tests Only
Edit `run_trial.py` and modify the `tests` list to comment out tests you don't want.

### Add More Output
Modify `run_trial.py` to save results to files, which you can download.

---

## Cost and Limits

- **Free tier**: 30 GPU hours per week
- **Session limit**: 9 hours per run (this project needs ~10 minutes)
- **Parallel runs**: Max 1 GPU kernel at a time on free tier
- **Storage**: Outputs cleared after session, download what you need

---

## Quick Reference Commands

```bash
# Push and run
kaggle kernels push

# Check status
kaggle kernels status riadmashrubshourov/binius-ntt-gpu-trial

# View output
kaggle kernels output riadmashrubshourov/binius-ntt-gpu-trial

# Download output files
kaggle kernels output riadmashrubshourov/binius-ntt-gpu-trial -p ./results/

# List your kernels
kaggle kernels list --mine

# Pull kernel back to local (if you edited on web)
kaggle kernels pull riadmashrubshourov/binius-ntt-gpu-trial
```

---

## Next Steps After Trial Run

Once the trial run succeeds:

1. **Run benchmarks**: Modify `run_trial.py` to run more extensive benchmarks
2. **Test different sizes**: Adjust test parameters in the script
3. **Profile GPU usage**: Add CUDA profiling commands
4. **Export results**: Save benchmark data to CSV for analysis

---

## Ready to Run!

Execute this now:
```bash
conda activate cuda
cd /home/rms/repos/binius-NTT
kaggle kernels push
```

Then monitor at: https://www.kaggle.com/code/riadmashrubshourov/binius-ntt-gpu-trial
