# NTT Implementation Details

## Two NTT Implementations

This project contains **two distinct NTT implementations** that serve different purposes:

### 1. Additive NTT (`additive_ntt.cuh`) - **PRIMARY FOR BINARY FIELDS**

**Purpose**: Specialized for binary tower fields (characteristic 2) used in Binius prover

**Algorithm Details**:
- **Butterfly Operation** (additive): `u = u + w*v; v = u + v`
  - Uses field addition (XOR in binary fields) instead of subtraction
  - Twiddle factor multiplication with additive structure
  
- **Subspace Polynomial Evaluation**: `f(x) = x² + c*x` where c is a constant
  - Maps elements via subspace polynomials specific to binary fields
  - Precomputes subspace evaluations for all stages

- **Twiddle Factor Calculation**: 
  - Computed dynamically from precomputed constants
  - Based on binary decomposition and coset indices
  - Formula: `sum of constants[stage][k]` where k bits are set in `(coset << offset) | butterfly_block`

- **Multi-rate Support**: Supports log_rate parameter (0, 1, 2) for different expansion rates
  - Transforms input of size 2^log_h to output of size 2^(log_h + log_rate)
  - Replicates input 2^log_rate times before transformation

**Key Features**:
- In-place butterfly operations on shared memory
- Handles inputs up to 2^30 elements
- Multi-kernel launch for large inputs (max 11-log_rate stages per kernel)
- Works with FanPaarTowerField<HEIGHT> types

**Code Structure**:
```
AdditiveNTTKernelParams → additive_ntt_kernel (GPU) → AdditiveNTT (host class)
```

**Test Coverage**: Tested for log_h from 1 to 30 with log_rate 0 and 2

---

### 2. Standard Radix-2 NTT (`gpuntt.cuh`) - **FOR PRIME FIELDS**

**Purpose**: Traditional NTT for prime fields (Baby Bear, etc.)

**Algorithm Details**:
- **DIF Butterfly** (decimation-in-frequency): `u' = u + v; v' = (u - v) * w`
  - Standard multiplicative group structure
  - Uses subtraction and multiplication (not XOR)

- **Twiddle Factors**: Powers of primitive root ω
  - Precomputed as ω^0, ω^1, ω^2, ..., ω^(n/2-1)
  - Stored in bit-reversed order for coalesced access

- **Bit Reversal**: Required for correct output ordering
  - Input may need bit-reversal before transformation
  - Twiddles are stored in bit-reversed order
  - Uses `bit_reverse_in_place_ker` GPU kernel

**Key Features**:
- Supports fields with multiplicative structure (Baby Bear: 2^31 - 2^27 + 1)
- Root of unity based (ω^(2^n) = 1)
- Max 11 stages per kernel launch
- Bit-reversal optimizations

**Code Structure**:
```
NTTConfRad2<E> → bit_reverse_in_place_ker → ntt_kernel (GPU) → NTT (host class)
```

**Test Coverage**: Tested with Baby Bear field for log sizes 1 to 27

---

## Which NTT is Actually Used?

### In Production Code:
- **Additive NTT is the PRIMARY implementation** for Binius
- Used for binary tower field F_{2^128} operations
- This is what Binius prover requires (binary field arithmetic)

### Standard NTT:
- Provided for **compatibility and benchmarking** with prime field work
- Used in tests and benchmarks with Baby Bear field
- NOT the main focus for Binius prover

### Evidence:
1. **Library structure**: `ulvt_gpu` library includes both, but binary fields dominate
2. **Sumcheck integration**: Sumcheck protocol doesn't use NTT directly, operates on evaluations
3. **Test naming**: Additive NTT tests are more comprehensive (up to 2^30)
4. **Documentation**: README mentions "additive NTT" as a key feature

---

## Detailed Kernel Flow: Additive NTT

### Stage 1: Setup and Memory Transfer
```
CPU: Precompute subspace evaluations → GPU memory
CPU: Input data (2^log_h elements) → GPU pitched memory (replicated 2^log_rate times)
```

### Stage 2: Kernel Execution (per kernel launch)
```cuda
Thread mapping: exec_id = f(threadIdx, blockIdx, gridDim)
Load from global to shared: uv_mem[local] = data_io[global_offset]

For each stage from end_stage-1 down to start_stage:
    1. Calculate twiddle from precomputed constants (based on coset, stage, butterfly_block)
    2. Map thread to butterfly pair (u, v) in shared memory
    3. Execute: u = u + w*v; v = u + v  (additive butterfly)
    4. __syncthreads()
    
Write back to global: data_io[global_offset] = uv_mem[local]
```

### Stage 3: Multi-kernel Coordination
- Large transforms split into multiple kernel calls
- Each kernel processes max (11 - log_rate) stages
- Kernels launched in **reverse order** (high stages to low stages)
- Example for log_h=14: kernel[1] does stages 13-9, kernel[0] does stages 8-0

### Stage 4: Output Collection
```
GPU pitched memory (2^log_rate rows) → CPU contiguous memory via cudaMemcpy2D
```

---

## Configuration System

Both NTTs use sophisticated kernel launch configurations stored in `nttconf.cu`:

### KERNEL_LAUNCH_CONFIGS[log_length][kernel_num][grid/block]
- Defines block dimensions and grid dimensions for each input size
- Optimized for different GPU occupancy at different scales
- Example: log_length=12 uses 2 kernels with different configurations

### MAX_STAGES_PER_KERNEL = 11
- Limits how many butterfly stages can run in one kernel
- Due to shared memory and thread synchronization constraints
- Additive NTT effective limit: (11 - log_rate)

### Shared Memory Usage
- MAX_SHARED_MEM = 8KB (configurable)
- Stores intermediate butterfly values
- Critical for performance (avoids global memory accesses)

---

## Performance Characteristics

### Additive NTT Optimizations:
1. **Coalesced memory access** via pitched memory layout
2. **Shared memory butterflies** minimize global memory traffic
3. **Dynamic twiddle calculation** trades computation for memory bandwidth
4. **Multi-kernel splitting** handles arbitrarily large inputs

### Standard NTT Optimizations:
1. **Bit-reversed twiddle storage** for sequential access patterns
2. **Precomputed twiddles** reduce online computation
3. **In-place transformations** minimize memory footprint

---

## Field Types Supported

### Additive NTT:
- `FanPaarTowerField<0>` through `FanPaarTowerField<5>` (1-bit to 32-bit)
- `FanPaarTowerField<7>` (128-bit, main target for sumcheck)
- Characteristic 2 fields only

### Standard NTT:
- `BB31` (Baby Bear: prime field mod 2^31 - 2^27 + 1)
- Any field with multiplicative structure and suitable root of unity
- Characteristic p > 2

---

## Integration with Sumcheck

The sumcheck protocol (`/src/ulvt/sumcheck/`) **does not directly call NTT**:
- Operates on multilinear polynomial evaluations
- Uses fold operations instead of transforms
- Field arithmetic uses same binary tower implementation
- NTT could be used for polynomial multiplication (not currently implemented)

The connection is through **shared field arithmetic**:
- Both use `binary_tower.cuh` for F_{2^128} operations
- Both optimize for GPU parallel processing
- Complementary tools for Binius prover pipeline
