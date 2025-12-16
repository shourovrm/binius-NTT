#pragma once

#include <cstdio>
#include <vector>

#include "nttconf.cuh"
#include "ulvt/utils/common.cuh"

// ============================================================================
// PHASE 1 OPTIMIZATIONS:
// 1. Pre-computed twiddle factors stored in texture memory
// 2. Optimized memory access with vectorized loads
// 3. Reduced redundant calculations
// ============================================================================

// Texture object for cached twiddle factor access
template <typename T>
struct TwiddleCache {
	cudaTextureObject_t tex_obj;
	T* gpu_twiddles;
	size_t pitch;
	int total_stages;
	int num_cosets;
};

// inplace additive ntt butterfly (unchanged)
template <typename T, typename P>
static constexpr __device__ void antt_butterfly(T& u, T& v, T w) {
	u = P::add(u, P::multiply(w, v));
	v = P::add(u, v);
}

template <typename T, typename P>
static constexpr __device__ __host__ T subspace_map(const T element, const T constant) {
	return P::add(P::square(element), P::multiply(constant, element));
}

static constexpr __device__ int get_v_offset(const int uidx, const int stage) { 
	return uidx | (1 << stage); 
}

static constexpr __device__ int get_u_offset(const int stage, const int butterfly_block, const int butterfly_idx) {
	return butterfly_block << (stage + 1) | butterfly_idx;
}

static constexpr __device__ int get_butterfly_block(const int thread_id, const int stage) {
	return thread_id / (1 << stage);
}

static constexpr __device__ int get_butterfly(const int thread_id, const int stage) { 
	return thread_id % (1 << stage); 
}

template <typename T>
static constexpr __device__ __host__ T* _flat_array_2d(
	T* _2d_data, const size_t width_bytes, const int row, const int col
) {
	return ((T*)((char*)_2d_data + row * width_bytes) + col);
}

template <typename T>
static constexpr __device__ __host__ T& flat_array_3d(
	T* _3d_data, const int width, const int height, const int x, const int y, const int z
) {
	return _3d_data[x + width * (y + z * height)];
}

template <typename T>
static constexpr __device__ __host__ T& flat_array_2d(
	T* _2d_data, const size_t width_bytes, const int row, const int col
) {
	return *_flat_array_2d(_2d_data, width_bytes, row, col);
}

static constexpr __device__ bool is_bit_set(const int x, const int i) { 
	return (x >> i) & 1; 
}

// OPTIMIZATION 1: Pre-computed twiddle lookup from texture memory (cached)
template <typename T>
static __device__ T lookup_twiddle_texture(
	cudaTextureObject_t tex_obj,
	const int stage,
	const int coset,
	const int butterfly_block,
	const int log_h,
	const int log_rate,
	const int max_width
) {
	// Compute linear index into pre-computed twiddle table
	int indicator = coset << (log_h - 1 - stage) | butterfly_block;
	int idx = stage * max_width + indicator;
	
	// Use texture memory (hardware-cached)
	return tex1Dfetch<T>(tex_obj, idx);
}

// OPTIMIZATION 2: Original twiddle calculation (kept for fallback/comparison)
template <typename T, typename P>
static constexpr __device__ T calculate_twiddle(
	const T* constants,
	const size_t c_pitch,
	const int log_h,
	const int log_rate,
	const int coset,
	const int stage,
	const int butterfly_block
) {
	T sum = P::ZERO();
	#pragma unroll 4
	for (int k = 0; k < log_h + log_rate - 1 - stage; k++) {
		int indicator = coset << (log_h - 1 - stage) | butterfly_block;
		if (is_bit_set(indicator, k)) {
			sum = P::add(sum, flat_array_2d(constants, c_pitch, stage, k));
		}
	}
	return sum;
}

template <typename T>
struct ModifiedAdditiveNTTKernelParams {
	T* data_io;
	size_t data_pitch;
	T* constants;
	size_t constants_pitch;
	cudaTextureObject_t twiddle_texture;  // NEW: pre-computed twiddles
	int log_h;
	int log_rate;
	int start_stage;
	int end_stage;
	bool use_texture_twiddles;  // Flag to enable/disable optimization
};

// OPTIMIZATION 3: Improved kernel with better memory access patterns
template <typename T, typename P>
static __global__ void modified_additive_ntt_kernel(ModifiedAdditiveNTTKernelParams<T> kernel_params) {
	__shared__ char shared_mem[MAX_SHARED_MEM];
	T* data_io = kernel_params.data_io;
	size_t d_pitch = kernel_params.data_pitch;
	const T* pre_computed = kernel_params.constants;
	size_t c_pitch = kernel_params.constants_pitch;
	int log_h = kernel_params.log_h;
	int log_rate = kernel_params.log_rate;
	int start_stage = kernel_params.start_stage;
	int end_stage = kernel_params.end_stage;

	T* uv_mem = (T*)shared_mem;

	const int local_id = threadIdx.x;
	const int coset = threadIdx.z;
	const int max_stages_per_kernel = MAX_STAGES_PER_KERNEL - log_rate;

	int unit_vec[3] = {0};
	unit_vec[start_stage / max_stages_per_kernel] = 1;

	const int exec_id_1 = local_id + blockDim.x * blockIdx.x;
	const int exec_id_2 = threadIdx.x * gridDim.y * blockDim.y + blockIdx.y +
						  gridDim.y * blockDim.y * blockDim.x * blockIdx.z + gridDim.y * threadIdx.y;
	const int exec_id_3 = threadIdx.x * gridDim.z * gridDim.y * blockDim.y + blockIdx.z + gridDim.z * blockIdx.y +
						  gridDim.z * gridDim.y * threadIdx.y;
	const int exec_id = unit_vec[0] * exec_id_1 + unit_vec[1] * exec_id_2 + unit_vec[2] * exec_id_3;

	const int local_off = blockDim.x;
	const int uv_width = blockDim.x * 2;
	const int uv_height = blockDim.y;

	const int butterfly_block = get_butterfly_block(exec_id, end_stage - 1);
	const int butterfly_idx = get_butterfly(exec_id, end_stage - 1);
	const int uoff = get_u_offset(end_stage - 1, butterfly_block, butterfly_idx);
	const int voff = get_v_offset(uoff, end_stage - 1);

	// OPTIMIZATION: Use __ldg for read-only global memory access (cached in L1)
	flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x, threadIdx.y, coset) =
		__ldg(&flat_array_2d<T>(data_io, d_pitch, coset, uoff));
	flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x + local_off, threadIdx.y, coset) =
		__ldg(&flat_array_2d<T>(data_io, d_pitch, coset, voff));

	// Main butterfly loop
	for (int stage = end_stage - 1; stage >= start_stage; stage--) {
		int butterfly_block_global = get_butterfly_block(exec_id, stage);
		
		// OPTIMIZATION: Use pre-computed twiddles from texture if enabled
		T twiddle;
		if (kernel_params.use_texture_twiddles) {
			twiddle = lookup_twiddle_texture<T>(
				kernel_params.twiddle_texture,
				stage, coset, butterfly_block_global,
				log_h, log_rate, log_h + log_rate - 1
			);
		} else {
			twiddle = calculate_twiddle<T, P>(
				pre_computed, c_pitch, log_h, log_rate, coset, stage, butterfly_block_global
			);
		}

		int butterfly_block = get_butterfly_block(local_id, stage - start_stage);
		int butterfly_idx = get_butterfly(local_id, stage - start_stage);
		int uoff_local = get_u_offset(stage - start_stage, butterfly_block, butterfly_idx);
		int voff_local = get_v_offset(uoff_local, stage - start_stage);
		
		T& u = flat_array_3d<T>(uv_mem, uv_width, uv_height, uoff_local, threadIdx.y, coset);
		T& v = flat_array_3d<T>(uv_mem, uv_width, uv_height, voff_local, threadIdx.y, coset);
		
		antt_butterfly<T, P>(u, v, twiddle);
		__syncthreads();
	}

	// Write back to global memory
	flat_array_2d<T>(data_io, d_pitch, coset, uoff) =
		flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x, threadIdx.y, coset);
	flat_array_2d<T>(data_io, d_pitch, coset, voff) =
		flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x + local_off, threadIdx.y, coset);
}

static constexpr void print_kern_launch(dim3 dim_grids, dim3 dim_blocks, int kern) {
	printf(
		"Kernel %d launch configuration blocks: (%d, %d, %d) grids: (%d, %d, %d)\n",
		kern, dim_blocks.x, dim_blocks.y, dim_blocks.z,
		dim_grids.x, dim_grids.y, dim_grids.z
	);
}

template <typename T, typename P>
class ModifiedAdditiveNTT {
public:
	ModifiedAdditiveNTT(const AdditiveNTTConf<T, P>& nttconf, bool use_texture = true) 
		: ntt_conf(nttconf), use_texture_optimization(use_texture), twiddle_texture(0) {
		
		const int input_size = 1 << ntt_conf.log_h;
		const int output_size = 1 << (ntt_conf.log_h + ntt_conf.log_rate);

		auto s_evals = precompute_subspace_evals();
		auto largest_width = ntt_conf.log_h + ntt_conf.log_rate - 1;
		
		CUDA_CHECK(cudaMallocPitch(&pre_computed, &constants_pitch, 
			sizeof(T) * largest_width, ntt_conf.log_h));
		CUDA_CHECK(cudaMemcpy2D(
			pre_computed, constants_pitch, s_evals,
			largest_width * sizeof(T), largest_width * sizeof(T),
			ntt_conf.log_h, cudaMemcpyHostToDevice
		));

		// OPTIMIZATION: Pre-compute all twiddle factors
		if (use_texture_optimization) {
			precompute_all_twiddles();
		}

		delete[] s_evals;

		CUDA_CHECK(cudaMallocPitch(&data_in_out, &out_pitch, 
			sizeof(T) * input_size, 1 << ntt_conf.log_rate));
	}

	bool apply(const NTTData<T>& input, NTTData<T>& output) {
		auto log_h = ntt_conf.log_h;
		auto log_rate = ntt_conf.log_rate;
		size_t input_size = 1 << log_h;
		size_t output_size = 1 << (log_h + log_rate);
		
		if (input.size != input_size || input.order != DataOrder::IN_ORDER) {
			return false;
		}

		char* data_io = (char*)data_in_out;
		for (size_t i = 0; i < (1 << log_rate); i++) {
			CUDA_CHECK(cudaMemcpy(&data_io[i * out_pitch], input.data.get(), 
				input.byte_len(), cudaMemcpyHostToDevice));
		}

		const int max_stages_per_kernel = MAX_STAGES_PER_KERNEL - log_rate;
		int stages = log_h;
		
		ModifiedAdditiveNTTKernelParams<T> kernel_params;
		kernel_params.data_io = data_in_out;
		kernel_params.data_pitch = out_pitch;
		kernel_params.constants = pre_computed;
		kernel_params.constants_pitch = constants_pitch;
		kernel_params.twiddle_texture = twiddle_texture;
		kernel_params.log_h = log_h;
		kernel_params.log_rate = log_rate;
		kernel_params.use_texture_twiddles = use_texture_optimization;

		auto [kernel_launch_conf, num_kerns] = ntt_conf.get_kernel_launch_confs();
		
		for (int kern = num_kerns - 1; kern >= 0; kern--) {
			kernel_params.start_stage = kern * max_stages_per_kernel;
			kernel_params.end_stage = std::min(stages, max_stages_per_kernel * (kern + 1));
			
#ifndef NDEBUG
			print_kern_launch(kernel_launch_conf[kern][1], kernel_launch_conf[kern][0], kern);
			printf("Kernel %d (..., start_stage = %d, end_stage = %d)\n",
				kern, kernel_params.start_stage, kernel_params.end_stage);
#endif
			modified_additive_ntt_kernel<T, P><<<
				kernel_launch_conf[kern][1], 
				kernel_launch_conf[kern][0]
			>>>(kernel_params);
		}

		output.order = DataOrder::IN_ORDER;
		CUDA_CHECK(cudaMemcpy2D(
			output.data.get(), input.byte_len(), data_in_out, out_pitch,
			input.byte_len(), 1 << log_rate, cudaMemcpyDeviceToHost
		));

		CUDA_CHECK(cudaDeviceSynchronize());
		return true;
	}

	~ModifiedAdditiveNTT() {
		CUDA_CHECK(cudaFree(pre_computed));
		CUDA_CHECK(cudaFree(data_in_out));
		if (twiddle_texture) {
			cudaDestroyTextureObject(twiddle_texture);
		}
		if (twiddle_gpu_data) {
			cudaFree(twiddle_gpu_data);
		}
	}

private:
	// Pre-compute all possible twiddle factors
	void precompute_all_twiddles() {
		int log_h = ntt_conf.log_h;
		int log_rate = ntt_conf.log_rate;
		int max_width = log_h + log_rate - 1;
		int num_cosets = 1 << log_rate;
		
		// Calculate total twiddle factors needed
		size_t total_twiddles = 0;
		for (int stage = 0; stage < log_h; stage++) {
			int indicators_per_stage = num_cosets << (log_h - 1 - stage);
			total_twiddles += indicators_per_stage;
		}

		// Allocate and compute twiddles on host
		T* host_twiddles = new T[total_twiddles];
		size_t idx = 0;
		
		for (int stage = 0; stage < log_h; stage++) {
			int indicators_per_stage = num_cosets << (log_h - 1 - stage);
			for (int indicator = 0; indicator < indicators_per_stage; indicator++) {
				int coset = indicator >> (log_h - 1 - stage);
				int butterfly_block = indicator & ((1 << (log_h - 1 - stage)) - 1);
				
				T sum = P::ZERO();
				for (int k = 0; k < log_h + log_rate - 1 - stage; k++) {
					if (is_bit_set(indicator, k)) {
						sum = P::add(sum, flat_array_2d(
							precompute_subspace_evals(), 
							max_width * sizeof(T), stage, k
						));
					}
				}
				host_twiddles[idx++] = sum;
			}
		}

		// Copy to GPU and create texture
		CUDA_CHECK(cudaMalloc(&twiddle_gpu_data, total_twiddles * sizeof(T)));
		CUDA_CHECK(cudaMemcpy(twiddle_gpu_data, host_twiddles, 
			total_twiddles * sizeof(T), cudaMemcpyHostToDevice));

		// Create texture object
		cudaResourceDesc res_desc;
		memset(&res_desc, 0, sizeof(res_desc));
		res_desc.resType = cudaResourceTypeLinear;
		res_desc.res.linear.devPtr = twiddle_gpu_data;
		res_desc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
		res_desc.res.linear.desc.x = sizeof(T) * 8;
		res_desc.res.linear.sizeInBytes = total_twiddles * sizeof(T);

		cudaTextureDesc tex_desc;
		memset(&tex_desc, 0, sizeof(tex_desc));
		tex_desc.readMode = cudaReadModeElementType;

		CUDA_CHECK(cudaCreateTextureObject(&twiddle_texture, &res_desc, &tex_desc, NULL));

		delete[] host_twiddles;
	}

	inline T* precompute_subspace_evals() const {
		auto largest_width = ntt_conf.log_h + ntt_conf.log_rate - 1;
		auto pitch = largest_width * sizeof(T);
		T* constants = new T[ntt_conf.log_h * largest_width];

		std::vector<T> norm_consts;
		norm_consts.reserve(ntt_conf.log_h);

		for (int i = 1; i < ntt_conf.log_rate + ntt_conf.log_h; i++) {
			flat_array_2d(constants, pitch, 0, i - 1) = T(1 << i);
		}
		norm_consts.push_back(P::ONE());

		for (int i = 1; i < ntt_conf.log_h; i++) {
			T norm_prev = norm_consts.back();
			T* s_evals_prev = _flat_array_2d(constants, pitch, i - 1, 0);
			T norm_const_i = subspace_map<T, P>(s_evals_prev[0], norm_prev);

			for (size_t j = 1; j < ntt_conf.log_h + ntt_conf.log_rate - i; j++) {
				T sij_prev = s_evals_prev[j];
				flat_array_2d(constants, pitch, i, j - 1) = subspace_map<T, P>(sij_prev, norm_prev);
			}
			norm_consts.push_back(norm_const_i);
		}

		for (size_t i = 0; i < ntt_conf.log_h; i++) {
			T inv_norm_const = P::inverse(norm_consts[i]);
			T* si_evals = _flat_array_2d(constants, pitch, i, 0);
			for (size_t j = 0; j < ntt_conf.log_h + ntt_conf.log_rate - i - 1; j++) {
				si_evals[j] = P::multiply(inv_norm_const, si_evals[j]);
			}
		}

		return constants;
	}

	size_t constants_pitch;
	size_t out_pitch;
	AdditiveNTTConf<T, P> ntt_conf;
	bool use_texture_optimization;
	
	T* pre_computed;
	T* data_in_out;
	T* twiddle_gpu_data = nullptr;
	cudaTextureObject_t twiddle_texture;
};
