/**
 * test_dynamic_shared_config.cu — Phase 3.4
 *
 * Test how dynamic shared memory (extern __shared__) affects
 * the window base computation compared to static shared memory.
 *
 * With static shared, the compiler knows the size at compile time
 * and emits UMOV with the exact size. With dynamic shared, the
 * size is unknown — does the compiler:
 *   a) Use a kernel parameter constant for the size?
 *   b) Use a different base computation (runtime ULEA)?
 *   c) Fall back to a different addressing mode entirely?
 */
#include <cstdio>

// ============================================================
// Control: static shared (known size)
// ============================================================
__global__ void k_static_shared_256(int *out, int N) {
    __shared__ int smem[64];  // 256 bytes
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_static_shared_1K(int *out, int N) {
    __shared__ int smem[256];  // 1 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

// ============================================================
// Dynamic shared (size unknown at compile time)
// ============================================================
__global__ void k_dynamic_shared(int *out, int N) {
    extern __shared__ int dyn_smem[];
    int tid = threadIdx.x;
    if (tid < N) dyn_smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = dyn_smem[tid];
}

// ============================================================
// Dynamic shared + static local vars (to test interaction)
// ============================================================
__global__ void k_dynamic_shared_with_stack(int *out, int N) {
    extern __shared__ int dyn_smem[];
    int local_arr[32];
    int tid = threadIdx.x;
    if (tid < N) dyn_smem[tid] = tid;
    for (int i = 0; i < 32; i++) local_arr[i] = tid + i;
    __syncthreads();
    if (tid < N) out[tid] = dyn_smem[tid] + local_arr[tid & 31];
}

// ============================================================
// Export dynamic shared base via cvta
// ============================================================
__device__ __noinline__ unsigned long long get_shared_base_dyn(unsigned int offset) {
    unsigned long long r;
    asm("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;" : "=l"(r) : "r"(offset));
    return r;
}

__global__ void k_export_dynamic_shared_base(int *out, int N) {
    extern __shared__ int dyn_smem[];
    int tid = threadIdx.x;
    if (tid < N) dyn_smem[tid] = tid;
    __syncthreads();

    unsigned long long base = get_shared_base_dyn(0);
    if (tid == 0) {
        out[0] = (int)(base & 0xFFFFFFFF);
        out[1] = (int)(base >> 32);
    }
}

// ============================================================
// Also test: two dynamic shared arrays (common pattern)
// In CUDA, you manually partition extern __shared__ memory
// ============================================================
__global__ void k_two_dynamic_arrays(int *out, int N) {
    extern __shared__ int dyn_smem[];
    int *arr1 = dyn_smem;                  // first half
    int *arr2 = &dyn_smem[blockDim.x];     // second half
    int tid = threadIdx.x;
    if (tid < N) {
        arr1[tid] = tid;
        arr2[tid] = tid * 2;
    }
    __syncthreads();
    if (tid < N) {
        out[tid] = arr1[tid] + arr2[tid];
    }
}
