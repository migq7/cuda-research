/**
 * test_mbarrier_shared_reserved.cu — Phase 1.3 supplement
 *
 * Hypothesis: The low 1 KB (0x000–0x3FF) at the start of each CTA's shared
 * memory window is reserved for hardware-managed storage (suspected: mbarrier
 * entries). User __shared__ variables start at offset 0x400.
 *
 * Tier 1: inline PTX mbarrier — compiler sees plain shared vars.
 * Tier 2: cuda::barrier — compiler-recognized hardware mbarrier. KEY TEST:
 *   If mbarrier state is placed in the reserved 0x000-0x3FF region, user
 *   variable ULEA offsets should stay at 0x400 regardless of barrier count.
 *   If barrier state occupies user space, offsets increase proportionally.
 */
#include <cstdio>
#include <cuda/barrier>

using cuda_barrier = cuda::barrier<cuda::thread_scope_block>;

// ============================================================
// Tier 1: Inline PTX (compiler-agnostic plain shared vars)
// ============================================================
__device__ __forceinline__ void mbar_init(unsigned int so, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(so), "r"(count));
}

__global__ void k_t1_baseline(int *out, int N) {
    __shared__ int arr[256];
    int tid = threadIdx.x;
    if (tid < N) arr[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = arr[tid];
}

__global__ void k_t1_two_barriers(int *out, int N) {
    __shared__ unsigned long long b1_st, b2_st;
    __shared__ int arr[64];
    int tid = threadIdx.x;
    if (tid == 0) {
        mbar_init(__cvta_generic_to_shared(&b1_st), 1);
        mbar_init(__cvta_generic_to_shared(&b2_st), 1);
    }
    __syncthreads();
    if (tid < N) arr[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = arr[tid];
}

// ============================================================
// Tier 2: cuda::barrier — compiler-recognized mbarrier
// ============================================================
__global__ void k_t2_baseline(int *out, int N) {
    __shared__ int arr[256];
    int tid = threadIdx.x;
    if (tid < N) arr[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = arr[tid];
}

__global__ void k_t2_one_barrier(int *out, int N) {
    __shared__ cuda_barrier bar;
    __shared__ int arr[256];
    int tid = threadIdx.x;
    if (tid == 0) init(&bar, 1);
    __syncthreads();
    if (tid < N) arr[tid] = tid;
    __syncthreads();
    if (tid == 0) bar.arrive_and_wait();
    if (tid < N) out[tid] = arr[tid];
}

__global__ void k_t2_three_barriers(int *out, int N) {
    __shared__ cuda_barrier b1, b2, b3;
    __shared__ int arr[64];
    int tid = threadIdx.x;
    if (tid == 0) { init(&b1, 1); init(&b2, 1); init(&b3, 1); }
    __syncthreads();
    if (tid < N) arr[tid] = tid;
    __syncthreads();
    if (tid == 0) { b1.arrive_and_wait(); b2.arrive_and_wait(); b3.arrive_and_wait(); }
    if (tid < N) out[tid] = arr[tid];
}

__global__ void k_t2_barrier_between_arrays(int *out, int N) {
    __shared__ int arr1[64];
    __shared__ cuda_barrier bar;
    __shared__ int arr2[64];
    int tid = threadIdx.x;
    if (tid == 0) init(&bar, 1);
    __syncthreads();
    if (tid < N) { arr1[tid] = tid; arr2[tid] = tid * 2; }
    __syncthreads();
    if (tid == 0) bar.arrive_and_wait();
    if (tid < N) out[tid] = arr1[tid] + arr2[tid];
}
