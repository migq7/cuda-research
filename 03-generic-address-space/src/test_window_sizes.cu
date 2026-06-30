/**
 * test_window_sizes.cu — Phase 1.3
 *
 * Vary shared memory allocation sizes across multiple kernels to
 * reverse-engineer the ULEA/UMOV window computation formula.
 *
 * Also varies local stack usage to map stack frame layout.
 *
 * Key comparisons across SASS outputs:
 *   UMOV constant, ULEA shift amount, VADD stack frame immediate,
 *   and whether the pattern changes at boundary sizes.
 */
#include <cstdio>

// ============================================================
// Shared memory size sweep: kernels with different __shared__ sizes
// ============================================================
__global__ void k_shared_64B(int *out, int N) {
    __shared__ int smem[16];   // 64 bytes
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_256B(int *out, int N) {
    __shared__ int smem[64];   // 256 bytes
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_1KB(int *out, int N) {
    __shared__ int smem[256];  // 1 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_4KB(int *out, int N) {
    __shared__ int smem[1024]; // 4 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_16KB(int *out, int N) {
    __shared__ int smem[4096]; // 16 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_32KB(int *out, int N) {
    __shared__ int smem[8192]; // 32 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_48KB(int *out, int N) {
    // sm_90 max static shared = 48 KB
    __shared__ int smem[12288]; // 48 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

// ============================================================
// Also test: same kernel with shared AND additional stack usage
//   to see if shared window computation changes with stack pressure
// ============================================================
__global__ void k_shared_1KB_plus_stack(int *out, int N) {
    __shared__ int smem[256];
    int local_arr[64];  // 256 bytes of stack
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    for (int i = 0; i < 64; i++) local_arr[i] = i;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid] + local_arr[tid & 63];
}

// ============================================================
// Local memory size sweep
// ============================================================
__global__ void k_local_none(int *out, int N) {
    int tid = threadIdx.x;
    if (tid < N) out[tid] = tid;
}

__global__ void k_local_16B(int *out, int N) {
    int a, b, c, d;
    int tid = threadIdx.x;
    a = tid; b = a + 1; c = b + 1; d = c + 1;
    if (tid < N) out[tid] = a + b + c + d;
}

__global__ void k_local_256B(int *out, int N) {
    int arr[64];
    int tid = threadIdx.x;
    for (int i = 0; i < 64; i++) arr[i] = tid + i;
    if (tid < N) out[tid] = arr[tid & 63];
}

__global__ void k_local_1KB(int *out, int N) {
    int arr[256];
    int tid = threadIdx.x;
    for (int i = 0; i < 256; i++) arr[i] = tid + i;
    if (tid < N) out[tid] = arr[tid & 255];
}

__global__ void k_local_4KB(int *out, int N) {
    int arr[1024];
    int tid = threadIdx.x;
    for (int i = 0; i < 1024; i++) arr[i] = tid + i;
    if (tid < N) out[tid] = arr[tid & 1023];
}

// ============================================================
// Vary both shared AND local simultaneously
// ============================================================
__global__ void k_combined_1KB_256B(int *out, int N) {
    __shared__ int smem[256];  // 1 KB shared
    int arr[64];                // 256 B local
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    for (int i = 0; i < 64; i++) arr[i] = i;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid] + arr[tid & 63];
}

// ============================================================
// Expose shared base via cvta and store to global
// ============================================================
__device__ __noinline__ unsigned long long get_shared_base(unsigned int offset) {
    unsigned long long r;
    asm("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;" : "=l"(r) : "r"(offset));
    return r;
}

__global__ void k_export_shared_base(int *out, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    // Export the full 64-bit generic pointer for smem[0]
    unsigned long long base = get_shared_base(0);
    if (tid == 0) {
        out[0] = (int)(base & 0xFFFFFFFF);
        out[1] = (int)(base >> 32);
    }
}

__global__ void k_export_local_ptr(int *out, int N) {
    int local_val = threadIdx.x;
    unsigned long long gen;
    asm("cvta.local.u64 %0, %1;" : "=l"(gen) : "l"((unsigned long long)&local_val));
    if (threadIdx.x == 0) {
        out[0] = (int)(gen & 0xFFFFFFFF);
        out[1] = (int)(gen >> 32);
    }
}
