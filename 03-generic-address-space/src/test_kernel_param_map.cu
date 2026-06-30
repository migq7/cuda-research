/**
 * test_kernel_param_map.cu — Phase 3.1
 *
 * Systematically catalog the constant bank (c[0x0][*]) layout
 * by varying kernel parameter count, type, and size.
 *
 * Compile kernels with different parameter lists and compare
 * which c[0x0][offsets] are used. Kernel-param-independent
 * offsets (like c[0x28]=stack_ptr, c[0x20]=local_window)
 * should be stable. Kernel-param-dependent offsets shift
 * based on parameter count and alignment.
 */
#include <cstdio>

__device__ unsigned long long g_dump[32];

// ============================================================
// Varying kernel parameter lists
//  — each kernel does a minimal shared/local/global access
//    to force the compiler to emit the full prologue
// ============================================================

__global__ void k0_empty() {
    // Minimal kernel — shows which constants are always loaded
}

__global__ void k1_one_int(int a) {
    int tid = threadIdx.x;
    if (tid == 0) g_dump[0] = a;
}

__global__ void k2_two_ints(int a, int b) {
    int tid = threadIdx.x;
    if (tid == 0) g_dump[0] = a + b;
}

__global__ void k3_int_ptr(int *a, int b) {
    int tid = threadIdx.x;
    // Uses a pointer param → forces address computation
    if (tid == 0) g_dump[0] = a[tid] + b;
}

__global__ void k4_two_ptrs(int *a, int *b) {
    int tid = threadIdx.x;
    if (tid == 0) g_dump[0] = a[tid] + b[tid];
}

__global__ void k5_three_ptrs(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if (tid == 0) g_dump[0] = a[tid] + b[tid] + c[tid];
}

__global__ void k6_mixed(int a, int *b, float c, int *d) {
    int tid = threadIdx.x;
    if (tid == 0) g_dump[0] = a + b[tid] + (int)c + d[tid];
}

__global__ void k7_large_params(int *a, int *b, int *c, int *d, int *e, int *f) {
    int tid = threadIdx.x;
    if (tid == 0) {
        g_dump[0] = a[tid] + b[tid] + c[tid] + d[tid] + e[tid] + f[tid];
    }
}

__global__ void k8_with_shared(int *a) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    smem[tid] = a[tid];
    __syncthreads();
    if (tid == 0) g_dump[0] = smem[0];
}

__global__ void k9_with_local(int *a) {
    int arr[64];
    int tid = threadIdx.x;
    for (int i = 0; i < 64; i++) arr[i] = a[tid] + i;
    if (tid == 0) g_dump[0] = arr[0];
}

__global__ void k10_with_both(int *a, int *b) {
    __shared__ int smem[256];
    int arr[64];
    int tid = threadIdx.x;
    smem[tid] = a[tid];
    arr[tid] = b[tid];
    __syncthreads();
    if (tid == 0) g_dump[0] = smem[0] + arr[0];
}
