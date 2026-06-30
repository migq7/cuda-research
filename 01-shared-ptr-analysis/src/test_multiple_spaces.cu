/**
 * test_multiple_spaces.cu
 *
 * 测试目的: 同一个 __noinline__ __device__ 函数，从不同地址空间
 * (shared / global / local) 调用，观察编译器是否生成多个函数版本。
 *
 * 如果编译器为每个调用点生成专用克隆（per-call-site versioning），
 * 那么每个克隆可以使用特定的地址空间指令。
 */

#include <cstdio>

// 通用 device 函数 —— 禁止内联
__device__ __noinline__ int generic_read(int *p) {
    return *p;
}

__device__ __noinline__ void generic_write(int *p, int val) {
    *p = val;
}

// 从 kernel 中用 shared memory pointer 调用
__global__ void kernel_shared(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = global_in[tid];
    __syncthreads();

    int v = generic_read(&smem[tid]);       // 调用点 A: shared
    generic_write(&smem[tid], v + 1);        // 调用点 B: shared
    __syncthreads();
    if (tid < N) global_out[tid] = smem[tid];
}

// 从 kernel 中用 global memory pointer 调用
__global__ void kernel_global(int *global_out, int *global_in, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    int v = generic_read(&global_in[tid]);  // 调用点 C: global
    generic_write(&global_out[tid], v + 1);  // 调用点 D: global
}

// 从 kernel 中用 local memory pointer 调用
__global__ void kernel_local(int *global_out, int *global_in, int N) {
    int local_var = 0;
    int tid = threadIdx.x;
    if (tid < N) local_var = global_in[tid];

    int v = generic_read(&local_var);        // 调用点 E: local
    generic_write(&local_var, v + 1);         // 调用点 F: local

    if (tid < N) global_out[tid] = local_var;
}

// 混合调用: 同一个 kernel 中从 shared 和 global 都调用
__global__ void kernel_mixed(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = global_in[tid];
    __syncthreads();

    // 从 shared 调用
    int v1 = generic_read(&smem[tid]);

    // 从 global 调用 (同一个函数，不同地址空间)
    int v2 = generic_read(&global_in[tid]);

    __syncthreads();
    if (tid < N) global_out[tid] = v1 + v2;
}
