/**
 * test_template.cu
 *
 * 测试目的: 使用模板参数来编码地址空间，作为对比方案。
 * 观察模板实例化是否能生成特定地址空间的指令（vs 泛型指针方案）。
 *
 * 注意: CUDA 不允许直接将地址空间作为模板参数，
 * 但可以通过类型特征或宏来模拟。
 */

#include <cstdio>

// ============================================================
// [Approach 1] 通过指针类型标记 —— 使用 __restrict__ 限定
// ============================================================

// 标记为 shared memory 指针（通过命名约定，实际上仍是泛型指针）
__device__ __noinline__ int read_shared_ptr(int * __restrict__ p) {
    return *p;
}

__device__ __noinline__ int read_global_ptr(const int * __restrict__ p) {
    return *p;
}

// ============================================================
// [Approach 2] 模板函数 —— 不同实例化可能有不同优化
// ============================================================
template <int AddrSpace>
__device__ __noinline__ int read_tmpl(int *p) {
    // AddrSpace 是一个 hint，但实际上不能影响指针类型
    // 编译器可能根据调用上下文优化
    return *p;
}

// 显式实例化
template __device__ __noinline__ int read_tmpl<0>(int *p); // shared hint
template __device__ __noinline__ int read_tmpl<1>(int *p); // global hint

// ============================================================
// [Approach 3] 不同签名的函数 —— 模拟地址空间重载
//              (CUDA 不支持基于地址空间的 C++ 重载)
// ============================================================

// 理论上，如果 CUDA 支持地址空间重载，会写成:
// __device__ int read(int __shared__ *p);
// __device__ int read(int __global__ *p);
// 但实际上不支持，所以只能用不同函数名

// ============================================================
// [Approach 4] 使用 asm 内联明确指定地址空间
// ============================================================
__device__ __noinline__ int read_with_asm_shared(int *p) {
    int result;
    // sm_90: shared memory pointer is 32-bit addressable, use "r" for 32-bit
    unsigned int addr_lo = (unsigned int)(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.s32 %0, [%1];" : "=r"(result) : "r"(addr_lo));
    return result;
}

__device__ __noinline__ int read_with_asm_global(const int *p) {
    int result;
    // sm_90: global pointer is 64-bit, use "l" for 64-bit
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(result) : "l"((unsigned long long)p));
    return result;
}

// ============================================================
// Kernel 测试
// ============================================================
__global__ void kernel_template(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = global_in[tid];
    __syncthreads();

    // Approach 1: 命名约定
    int v1 = read_shared_ptr(&smem[tid]);
    int v2 = read_global_ptr(&global_in[tid]);

    // Approach 2: 模板
    int v3 = read_tmpl<0>(&smem[tid]);
    int v4 = read_tmpl<1>(&global_in[tid]);

    // Approach 4: 内联 asm
    int v5 = read_with_asm_shared(&smem[tid]);
    int v6 = read_with_asm_global(&global_in[tid]);

    __syncthreads();
    if (tid < N) {
        global_out[tid] = v1 + v2 + v3 + v4 + v5 + v6;
    }
}
