/**
 * cvta_to_global.cu — 测试 cvta.to.global 在 SASS 层面的实现
 *
 * cvta.to.global.u64 从泛型指针提取 global 地址。
 * 对于 global memory，泛型指针的低位就是实际地址（高位标记地址空间为 global），
 * 所以 cvta.to.global 理论上接近 no-op。
 *
 * 测试 sm_90 SASS 中是否有专门指令，还是被编译器优化掉。
 */

#include <cstdio>

// ============================================================
// Test A: __noinline__ 函数中用 cvta.to.global 提取地址
// ============================================================
__device__ __noinline__ unsigned long long extract_global_addr(const int *generic_ptr) {
    unsigned long long addr;
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(addr) : "l"((unsigned long long)generic_ptr));
    return addr;
}

// ============================================================
// Test B: cvta.to.global 后立即访存
// ============================================================
__device__ __noinline__ int read_via_cvta_to_global(const int *generic_ptr) {
    unsigned long long addr;
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(addr) : "l"((unsigned long long)generic_ptr));
    int result;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(result) : "l"(addr));
    return result;
}

// ============================================================
// Test C: 编译器原生 cvta.to.global（让编译器自由优化）
// ============================================================
__device__ __noinline__ int read_global_native(const int *p) {
    return *p;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_cvta_to_global(int *global_out, int *global_in, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;

    // Test A: 提取 global 地址
    unsigned long long addr = extract_global_addr(&global_in[tid]);

    // Test B: cvta.to.global + ld.global
    int v1 = read_via_cvta_to_global(&global_in[tid]);

    // Test C: 原生读取
    int v2 = read_global_native(&global_in[tid]);

    // 将提取的地址低32位写入输出（防止被优化掉）
    global_out[tid] = v1 + v2 + (int)(addr & 0xFFFFFFFF);
}
