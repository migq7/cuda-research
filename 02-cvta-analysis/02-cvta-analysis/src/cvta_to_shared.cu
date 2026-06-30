/**
 * cvta_to_shared.cu — 测试 cvta.to.shared 在 SASS 层面的实现
 *
 * 核心问题: cvta.to.shared.u64 (泛型指针→shared地址提取)
 *           在 sm_90 SASS 中是独立指令还是基础指令组合？
 *
 * 测试策略:
 *   A) 转换后立即访存 → 观察是否被融合为一条访存指令
 *   B) 转换后存入 global memory → 强制保留转换操作
 *   C) 通过 __noinline__ 函数边界 → 观察 ABI 层面的处理
 */

#include <cstdio>

// ============================================================
// Test A: __noinline__ 函数接收泛型指针，用 cvta.to.shared 提取后访存
// ============================================================
__device__ __noinline__ void write_via_cvta_to_shared(int *generic_ptr, int val) {
    // cvta.to.shared 从泛型指针中提取 shared memory 偏移
    unsigned int shared_addr;
    asm volatile("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
                 : "=r"(shared_addr) : "l"((unsigned long long)generic_ptr));
    // 通过提取出的 shared 地址写入
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(shared_addr), "r"(val));
}

// ============================================================
// Test B: 更接近真实用例 —— 编译器原生 cvta.to.shared
//         观察编译器如何生成 cvta.to.shared + st.shared
// ============================================================
__device__ __noinline__ void write_native_cvta_shared(int *generic_ptr, int val) {
    // 让编译器自行生成 cvta.to.shared（不用 asm）
    // 使用 volatile 指针阻止编译器优化掉转换
    volatile int *vp = (volatile int *)generic_ptr;
    *vp = val;
}

// ============================================================
// Test C: 将 cvta.to.shared 的结果存入 global memory
//         阻止编译器将转换与后续访存融合
// ============================================================
__device__ __noinline__ unsigned int extract_shared_offset(int *generic_ptr) {
    unsigned int offset;
    asm volatile("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
                 : "=r"(offset) : "l"((unsigned long long)generic_ptr));
    return offset;
}

// ============================================================
// Test D: 纯 PTX cvta.to.shared（不用 asm wrapper，让编译器自由发挥）
//         观察编译器自身的优化策略
// ============================================================
__device__ __noinline__ int read_shared_native(int *p) {
    // 直接解引用 —— 编译器会插入 cvta.to.shared
    return *p;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_cvta_to_shared(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;

    // 初始化 shared memory
    if (tid < N) {
        smem[tid] = global_in[tid];
    }
    __syncthreads();

    // Test A: 手动 asm cvta.to.shared
    int v1 = smem[tid] + 1;
    write_via_cvta_to_shared(&smem[tid], v1);

    // Test B: 编译器原生 cvta.to.shared
    int v2 = smem[tid] + 2;
    write_native_cvta_shared(&smem[tid], v2);

    // Test C: 提取 shared offset 存入 global
    unsigned int offset = extract_shared_offset(&smem[tid]);
    __syncthreads();

    if (tid < N) {
        global_out[tid] = smem[tid] + (int)offset;
    }
}
