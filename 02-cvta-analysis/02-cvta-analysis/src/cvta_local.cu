/**
 * cvta_local.cu — 测试 cvta.local / cvta.to.local 在 SASS 层面的实现
 *
 * Local memory 是每线程私有的栈空间。cvta.local 和 cvta.to.local
 * 在 local↔generic 之间转换。
 *
 * local memory 地址空间的特点：
 *   - 每个线程有独立的 local memory 空间
 *   - 64-bit 泛型指针通过高位 bit 标记 local 空间
 *   - cvta.to.local 提取 local 偏移
 */

#include <cstdio>

// ============================================================
// Test A: cvta.local —— local 地址 → 泛型指针
// ============================================================
__device__ __noinline__ unsigned long long wrap_local_to_generic(void *local_ptr) {
    unsigned long long generic_ptr;
    asm volatile("cvta.local.u64 %0, %1;" : "=l"(generic_ptr) : "l"((unsigned long long)local_ptr));
    return generic_ptr;
}

// ============================================================
// Test B: cvta.to.local —— 泛型指针 → local 地址
// ============================================================
__device__ __noinline__ unsigned long long extract_local_addr(unsigned long long generic_ptr) {
    unsigned long long local_addr;
    asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(local_addr) : "l"(generic_ptr));
    return local_addr;
}

// ============================================================
// Test C: 通过 local 泛型指针读写
// ============================================================
__device__ __noinline__ void write_via_local_generic(unsigned long long generic_ptr, int val) {
    unsigned long long addr;
    asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(addr) : "l"(generic_ptr));
    asm volatile("st.local.u32 [%0], %1;" :: "l"(addr), "r"(val));
}

// ============================================================
// Kernel: 创建 local 变量，测试 cvta.local 双向转换
// ============================================================
__global__ void kernel_cvta_local(int *global_out, int N) {
    int local_var = threadIdx.x;  // 栈变量 (local memory)

    // Test A: local → generic
    unsigned long long gen = wrap_local_to_generic((void *)&local_var);

    // Test B: generic → local
    unsigned long long back = extract_local_addr(gen);

    // Test C: 通过 local 泛型指针写入
    local_var += 1;
    write_via_local_generic(gen, local_var);

    if (threadIdx.x < N) {
        // 输出回收的地址用于验证（同时防止优化）
        global_out[threadIdx.x] = local_var + (int)(gen & 0xFF) + (int)(back & 0xFF);
    }
}
