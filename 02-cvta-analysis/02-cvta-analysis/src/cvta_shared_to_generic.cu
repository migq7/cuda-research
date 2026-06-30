/**
 * cvta_shared_to_generic.cu — 测试 cvta.shared（shared→泛型）在 SASS 层面的实现
 *
 * cvta.shared.u64 将 shared memory 的 32-bit 偏移包装为 64-bit 泛型指针。
 * 泛型指针的高位 bit 编码了 "shared" 地址空间信息。
 *
 * 关键问题: sm_90 如何设置泛型指针的高位？
 *   - 有专门的 CVTA 指令设置高位？
 *   - 还是通过 OR/SHF 等基础指令组合？
 */

#include <cstdio>

// ============================================================
// Test A: cvta.shared 包装 shared 地址为泛型指针
//         存入 global memory 以阻止优化
// ============================================================
__device__ __noinline__ unsigned long long wrap_shared_to_generic(unsigned int shared_offset) {
    unsigned long long generic_ptr;
    asm volatile("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;"
                 : "=l"(generic_ptr) : "r"(shared_offset));
    return generic_ptr;
}

// ============================================================
// Test B: cvta.shared + 传递给 __noinline__ 函数
//         观察泛型指针如何穿越函数边界
// ============================================================
__device__ __noinline__ void write_through_generic(unsigned long long generic_ptr, int val) {
    // 通过泛型指针写入（编译器需自行插入 cvta.to.shared 提取地址）
    int *p = (int *)(unsigned long)generic_ptr;  // 低位就是 shared offset
    // 使用 asm 确保走 shared 路径
    unsigned int addr;
    asm volatile("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
                 : "=r"(addr) : "l"(generic_ptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Test C: 原生编译器行为 —— 从 kernel 传 shared 指针给 __noinline__
//         编译器必须自行插入 cvta.shared
// ============================================================
__device__ __noinline__ int read_generic_ptr(int *p) {
    return *p;  // 编译器会在调用点包装为泛型指针
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_cvta_shared_to_generic(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;

    if (tid < N) smem[tid] = global_in[tid];
    __syncthreads();

    // Test A: 提取 shared 偏移，包装为泛型指针
    // shared memory 变量的地址是它们在 shared memory 空间内的偏移
    unsigned int shared_off = (unsigned int)(__cvta_generic_to_shared(&smem[tid]));
    unsigned long long generic_ptr = wrap_shared_to_generic(shared_off);

    // Test B: 通过包装后的泛型指针写入
    int v1 = smem[tid] + 1;
    write_through_generic(generic_ptr, v1);

    // Test C: 编译器原生 —— 传 shared 指针给 __noinline__ 函数
    int v2 = read_generic_ptr(&smem[tid]);

    __syncthreads();

    if (tid < N) {
        global_out[tid] = smem[tid] + v2 + (int)(generic_ptr & 0xFF);
    }
}
