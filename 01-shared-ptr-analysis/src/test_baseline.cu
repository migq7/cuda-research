/**
 * test_baseline.cu
 *
 * 测试目的: 对比 __device__ 函数在内联 vs 非内联情况下，
 * 对 shared memory pointer 参数的访问指令差异。
 *
 * 核心问题: 当 foo(int *p) 被传入 shared memory 指针时，
 * __noinline__ 版本无法通过内联推断地址空间，编译器如何处理？
 *   - 使用泛型 ld/st？
 *   - 还是生成特定地址空间的指令（通过其他方式推断）？
 */

#include <cstdio>

// ============================================================
// [Case 1] 可内联的 device 函数 —— 编译器可以通过内联知道 p 来自 shared memory
// ============================================================
__device__ int read_ptr_inline(int *p) {
    return *p;  // 内联后编译器知道 p 指向 shared memory
}

// ============================================================
// [Case 2] 禁止内联的 device 函数 —— 编译器无法通过内联推断地址空间
// ============================================================
__device__ __noinline__ int read_ptr_noinline(int *p) {
    return *p;  // 编译器该如何生成 ld 指令？
}

// ============================================================
// [Case 3] 写操作 —— 同样对比 inline / noinline
// ============================================================
__device__ void write_ptr_inline(int *p, int val) {
    *p = val;
}

__device__ __noinline__ void write_ptr_noinline(int *p, int val) {
    *p = val;
}

// ============================================================
// [Case 4] 同时访问 global 和 shared —— 测试同一函数内多地址空间
// ============================================================
__device__ __noinline__ int read_both(int *p_shared, int *p_global) {
    // p_shared 指向 shared memory, p_global 指向 global memory
    // 编译器如何为两个指针生成不同的 ld 指令？
    return *p_shared + *p_global;
}

// ============================================================
// Kernel: 调用上述函数，传入 shared memory 指针
// ============================================================
__global__ void kernel_baseline(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];

    int tid = threadIdx.x;

    // 初始化 shared memory
    if (tid < N) {
        smem[tid] = global_in[tid];
    }
    __syncthreads();

    // ---- Case 1: inline read ----
    int v1 = read_ptr_inline(&smem[tid]);

    // ---- Case 2: noinline read ----
    int v2 = read_ptr_noinline(&smem[tid]);

    // ---- Case 3: inline write ----
    write_ptr_inline(&smem[tid], v1 + 1);

    // ---- Case 4: noinline write ----
    write_ptr_noinline(&smem[tid], v2 + 2);

    __syncthreads();

    // ---- Case 5: read_both (shared + global) ----
    int v3 = read_both(&smem[tid], &global_in[tid]);

    if (tid < N) {
        global_out[tid] = smem[tid] + v3;
    }
}
