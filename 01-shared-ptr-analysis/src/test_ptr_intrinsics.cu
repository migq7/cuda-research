/**
 * test_ptr_intrinsics.cu
 *
 * 测试目的: 使用 CUDA 指针属性 intrinsic 在运行时判断地址空间。
 * CUDA 5.0+ 提供: __isGlobal(), __isShared(), __isLocal(), __isConstant()
 *
 * 这可以在 __noinline__ 函数内部实现手动地址空间分发。
 */

#include <cstdio>

// ============================================================
// 使用 intrinsic 做运行时地址空间判断
// ============================================================
__device__ __noinline__ int read_with_dispatch(int *p) {
    // 使用 CUDA intrinsic 判断指针类型
    // 注意: 这些 intrinsic 在 PTX 层面通过指针高位 bit 判断，不是真的运行时开销很大
    if (__isShared(p)) {
        return *p;  // 编译器可以推断此分支 p 是 shared
    } else if (__isGlobal(p)) {
        return *p;  // 编译器可以推断此分支 p 是 global
    } else {
        return *p;  // local 或其他
    }
}

// 使用 switch-case 风格
__device__ __noinline__ int read_with_switch(int *p) {
    // 注意: (unsigned long long) 转换在 64-bit 寻址下需要
    unsigned int high_bits = __cvta_generic_to_shared(p) != 0 ? 1 :
                              __cvta_generic_to_global(p) != 0 ? 2 : 0;

    // 但实际上 CUDA 没有这样的转换函数……
    // 正确的做法是用 __isShared / __isGlobal
    if (__isShared(p)) {
        return *p;
    }
    // fallthrough: default generic access
    return *p;
}

// ============================================================
// 对比: 直接用泛型指针访问（不加任何 hint）
// ============================================================
__device__ __noinline__ int read_generic_only(int *p) {
    return *p;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_intrinsics(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = global_in[tid];
    __syncthreads();

    // 用 intrinsic dispatch 读 shared
    int v1 = read_with_dispatch(&smem[tid]);

    // 用 intrinsic dispatch 读 global
    int v2 = read_with_dispatch(&global_in[tid]);

    // 纯泛型读 shared
    int v3 = read_generic_only(&smem[tid]);

    __syncthreads();
    if (tid < N) {
        global_out[tid] = v1 + v2 + v3;
    }
}
