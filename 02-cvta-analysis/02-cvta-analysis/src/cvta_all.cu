/**
 * cvta_all.cu — 综合测试：所有 cvta 变体在同一编译单元内
 *
 * 目的是观察编译器在不同地址空间转换间是否有统一的指令模式。
 * 将所有变体放在一起，便于在同一个 PTX/SASS 文件中对比。
 */

#include <cstdio>

// ============================================================
// global → generic
// ============================================================
__device__ __noinline__ unsigned long long cvta_global_u64(const int *p) {
    unsigned long long r;
    asm volatile("cvta.global.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

// ============================================================
// generic → global
// ============================================================
__device__ __noinline__ unsigned long long cvta_to_global_u64(unsigned long long p) {
    unsigned long long r;
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(r) : "l"(p));
    return r;
}

// ============================================================
// shared → generic
// ============================================================
__device__ __noinline__ unsigned long long cvta_shared_u64(unsigned int shared_off) {
    unsigned long long r;
    asm volatile("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;"
                 : "=l"(r) : "r"(shared_off));
    return r;
}

// ============================================================
// generic → shared
// ============================================================
__device__ __noinline__ unsigned int cvta_to_shared_u64(unsigned long long p) {
    unsigned int r;
    asm volatile("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
                 : "=r"(r) : "l"(p));
    return r;
}

// ============================================================
// local → generic
// ============================================================
__device__ __noinline__ unsigned long long cvta_local_u64(const int *p) {
    unsigned long long r;
    asm volatile("cvta.local.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

// ============================================================
// generic → local
// ============================================================
__device__ __noinline__ unsigned long long cvta_to_local_u64(unsigned long long p) {
    unsigned long long r;
    asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(r) : "l"(p));
    return r;
}

// ============================================================
// const → generic (constant memory)
//   cvta.const.u64 在 sm_90 ptxas 中需要 .u32 源操作数
//   改为让编译器自行生成，通过 __noinline__ 函数间接访问 __constant__
// ============================================================
__device__ __noinline__ int read_const_ptr(const int *p) {
    return *p;  // 编译器在此生成 cvta.const + ld.const
}

// ============================================================
// generic → const
// ============================================================
__device__ __noinline__ unsigned long long cvta_to_const_u64(unsigned long long p) {
    // cvta.to.const 在 sm_90 中可能不被 ptxas 支持
    // 这里返回原值以便观察编译器如何生成对应指令
    unsigned long long r;
    asm volatile("cvta.global.u64 %0, %1;" : "=l"(r) : "l"(p));
    return r;
}

// ============================================================
// Kernel
// ============================================================
__constant__ int cdata[64] = {0};

__global__ void kernel_cvta_all(int *global_out, int *global_in, int N) {
    __shared__ int smem[256];
    int local_var = 0;
    int tid = threadIdx.x;

    if (tid < N) smem[tid] = global_in[tid];

    // global ↔ generic
    unsigned long long g1 = cvta_global_u64(&global_in[tid]);
    unsigned long long g2 = cvta_to_global_u64(g1);

    // shared ↔ generic
    unsigned long long s1 = cvta_shared_u64((unsigned int)(tid * 4));
    unsigned int s2 = cvta_to_shared_u64(s1);

    // local ↔ generic
    unsigned long long l1 = cvta_local_u64(&local_var);
    unsigned long long l2 = cvta_to_local_u64(l1);

    // --- const → generic (native compiler) ---
    int cv = read_const_ptr(&cdata[tid & 63]);

    __syncthreads();
    if (tid < N) {
        global_out[tid] = (int)(g1 + g2 + s1 + s2 + l1 + l2) + cv + smem[tid];
    }
}
