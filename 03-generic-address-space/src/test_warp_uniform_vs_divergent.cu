/**
 * test_warp_uniform_vs_divergent.cu — Phase 2.3
 *
 * Compare shared memory access when the generic pointer is
 * warp-uniform (all threads share the same upper bits) vs
 * warp-divergent (each thread has a potentially different window).
 *
 * Also test: cvta.shared with per-thread vs per-warp offsets.
 *
 * Key observation: does warp-uniform path use UR* (uniform regs)
 * while divergent path uses R* (per-thread regs)?
 */
#include <cstdio>

// ============================================================
// Test A: uniform access — all threads use the same shared base
// ============================================================
__device__ __noinline__ unsigned long long make_shared_ptr_uniform(unsigned int offset) {
    // Offset is same for all threads in the warp
    unsigned long long r;
    asm("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;" : "=l"(r) : "r"(offset));
    return r;
}

__device__ __noinline__ void write_via_generic_uniform(unsigned long long gen_ptr, int val) {
    unsigned int addr;
    asm("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %0; cvt.u32.u64 %1, tmp; }"
        : "+l"(gen_ptr), "=r"(addr));
    asm("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Test B: divergent access — each thread uses a different offset
// ============================================================
__device__ __noinline__ unsigned long long make_shared_ptr_divergent(unsigned int offset) {
    unsigned long long r;
    asm("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;" : "=l"(r) : "r"(offset));
    return r;
}

__device__ __noinline__ void write_via_generic_divergent(unsigned long long gen_ptr, int val) {
    unsigned int addr;
    asm("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %0; cvt.u32.u64 %1, tmp; }"
        : "+l"(gen_ptr), "=r"(addr));
    asm("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Test C: local memory — same offset vs thread-id offset
// ============================================================
__device__ __noinline__ unsigned long long make_local_ptr(int *p) {
    unsigned long long r;
    asm("cvta.local.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

__device__ __noinline__ int read_via_local_generic(unsigned long long gen_ptr) {
    unsigned long long addr;
    asm("cvta.to.local.u64 %0, %1;" : "=l"(addr) : "l"(gen_ptr));
    int result;
    asm("ld.local.u32 %0, [%1];" : "=r"(result) : "l"(addr));
    return result;
}

// ============================================================
// Test D: global memory — warp-uniform vs divergent
// ============================================================
__device__ __noinline__ unsigned long long make_global_ptr(const int *p) {
    unsigned long long r;
    asm("cvta.global.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

__device__ __noinline__ int read_via_global_generic(unsigned long long gen_ptr) {
    unsigned long long addr;
    asm("cvta.to.global.u64 %0, %1;" : "=l"(addr) : "l"(gen_ptr));
    int result;
    asm("ld.global.u32 %0, [%1];" : "=r"(result) : "l"(addr));
    return result;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_warp_uniform(int *out, int *in, int N) {
    __shared__ int smem[256];
    int local_val = threadIdx.x;
    int tid = threadIdx.x;

    if (tid < N) smem[tid] = in[tid];
    __syncthreads();

    // --- Test A: uniform shared access ---
    // All threads use offset 0 (same shared location)
    unsigned long long gen_shared_uniform = make_shared_ptr_uniform(0);
    write_via_generic_uniform(gen_shared_uniform, 100);

    // --- Test B: divergent shared access ---
    // Each thread uses offset tid*4 (different shared locations)
    unsigned long long gen_shared_divergent = make_shared_ptr_divergent(tid * 4);
    write_via_generic_divergent(gen_shared_divergent, 200 + tid);

    __syncthreads();

    // --- Test C: local memory — same ptr vs divergent ptrs ---
    unsigned long long gen_local = make_local_ptr(&local_val);
    // Thread 0 exports its local ptr to shared memory
    __shared__ unsigned long long export_local_ptr;
    if (tid == 0) export_local_ptr = gen_local;
    __syncthreads();

    // All threads read thread 0's local via the exported generic ptr
    // (warp-uniform: same gen_ptr for all threads)
    int val1 = read_via_local_generic(export_local_ptr);

    // All threads read their OWN local (divergent: different gen_ptr per thread)
    int val2 = read_via_local_generic(gen_local);

    // --- Test D: global memory ---
    unsigned long long gen_global_uniform = make_global_ptr(&in[0]);
    unsigned long long gen_global_divergent = make_global_ptr(&in[tid]);

    int gval1 = read_via_global_generic(gen_global_uniform);
    int gval2 = read_via_global_generic(gen_global_divergent);

    if (tid < N) {
        out[tid] = smem[tid] + val1 + val2 + gval1 + gval2;
    }
}
