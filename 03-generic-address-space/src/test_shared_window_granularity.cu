/**
 * test_shared_window_granularity.cu — Phase 1.3 supplement
 *
 * Probe the shared memory window granularity from multiple angles
 * to confirm the 16 MB per CTA interpretation of ULEA(0x400, 0x18).
 *
 * Tests:
 *   A: No shared memory — ULEA should be absent
 *   B: Single-block kernel — CgaCtaId is always 0, does ULEA still appear?
 *   C: blockIdx.x used in shared offset — SR_CTAID.X vs SR_CgaCtaId in SASS?
 *   D: Two disjoint __shared__ arrays — one ULEA or two?
 *   E: Export raw cvta.shared output (full 64-bit pointer with SR_SWINHI)
 *      and compare upper 32 bits pattern against ULEA output
 *   F: Three kernels with cascading shared sizes at the 48KB boundary
 *      to check if UMOV changes near the limit
 *   G: Use __cvta_generic_to_shared() builtin vs asm cvta.to.shared
 *      to observe if the compiler inserts ULEA in both cases
 */
#include <cstdio>

__device__ unsigned long long g_storage[8];

// ============================================================
// Test A: No shared memory — should not generate ULEA/UMOV
// ============================================================
__global__ void k_no_shared(int *out, int N) {
    int tid = threadIdx.x;
    if (tid < N) out[tid] = tid;
}

// ============================================================
// Test B: Single-block kernel (gridDim = 1 in practice)
// ============================================================
__global__ void k_single_block(int *out, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

// ============================================================
// Test C: Use blockIdx.x — does SASS add SR_CTAID.X load?
//   Compare: kernel that uses blockIdx in shared vs one that doesn't
// ============================================================
__global__ void k_use_blockidx(int *out, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid < N) smem[tid] = tid + bid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_no_blockidx(int *out, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

// ============================================================
// Test D: Two disjoint __shared__ arrays
// ============================================================
__global__ void k_two_shared_arrays(int *out, int N) {
    __shared__ int arr1[256];
    __shared__ float arr2[128];
    int tid = threadIdx.x;
    if (tid < N) {
        arr1[tid] = tid;
        arr2[tid] = (float)tid;
    }
    __syncthreads();
    if (tid < N) out[tid] = arr1[tid] + (int)arr2[tid];
}

// ============================================================
// Test E: Export raw cvta.shared output (contains SR_SWINHI)
//   via __noinline__ function to prevent IPO elimination
// ============================================================
__device__ __noinline__ unsigned long long export_shared_generic(unsigned int offset) {
    unsigned long long r;
    asm("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;" : "=l"(r) : "r"(offset));
    return r;
}

__device__ __noinline__ unsigned long long export_local_generic(void *p) {
    unsigned long long r;
    asm("cvta.local.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

__device__ __noinline__ unsigned long long export_global_generic(const void *p) {
    unsigned long long r;
    asm("cvta.global.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

__global__ void k_export_all_generic_ptrs(int *out, int N) {
    __shared__ int smem[256];
    int local_val = threadIdx.x;
    int tid = threadIdx.x;

    // Export cvta results for comparison
    unsigned long long gen_shared  = export_shared_generic(0);     // offset 0
    unsigned long long gen_shared2 = export_shared_generic(1024);  // offset 0x400
    unsigned long long gen_local   = export_local_generic((void*)&local_val);
    unsigned long long gen_global  = export_global_generic((const void*)&out[tid]);

    if (tid == 0) {
        g_storage[0] = gen_shared;
        g_storage[1] = gen_shared2;
        g_storage[2] = gen_local;
        g_storage[3] = gen_global;
    }
    __syncthreads();

    if (tid < N) {
        smem[tid] = tid;
        __syncthreads();
        out[tid] = smem[tid] + local_val;
    }
}

// ============================================================
// Test F: Max shared memory boundary test (48KB)
//   Also test 32KB and 16KB for comparison
// ============================================================
__global__ void k_shared_max_48KB(int *out, int N) {
    __shared__ int smem[12288];  // 48 KB = sm_90 max static shared
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_mid_32KB(int *out, int N) {
    __shared__ int smem[8192];   // 32 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

__global__ void k_shared_mid_16KB(int *out, int N) {
    __shared__ int smem[4096];   // 16 KB
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}

// ============================================================
// Test G: __cvta_generic_to_shared() builtin vs asm cvta.to.shared
//   Does the builtin path also generate ULEA at the call site?
// ============================================================
__device__ __noinline__ void write_shared_builtin(void *p, int val) {
    unsigned int addr = __cvta_generic_to_shared(p);
    asm("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

__device__ __noinline__ void write_shared_asm_cvta(void *p, int val) {
    unsigned int addr;
    asm("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"((unsigned long long)p));
    asm("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

__global__ void k_builtin_vs_asm_cvta(int *out, int N) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    if (tid < N) smem[tid] = tid;
    __syncthreads();

    // Test both paths
    write_shared_builtin(&smem[tid], 100);
    write_shared_asm_cvta(&smem[tid], 200);

    __syncthreads();
    if (tid < N) out[tid] = smem[tid];
}
