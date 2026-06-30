/**
 * test_shared_cross_cta_ptr.cu — Phase 2.2
 *
 * Multi-block kernel. Block 0 constructs a generic shared pointer,
 * writes it to global memory. Block 1 reads it back and attempts
 * to dereference it via a __noinline__ function.
 *
 * Key questions:
 *   - Is the shared window truly per-CTA (via CgaCtaId)?
 *   - Does block 1's dereference use block 1's CgaCtaId or
 *     the pointer's embedded block-0 window ID?
 *   - If the compiler uses block 1's CgaCtaId, the SASS will
 *     show hardware-level CTA isolation.
 */
#include <cstdio>

// ============================================================
// Write to shared memory via a generic pointer
//   Caller provides the full 64-bit generic pointer
//   This decouples the window computation from the caller's CTA context
// ============================================================
__device__ __noinline__ void write_shared_via_generic(unsigned long long gen_ptr, int val) {
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"(gen_ptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Read from shared memory via a generic pointer
// ============================================================
__device__ __noinline__ int read_shared_via_generic(unsigned long long gen_ptr) {
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"(gen_ptr));
    int result;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(result) : "r"(addr));
    return result;
}

// ============================================================
// Construct a generic shared pointer from a shared offset
// ============================================================
__device__ __noinline__ unsigned long long make_shared_generic_ptr(unsigned int offset) {
    unsigned long long r;
    asm("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;" : "=l"(r) : "r"(offset));
    return r;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_shared_cross_cta(int *out, int *in, int N) {
    __shared__ int smem[256];
    __shared__ unsigned long long shared_ptr_storage;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Initialize
    if (tid < N) smem[tid] = in[tid] + bid * 1000;
    __syncthreads();

    if (bid == 0) {
        // Block 0: capture the full generic pointer to smem[0]
        shared_ptr_storage = make_shared_generic_ptr(0);
        smem[0] = 0xDEAD;  // marker value
    }
    __syncthreads();

    // --- Write via global memory (simulates cross-CTA pointer sharing) ---
    // Block 0 writes its generic shared ptr to global memory
    if (bid == 0 && tid == 0) {
        out[0] = (int)(shared_ptr_storage & 0xFFFFFFFF);
        out[1] = (int)(shared_ptr_storage >> 32);
    }

    // Block 1 reads the foreign shared ptr from global memory
    // and attempts to write through it
    unsigned long long foreign_shared_ptr = 0;
    if (bid == 1 && tid == 0) {
        unsigned int lo = (unsigned int)out[0];
        unsigned int hi = (unsigned int)out[1];
        asm("cvt.u64.u32 %0, %1;" : "=l"(foreign_shared_ptr) : "r"(lo));
        unsigned long long hi64;
        asm("cvt.u64.u32 %0, %1;" : "=l"(hi64) : "r"(hi));
        asm("shl.b64 %0, %0, 32;" : "+l"(hi64));
        asm("or.b64 %0, %0, %1;" : "+l"(foreign_shared_ptr) : "l"(hi64));
    }

    // Block 1 thread 0 broadcasts foreign ptr via shared memory
    if (bid == 1) {
        if (tid == 0) shared_ptr_storage = foreign_shared_ptr;
        __syncthreads();
        foreign_shared_ptr = shared_ptr_storage;

        // Probe: write to the foreign shared ptr
        write_shared_via_generic(foreign_shared_ptr, 0xBEEF + tid);
        __syncthreads();

        // Read back (from block 1's OWN smem[0] as comparison)
        int own_val = smem[0];
        int foreign_val = read_shared_via_generic(foreign_shared_ptr);

        if (tid == 0) {
            out[2] = own_val;
            out[3] = foreign_val;
        }
    }

    __syncthreads();

    // Block 0 reads its own smem[0] (may have been modified by block 1?)
    if (bid == 0 && tid == 0) {
        out[4] = smem[0];
    }
}
