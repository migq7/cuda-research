/**
 * test_cgactaid_generic_shared.cu
 *
 * Target SM: sm_90
 * Mode: Disassembly-only (compile with scripts/compile.sh, inspect build/sass/)
 *
 * Probes: What exactly goes into the 64-bit generic shared pointer?
 *   - Is the low 32 bits a pure shared-internal byte offset?
 *   - Or does it include UR4 / CgaCtaId / window base?
 *   - How does cvta.to.shared extract the address?
 *   - How does the extracted address differ from the direct STS path?
 *
 * Approach:
 *   Declare two __shared__ arrays (arr_A, arr_B) with a gap between them.
 *   Construct generic ptrs to arr_A[0], arr_B[0] and fixed offset 0.
 *   Dump raw 64-bit values.  Compare lo32 differences.
 *   Use __noinline__ dereference to force cvta.to.shared in SASS.
 *   Use inline asm mvta.to.shared to force ptxas to keep the conversion.
 *
 * Predicted answer (from prior analysis):
 *   lo32 = pure shared byte offset (NOT including UR4/CgaCtaId/window base)
 *   hi32 = SR_SWINHI (fixed shared-space tag)
 *   cvta.to.shared = NO-OP (MOV hi32-> discard, MOV lo32 -> pass through)
 *   STS [lo32] uses the offset directly; hardware adds window base implicitly
 */
#include <cstdio>
#include <cstdint>

__device__ unsigned long long g_dump[32];

// ============================================================
// Shared arrays declared inside the kernel for clarity.
// The compiler will lay them out sequentially:
//   arr_A at offset 0x400 (UR4 base)
//   gap_A at offset 0x440 (0x400 + sizeof(arr_A))
//   arr_B at offset 0x540 (0x440 + sizeof(gap_A))
// (gap_A may be optimized away if unreferenced — see SASS)
// ============================================================

// ============================================================
// Generic pointer constructors — cvta.shared.u64
// These are __noinline__ so the SASS is easy to read.
// ============================================================
__device__ __noinline__ uint64_t make_generic_shared(void *p) {
    uint64_t r;
    asm volatile("cvta.shared.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

__device__ __noinline__ uint64_t make_generic_shared_from_offset(unsigned int offset) {
    uint64_t r;
    asm volatile("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;"
                 : "=l"(r) : "r"(offset));
    return r;
}

// ============================================================
// Generic pointer destructor — cvta.to.shared.u64
// ============================================================
__device__ __noinline__ unsigned int cvta_to_shared(uint64_t gen_ptr) {
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"(gen_ptr));
    return addr;
}

// ============================================================
// Dereference via generic pointer (caller passes 64-bit ptr)
// These force ptxas to use LD.E / ST.E or STS/LDS with the
// extracted address.
// ============================================================
__device__ __noinline__ int read_generic(uint64_t gen_ptr) {
    unsigned int addr = cvta_to_shared(gen_ptr);
    int result;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(result) : "r"(addr));
    return result;
}

__device__ __noinline__ void write_generic(uint64_t gen_ptr, int val) {
    unsigned int addr = cvta_to_shared(gen_ptr);
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Direct shared access (no generic pointers) for comparison
// ============================================================
__device__ __noinline__ void write_shared_direct(unsigned int offset, int val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(val));
}

__device__ __noinline__ int read_shared_direct(unsigned int offset) {
    int result;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(result) : "r"(offset));
    return result;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_cgactaid(int *out) {
    __shared__ int arr_A[16];        // 64 bytes
    __shared__ char gap_A[256];      // 256 bytes padding
    __shared__ int arr_B[16];        // 64 bytes, at offset 0x140 from arr_A

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // --- Initialize arrays with known values ---
    if (tid < 16) {
        arr_A[tid] = 0xAAAA0000 + bid * 1000 + tid;
        arr_B[tid] = 0xBBBB0000 + bid * 1000 + tid;
    }
    __syncthreads();

    if (tid == 0) {
        // ==== A. Raw generic pointer values ====

        // A1: generic ptr to arr_A[0], constructed from the address
        uint64_t gen_A0 = make_generic_shared((void*)&arr_A[0]);

        // A2: generic ptr to arr_A[8] (offset 32 bytes from arr_A[0])
        uint64_t gen_A8 = make_generic_shared((void*)&arr_A[8]);

        // A3: generic ptr to arr_B[0], constructed from the address
        uint64_t gen_B0 = make_generic_shared((void*)&arr_B[0]);

        // A4: generic ptr constructed from raw offset 0
        uint64_t gen_raw0 = make_generic_shared_from_offset(0);

        // A5: generic ptr constructed from raw offset 0x140 (= arr_B offset)
        uint64_t gen_raw_140 = make_generic_shared_from_offset(0x140);

        // Dump raw 64-bit values
        g_dump[0] = gen_A0;
        g_dump[1] = gen_A8;
        g_dump[2] = gen_B0;
        g_dump[3] = gen_raw0;
        g_dump[4] = gen_raw_140;

        // ==== B. cvta.to.shared extraction ====

        // B1: extract from gen_A0 → should be arr_A[0] offset
        unsigned int addr_A0 = cvta_to_shared(gen_A0);

        // B2: extract from gen_B0 → should be some larger offset
        unsigned int addr_B0 = cvta_to_shared(gen_B0);

        // B3: extract from gen_raw0 → should be 0
        unsigned int addr_raw0 = cvta_to_shared(gen_raw0);

        // B4: extract from gen_raw_140 → should be 0x140
        unsigned int addr_raw140 = cvta_to_shared(gen_raw_140);

        g_dump[5] = addr_A0;
        g_dump[6] = addr_B0;
        g_dump[7] = addr_raw0;
        g_dump[8] = addr_raw140;

        // ==== C. Use raw-offset generic ptrs to write/read ====
        // Write 0xCAFE to the location pointed to by gen_raw0 (i.e. offset 0)
        write_generic(gen_raw0, 0xCAFE0000 + bid);

        // Write 0xBEEF to the location at gen_raw_140 (i.e. offset 0x140)
        write_generic(gen_raw_140, 0xBEEF0000 + bid);

        // Write 0xDEAD to arr_A[0] via generic ptr
        write_generic(gen_A0, 0xDEAD0000 + bid);

        __syncthreads();

        // Read back
        int val_raw0  = read_generic(gen_raw0);
        int val_raw140 = read_generic(gen_raw_140);
        int val_A0     = read_generic(gen_A0);

        // Also read via direct shared
        int val_A0_direct = arr_A[0];
        int val_B0_direct = arr_B[0];

        g_dump[9]  = val_raw0;
        g_dump[10] = val_raw140;
        g_dump[11] = val_A0;
        g_dump[12] = val_A0_direct;
        g_dump[13] = val_B0_direct;

        // ==== E. Direct STS/LDS for reference ====
        // These use no generic pointers — compare their SASS addressing
        write_shared_direct(0, 0x11111111);
        int val_direct = read_shared_direct(0);
        g_dump[14] = val_direct;
    }

    // Export to global for inspection
    if (tid < 16) {
        out[tid] = g_dump[tid];
    }

    // Fill g_dump slots > 15 with block info for cross-CTA verification
    if (tid == 1) {
        g_dump[15] = bid;
        g_dump[16] = (unsigned int)(uint64_t)&arr_A[0];   // direct shared address (lo32)
        g_dump[17] = (unsigned int)(uint64_t)&arr_B[0];   // direct shared address (lo32)
        g_dump[18] = (unsigned int)(uint64_t)&arr_A[8];   // direct shared address (lo32)
    }
}
