/**
 * test_isspacep_const_fold.cu — Phase 1.1
 *
 * Probe the generic address space tag bits by passing known 64-bit integer
 * literals through __isspacep_* intrinsics.
 *
 * EXPECTED (from comment hypothesis): The compiler should constant-fold
 *   the branch — the surviving SASS branch reveals which bit patterns
 *   the compiler considers shared/global/local.
 *
 * ACTUAL RESULT: The pointer construction (make_ptr_with_tag) IS
 *   constant-folded (tag and offset stored as UR immediates), but the
 *   isspacep.* checks are NOT constant-folded — ptxas does not know
 *   the hardware's address space boundary values, so it emits actual
 *   QSPC.E.{S,G,L} runtime instructions via CALL.REL.NOINC.
 *
 * VALUE OF THIS TEST: Despite the failed hypothesis, this test revealed:
 *   1. QSPC.E.S / QSPC.E.L are single-instruction tag checks
 *   2. QSPC.E.G uses a dual check: tag test OR range comparison vs c[0xd0]
 *   3. The cvta.shared lowering (S2R SR_SWINHI) is confirmed
 *   4. The cvta.local lowering (IADD3 add/subtract of c[0x20]) is confirmed
 *   5. The cvta.global lowering (identity NO-OP) is confirmed
 */
#include <cstdio>

__device__ unsigned long long g_storage[32];

// ============================================================
// isspacep.* via inline PTX asm — compiles to QSPC.E.* in SASS
// ============================================================
__device__ __noinline__ int is_shared_space(unsigned long long ptr) {
    // Hack: use inline asm to test isspacep.shared via PTX
    unsigned int result;
    asm volatile(
        "{ .reg .pred p;"
        "  isspacep.shared p, %1;"
        "  selp.u32 %0, 1, 0, p;"
        "}" : "=r"(result) : "l"(ptr));
    return result;
}

__device__ __noinline__ int is_global_space(unsigned long long ptr) {
    unsigned int result;
    asm volatile(
        "{ .reg .pred p;"
        "  isspacep.global p, %1;"
        "  selp.u32 %0, 1, 0, p;"
        "}" : "=r"(result) : "l"(ptr));
    return result;
}

__device__ __noinline__ int is_local_space(unsigned long long ptr) {
    unsigned int result;
    asm volatile(
        "{ .reg .pred p;"
        "  isspacep.local p, %1;"
        "  selp.u32 %0, 1, 0, p;"
        "}" : "=r"(result) : "l"(ptr));
    return result;
}

// ============================================================
// Test B: produce known-tagged pointers via cvta.*
// ============================================================
__device__ __noinline__ unsigned long long make_shared_generic(unsigned int offset) {
    unsigned long long r;
    asm("cvt.u64.u32 %0, %1; cvta.shared.u64 %0, %0;" : "=l"(r) : "r"(offset));
    return r;
}

__device__ __noinline__ unsigned long long make_local_generic(void *p) {
    unsigned long long r;
    asm("cvta.local.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

__device__ __noinline__ unsigned long long make_global_generic(const void *p) {
    unsigned long long r;
    asm("cvta.global.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

// ============================================================
// Test C: manually construct generic pointers with known upper bits
//   and test isspacep on them
// ============================================================
__device__ __noinline__ unsigned long long make_ptr_with_upper(unsigned int upper32, unsigned int lower32) {
    unsigned long long r;
    asm volatile(
        "cvt.u64.u32 %0, %2;"
        "cvt.u64.u32 %1, %1;"
        "shl.b64 %1, %1, 32;"
        "add.u64 %0, %1, %0;"
        : "=l"(r) : "r"(upper32), "r"(lower32));
    return r;
}

__device__ __noinline__ unsigned long long make_ptr_with_tag(unsigned int tag, unsigned int offset) {
    unsigned long long r;
    unsigned long long lo;
    asm volatile(
        "cvt.u64.u32 %0, %2;"
        "cvt.u64.u32 %1, %3;"
        "shl.b64 %0, %0, 32;"
        "add.u64 %0, %0, %1;"
        : "=&l"(r), "=l"(lo) : "r"(tag), "r"(offset));
    return r;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_isspacep(int *out, int *in, int N) {
    int local_val = 0;
    int tid = threadIdx.x;

    int *global_ptr = &in[tid];
    unsigned int shared_off = (unsigned int)(tid * 4);

    // --- Test A: classify known-tagged pointers ---
    unsigned long long gen_shared = make_shared_generic(shared_off);
    unsigned long long gen_global  = make_global_generic((const void*)global_ptr);
    unsigned long long gen_local   = make_local_generic((void*)&local_val);

    // Store raw values so we can see bit patterns in PTX/SASS
    g_storage[0] = gen_shared;
    g_storage[1] = gen_global;
    g_storage[2] = gen_local;

    int is_sh  = is_shared_space(gen_shared);
    int is_sh2 = is_shared_space(gen_global);
    int is_sh3 = is_shared_space(gen_local);

    int is_gl  = is_global_space(gen_shared);
    int is_gl2 = is_global_space(gen_global);
    int is_gl3 = is_global_space(gen_local);

    int is_lo  = is_local_space(gen_shared);
    int is_lo2 = is_local_space(gen_global);
    int is_lo3 = is_local_space(gen_local);

    // --- Test C: manually constructed pointers ---
    // Try upper32 = 0 (should be global), 1, 2, 0xFF, 0xFFFF
    unsigned long long manual_ptrs[6];
    unsigned int upper_tags[] = {0, 1, 2, 0xFF, 0xFFFF, 0x7FFFFFFF};
    for (int i = 0; i < 6; i++) {
        manual_ptrs[i] = make_ptr_with_tag(upper_tags[i], 0x100);
    }

    int manual_sh[6], manual_gl[6], manual_lo[6];
    for (int i = 0; i < 6; i++) {
        manual_sh[i] = is_shared_space(manual_ptrs[i]);
        manual_gl[i] = is_global_space(manual_ptrs[i]);
        manual_lo[i] = is_local_space(manual_ptrs[i]);
    }

    // --- Store results to global memory ---
    // (prevents optimization, and nvdisasm can show constant propagation)
    if (tid == 0) {
        out[0]  = is_sh;
        out[1]  = is_sh2;
        out[2]  = is_sh3;
        out[3]  = is_gl;
        out[4]  = is_gl2;
        out[5]  = is_gl3;
        out[6]  = is_lo;
        out[7]  = is_lo2;
        out[8]  = is_lo3;
        for (int i = 0; i < 6; i++) {
            out[10 + i] = manual_sh[i];
            out[16 + i] = manual_gl[i];
            out[22 + i] = manual_lo[i];
        }
    }
}
