/**
 * test_generic_ptr_surgery.cu — Phase 1.2
 *
 * Take a known-valid generic pointer (from cvta.shared/cvta.local),
 * surgically modify bits via inline asm, then dereference with
 * space-specific PTX instructions (ld.shared, st.shared, etc.).
 *
 * The SASS reveals:
 *   - Does ptxas reject space-specific ld/st with modified ptrs?
 *   - Does it switch to generic LD.E/ST.E when bits are modified?
 *   - Which bit modifications cause the compiler to lose trust?
 */
#include <cstdio>

// ============================================================
// Control: unmodified shared generic ptr — space-specific store
// ============================================================
__device__ __noinline__ void write_shared_clean(int *p, int val) {
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"((unsigned long long)p));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Surgery A: clear upper byte of shared generic ptr
//   Use and.b64 with mask 0x00FFFFFFFFFFFFFF to zero bits [63:56]
// ============================================================
__device__ __noinline__ void write_shared_clear_upper(int *p, int val) {
    unsigned long long ptr = (unsigned long long)p;
    unsigned long long masked;
    // Mask = 0x00FFFFFFFFFF — clear upper byte
    unsigned long long mask;
    asm("mov.u64 %0, 0x00FFFFFFFFFFFFFF;" : "=l"(mask));
    asm("and.b64 %0, %1, %2;" : "=l"(masked) : "l"(ptr), "l"(mask));
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %0; cvt.u32.u64 %1, tmp; }"
        : "+l"(masked), "=r"(addr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Surgery B: set bit 32 (boundary between lower/upper 32)
//   Use or.b64 with 0x100000000 to set bit 32
// ============================================================
__device__ __noinline__ void write_shared_set_bit32(int *p, int val) {
    unsigned long long ptr = (unsigned long long)p;
    unsigned long long modified;
    unsigned long long bit32;
    asm("mov.u64 %0, 0x100000000;" : "=l"(bit32));
    asm("or.b64 %0, %1, %2;" : "=l"(modified) : "l"(ptr), "l"(bit32));
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %0; cvt.u32.u64 %1, tmp; }"
        : "+l"(modified), "=r"(addr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Surgery C: replace upper 32 bits with a constant (local-like tag)
//   Extract lower 32 bits, zero-clear upper 32, OR with tag
// ============================================================
__device__ __noinline__ void write_shared_inject_local_upper(int *p, int val) {
    unsigned long long ptr = (unsigned long long)p;

    // Extract lower 32 bits
    unsigned int lo;
    asm("cvt.u32.u64 %0, %1;" : "=r"(lo) : "l"(ptr));

    // Build 64-bit from lo (effectively zero-extending)
    unsigned long long lo64;
    asm("cvt.u64.u32 %0, %1;" : "=l"(lo64) : "r"(lo));

    // Build upper: tag << 32
    unsigned long long tag64;
    unsigned int tag = 0x01000000;
    asm("cvt.u64.u32 %0, %1;" : "=l"(tag64) : "r"(tag));
    asm("shl.b64 %0, %0, 32;" : "+l"(tag64));

    // Combine: tag_upper | lo
    unsigned long long reconstructed;
    asm("or.b64 %0, %1, %2;" : "=l"(reconstructed) : "l"(tag64), "l"(lo64));

    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %0; cvt.u32.u64 %1, tmp; }"
        : "+l"(reconstructed), "=r"(addr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Surgery D: local ptr — zero upper bits, try st.shared
// ============================================================
__device__ __noinline__ void write_local_as_shared(void *p, int val) {
    unsigned long long ptr = (unsigned long long)p;

    // Extract lower 32 bits (strip local tag)
    unsigned int lo;
    asm("cvt.u32.u64 %0, %1;" : "=r"(lo) : "l"(ptr));

    // Zero-extend to 64-bit (creates a global-like pointer)
    unsigned long long global_like;
    asm("cvt.u64.u32 %0, %1;" : "=l"(global_like) : "r"(lo));

    // Try to store via st.shared using this modified ptr
    unsigned int addr;
    asm volatile(
        "{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %0; cvt.u32.u64 %1, tmp; }"
        : "+l"(global_like), "=r"(addr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(val));
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_ptr_surgery(int *out, int *in, int N) {
    __shared__ int smem[256];
    int local_val = 0;
    int tid = threadIdx.x;

    if (tid < N) smem[tid] = in[tid];
    __syncthreads();

    // Test 1: clean shared access (control)
    write_shared_clean(&smem[tid], 1);

    // Test 2: clear upper byte of shared generic ptr
    write_shared_clear_upper(&smem[tid], 2);

    // Test 3: set bit 32
    write_shared_set_bit32(&smem[tid], 3);

    // Test 4: inject local-like upper bits into shared ptr
    write_shared_inject_local_upper(&smem[tid], 4);

    // Test 5: treat local ptr as shared
    write_local_as_shared((void*)&local_val, 5);

    __syncthreads();

    if (tid < N) {
        out[tid] = smem[tid] + local_val;
    }
}
