/**
 * test_local_cross_thread_ptr.cu — Phase 2.1
 *
 * Thread 0 constructs a generic pointer to its local variable.
 * The pointer is broadcast to all threads via shared memory.
 * All threads (including non-0) call a __noinline__ function
 * that dereferences the pointer via st.local.
 *
 * Key question in SASS: Does the __noinline__ function use SR_TID.X
 * when converting the generic pointer back to a local address?
 * If yes → runtime enforces per-thread isolation.
 * If no → the pointer itself encodes thread identity,
 *         and cross-thread local access is architecturally possible.
 */
#include <cstdio>

// ============================================================
// Function that receives a generic pointer and writes via st.local
// ============================================================
__device__ __noinline__ void write_local_via_generic(unsigned long long gen_ptr, int val) {
    unsigned long long addr;
    asm("cvta.to.local.u64 %0, %1;" : "=l"(addr) : "l"(gen_ptr));
    asm("st.local.u32 [%0], %1;" :: "l"(addr), "r"(val));
}

// ============================================================
// Control: store to own local var (same thread)
// ============================================================
__device__ __noinline__ void write_own_local(void *local_ptr, int val) {
    unsigned long long gen;
    asm("cvta.local.u64 %0, %1;" : "=l"(gen) : "l"((unsigned long long)local_ptr));
    write_local_via_generic(gen, val);
}

// ============================================================
// Cross-thread: write to a generic pointer from another thread
// ============================================================
__device__ __noinline__ void write_foreign_local(unsigned long long gen_ptr, int val) {
    // This receives a generic ptr that originated in thread 0
    write_local_via_generic(gen_ptr, val);
}

// ============================================================
// Read-only version (observe IPO behavior)
// ============================================================
__device__ __noinline__ int read_local_via_generic(unsigned long long gen_ptr) {
    unsigned long long addr;
    asm("cvta.to.local.u64 %0, %1;" : "=l"(addr) : "l"(gen_ptr));
    int result;
    asm("ld.local.u32 %0, [%1];" : "=r"(result) : "l"(addr));
    return result;
}

// ============================================================
// Kernel
// ============================================================
__global__ void kernel_local_cross_thread(int *out, int N) {
    __shared__ unsigned long long shared_gen_ptr;
    __shared__ int shared_before;
    int local_val = threadIdx.x + 1000;  // unique per thread
    int tid = threadIdx.x;

    // --- Thread 0: construct generic local ptr and publish ---
    if (tid == 0) {
        unsigned long long gen;
        asm("cvta.local.u64 %0, %1;" : "=l"(gen) : "l"((unsigned long long)&local_val));
        shared_gen_ptr = gen;        // broadcast to all threads
        shared_before = local_val;   // capture value before write
    }

    __syncthreads();

    // --- All threads: write to thread 0's local space ---
    // This is the key probe. If local memory is per-thread isolated,
    // the hardware should route the st.local to thread 0's stack.
    unsigned long long t0_gen_ptr = shared_gen_ptr;
    write_foreign_local(t0_gen_ptr, tid * 100);  // each thread writes different value

    // --- Read-back attempt ---
    // Thread 0 reads its own local var (may have been overwritten by itself)
    // Other threads read via the shared generic pointer
    int readback;
    if (tid == 0) {
        readback = local_val;  // direct read
    } else {
        readback = read_local_via_generic(t0_gen_ptr);  // cross-thread read
    }

    // --- Thread 0 also reads the shared generic ptr (self-read) ---
    int self_read;
    if (tid == 0) {
        // Wait for all writes
        __threadfence_block();
        self_read = read_local_via_generic(t0_gen_ptr);
    } else {
        self_read = 0;
    }

    __syncthreads();

    if (tid < N) {
        out[tid] = readback + self_read + shared_before;
    }
}
