/**
 * probe_local_memory.cu  (v2)
 *
 * Runtime verification of local memory addressing.
 * Build:  nvcc -arch=sm_87 -O3 -o probe_local_memory probe_local_memory.cu
 *
 * Tests T0–T3 only (avoids warp-divergence issues with 256 threads).
 * No __syncthreads() before independent global writes.
 */
#include <cstdio>

__global__ void probe(unsigned long long *g_gen, unsigned int *g_local,
                      unsigned int *g_hi, int *cross) {
    int a = 0xAAAA + threadIdx.x;
    int b = 0xBBBB + threadIdx.x;
    int tid = threadIdx.x;

    // Local-space addresses
    unsigned int addr_a = (unsigned int)(size_t)&a;
    unsigned int addr_b = (unsigned int)(size_t)&b;

    // Generic pointer via cvta.local
    unsigned long long gen_a, gen_b;
    asm volatile("cvta.local.u64 %0, %1;" : "=l"(gen_a) : "l"((unsigned long long)&a));
    asm volatile("cvta.local.u64 %0, %1;" : "=l"(gen_b) : "l"((unsigned long long)&b));

    unsigned int lo_a = (unsigned int)(gen_a & 0xFFFFFFFFULL);
    unsigned int lo_b = (unsigned int)(gen_b & 0xFFFFFFFFULL);
    unsigned int hi_a = (unsigned int)(gen_a >> 32);
    unsigned int hi_b = (unsigned int)(gen_b >> 32);

    // Store per-thread data (no __syncthreads needed — each thread writes
    // to its own slot)
    g_gen[tid]   = gen_a;
    g_local[tid] = addr_a;
    g_hi[tid]    = hi_a;

    // Cross-thread test
    __shared__ unsigned long long t0_ptr;
    if (tid == 0) t0_ptr = gen_a;
    __syncthreads();

    if (tid == 1) {
        // Decode T0's generic ptr
        unsigned long long decoded;
        asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(decoded) : "l"(t0_ptr));
        int val;
        asm volatile("ld.local.u32 %0, [%1];" : "=r"(val) : "r"((unsigned int)decoded));
        cross[0] = val;

        // Decode T1's own generic ptr
        asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(decoded) : "l"(gen_a));
        asm volatile("ld.local.u32 %0, [%1];" : "=r"(val) : "r"((unsigned int)decoded));
        cross[1] = val;
    }
}

int main() {
    const int N = 256;

    unsigned long long *d_gen, h_gen[N];
    unsigned int *d_local, h_local[N];
    unsigned int *d_hi, h_hi[N];
    int *d_cross, h_cross[4] = {0};

    cudaMalloc(&d_gen,   N * sizeof(unsigned long long));
    cudaMalloc(&d_local, N * sizeof(unsigned int));
    cudaMalloc(&d_hi,    N * sizeof(unsigned int));
    cudaMalloc(&d_cross, 4 * sizeof(int));

    cudaMemset(d_gen,   0, N * sizeof(unsigned long long));
    cudaMemset(d_local,  0, N * sizeof(unsigned int));
    cudaMemset(d_hi,    0, N * sizeof(unsigned int));
    cudaMemset(d_cross, 0, 4 * sizeof(int));

    probe<<<1, N>>>(d_gen, d_local, d_hi, d_cross);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gen,   d_gen,   N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_local, d_local, N * sizeof(unsigned int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hi,    d_hi,    N * sizeof(unsigned int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cross, d_cross, 4 * sizeof(int),                 cudaMemcpyDeviceToHost);

    // Print raw data for first 4 threads
    printf("=== Raw Data (first 4 threads) ===\n");
    for (int i = 0; i < 4; i++) {
        unsigned int lo = (unsigned int)(h_gen[i] & 0xFFFFFFFFULL);
        unsigned int hi = (unsigned int)(h_gen[i] >> 32);
        printf("  T%d: gen=0x%08x_%08x  local=0x%08x  hi=0x%08x\n",
               i, hi, lo, h_local[i], h_hi[i]);
    }

    // 1. Are gen_lo values uniform?
    printf("\n=== gen_lo uniformity ===\n");
    unsigned int lo0 = (unsigned int)(h_gen[0] & 0xFFFFFFFFULL);
    int ok = 1;
    for (int i = 1; i < N; i++) {
        unsigned int lo = (unsigned int)(h_gen[i] & 0xFFFFFFFFULL);
        if (lo != lo0) { ok = 0; printf("  T%d differs: 0x%08x vs 0x%08x\n", i, lo, lo0); break; }
    }
    printf("  gen_lo ALL EQUAL: %s (value = 0x%08x)\n", ok ? "YES" : "NO", lo0);

    // 2. Are gen_hi values uniform?
    printf("\n=== gen_hi uniformity ===\n");
    unsigned int hi0 = h_hi[0];
    ok = 1;
    for (int i = 1; i < N; i++) {
        if (h_hi[i] != hi0) { ok = 0; printf("  T%d differs: 0x%08x vs 0x%08x\n", i, h_hi[i], hi0); break; }
    }
    printf("  gen_hi ALL EQUAL: %s (value = 0x%08x)\n", ok ? "YES" : "NO", hi0);

    // 3. Are local addresses uniform?
    printf("\n=== local_addr uniformity ===\n");
    unsigned int la0 = h_local[0];
    ok = 1;
    for (int i = 1; i < N; i++) {
        if (h_local[i] != la0) { ok = 0; printf("  T%d differs: 0x%08x vs 0x%08x\n", i, h_local[i], la0); break; }
    }
    printf("  local_addr ALL EQUAL: %s (value = 0x%08x)\n", ok ? "YES" : "NO", la0);

    // 4. Window base = gen_lo - local_addr
    printf("\n=== Window Base (gen_lo - local_addr) ===\n");
    unsigned int wb0 = lo0 - la0;
    ok = 1;
    for (int i = 1; i < N; i++) {
        unsigned int lo = (unsigned int)(h_gen[i] & 0xFFFFFFFFULL);
        unsigned int wb = lo - h_local[i];
        if (wb != wb0) { ok = 0; printf("  T%d wb=0x%08x (expected 0x%08x)\n", i, wb, wb0); }
    }
    printf("  Window base uniform: %s (value = 0x%08x = %u)\n", ok ? "YES" : "NO", wb0, wb0);

    // 5. Cross-thread test
    printf("\n=== Cross-Thread Read ===\n");
    printf("  T1 read via T0 ptr: 0x%08x\n", h_cross[0]);
    printf("  T1 read via own ptr: 0x%08x\n", h_cross[1]);
    if (h_cross[0] == h_cross[1])
        printf("  => Same value: cross-thread access NOT possible (hardware isolates)\n");
    else
        printf("  => Different values: cross-thread access MAY work\n");

    // 6. Model conclusion
    printf("\n=== Model Conclusion ===\n");
    printf("  gen_lo uniform: %s\n", lo0 == (unsigned int)(h_gen[1] & 0xFFFFFFFFULL) ? "YES" : "NO");
    printf("  local_addr uniform: %s\n", la0 == h_local[1] ? "YES" : "NO");
    if (la0 == h_local[1])
        printf("  => R1 likely uniform; STL/LDL hardware adds per-thread offset.\n");
    else
        printf("  => R1 likely per-thread (c[0x28] delivers thread-specific value).\n");

    cudaFree(d_gen); cudaFree(d_local); cudaFree(d_hi); cudaFree(d_cross);
    return 0;
}
