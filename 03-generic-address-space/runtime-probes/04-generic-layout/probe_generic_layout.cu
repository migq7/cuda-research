/**
 * probe_generic_layout.cu
 *
 * Target SM: sm_87 (Jetson Orin)
 * Mode: Runtime
 *
 * Probes: What is the actual layout of different address spaces in the
 *         64-bit generic address space on sm_87?
 *
 * For each space (global, shared, local, constant), construct a generic
 * pointer via cvta.* and export the full 64-bit value. Compare upper and
 * lower 32 bits to determine:
 *   - Relative ordering (which space is at higher/lower addresses)
 *   - Whether spaces occupy non-overlapping ranges
 *   - Where the global threshold lies
 */
#include <cstdio>

__constant__ int c_const_data[64];
__constant__ float c_const_data2[32];  // second array at offset 256 from first

__global__ void probe(unsigned long long *out_gen,
                      unsigned int *out_lo,
                      unsigned int *out_hi,
                      int *out_in, int N) {
    __shared__ int smem[256];
    int local_val = threadIdx.x;
    int tid = threadIdx.x;

    smem[tid] = out_in[tid];
    __syncthreads();

    if (tid == 0) {
        unsigned long long gen_global, gen_shared, gen_local, gen_const0, gen_const1;

        asm volatile("cvta.global.u64 %0, %1;"
                     : "=l"(gen_global) : "l"((unsigned long long)out_in));
        asm volatile("cvta.shared.u64 %0, %1;"
                     : "=l"(gen_shared) : "l"((unsigned long long)(size_t)&smem[0]));
        asm volatile("cvta.local.u64 %0, %1;"
                     : "=l"(gen_local) : "l"((unsigned long long)(size_t)&local_val));
        asm volatile("cvta.const.u64 %0, %1;"
                     : "=l"(gen_const0) : "l"((unsigned long long)(size_t)&c_const_data[0]));
        asm volatile("cvta.const.u64 %0, %1;"
                     : "=l"(gen_const1) : "l"((unsigned long long)(size_t)&c_const_data2[0]));

        out_gen[0] = gen_global;  out_lo[0] = (unsigned int)(gen_global  & 0xFFFFFFFFULL); out_hi[0] = (unsigned int)(gen_global  >> 32);
        out_gen[1] = gen_shared;  out_lo[1] = (unsigned int)(gen_shared  & 0xFFFFFFFFULL); out_hi[1] = (unsigned int)(gen_shared  >> 32);
        out_gen[2] = gen_local;   out_lo[2] = (unsigned int)(gen_local   & 0xFFFFFFFFULL); out_hi[2] = (unsigned int)(gen_local   >> 32);
        out_gen[3] = gen_const0;  out_lo[3] = (unsigned int)(gen_const0  & 0xFFFFFFFFULL); out_hi[3] = (unsigned int)(gen_const0  >> 32);
        out_gen[4] = gen_const1;  out_lo[4] = (unsigned int)(gen_const1  & 0xFFFFFFFFULL); out_hi[4] = (unsigned int)(gen_const1  >> 32);
    }
}

int main() {
    const int N = 5;  // global, shared, local, const0, const1
    unsigned long long h_gen[N];
    unsigned int h_lo[N], h_hi[N];
    unsigned long long *d_gen;
    unsigned int *d_lo, *d_hi;
    int *d_in, *d_out;

    int h_in[256];
    for (int i = 0; i < 256; i++) h_in[i] = i;
    cudaMalloc(&d_in,  256 * sizeof(int));
    cudaMalloc(&d_out, 256 * sizeof(int));
    cudaMalloc(&d_gen, N * sizeof(unsigned long long));
    cudaMalloc(&d_lo,  N * sizeof(unsigned int));
    cudaMalloc(&d_hi,  N * sizeof(unsigned int));
    cudaMemcpy(d_in, h_in, 256 * sizeof(int), cudaMemcpyHostToDevice);

    probe<<<1, 32>>>(d_gen, d_lo, d_hi, d_out, 32);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gen, d_gen, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lo,  d_lo,  N * sizeof(unsigned int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hi,  d_hi,  N * sizeof(unsigned int),        cudaMemcpyDeviceToHost);

    const char *names[] = {"Global", "Shared", "Local", "Const[0]", "Const[1]"};

    printf("=== Generic Address Space Layout (sm_87) ===\n\n");
    printf("%-10s  %-20s  %-10s  %-10s\n", "Space", "generic_ptr (hex)", "upper32", "lower32");
    printf("----------  --------------------  ----------  ----------\n");
    for (int i = 0; i < N; i++) {
        printf("%-10s  0x%08x_%08x  0x%08x  0x%08x\n",
               names[i], h_hi[i], h_lo[i], h_hi[i], h_lo[i]);
    }

    // Sort by generic pointer value (descending)
    unsigned long long sorted_gen[5];
    const char *sorted_names[5];
    for (int i = 0; i < N; i++) { sorted_gen[i] = h_gen[i]; sorted_names[i] = names[i]; }
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < N; j++)
            if (sorted_gen[i] < sorted_gen[j]) {
                unsigned long long t = sorted_gen[i]; sorted_gen[i] = sorted_gen[j]; sorted_gen[j] = t;
                const char *tn = sorted_names[i]; sorted_names[i] = sorted_names[j]; sorted_names[j] = tn;
            }

    printf("\n=== Sorted by address (high to low) ===\n");
    for (int i = 0; i < N; i++) {
        unsigned int shi = (unsigned int)(sorted_gen[i] >> 32);
        unsigned int slo = (unsigned int)(sorted_gen[i] & 0xFFFFFFFFULL);
        printf("  %d. %-10s  0x%08x_%08x\n", i+1, sorted_names[i], shi, slo);
    }

    // === Verify cvta.const formula: gen = 2 * c[0xd0] + offset ===
    printf("\n=== cvta.const formula verification ===\n");
    unsigned long long gen0 = h_gen[3];
    unsigned long long gen1 = h_gen[4];

    // Difference between const[0] and const[1] should be the byte offset
    // c_const_data[64] = 256 bytes, so c_const_data2 is at offset 256 = 0x100
    unsigned long long diff = gen1 - gen0;
    printf("c_const_data2  - c_const_data[0] = 0x%016llx\n", (unsigned long long)diff);
    printf("Expected byte offset (64 ints):  0x%016llx (256)\n", (unsigned long long)256);
    printf("Match: %s\n\n", diff == 256 ? "YES" : "NO");

    // From formula: gen0 = 2 * c[0xd0] + 0, gen1 = 2 * c[0xd0] + 256
    // So c[0xd0] (low 32) = gen0_lo / 2
    unsigned int base_lo = h_lo[3] / 2;
    unsigned int base_hi = h_hi[3] / 2;
    printf("Derived c[0x0][0xd0] (from gen0 / 2):\n");
    printf("  lo = 0x%08x  hi = 0x%08x  → 0x%08x_%08x\n", base_lo, base_hi, base_hi, base_lo);

    // Verify: gen0_lo should equal 2 * base_lo
    printf("  gen0_lo = 0x%08x  →  2 * 0x%08x = 0x%08x  %s\n",
           h_lo[3], base_lo, base_lo * 2, h_lo[3] == base_lo * 2 ? "OK" : "FAIL");
    // Verify: gen1_lo should equal 2 * base_lo + 256
    printf("  gen1_lo = 0x%08x  →  2*0x%08x + 256 = 0x%08x  %s\n",
           h_lo[4], base_lo, base_lo * 2 + 256, h_lo[4] == base_lo * 2 + 256 ? "OK" : "FAIL");

    // Compare with global threshold boundary
    printf("\n=== Threshold check ===\n");
    printf("Global  gen = 0x%08x_%08x\n", h_hi[0], h_lo[0]);
    printf("Const   gen = 0x%08x_%08x\n", h_hi[3], h_lo[3]);
    printf("c[0xd0]       0x%08x_%08x\n", base_hi, base_lo);
    printf("Global < c[0xd0]: %s\n",
           (h_gen[0] < (((unsigned long long)base_hi << 32) | base_lo)) ? "YES" : "NO");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gen); cudaFree(d_lo); cudaFree(d_hi);
    return 0;
}
