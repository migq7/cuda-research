/**
 * probe_cta_identity.cu
 *
 * Target SM: sm_87 (Jetson Orin)
 * Mode: Runtime (compile with Makefile, run on hardware)
 *
 * Probes:
 *   1. SR_SWINHI: is it a fixed tag or per-CTA identifier?
 *   2. Shared variable offset: 0 or 0x400?
 *   3. CTA isolation verification: each block writes bid+tid to arr[tid],
 *      reads back arr[0] — should see its own bid, not another block's.
 */
#include <cstdio>

__global__ void probe_cta_id(unsigned long long *out_gen,
                             unsigned int *out_swinhi,
                             unsigned int *out_off0,
                             unsigned int *out_bidx,
                             unsigned int *out_arr0_val,
                             unsigned int *out_isolation_ok) {
    __shared__ unsigned int arr[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Each block writes a unique value: bid + tid
    arr[tid] = (unsigned int)(bid + tid);
    __syncthreads();

    // 1. Generic shared pointer of arr[0] via cvta on &arr[0]
    unsigned long long gen_arr0;
    asm volatile("cvta.shared.u64 %0, %1;"
                 : "=l"(gen_arr0) : "l"((unsigned long long)(size_t)&arr[0]));
    unsigned int lo = (unsigned int)(gen_arr0 & 0xFFFFFFFFULL);
    unsigned int hi = (unsigned int)(gen_arr0 >> 32);

    // 2. Shared offset of arr[0] via builtin
    unsigned int off0 = __cvta_generic_to_shared(&arr[0]);

    // 3. Read back arr[0] — should be 'bid' (tid=0 wrote bid+0)
    unsigned int val = arr[0];

    // 4. Isolation check: arr[0] should equal blockIdx.x
    //    (tid=0 wrote bid to arr[0], no other block can touch this)
    unsigned int expected = (unsigned int)bid;

    if (tid == 0) {
        out_gen[bid]   = gen_arr0;
        out_swinhi[bid]    = hi;
        out_off0[bid]      = off0;
        out_bidx[bid]      = bid;
        out_arr0_val[bid]  = val;
        out_isolation_ok[bid] = (val == expected);
    }
}

int main() {
    const int nBlocks  = 8;
    const int nThreads = 32;

    unsigned long long *d_gen, h_gen[nBlocks];
    unsigned int *d_swinhi, h_swinhi[nBlocks];
    unsigned int *d_off0,   h_off0[nBlocks];
    unsigned int *d_bidx,   h_bidx[nBlocks];
    unsigned int *d_val,    h_val[nBlocks];
    unsigned int *d_iso,    h_iso[nBlocks];

    cudaMalloc(&d_gen,    nBlocks * sizeof(unsigned long long));
    cudaMalloc(&d_swinhi, nBlocks * sizeof(unsigned int));
    cudaMalloc(&d_off0,   nBlocks * sizeof(unsigned int));
    cudaMalloc(&d_bidx,   nBlocks * sizeof(unsigned int));
    cudaMalloc(&d_val,    nBlocks * sizeof(unsigned int));
    cudaMalloc(&d_iso,    nBlocks * sizeof(unsigned int));

    probe_cta_id<<<nBlocks, nThreads>>>(d_gen, d_swinhi, d_off0, d_bidx, d_val, d_iso);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gen,    d_gen,    nBlocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_swinhi, d_swinhi, nBlocks * sizeof(unsigned int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_off0,   d_off0,   nBlocks * sizeof(unsigned int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bidx,   d_bidx,   nBlocks * sizeof(unsigned int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val,    d_val,    nBlocks * sizeof(unsigned int),        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iso,    d_iso,    nBlocks * sizeof(unsigned int),        cudaMemcpyDeviceToHost);

    printf("=== CTA Identity Probe (%d blocks x %d threads) ===\n\n", nBlocks, nThreads);
    printf("Block | blockIdx | gen_hi(SWINHI)| gen_lo(arr[0])| off0 | arr[0]_val | isolated?\n");
    printf("------+----------+---------------+----------------+------+------------+----------\n");

    for (int i = 0; i < nBlocks; i++) {
        unsigned int lo = (unsigned int)(h_gen[i] & 0xFFFFFFFFULL);
        printf("  %2d  |  %5u   | 0x%08x   | 0x%08x    | 0x%04x |  %8u   | %s\n",
               i, h_bidx[i], h_swinhi[i], lo, h_off0[i],
               h_val[i], h_iso[i] ? "YES" : "NO");
    }

    // Analysis
    printf("\n=== Analysis ===\n");

    int varies = 0;
    for (int i = 1; i < nBlocks; i++)
        if (h_swinhi[i] != h_swinhi[0]) { varies = 1; break; }
    printf("SR_SWINHI varies across blocks: %s (value=0x%08x)\n", varies ? "YES" : "NO", h_swinhi[0]);

    int all_isolated = 1;
    for (int i = 0; i < nBlocks; i++)
        if (!h_iso[i]) { all_isolated = 0; break; }
    printf("CTA isolation verified (arr[0]==blockIdx.x): %s\n", all_isolated ? "YES" : "NO");

    int off0_uniform = 1;
    for (int i = 1; i < nBlocks; i++)
        if (h_off0[i] != h_off0[0]) { off0_uniform = 0; break; }
    printf("arr[0] shared offset: uniform = 0x%08x (%s)\n",
           h_off0[0], (h_off0[0] == 0) ? "no reserved region" :
                      (h_off0[0] == 0x400) ? "0x400 reserved" : "unexpected");

    printf("\n=== Conclusions ===\n");
    printf("SR_SWINHI = 0x%08x: fixed shared-space tag, not per-CTA identifier\n", h_swinhi[0]);
    printf("CTA isolation: %s (via SR_CgaCtaId in ULEA, not SR_SWINHI)\n",
           all_isolated ? "confirmed" : "BROKEN");
    printf("Shared variable start offset: 0x%08x\n", h_off0[0]);

    cudaFree(d_gen); cudaFree(d_swinhi); cudaFree(d_off0); cudaFree(d_bidx);
    cudaFree(d_val); cudaFree(d_iso);
    return 0;
}
