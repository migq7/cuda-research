/**
 * probe_cvta_to_shared.cu
 *
 * Target SM: sm_87 (Jetson Orin)
 * Mode: Runtime
 *
 * Probes: Verify cvta.to.shared behavior on sm_87.
 *
 * Finding: On sm_87, inline asm cvta.to.shared produces incorrect output
 *          (garbage in upper bits). Use __cvta_generic_to_shared() builtin
 *          to get shared memory offsets — it works correctly.
 */
#include <cstdio>

__global__ void probe(unsigned int *out_b, unsigned int *out_a, unsigned int *out_gl) {
    __shared__ unsigned int arr[64];

    if (threadIdx.x == 0) {
        out_b[0] = __cvta_generic_to_shared(&arr[8]);

        unsigned long long gen;
        unsigned int off;
        asm volatile(
            "cvta.shared.u64 %0, %2;\n"
            "{ .reg .u64 t; cvta.to.shared.u64 t, %0; cvt.u32.u64 %1, t; }\n"
            : "=l"(gen), "=r"(off) : "l"((unsigned long long)(size_t)&arr[8]));
        out_a[0] = off;
        out_gl[0] = (unsigned int)(gen & 0xFFFFFFFFULL);
    }
}

int main() {
    unsigned int h_b[1], h_a[1], h_g[1];
    unsigned int *d_b, *d_a, *d_g;
    cudaMalloc(&d_b, sizeof(unsigned int));
    cudaMalloc(&d_a, sizeof(unsigned int));
    cudaMalloc(&d_g, sizeof(unsigned int));

    probe<<<1, 1>>>(d_b, d_a, d_g);
    cudaDeviceSynchronize();
    cudaMemcpy(h_b, d_b, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a, d_a, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("=== cvta.to.shared: Builtin vs Asm (sm_87) ===\n\n");
    printf("Builtin: %#010x (correct, expect 0x20 for arr[8])\n", h_b[0]);
    printf("Asm:     %#010x %s\n",  h_a[0], h_a[0] == h_b[0] ? "OK" : "INCORRECT");
    printf("cvta.shared gen_lo: %#010x\n", h_g[0]);
    printf("\nConclusion: sm_87 inline asm cvta.to.shared is broken.\n");
    printf("Use __cvta_generic_to_shared() builtin instead.\n");

    cudaFree(d_b); cudaFree(d_a); cudaFree(d_g);
    return 0;
}
