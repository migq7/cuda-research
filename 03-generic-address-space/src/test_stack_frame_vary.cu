/**
 * test_stack_frame_vary.cu — Phase 3.2
 *
 * Reverse-engineer the stack frame layout by varying local variable
 * usage and observing the VIADD R1, R1, immediate pattern at kernel entry.
 *
 * Also observe: relationship between R1 initialization (c[0x28]) and
 * the local window base (c[0x20]).
 */
#include <cstdio>

__device__ unsigned long long g_dump[32];

// ============================================================
// Stack frame size sweep
// ============================================================

__global__ void kf_empty() {
    // No locals — shows bare minimum frame
    g_dump[0] = 0;
}

__global__ void kf_1_int() {
    int a = 1;
    g_dump[0] = a;
}

__global__ void kf_2_int() {
    int a = 1, b = 2;
    g_dump[0] = a + b;
}

__global__ void kf_4_int() {
    int a = 1, b = 2, c = 3, d = 4;
    g_dump[0] = a + b + c + d;
}

__global__ void kf_8_int() {
    int a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8;
    g_dump[0] = a + b + c + d + e + f + g + h;
}

__global__ void kf_16_int() {
    int v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, v6=6, v7=7;
    int v8=8, v9=9, v10=10, v11=11, v12=12, v13=13, v14=14, v15=15;
    g_dump[0]=v0+v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15;
}

__global__ void kf_32_int() {
    int v[32];
    for (int i = 0; i < 32; i++) v[i] = i;
    g_dump[0] = v[0] + v[31];
}

__global__ void kf_64_int() {
    int v[64];
    for (int i = 0; i < 64; i++) v[i] = i;
    g_dump[0] = v[0] + v[63];
}

__global__ void kf_256_int() {
    int v[256];
    for (int i = 0; i < 256; i++) v[i] = i;
    g_dump[0] = v[0] + v[255];
}

__global__ void kf_1024_int() {
    int v[1024];
    for (int i = 0; i < 1024; i++) v[i] = i;
    g_dump[0] = v[0] + v[1023];
}

// ============================================================
// Test with __noinline__ calls (forces frame + caller/callee conventions)
// ============================================================
__device__ __noinline__ int callee_no_locals(int x) {
    return x + 1;
}

__device__ __noinline__ int callee_with_locals(int x) {
    int a = x;
    int b = a + 1;
    int c = b + 1;
    return c;
}

__device__ __noinline__ int callee_with_array(int x) {
    int arr[64];
    for (int i = 0; i < 64; i++) arr[i] = x + i;
    return arr[63];
}

__global__ void kf_with_noinline_calls(int *out) {
    int a = threadIdx.x;
    int r1 = callee_no_locals(a);
    int r2 = callee_with_locals(a);
    int r3 = callee_with_array(a);
    if (threadIdx.x == 0) {
        out[0] = r1 + r2 + r3;
    }
}

// ============================================================
// Test local var address vs R1 relationship
// ============================================================
__device__ __noinline__ unsigned long long get_local_ptr(void *p) {
    unsigned long long r;
    asm("cvta.local.u64 %0, %1;" : "=l"(r) : "l"((unsigned long long)p));
    return r;
}

__global__ void kf_local_ptr_dump(int *out) {
    int a = 0xAAAA;
    int b = 0xBBBB;
    unsigned long long pa = get_local_ptr(&a);
    unsigned long long pb = get_local_ptr(&b);
    if (threadIdx.x == 0) {
        out[0] = (int)(pa & 0xFFFFFFFF);
        out[1] = (int)(pa >> 32);
        out[2] = (int)(pb & 0xFFFFFFFF);
        out[3] = (int)(pb >> 32);
    }
}
