/**
 * test_register_pressure_spill.cu — Phase 3.3
 *
 * Escalate register demand to force spills into local memory.
 * Observing the spill point reveals the per-thread register budget
 * and how spill slots are allocated in the stack frame.
 */
#include <cstdio>

__device__ unsigned long long g_dump[64];

// ============================================================
// Escalating register pressure: 16, 32, 64, 128 live ints
// Each kernel keeps ALL ints live until the final computation.
// ============================================================

__global__ void kr_16_live(int *out) {
    int v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, v6=6, v7=7;
    int v8=8, v9=9, v10=10, v11=11, v12=12, v13=13, v14=14, v15=15;
    out[0] = v0+v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15;
}

__global__ void kr_32_live(int *out) {
    int v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, v6=6, v7=7;
    int v8=8, v9=9, v10=10, v11=11, v12=12, v13=13, v14=14, v15=15;
    int v16=16,v17=17,v18=18,v19=19,v20=20,v21=21,v22=22,v23=23;
    int v24=24,v25=25,v26=26,v27=27,v28=28,v29=29,v30=30,v31=31;
    out[0] = v0+v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15
           + v16+v17+v18+v19+v20+v21+v22+v23+v24+v25+v26+v27+v28+v29+v30+v31;
}

__global__ void kr_48_live(int *out) {
    int v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, v6=6, v7=7;
    int v8=8, v9=9, v10=10, v11=11, v12=12, v13=13, v14=14, v15=15;
    int v16=16,v17=17,v18=18,v19=19,v20=20,v21=21,v22=22,v23=23;
    int v24=24,v25=25,v26=26,v27=27,v28=28,v29=29,v30=30,v31=31;
    int v32=32,v33=33,v34=34,v35=35,v36=36,v37=37,v38=38,v39=39;
    int v40=40,v41=41,v42=42,v43=43,v44=44,v45=45,v46=46,v47=47;
    out[0] = v0+v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15
           + v16+v17+v18+v19+v20+v21+v22+v23+v24+v25+v26+v27+v28+v29+v30+v31
           + v32+v33+v34+v35+v36+v37+v38+v39+v40+v41+v42+v43+v44+v45+v46+v47;
}

__global__ void kr_64_live(int *out) {
    int v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, v6=6, v7=7;
    int v8=8, v9=9, v10=10, v11=11, v12=12, v13=13, v14=14, v15=15;
    int v16=16,v17=17,v18=18,v19=19,v20=20,v21=21,v22=22,v23=23;
    int v24=24,v25=25,v26=26,v27=27,v28=28,v29=29,v30=30,v31=31;
    int v32=32,v33=33,v34=34,v35=35,v36=36,v37=37,v38=38,v39=39;
    int v40=40,v41=41,v42=42,v43=43,v44=44,v45=45,v46=46,v47=47;
    int v48=48,v49=49,v50=50,v51=51,v52=52,v53=53,v54=54,v55=55;
    int v56=56,v57=57,v58=58,v59=59,v60=60,v61=61,v62=62,v63=63;
    out[0] = v0+v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15
           + v16+v17+v18+v19+v20+v21+v22+v23+v24+v25+v26+v27+v28+v29+v30+v31
           + v32+v33+v34+v35+v36+v37+v38+v39+v40+v41+v42+v43+v44+v45+v46+v47
           + v48+v49+v50+v51+v52+v53+v54+v55+v56+v57+v58+v59+v60+v61+v62+v63;
}

__global__ void kr_96_live(int *out) {
    // 96 live ints — should exceed register budget and spill
    int v[96];
    for (int i = 0; i < 96; i++) v[i] = i;
    int sum = 0;
    for (int i = 0; i < 96; i++) sum += v[i];
    out[0] = sum;
}

__global__ void kr_128_live(int *out) {
    int v[128];
    for (int i = 0; i < 128; i++) v[i] = i;
    int sum = 0;
    for (int i = 0; i < 128; i++) sum += v[i];
    out[0] = sum;
}

// ============================================================
// Test with __noinline__ calls + register pressure
// ============================================================
__device__ __noinline__ int heavy_callee(int a, int b, int c, int d) {
    int v0=a, v1=b, v2=c, v3=d;
    int v4=v0+v1, v5=v2+v3;
    int v6=v4+v5;
    int v7=v6+v0;
    return v7;
}

__global__ void kr_with_calls(int *out) {
    int a = threadIdx.x;
    // Force many live values across calls
    int r1 = heavy_callee(a, a+1, a+2, a+3);
    int r2 = heavy_callee(a+1, a+2, a+3, a+4);
    int r3 = heavy_callee(a+2, a+3, a+4, a+5);
    int r4 = heavy_callee(a+3, a+4, a+5, a+6);
    out[0] = r1 + r2 + r3 + r4;
}
