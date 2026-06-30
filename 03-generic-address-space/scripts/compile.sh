#!/bin/bash
#
# compile.sh — compile all generic address space probe kernels to PTX and SASS
#
set -uo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

CUDA_BIN="/usr/local/cuda/bin"
NVCC="${CUDA_BIN}/nvcc"
CUOBJDUMP="${CUDA_BIN}/cuobjdump"
NVDISASM="${CUDA_BIN}/nvdisasm"

ARCH="sm_90"
SRCDIR="src"
OUTDIR="build"
PTXDIR="${OUTDIR}/ptx"
SASSDIR="${OUTDIR}/sass"
CUBINDIR="${OUTDIR}/cubin"

TESTS=(
    "test_isspacep_const_fold"
    "test_generic_ptr_surgery"
    "test_window_sizes"
    "test_local_cross_thread_ptr"
    "test_shared_cross_cta_ptr"
    "test_warp_uniform_vs_divergent"
    "test_kernel_param_map"
    "test_stack_frame_vary"
    "test_register_pressure_spill"
    "test_dynamic_shared_config"
    "test_shared_window_granularity"
    "test_mbarrier_shared_reserved"
)

mkdir -p "${PTXDIR}" "${SASSDIR}" "${CUBINDIR}"

echo "============================================"
echo "  Generic Address Space Analysis — sm_90"
echo "  NVCC: $(${NVCC} --version | head -1)"
echo "============================================"
echo ""

PASS=0
FAIL=0
SKIP=0

for test in "${TESTS[@]}"; do
    src="${SRCDIR}/${test}.cu"
    if [[ ! -f "${src}" ]]; then
        echo "[SKIP] ${src} not found"
        ((SKIP++))
        continue
    fi

    echo "--- ${src} ---"

    ptx_out="${PTXDIR}/${test}.ptx"
    echo "  PTX -> ${ptx_out}"
    if ${NVCC} -arch="${ARCH}" -ptx -O3 -o "${ptx_out}" "${src}" 2>&1; then
        : # ok
    else
        echo "  [WARN] PTX failed, retry without -O3"
        ${NVCC} -arch="${ARCH}" -ptx -o "${ptx_out}" "${src}" 2>&1 || {
            echo "  [FAIL] PTX compilation failed"
            ((FAIL++))
            continue
        }
    fi

    cubin_out="${CUBINDIR}/${test}.cubin"
    echo "  CUBIN -> ${cubin_out}"
    if ${NVCC} -arch="${ARCH}" -cubin -O3 -o "${cubin_out}" "${src}" 2>&1; then
        : # ok
    else
        echo "  [WARN] CUBIN failed, retry without -O3"
        ${NVCC} -arch="${ARCH}" -cubin -o "${cubin_out}" "${src}" 2>&1 || {
            echo "  [FAIL] CUBIN compilation failed"
            ((FAIL++))
            continue
        }
    fi

    sass_out="${SASSDIR}/${test}.sass"
    if [[ -f "${cubin_out}" ]]; then
        echo "  SASS -> ${sass_out}"
        ${CUOBJDUMP} -sass "${cubin_out}" > "${sass_out}" 2>&1 || {
            ${NVDISASM} -ndf "${cubin_out}" > "${sass_out}" 2>&1
        }
    fi

    ((PASS++))
    echo ""

done

echo "============================================"
echo "  Results: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo ""
echo "  Analyze with:"
echo "    # Search SASS for CVTA patterns"
echo "    grep -n 'cvta\|LDS\|LDG\|LDL\|STS\|STG\|STL\|LD\.E\|ST\.E\|SR_\|ULEA\|c\[' build/sass/*.sass | head -200"
echo ""
echo "    # Search PTX for isspacep and cvta instructions"
echo "    grep -n 'isspacep\|cvta\.' build/ptx/*.ptx"
echo "============================================"
