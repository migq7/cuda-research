#!/bin/bash
#
# compile.sh — 编译 cvta 测试用例到 PTX 和 SASS
#
# 用法: bash scripts/compile.sh
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
    "cvta_to_shared"
    "cvta_to_global"
    "cvta_shared_to_generic"
    "cvta_local"
    "cvta_all"
)

mkdir -p "${PTXDIR}" "${SASSDIR}" "${CUBINDIR}"

echo "============================================"
echo "  CVTA Instruction Analysis — sm_90"
echo "  NVCC: $(${NVCC} --version | head -1)"
echo "============================================"
echo ""

for test in "${TESTS[@]}"; do
    src="${SRCDIR}/${test}.cu"
    if [[ ! -f "${src}" ]]; then
        echo "[SKIP] ${src} not found"
        continue
    fi

    echo "--- ${src} ---"

    # PTX
    ptx_out="${PTXDIR}/${test}.ptx"
    echo "  PTX -> ${ptx_out}"
    ${NVCC} -arch="${ARCH}" -ptx -O3 -o "${ptx_out}" "${src}" 2>&1 || {
        echo "  [WARN] PTX failed, retry without -O3"
        ${NVCC} -arch="${ARCH}" -ptx -o "${ptx_out}" "${src}" 2>&1
    }

    # CUBIN
    cubin_out="${CUBINDIR}/${test}.cubin"
    echo "  CUBIN -> ${cubin_out}"
    ${NVCC} -arch="${ARCH}" -cubin -O3 -o "${cubin_out}" "${src}" 2>&1 || {
        echo "  [WARN] CUBIN failed, retry without -O3"
        ${NVCC} -arch="${ARCH}" -cubin -o "${cubin_out}" "${src}" 2>&1
    }

    # SASS
    sass_out="${SASSDIR}/${test}.sass"
    if [[ -f "${cubin_out}" ]]; then
        echo "  SASS -> ${sass_out}"
        ${CUOBJDUMP} -sass "${cubin_out}" > "${sass_out}" 2>&1 || {
            ${NVDISASM} -ndf "${cubin_out}" > "${sass_out}" 2>&1
        }
    fi

    echo ""
done

echo "============================================"
echo "  Done. Analyze with:"
echo "    # Search PTX for cvta instructions"
echo "    grep -n 'cvta\.' build/ptx/*.ptx"
echo ""
echo "    # Search SASS for cvta-like instructions"
echo "    grep -in 'cvta\|LDS\|LDG\|LDL\|STS\|STG\|STL\|LEA\|IMAD\|SHF\|LOP3' build/sass/*.sass | head -100"
echo "============================================"
