#!/bin/bash
#
# compile.sh — 编译所有测试用例到 PTX 和 SASS，用于分析
#
# 用法: bash compile.sh
#

set -uo pipefail
# Note: NOT using -e so we can continue on individual compilation failures

# cd to project root (parent of scripts/)
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

CUDA_BIN="/usr/local/cuda/bin"
NVCC="${CUDA_BIN}/nvcc"
CUOBJDUMP="${CUDA_BIN}/cuobjdump"
NVDISASM="${CUDA_BIN}/nvdisasm"

ARCH="sm_90"              # 目标架构
SRCDIR="src"
OUTDIR="build"
PTXDIR="${OUTDIR}/ptx"
SASSDIR="${OUTDIR}/sass"
CUBINDIR="${OUTDIR}/cubin"

# 测试文件列表
TESTS=(
    "test_baseline"
    "test_multiple_spaces"
    "test_template"
    "test_ptr_intrinsics"
)

mkdir -p "${PTXDIR}" "${SASSDIR}" "${CUBINDIR}"

echo "============================================"
echo "  CUDA Shared Pointer Test Compilation"
echo "  Architecture: ${ARCH}"
echo "  NVCC: $(${NVCC} --version | head -1)"
echo "============================================"
echo ""

for test in "${TESTS[@]}"; do
    src="${SRCDIR}/${test}.cu"
    if [[ ! -f "${src}" ]]; then
        echo "[SKIP] ${src} not found"
        continue
    fi

    echo "----------------------------------------"
    echo "  Compiling: ${src}"
    echo "----------------------------------------"

    # ---- 1. 编译到 PTX ----
    # PTX 是虚拟 ISA，可以看到编译器意图
    ptx_out="${PTXDIR}/${test}.ptx"
    echo "  -> PTX: ${ptx_out}"
    ${NVCC} -arch="${ARCH}" -ptx -O3 -o "${ptx_out}" "${src}" 2>&1 || {
        echo "  [WARN] PTX compilation failed, trying without -O3"
        ${NVCC} -arch="${ARCH}" -ptx -o "${ptx_out}" "${src}" 2>&1
    }

    # ---- 2. 编译到 cubin (SASS) ----
    # cubin 包含机器码，需要用 cuobjdump 提取 SASS
    cubin_out="${CUBINDIR}/${test}.cubin"
    echo "  -> CUBIN: ${cubin_out}"
    ${NVCC} -arch="${ARCH}" -cubin -O3 -o "${cubin_out}" "${src}" 2>&1 || {
        echo "  [WARN] CUBIN compilation failed, trying without -O3"
        ${NVCC} -arch="${ARCH}" -cubin -o "${cubin_out}" "${src}" 2>&1
    }

    # ---- 3. 从 cubin 提取 SASS 反汇编 ----
    sass_out="${SASSDIR}/${test}.sass"
    if [[ -f "${cubin_out}" ]]; then
        echo "  -> SASS: ${sass_out}"
        ${CUOBJDUMP} -sass "${cubin_out}" > "${sass_out}" 2>&1 || {
            echo "  [WARN] cuobjdump failed, trying nvdisasm"
            ${NVDISASM} -ndf "${cubin_out}" > "${sass_out}" 2>&1
        }
    fi

    echo ""
done

echo "============================================"
echo "  Compilation complete!"
echo ""
echo "  Output directories:"
echo "    PTX:   ${PTXDIR}/"
echo "    SASS:  ${SASSDIR}/"
echo "    CUBIN: ${CUBINDIR}/"
echo ""
echo "  Next steps:"
echo "    1. Inspect PTX for generic ld vs ld.shared/ld.global"
echo "       grep -n 'ld\.' ${PTXDIR}/*.ptx"
echo "    2. Inspect SASS for LDS vs LDG vs LD instructions"
echo "       grep -n 'LDS\|LDG\| LD ' ${SASSDIR}/*.sass"
echo "    3. Check if __noinline__ functions exist as separate entries"
echo "       grep -n '\.entry\|\.func' ${PTXDIR}/*.ptx"
echo "============================================"
