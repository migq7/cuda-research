# AGENTS.md

## Environment

- Docker image: `nvidia/cuda:12.9.1-cudnn-devel-ubi8`
- **No physical GPU** — all analysis is done via *offline* disassembly (PTX, SASS, cubin). Never try to run kernels.
- NVCC at `/usr/local/cuda/bin/nvcc`, targets `sm_90`.

## What this repo is

A workspace for studying NVIDIA GPU architecture through offline disassembly. New analysis sub-projects may be added over time.

Three CUDA analysis projects studying how NVCC (CUDA 12.9) compiles PTX/SASS for sm_90 (Hopper):

- **`01-shared-ptr-analysis/`** — How `__device__ __noinline__` shared memory pointer parameters are compiled (generic vs. space-specific ld/st)
- **`02-cvta-analysis/`** — How `cvta.*` PTX address-space-conversion instructions lower to SASS
- **`03-generic-address-space/`** — Complete sm_90 generic address space design: window layout, cross-thread/CTA visibility, runtime resource management

No build system, no tests, no lint. Each project has its own `scripts/compile.sh`.

## Compiling

```bash
# From each project root:
bash scripts/compile.sh
```

Each script assumes CUDA at `/usr/local/cuda/bin/` and targets `sm_90`. Output goes to `build/ptx/`, `build/sass/`, `build/cubin/`.

## Directory quirks

- `02-cvta-analysis/02-cvta-analysis/` — nested duplicate directory. The actual project root is `02-cvta-analysis/02-cvta-analysis/` (run compile.sh from there, or `scripts/compile.sh` from the outer `02-cvta-analysis/`).
- Each `.sass` file is a `cuobjdump -sass` disassembly from a `.cubin`; `.ptx` files come from `nvcc -ptx`.

## Report convention

All research projects produce two reports, kept in sync:
- `REPORT.md` — English (written first)
- `REPORT_zh.md` — Chinese (translated from English)

When adding new analysis or updating an existing report, always update both.

**Evidence discipline**: Every claim in a report must be traceable to observable SASS/PTX evidence. When presenting an interpretation that cannot be directly confirmed from disassembly alone (e.g., runtime register values, hardware fault behavior, per-thread semantics), mark it explicitly as an inference with the evidence it rests on — never state unconfirmed assumptions as facts. Static disassembly shows what instructions the compiler emits, not what happens at runtime.

## Key files per project

- `01-shared-ptr-analysis/README.md` — quickstart and conclusions
- `01-shared-ptr-analysis/REPORT.md` — full analysis with PTX/SASS excerpts
- `02-cvta-analysis/02-cvta-analysis/REPORT.md` — full analysis with SASS disassembly for each cvta variant
- `03-generic-address-space/REPORT.md` — full analysis (English)
- `03-generic-address-space/REPORT_zh.md` — full analysis (Chinese)

## PTX inline asm notes

- `lop3.b64` and `shf.l.wrap.*` are **not** valid PTX instructions in CUDA 12.9 sm_90. Use `and.b64`/`or.b64` and `shl.b64`/`shr.b64` instead.
- `cvta.const` and `cvta.to.const` are not supported by ptxas on sm_90.
- `__isspacep_shared/local/global()` intrinsics compile to `isspacep.*` PTX, which lowers to `QSPC.E.{S,L,G}` SASS instructions.
