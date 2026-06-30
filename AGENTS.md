# AGENTS.md

## Environment

- Docker image: `nvidia/cuda:12.9.1-cudnn-devel-ubi8`
- **No physical GPU** — all analysis is done via *offline* disassembly (PTX, SASS, cubin). Never try to run kernels.
- NVCC at `/usr/local/cuda/bin/nvcc`, targets `sm_90`.

**First action on every session** — check the actual environment before doing anything:

```bash
# 1. Is CUDA toolkit present?
which nvcc && nvcc --version

# 2. Is there a GPU runtime? (determines if we can run kernels)
nvidia-smi 2>/dev/null || echo "No GPU runtime"

# 3. What SMs does this NVCC support?
nvcc --list-gpu-arch 2>/dev/null || echo "nvcc does not support --list-gpu-arch"

# 4. What SMs does the GPU hardware support? (if GPU present)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null
```

The answers determine:

| Capability available | What we can do |
|---|---|
| NVCC only, no GPU | Offline disassembly only (PTX/SASS analysis) |
| NVCC + GPU | Offline analysis + **runtime probes** (compile and run `.cu` files) |

Runtime probes must target the **GPU's actual SM version** (from `nvidia-smi`), not a hardcoded default.

## What this repo is

A workspace for studying NVIDIA GPU architecture through offline disassembly. New analysis sub-projects may be added over time.

Three CUDA analysis projects studying how NVCC (CUDA 12.9) compiles PTX/SASS for sm_90 (Hopper):

- **`01-shared-ptr-analysis/`** — How `__device__ __noinline__` shared memory pointer parameters are compiled (generic vs. space-specific ld/st)
- **`02-cvta-analysis/`** — How `cvta.*` PTX address-space-conversion instructions lower to SASS
- **`03-generic-address-space/`** — Complete sm_90 generic address space design: window layout, cross-thread/CTA visibility, runtime resource management

No build system, no tests, no lint. Each project has its own `scripts/compile.sh`.

## Adding a new research project

Every new sub-project follows this structure:

```
NN-project-name/
├── scripts/
│   └── compile.sh              # Builds all .cu → .ptx /.sass /.cubin
├── src/
│   ├── test_foo.cu             # One probe per concern
│   └── test_bar.cu
├── build/                      # Generated (git-ignored)
│   ├── ptx/
│   ├── sass/
│   └── cubin/
├── runtime-probes/             # Optional: runnable tests (if GPU available)
│   ├── Makefile                # SM ?= <target>, defaults to GPU's compute cap
│   └── 01-something/
│       ├── Makefile
│       └── probe_xxx.cu
├── REPORT.md                   # English (written first)
└── REPORT_zh.md                # Chinese (synced with English)
```

**Source annotation rule** — every `.cu` file must include a header comment that states:

1. **Target SM version(s)** the test was written for
2. **Analysis mode**: `Disassembly-only` (offline PTX/SASS) or `Runtime` (runs on hardware)
3. **What question the test probes**

Template:

```c
/**
 * test_xxx.cu
 *
 * Target SM: sm_90
 * Mode: Disassembly-only (compile with scripts/compile.sh, inspect build/sass/)
 *
 * Probes: <one-line description of what this test investigates>
 */
```

For runtime probes, add:

```c
/**
 * probe_xxx.cu
 *
 * Target SM: sm_87 (Jetson Orin)
 * Mode: Runtime (compile with Makefile, run on hardware)
 *
 * Probes: <one-line description>
 */
```

## Compiling

```bash
# Disassembly analysis (from each project root):
bash scripts/compile.sh

# Runtime probes (from runtime-probes/):
make          # uses SM from Makefile or environment
make SM=sm_90 # override SM
```

Each compile script assumes CUDA at `/usr/local/cuda/bin/`. Output goes to `build/ptx/`, `build/sass/`, `build/cubin/`.

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
