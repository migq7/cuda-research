# sm_90 Generic Address Space Design — Complete Analysis

**Date**: 2026-06-27
**Toolchain**: CUDA 12.9 (NVCC Build: V12.9.86)
**Target**: sm_90 (Hopper)
**Based on**: `03-generic-address-space/` test suite (10 probe kernels)

---

## Contents

1. [Executive Summary](#1-executive-summary)
2. [Generic Pointer Bit Layout](#2-generic-pointer-bit-layout)
   - 2.1. [Space Identification: `QSPC.E.*` Instructions](#21-space-identification-qspce-instructions)
   - 2.2. [64-bit Generic Pointer Structure](#22-64-bit-generic-pointer-structure)
   - 2.3. [Address Space Conversion Rules](#23-address-space-conversion-rules)
3. [Window Layout in Generic Address Space](#3-window-layout-in-generic-address-space)
   - 3.1. [Shared Memory Window](#31-shared-memory-window)
   - 3.2. [Local Memory Window](#32-local-memory-window)
   - 3.3. [Global Memory](#33-global-memory)
   - 3.4. [Constant Memory: Outside the Generic Address Space](#34-constant-memory-outside-the-generic-address-space)
   - 3.5. [Unified Window Layout Diagram](#35-unified-window-layout-diagram)
   - 3.6. [Generic Address Space from a Single CTA's Perspective](#36-generic-address-space-from-a-single-ctas-perspective)
4. [Cross-Thread and Cross-CTA Visibility](#4-cross-thread-and-cross-cta-visibility)
   - 4.1. [Local Memory: Architecturally NOT cross-thread accessible](#41-local-memory-architecturally-not-cross-thread-accessible)
   - 4.2. [Shared Memory: Architecturally NOT cross-CTA accessible](#42-shared-memory-architecturally-not-cross-cta-accessible)
   - 4.3. [`SR_SWINHI` — Constructor vs. Dereferencer](#43-sr_swinhi--constructor-vs-dereferencer)
   - 4.4. [Visibility Matrix](#44-visibility-matrix)
5. [Runtime Resource Management](#5-runtime-resource-management)
   - 5.1. [Constant Bank Layout](#51-constant-bank-layout)
   - 5.2. [Stack Frame Allocation](#52-stack-frame-allocation)
   - 5.3. [Shared Memory Window Formula](#53-shared-memory-window-formula)
   - 5.4. [Dynamic Shared Memory](#54-dynamic-shared-memory)
   - 5.5. [Special Register Catalog](#55-special-register-catalog)
6. [`CVTA` Is Not a Hardware Instruction](#6-cvta-is-not-a-hardware-instruction)
   - 6.1. [Pointer Surgery Resilience](#61-pointer-surgery-resilience)
7. [Compiler Strategy Under Generic Addressing](#7-compiler-strategy-under-generic-addressing)
   - 7.1. [The Root Cause: No Address-Space Qualifiers on Parameters](#71-the-root-cause-no-address-space-qualifiers-on-parameters)
   - 7.2. [Call-Site cvta Injection](#72-call-site-cvta-injection)
   - 7.3. [Inter-Procedural Optimization (IPO)](#73-inter-procedural-optimization-ipo)
   - 7.4. [Generic vs. Space-Specific Instruction Selection](#74-generic-vs-space-specific-instruction-selection)
   - 7.5. [No Per-Call-Site Function Cloning](#75-no-per-call-site-function-cloning)
   - 7.6. [Uniform Register Promotion](#76-uniform-register-promotion)
   - 7.7. [cvta Fusion at the Instruction Level](#77-cvta-fusion-at-the-instruction-level)
   - 7.8. [Summary: Compiler Decision Table](#78-summary-compiler-decision-table)
8. [Key Architectural Inferences](#8-key-architectural-inferences)
   - 8.1. [Why Cross-CTA Shared Access Is Not Possible](#81-why-cross-cta-shared-access-is-not-possible)
   - 8.2. [Why Local Memory Is Per-Thread](#82-why-local-memory-is-per-thread)
   - 8.3. [The `QSPC.E.G` Dual-Check Design](#83-the-qspceg-dual-check-design)
9. [Test File Inventory](#9-test-file-inventory)
10. [Open Questions](#10-open-questions)
11. [Runtime Verification (sm_87 Jetson Orin)](#11-runtime-verification-sm_87-jetson-orin)

---

## 1. Executive Summary

The sm_90 generic address space is a **tagged 64-bit pointer scheme** where the upper 32 bits encode an address space identifier (window ID), and the lower 32 bits encode the offset within that space. The hardware uses a dedicated instruction family (`QSPC.E.*`) to test pointer tags, and maps each space to a **window** in the virtual address space via per-CTA/per-thread window registers.

Key findings:
- **3 distinct spaces**: shared, local, global — each with unique tag encoding
- No `CVTA` SASS instruction exists — all conversions are arithmetic (ADD/SUB/MOV) or register reads
- Cross-thread local access and cross-CTA shared access are **not architecturally supported** via generic pointers
- Shared memory window granularity is fixed at 16 MB per CTA (`1 << 24`)
- Runtime configuration uses constant bank 0 for ABI constants and bank 4 for kernel arguments

---

## 2. Generic Pointer Bit Layout

### 2.1. Space Identification: `QSPC.E.*` Instructions

The PTX `isspacep.*` intrinsics lower to dedicated hardware space-query instructions:

| PTX Intrinsic | SASS Instruction | Operations |
|---------------|-----------------|------------|
| `isspacep.shared` | `QSPC.E.S P0, RZ, [Rptr]` | Single tag-bit test — sets predicate if shared space |
| `isspacep.local` | `QSPC.E.L P0, RZ, [Rptr]` | Single tag-bit test — sets predicate if local space |
| `isspacep.global` | `QSPC.E.G` + 64-bit range check | Tag test **OR** `(ptr >= threshold)` — dual condition |

**Critical finding for `isspacep.global`**: The SASS emits **two** tests OR-combined:
1. `QSPC.E.G` — tag bit test
2. `ISETP.GE.U32` / `ISETP.GE.U32.AND.EX` — 64-bit unsigned comparison against a threshold from `c[0x0][0xd0]`

This dual check implies that global addresses can be identified by both tag bits **and** by falling within a specific virtual address range. Either condition qualifies the pointer as "global".

From `test_isspacep_const_fold.sass`:
```sass
/*0db0*/  QSPC.E.G P0, RZ, [R18]                  ; tag test
/*0dc0*/  ULDC.64 UR16, c[0x0][0xd0]              ; threshold constant
/*0de0*/  ISETP.GE.U32.AND P2, PT, R18, UR16, PT  ; range check (lo)
/*0e00*/  ISETP.GE.U32.AND.EX P2, PT, R19, UR17, PT, P2  ; range check (hi)
/*0e20*/  PLOP3.LUT P0, PT, P0, P1, PT, 0x20, 0x0  ; P0 = tag OR range
```

#### 2.1.1. How QSPC Is Generated

`QSPC.E.*` instructions do **not** appear automatically from ordinary pointer dereference, `cvta` operations, or generic pointer passing through function boundaries. They are generated **only** when the programmer explicitly requests a space query, via two equivalent paths:

1. **C++ builtin** (no `#include` required — compiler-recognized):
   ```cpp
   if (__isspacep_shared(p))  { /* ... */ }   // → isspacep.shared → QSPC.E.S
   if (__isspacep_global(p))  { /* ... */ }   // → isspacep.global → QSPC.E.G
   if (__isspacep_local(p))   { /* ... */ }   // → isspacep.local  → QSPC.E.L
   ```

2. **Inline PTX assembly**:
   ```cpp
   asm("isspacep.shared p, %1; selp.u32 %0, 1, 0, p;" ...);
   ```

Builtins and inline asm are equivalent — the builtin is expanded by the compiler into the same PTX `isspacep.*` instruction.

| Code Pattern | Resulting SASS | QSPC? |
|---|---|:---:|
| `int *p = &smem[tid]; *p = val;` | `STS [Rx+UR4], Ry` | No |
| `__noinline__` function dereferencing a generic pointer | `ST.E desc[...], Ry` | No |
| `cvta.shared.u64` (construct generic shared ptr) | `S2R SR_SWINHI` + `MOV` | No |
| `cvta.to.shared.u64` (extract shared offset) | `MOV` (NO-OP, take low 32b) | No |
| Generic pointer passed across function call boundary | `ST.E`/`LD.E` (no space check) | No |
| **`__isspacep_shared(p)`** | **`QSPC.E.S`** | **Yes** |
| **`__isspacep_global(p)`** | **`QSPC.E.G`** + range check | **Yes** |
| **`__isspacep_local(p)`** | **`QSPC.E.L`** | **Yes** |

The compiler never auto-inserts QSPC as part of address conversion, pointer passing, or generic load/store paths. This also explains the asymmetry in `QSPC.E.G`'s dual-check design: `QSPC.E.S` and `QSPC.E.L` test tag bits that were intentionally set by the compiler at pointer construction time, so a single tag test suffices. `QSPC.E.G` additionally checks the address range because global pointers can originate from `cudaMalloc` (low addresses with no tag bit set) and still need to be recognized as global.

#### 2.1.2. Judgment Principle — How QSPC Determines Address Space

All three QSPC variants read the **full 64-bit pointer** from a register pair and test it against hardware knowledge of the generic address space layout. The operand `[R18]` in `QSPC.E.S P0, RZ, [R18]` denotes the 64-bit register pair `{R18, R19}` — the SASS treats consecutive registers as a 64-bit value.

**QSPC.E.S — "Is this pointer in the shared space range?"**

```sass
QSPC.E.S P0, RZ, [R18]    ; P0 = 1 if {R18,R19} ∈ shared space
```

Judgment mechanism: The upper 32 bits of a shared generic pointer carry `SR_SWINHI`, a **fixed shared-space tag** — runtime evidence on sm_87 confirms this is a constant value (0x0001ffff on that architecture) that does NOT vary per CTA. The per-CTA window isolation is provided by `SR_CgaCtaId` in the ULEA computation, not by `SR_SWINHI`. The shared space occupies a well-defined, contiguous region in the high portion of the generic address space — one 16 MB window per CTA. QSPC.E.S likely compares the upper bits against the hardware's known shared-window range boundaries in a single cycle. No additional constant-bank lookup is needed because the shared window boundaries are fixed by the architecture (number of CTAs × 16 MB), not by the kernel.

**QSPC.E.L — "Is this pointer in the local space range?"**

```sass
QSPC.E.L P0, RZ, [R18]    ; P0 = 1 if {R18,R19} ∈ local space
```

Judgment mechanism: Same principle — the upper 32 bits of a local generic pointer carry the local window base (`c[0x24]` upper, with carry from the addition of `c[0x20]` and the stack offset). Local space occupies yet another contiguous address range. QSPC.E.L checks the upper bits against the local-space boundaries without consulting constant banks at runtime.

Like shared, local space boundaries are architecturally fixed ranges, so the hardware can test membership with a single range-comparison operation embedded in the QSPC instruction microarchitecture.

**QSPC.E.G — "Is this pointer in the global space range?"**

```sass
QSPC.E.G P0, RZ, [R18]                   ; step 1: tag test
ULDC.64 UR16, c[0x0][0xd0]               ; step 2: load threshold
ISETP.GE.U32.AND P2, PT, R18, UR16, PT     ; step 2a: lo >= threshold?
ISETP.GE.U32.AND.EX P2, PT, R19, UR17, PT, P2  ; step 2b: hi >= threshold? (extended)
PLOP3.LUT P0, PT, P0, P1, PT, 0x20, 0x0    ; P0 = tag_test OR range_test
```

Global is the only space that requires a **two-part judgment**, because a pointer can be global in two distinct ways:

| Case | Reason | Checked by |
|------|--------|-----------|
| Pointer was explicitly tagged as global (compiler set global tag bits at construction) | Normal path — all three spaces use tags | `QSPC.E.G` tag test |
| Pointer came from `cudaMalloc` or host-side allocation, returned as an untagged low address | Backward compatibility — legacy pointers have no space tag | `ISETP.GE.U32.*` range check against threshold `c[0xd0]` |

The threshold `c[0x0][0xd0]` is a 64-bit runtime constant. Any pointer below this threshold is considered global regardless of its tag bits. This preserves compatibility with the pre-generic-addressing era (Kepler and earlier), where all pointers were de facto global. The OR-combination via `PLOP3.LUT` ensures either condition suffices.

**Underlying principle — non-overlapping ranges in a flat 64-bit space**

```
64-bit Generic Address Space
══════════════════════════════════════
 ↑ 0xFFFF_FFFF_FFFF_FFFF
 │  ┌──────────────────────┐
 │  │  TAGGED REGION       │  ← QSPC.E.S / QSPC.E.L: single range check
 │  │  (shared + local,    │
 │  │   non-overlapping)   │
 │  ├──────────────────────┤  ← c[0x0][0xd0] threshold (runtime)
 │  │  GLOBAL              │  ← QSPC.E.G tag test OR ptr < threshold
 │  │  (identity-mapped)   │
 ↓ 0x0000_0000_0000_0000
```

All three spaces occupy **non-overlapping, architecturally defined ranges**. The QSPC hardware instructions exploit this by performing a single-cycle range comparison on the pointer's upper bits. The only exception is `QSPC.E.G`'s range fallback, which exists solely to handle legacy untagged pointers in the low address range.

### 2.2. 64-bit Generic Pointer Structure

```
┌────────────────────┬────────────────────┐
│  Upper 32 bits     │  Lower 32 bits     │
│  (Window / Tag)    │  (Offset)          │
├────────────────────┼────────────────────┤
│  Global: identity  │  physical address  │
│         (no tag)   │                    │
│  Shared: SR_SWINHI │  shared offset     │
│  Local:  window    │  stack offset      │
│         base       │                    │
└────────────────────┴────────────────────┘
```

| Address Space | Upper 32 bits source | Lower 32 bits | cvta.to (extract) | cvta.from (construct) |
|---------------|---------------------|---------------|-------------------|----------------------|
| **Global** | same as lower 32 (identity) | physical address | NO-OP (identity) | NO-OP (identity) |
| **Shared** | `SR_SWINHI` (fixed shared-space tag) | shared memory offset | NO-OP `MOV` — low 32 bits used directly | `S2R SR_SWINHI` + MOV |
| **Local** | `c[0x0][0x24]:c[0x0][0x20]` window base | stack frame offset | `IADD3 ptr - UR_base` subtraction | `IADD3 ptr + UR_base` addition |

### 2.3. Address Space Conversion Rules

The following table consolidates all space ↔ generic conversion rules, SASS implementations, and runtime cost for each address space on sm_90.

#### Space → Generic (cvta.from)

Constructing a 64-bit generic pointer from a space-specific address:

| Space | PTX | SASS Implementation | Cost |
|-------|-----|--------------------|------|
| **Global** | `cvta.global.u64` | Identity NO-OP — same bit pattern | 0 |
| **Shared** | `cvta.shared.u64` | `S2R SR_SWINHI` (fixed tag → upper 32) + `MOV` offset → lower 32 | 1 special register read |
| **Local** | `cvta.local.u64` | `IADD3 R0, P0, R1, UR4, RZ` (R1 + c[0x20] → lower 32); `IADD3.X R2, RZ, UR5, RZ, P0` (c[0x24] + carry → upper 32) | 2 integer ALU ops |
| **Constant** | `cvta.const.u64` | **Not supported** on sm_90 ptxas | — |

#### Generic → Space (cvta.to)

Extracting a space-specific address from a 64-bit generic pointer:

| Space | PTX | SASS Implementation | Cost |
|-------|-----|--------------------|------|
| **Global** | `cvta.to.global.u64` | Identity NO-OP | 0 |
| **Shared** | `cvta.to.shared.u64` | Identity NO-OP — low 32 bits used directly as shared offset | 0 |
| **Local** | `cvta.to.local.u64` | `IADD3 R3, R6, -UR4, RZ` (gen_lo - c[0x20] → stack offset) | 1 integer ALU op |
| **Constant** | `cvta.to.const.u64` | **Not supported** on sm_90 ptxas | — |

#### Space-Specific Load/Store (no generic pointer involved)

When the compiler statically knows the address space, it bypasses the generic pointer entirely:

| Op | Space | SASS |
|----|-------|------|
| Load | Global | `LDG.E desc[UR][Rx.64], Ry` |
| Load | Shared | `LDS Ry, [Rx+UR4]` |
| Load | Local | `LDL Ry, [Rx]` |
| Load | Constant | `LDC Ry, c[bank][Rx]` |
| Store | Global | `STG.E desc[UR][Rx.64], Ry` |
| Store | Shared | `STS [Rx+UR4], Ry` |
| Store | Local | `STL [Rx], Ry` |

#### Generic Load/Store (runtime space dispatch)

When the compiler **cannot** statically determine the address space, it falls back to the generic path, and the hardware dispatches based on pointer tags at runtime:

| Op | SASS |
|----|------|
| Load | `LD.E desc[UR][Rx.64], Ry` |
| Store | `ST.E desc[UR][Rx.64], Ry` |

#### Summary Diagram

```
                        cvta.shared   cvta.to.shared
SHARED offset ◄───────────────────► (SR_SWINHI << 32) | offset
 (32-bit)        NO-OP / S2R          (64-bit generic)

                        cvta.local    cvta.to.local
LOCAL  offset ◄───────────────────► window_base + offset
 (32-bit)       IADD3 add/sub         (64-bit generic)

                        cvta.global   cvta.to.global
GLOBAL addr  ◄──────────────────────► identity
 (64-bit)          NO-OP / NO-OP       (64-bit generic)

                        cvta.const    cvta.to.const
CONSTANT     ◄──────────────────────► NOT SUPPORTED (sm_90)
 c[bank][off]                            (64-bit generic)
 (separate addr space — never converted to generic)
```

> **Note**: On sm_87 (Jetson Orin, runtime-verified), inline asm `cvta.to.shared.u64` produces incorrect output (upper bits polluted). Use `__cvta_generic_to_shared()` builtin instead. `st.shared.u32` via asm works correctly with a valid offset.

---

## 3. Window Layout in Generic Address Space

### 3.1. Shared Memory Window

Shared memory uses a **windowed virtual address scheme**:

```sass
S2UR   UR5, SR_CgaCtaId        ; UR5 = per-CTA ID
UMOV   UR4, 0x400              ; UR4 = 0x400 (window granularity constant)
ULEA   UR4, UR5, UR4, 0x18     ; UR4 = (UR5 << 24) | UR4  → shared window base
```

- **Window size per CTA**: 16 MB (`1 << 24` bytes), encoded by the ULEA shift of `0x18`
- **Window granularity**: `0x400` = 1024 (likely the minimum shared memory allocation granularity in bytes)
- **UR4 (uniform register)** becomes the shared memory window base address used in `STS [Rx+UR4]` / `LDS [Rx+UR4]`

This pattern is **identical across all shared memory sizes** (64B to 48KB) — confirming the window is a virtual construct, not physically sized by the allocation.

`SR_SWINHI` provides the upper 32 bits of the shared generic pointer — a **fixed shared-space tag**, not a per-CTA identifier. Runtime evidence on sm_87 confirms this is a constant value (0x0001ffff) that does not vary across blocks:
```sass
S2R   R5, SR_SWINHI     ; upper 32 bits = shared window high register
MOV   R6, R4             ; lower 32 bits = shared offset
; → 64-bit generic shared pointer = (R5, R6)
```

### 3.2. Local Memory Window

Local memory uses a per-thread isolation model with a 64-bit window base constant from bank 0:

```sass
LDC   R1, c[0x0][0x28]       ; initial stack pointer (uniform across threads)
ULDC  UR4, c[0x0][0x20]      ; local window base (lower 32)
ULDC  UR5, c[0x0][0x24]      ; local window base (upper 32)
VIADD R1, R1, -frame_size    ; allocate stack frame (optional)
```

The 64-bit local window base `c[0x0][0x24]:c[0x0][0x20]` is a **uniform constant** — identical for all threads in the kernel. The stack pointer `R1` from `c[0x0][0x28]` is also **uniform across all threads** (confirmed by runtime testing on sm_87 — 256 threads all received the same `R1` value).

> **Runtime evidence**: A runtime probe on Jetson Orin (sm_87) with 256 threads confirmed that `R1` (and consequently `gen_lo = R1 + UR4`, `gen_hi`, and the stack-relative `local_addr` of the first local variable) are all invariant across threads. Thread isolation for local memory does **not** arise from per-thread R1 values — it is enforced entirely at the `STL`/`LDL` hardware level.

**`cvta.local.u64`** (local→generic):
```sass
IADD3   R0, P0, R1, UR4, RZ          ; gen_lo = R1 + window_base_lo  (uniform: both operands uniform)
IADD3.X R2, RZ, UR5, RZ, P0, !PT     ; gen_hi = window_base_hi + carry
```

**`cvta.to.local.u64`** (generic→local):
```sass
IADD3   R3, R6, -UR4, RZ             ; addr = gen_lo - window_base_lo
STL     [R3], R2                       ; local store (HW scopes to current thread)
; or
LDL     R5, [R5]                       ; local load  (HW scopes to current thread)
```

The subtraction `gen_lo - UR4` strips the window base, recovering the stack-relative offset. Since `gen_lo` and `UR4` are both uniform, every thread computes the same stack-relative address for a given local variable. However, `STL`/`LDL` hardware unconditionally scopes to the executing thread — a thread dereferencing another thread's generic local pointer accesses its **own** storage at that offset, not the foreign thread's. The isolation is at the instruction level, not the address level.

### 3.3. Global Memory

Global memory has **no window translation** — the 64-bit generic pointer is the physical virtual address:

```sass
; cvta.global.u64 → IADD3/IADD3.X (64-bit add of global window base)
LDC   R0, c[0x0][0x20]       ; global window base (lower)
ULDC  UR4, c[0x0][0x24]      ; global window base (upper)
IADD3 R0, P0, R0, R4, RZ     ; add pointer to window base
IADD3.X R11, R5, UR4, RZ, P0, !PT

; cvta.to.global.u64 → NO-OP (identity)
HFMA2.MMA R7, -RZ, RZ, 0, 0 ; NOP placeholder
; pointer passes through unchanged
```

When the input is already a global pointer, `cvta.global` is identical to `cvta.to.global` (both NO-OPs).

### 3.4. Constant Memory: Outside the Generic Address Space

Constant memory does **not** participate in the 64-bit generic pointer addressing scheme. Unlike shared, local, and global — which all map addresses into the unified generic space via `cvta.*` conversion — constant memory uses an independent `<bank, offset>` addressing mode that never passes through a 64-bit generic pointer.

**Space-specific access** uses the `LDC` instruction with a bank-indexed operand:

```sass
LDC R10, c[0x3][R10]       ; bank 3, byte offset from R10
```

The `c[0x3]` is not a generic address range — it is a separate addressing mode encoded directly in the SASS opcode. The instruction tells the hardware "read from constant bank 3 at byte offset R10," bypassing the generic pointer dispatch logic that `LD.E`/`ST.E` use for shared/local/global.

**Evidence that no constant-to-generic conversion exists on sm_90**:

1. **`cvta.const` and `cvta.to.const` are rejected by ptxas** — attempting to use them in `cvta_all.cu` produced a compilation error. If no PTX cvta.const exists, there is no SASS to lower.

2. **IPO eliminates constant pointers across function boundaries** — when `read_const_ptr(const int *p)` is called with a `__constant__` argument, the compiler performs the `LDC c[0x3][...]` at the **call site** and passes only the value (a 32-bit int), not the pointer. The callee's signature in PTX is `.param .b32` (value), not `.param .b64` (generic pointer):
   ```ptx
   .func (.param .b32 func_retval0) _Z14read_const_ptrPKi(
       .param .b32 _Z14read_const_ptrPKi_param_0   ; ← .b32 value, not .b64 generic ptr
   )
   ```

3. **PTX uses direct symbol addressing for constants** — `mov.u64 %rd19, cdata;` references the constant symbol directly without involving `cvta`:
   ```ptx
   .const .align 4 .b8 cdata[256];
   mov.u64 %rd19, cdata;
   ```

4. **`QSPC.E.C` (constant space query) could not be tested** — since `ptxas` rejects `cvta.const`, there is no way to construct a 64-bit generic pointer tagged as "constant" to test with `isspacep.const`. The instruction may or may not exist.

**Architectural reason**: Constant memory is fundamentally different from the other spaces:
- It is **read-only** and **uniform** across all threads in the kernel
- It is accessed via a dedicated cache path (constant cache) separate from the L1/shared memory subsystem
- Unlike shared/local pointers that must flow through generic `int *` function parameters, constant data is rarely passed by pointer — it is typically accessed directly by name
- The bank index baked into `LDC` is sufficient addressing — there is no need for a 64-bit virtual address

```
Shared / Local / Global space               Constant space
══════════════════════════                  ═════════════
  Accessed via generic 64-bit ptr            Accessed via LDC c[bank][offset]
  cvta.* converts spaces ↔ generic            No cvta.const on sm_90
  QSPC.E.{S,G,L} for space detection          QSPC.E.C unverified
  STS/LDS/STG are space-specific              LDC is space-specific
```

### 3.5. Unified Window Layout Diagram

> **Note on ordering**: Global is confirmed to occupy low addresses (below threshold `c[0x0][0xd0]`). Shared and local both occupy tagged high-address ranges, but their **relative vertical ordering is not determinable** from static SASS — the diagram groups them together in the tagged region without implying a specific top-to-bottom sequence.

```
64-bit Generic Virtual Address Space (sm_90)
══════════════════════════════════════════════════════════════
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  TAGGED HIGH-ADDRESS REGION                                │
│  (exact internal ordering unknown — shared and local       │
│   occupy non-overlapping sub-ranges in this region)        │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SHARED MEMORY WINDOWS (per-CTA, 16 MB each)         │  │
│  │  Tag source: SR_SWINHI                               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LOCAL MEMORY WINDOWS (per-thread)                   │  │
│  │  Tag source: c[0x24]:c[0x20] + R1_stack               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
├────────────────────────────────────────────────────────────┤  ← c[0x0][0xd0] threshold
│                                                            │
│  GLOBAL MEMORY                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Identity-mapped 64-bit address space                 │  │
│  │  (no window translation — direct physical addressing)│  │
│  │  Also catches untagged legacy pointers (cudaMalloc)   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘

CONSTANT MEMORY — separate address space, not in generic pointer range
┌──────┐ ┌──────┐ ┌──────┐
│Bank 0│ │Bank 1│ │Bank N│  (c[bank][offset] indexed)
└──────┘ └──────┘ └──────┘

---

### 3.6. Generic Address Space from a Single CTA's Perspective

This section synthesizes the above window layout into a coherent picture: what does the full 64-bit generic address space look like from the viewpoint of a single CTA executing on an SM?

#### 3.6.1. Why the CTA Must Compute Its Own Shared Base

The shared memory window base is not a compile-time constant — it depends on the CTA's runtime identity:

```sass
S2UR  UR5, SR_CgaCtaId        ; UR5 = CTA identifier (assigned by hardware scheduler)
UMOV  UR4, 0x400              ; UR4 = 0x400 (window granularity constant)
ULEA  UR4, UR5, UR4, 0x18     ; UR4 = (CgaCtaId << 24) | 0x400
```

`SR_CgaCtaId` is a **read-only hardware special register**. Its value is assigned dynamically by the hardware scheduler when the CTA is launched onto an SM. The CTA does not know its ID at compile time — the ID depends on which SM it lands on, in what order CTAs are dispatched, and how many CTAs are co-resident on the same SM. Therefore:

- The CTA **must** read `SR_CgaCtaId` at runtime to compute its own window base
- The window base `UR4` is stored in a **uniform register** (shared across all threads in a warp) because all threads in the same warp belong to the same CTA
- The result `UR4 = (CgaCtaId << 24) | 0x400` provides a unique 16 MB virtual address range for this CTA's shared memory

The same mechanism for per-CTA window isolation is handled by `ULEA` using `SR_CgaCtaId` — `SR_SWINHI` (see above) is a fixed tag, not a CTA identifier.

#### 3.6.2. The Address Space as Seen by One CTA

From a single CTA's viewpoint, the 64-bit generic address space partitions into these regions:

```
High addresses (tagged region — shared & local, non-overlapping,
                internal ordering not determinable from static SASS)
┌────────────────────────────────────────────┐
│  SHARED MEMORY — THIS CTA ONLY (16 MB window)     │
│  Base = (CgaCtaId << 24) | 0x400                 │
│  Generic ptr upper = SR_SWINHI                    │
│  STS/LDS: implicitly scoped to THIS CTA           │
├────────────────────────────────────────────┤
│  LOCAL MEMORY WINDOWS (per-thread, opaque to other threads)    │
│  Base: c[0x24]:c[0x20] (uniform constant)           │
│  Offset: R1 stack ptr from c[0x28] (uniform across threads)│
│  STL/LDL: implicitly scoped to current thread                  │
├────────────────────────────────────────────┤  ← c[0x0][0xd0] threshold
│  GLOBAL MEMORY (shared by all CTAs, all threads)              │
│  Identity-mapped 64-bit space                                  │
├────────────────────────────────────────────┤
│  CONSTANT MEMORY (separate space, bank-indexed)               │
│  Bank 0 = ABI + kernel params, Bank 4 = kernel arg data        │
└────────────────────────────────────────────┘
Low addresses (0x0)
```

#### 3.6.3. Per-Space Access Rules Summary

| Operation | Address Operand | Space Targeted | Cross-CTA? | Cross-Thread? |
|-----------|----------------|----------------|------------|---------------|
| `LDG`/`STG` via desc | 32b offset + 64b descriptor | Global | Yes | Yes |
| `LDS`/`STS` | 32b offset (+ uniform base UR4) | **This CTA's shared** | No | Yes (within CTA) |
| `LDL`/`STL` | 32b offset (stack-relative) | **This thread's local** | No | No |
| `LDC` | bank# + offset | Constant | Same for all | Same for all |
| `LD.E`/`ST.E` (generic) | 64b tagged pointer | Dispatched by hardware from pointer tags | Unknown† | Unknown† |

> † Whether `LD.E`/`ST.E` allows cross-CTA or cross-thread access through manually-tagged generic pointers depends on hardware protection logic that is not observable from static SASS. See section 3.5.5 below.

#### 3.6.4. How Pointer Tags Are Used vs. Ignored

The 64-bit generic pointer carries tag information in its upper 32 bits:

| Space | Upper 32 bits | Used by `QSPC.E.*`? | Used by space-specific ld/st? |
|-------|--------------|---------------------|------------------------------|
| Global | 0 or matching lower 32 (identity) | Yes (`QSPC.E.G`) | Not needed (LDG/STG use descriptor) |
| Shared | `SR_SWINHI` (fixed space tag) | Yes (`QSPC.E.S`) | **No** — `STS`/`LDS` use only the 32-bit offset, window is implicit |
| Local | `c[0x24]` (window base upper) | Yes (`QSPC.E.L`) | **No** — `STL`/`LDL` use computed stack offset, window is implicit |
| Generic `ST.E`/`LD.E` | All 64 bits matter | Dispatch source | Hardware decodes tags to route to correct space |

This is the central tension in the design: **the generic pointer carries complete space-and-owner identity, but the space-specific hardware instructions (`STS`, `LDS`, `STL`, `LDL`) ignore the tag bits entirely.** The window is determined by the *executing context* (current CTA for shared, current thread for local), not by the pointer.

The tags serve three purposes:
1. **ABI correctness**: Functions receiving `int *` can call `__isspacep_shared(p)` to determine what space the pointer targets — compiled to `QSPC.E.S`
2. **Generic load/store dispatch**: `LD.E`/`ST.E` read the tags to route to the correct memory subsystem
3. **Pointer provenance tracking in PTX**: The compiler can constant-fold `isspacep` when it statically knows the pointer's origin

But for the hot path (`STS`/`LDS`/`STL`/`LDL`), the tags are stripped, making dereference zero-overhead.

#### 3.6.5. Can a CTA Access Another CTA's Shared Memory by Manipulating Pointer Bits?

This question decomposes into two scenarios:

**Scenario A: Using `STS`/`LDS` (space-specific instructions).**

Impossible. `STS [R7], R0` only takes a 32-bit offset — the hardware routes this to the current CTA's shared window unconditionally. `cvta.to.shared` is a NO-OP in SASS: the instruction simply takes the low 32 bits of whatever 64-bit value you provide and uses it as the offset. There is no instruction-level mechanism to redirect `STS`/`LDS` to a foreign CTA.

**Scenario B: Using `ST.E`/`LD.E` (generic load/store).**

Hypothetically possible at the instruction level, because `ST.E` takes the full 64-bit tagged pointer and dispatches based on the tags. If you manually construct a 64-bit value with the shared-space tag in the upper 32 bits and pass it to `ST.E`, the hardware would decode the tags and see "this targets shared space."

Whether the hardware *permits* this cross-CTA access or *faults* depends on the GPU's protection model, which is not observable from static SASS disassembly. The hardware has all the information needed to detect the violation (the executing CTA's `CgaCtaId` vs. the shared window computed by ULEA), but whether it actually checks is implementation-defined.

However, the **compiler's behavior** provides a strong hint: when a `__noinline__` function receives a generic pointer that might target multiple spaces, the compiler emits `ST.E`/`LD.E` (not `STS`/`LDS`). This suggests NVIDIA intends `ST.E`/`LD.E` to be the correct way to handle pointers of unknown provenance — and that the hardware-level dispatch is trusted to route correctly. But the compiler never *deliberately* constructs cross-CTA generic pointers, so this path is not exercised in normal code.

**Conclusion**: `SR_CgaCtaId` is read-only and cannot be modified by user code. Modifying the generic pointer's upper bits alone cannot bypass `STS`/`LDS` isolation (they ignore the bits). Whether `ST.E`/`LD.E` enforces CTA-level protection is a hardware implementation detail not answerable from offline disassembly.

---

## 4. Cross-Thread and Cross-CTA Visibility

### 4.1. Local Memory: Architecturally NOT cross-thread accessible

**Evidence** — from `test_local_cross_thread_ptr.sass`:

When thread 0 constructs a generic pointer to its local variable and thread 1 attempts to dereference it:
1. The generic pointer's **upper 32 bits are stripped** before reaching the callee (only the low 32 bits are loaded from shared memory: `LDS R6, [UR4]`)
2. `cvta.to.local` in the callee is a **pure subtraction** of a thread-uniform constant: `IADD3 R3, R6, -UR4, RZ` where `UR4 = c[0x0][0x20]`
3. The resulting address is used with `STL [R3]` / `LDL [R5]` — which **always targets the executing thread's own local memory**
4. No `SR_TID` or any per-thread register is consulted in the callee

**Result**: Thread 1 dereferencing thread 0's generic local pointer simply accesses **thread 1's own local memory** at the offset encoded in the pointer. The hardware has no mechanism to redirect `STL`/`LDL` to another thread's local memory window.

### 4.2. Shared Memory: Architecturally NOT cross-CTA accessible

**Evidence** — from `test_shared_cross_cta_ptr.sass`:

When CTA 0 constructs a generic shared pointer and CTA 1 attempts to dereference it:
1. `cvta.to.shared` is a **complete NO-OP** in SASS — the low 32 bits are used directly:
   ```sass
   MOV  R7, R2        ; raw offset from generic pointer
   STS  [R7], R0      ; store to shared[R7]
   ```
2. The upper 32 bits (containing the fixed shared-space tag from `SR_SWINHI`) are **discarded**
3. `STS`/`LDS` instructions use only the offset — the shared memory window is determined **implicitly by the hardware** based on the executing CTA's `CgaCtaId`
4. No `SR_CgaCtaId` or `SR_SWINHI` is used in the dereference callee

**Result**: CTA 1 dereferencing CTA 0's generic shared pointer accesses **CTA 1's own shared memory** at the offset encoded in the pointer. The hardware routes `STS`/`LDS` to the current CTA's window regardless of the pointer's origin.

### 4.3. `SR_SWINHI` — Constructor vs. Dereferencer

`SR_SWINHI` is a **fixed shared-space tag**, not a per-CTA identifier (runtime-confirmed on sm_87: constant value 0x0001ffff across all blocks). It is read **only when constructing** a generic shared pointer and is never consulted during dereference:

```sass
; CONSTRUCTOR (make_shared_generic_ptr):
S2R  R4, SR_SWINHI         ; read fixed shared-space tag
MOV  R2, RZ                  ; offset = 0
MOV  R3, R4                  ; return (offset=0, upper=fixed tag)

; DEREFERENCER (write_shared_via_generic):
MOV  R7, R2                  ; take only the low 32 bits → offset
STS  [R7], R0                ; hardware determines CTA window from CgaCtaId
```

### 4.4. Visibility Matrix

| Accessor \ Target | Thread 0 Local | Thread N Local | CTA 0 Shared | CTA N Shared | Global |
|--------------------|---------------|---------------|-------------|-------------|--------|
| **Thread 0** | RW (own) | Access to own local at offset (not target's) | RW | Access to own CTA's shared | RW |
| **Thread N** | Access to own local at offset | RW (own) | RW | Access to own CTA's shared | RW |
| **CTA 0** | N/A | N/A | RW | Access to own CTA's shared | RW |
| **CTA N** | N/A | N/A | Access to own CTA's shared | RW | RW |

The generic pointer's upper 32 bits encode window identity, but the **dereference hardware ignores them** for space-specific instructions (STS/LDS/STL/LDL). The window is determined by the executing context: for shared memory, the current CTA's window; for local memory, the current thread's storage — enforced at the STL/LDL hardware level. Since `R1` (stack pointer from `c[0x0][0x28]`) and the window base are both uniform, a generic local pointer encodes the same stack offset for all threads; `STL`/`LDL` isolation arises from hardware scoping, not from pointer differentiation.

---

## 5. Runtime Resource Management

### 5.1. Constant Bank Layout

NVIDIA GPUs organize constant memory into independently addressable **banks**, each 64 KB. The syntax `c[bank][offset]` identifies a specific bank (0–7 or more) and a byte offset within that 64 KB region. Three banks were observed across all projects in this repository:

| Bank | Content | Evidence |
|------|---------|----------|
| `c[0x0]` | Kernel ABI context + user parameters | Every kernel (systematic sweep via `test_kernel_param_map.cu`) |
| `c[0x3]` | `__constant__` variable data | `02-cvta-analysis/cvta_all.cu`: `__constant__ int cdata[64]` → `LDC R10, c[0x3][R10]` |
| `c[0x4]` | `__device__` global variable address | `k1`–`k10`: address of `g_dump` → `LDC.64 R*, c[0x4][RZ]` |

Banks `c[0x1]` and `c[0x2]` were never referenced in any tested kernel — they may be empty or reserved for overflow when bank 0's parameter space is exhausted.

#### Bank 0 — Internal Layout

Bank 0 is the primary bank. Its lower half (offsets 0x000–0x20F) holds the **execution context header** — runtime-computed values that describe the kernel's resource environment. Its upper half (offsets 0x210+) holds **user kernel parameters**.

```
c[0x0] Bank (64 KB)
┌──────────────────────────────────────────┐
│ 0x000: Implicit params (blockDim.x etc.)   │  Execution
│ 0x020: Local/global window base (lo 32)    │  context
│ 0x024: Local/global window base (hi 32)    │  header
│ 0x028: Stack pointer R1 (uniform)           │  (runtime-
│  ...   (unmapped region — config data?)    │   computed)
│ 0x0D0: Global addr-space threshold         │
│  ...                                        │
│ 0x208: Global memory descriptor base        │
├──────────────────────────────────────────┤
│ 0x210: User kernel param slot 0 (arg0)     │  User
│ 0x218: User kernel param slot 1 (arg1)     │  kernel
│  ...   (each slot 0x8, sequential)          │  parameters
└──────────────────────────────────────────┘
```

The execution context header is **not** an extension of the user parameter list — its values come from the launch configuration (`blockDim`, grid dimensions) and from runtime resource calculations (stack size per thread, local/shared memory allocation, window base addresses). The user parameters follow at 0x210, forming the tail of the bank.

**ABI / Runtime Constants (invariant offsets)**:

| Offset | Width | Content |
|--------|-------|---------|
| `c[0x0][0x00]` | 32-bit | `blockDim.x` (grid dimension) |
| `c[0x0][0x20]` | 32-bit | Local/global window base (lower 32) |
| `c[0x0][0x24]` | 32-bit | Local/global window base (upper 32) |
| `c[0x0][0x28]` | 32-bit | Initial stack pointer (R1) — uniform across threads — **present in every kernel** |
| `c[0x0][0xd0]` | 64-bit | Global address space range threshold (for `QSPC.E.G` fallback) |
| `c[0x0][0x208]` | 64-bit | Global memory descriptor base (for `STG.E desc[UR][...]`) |

**Kernel Parameter Slots (vary by arg count)**:

Kernel arguments occupy offsets starting at `0x210`, incrementing by `0x8` per slot:

| Kernel | Param | Offset | Type |
|--------|-------|--------|------|
| `k1(int a)` | a | `0x210` | 32-bit int |
| `k2(int a, int b)` | a, b | `0x210` | 64-bit pair (two ints packed) |
| `k3(int *a, int b)` | a | `0x210` | 64-bit ptr |
| | b | `0x218` | 32-bit int |
| `k5(int*, int*, int*)` | ptr0, ptr1, ptr2 | `0x210`, `0x218`, `0x220` | 3×64-bit ptrs |
| `k7(int*,..., int*)` | 6 ptrs | `0x210`–`0x238` | 6×64-bit ptrs |

Each 64-bit pointer or 32-bit scalar occupies one `0x8`-aligned slot. The pattern is sequential from `0x210`.

#### Slot Allocation Rules

The global memory descriptor is at `c[0x0][0x208]` (a fixed ABI offset loaded with `ULDC.64` and used as the `desc[UR]` operand in `STG.E desc[UR][Rx.64], Ry`). Unlike the descriptor, user parameter slots start at `c[0x0][0x210]` and are accessed as either 32-bit (`ULDC`) or 64-bit (`LDC.64`) depending on the parameter type.

**Packing rule**: Two consecutive 32-bit scalar parameters can be **packed** into a single 8-byte slot. `k2(int a, int b)` references only `c[0x0][0x210]` (loaded as `LDC.64`) — the second 32-bit scalar does not consume a separate slot. A 64-bit pointer always claims a full slot and forces the next parameter to the next 8-byte boundary:

```
k2(int a, int b):            | a(4B) + b(4B) |          ← packed at slot 0 (0x210)
k3(int *a, int b):           | a(8B ptr)      | b(4B)    ← slot 0 (0x210) + slot 1 (0x218)
k4(int *a, int *b):          | a(8B ptr)      | b(8B)    ← slot 0 + slot 1
```

#### Bank 3 — `__constant__` Variable Storage

Observed in `02-cvta-analysis/cvta_all.cu`, where `__constant__ int cdata[64]` is accessed via `LDC R10, c[0x3][R10]`. The compiler computes a byte offset into bank 3 (using `LOP3.LUT` to compute `cdata_offset & 0xFC | 0xC0`) and issues `LDC` with that offset. Bank 3 holds all `__constant__` variables declared in the module.

#### Bank 4 — `__device__` Global Variable Pointer

```sass
LDC.64  R*, c[0x4][RZ]     ; address of __device__ global variables
```

`c[0x4][0x0]` holds a pointer to the kernel's output destination — the address of the `__device__` global variables that the kernel writes to (e.g., `g_dump`, `g_storage`). The offset is always `RZ` (zero register), meaning the value is at the start of bank 4. This is separate from bank 0's parameter slots.

### 5.2. Stack Frame Allocation

The stack frame is allocated at kernel entry via:
```sass
LDC   R1, c[0x0][0x28]      ; load initial stack pointer
VIADD R1, R1, immediate      ; allocate frame (R1 -= frame_size)
```

**Frame size rules** (from `test_stack_frame_vary.sass` and `test_window_sizes.sass`):

| Kernel | Locals | VIADD | Frame Size |
|--------|--------|-------|------------|
| `kf_empty` | none | none | 0 |
| `kf_1_int` through `kf_1024_int` | optimized away | none | 0 |
| `kf_local_ptr_dump` | 2 ints (address taken) | `0xfffffff8` (-8) | 8 bytes |
| `k_combined_1KB_256B` | `int arr[64]` | `0xffffff00` (-256) | 256 bytes |
| `k_local_1KB` | `int arr[256]` | `0xfffffc00` (-1024) | 1024 bytes |
| `k_local_4KB` | `int arr[1024]` | `0xfffff000` (-4096) | 4096 bytes |

Key observations:
- Scalar locals that can be held in registers → **no stack frame**
- Arrays that survive optimization → frame = `sizeof(array)` rounded to alignment
- Address-taken locals (e.g., `&a` passed to function) → forced stack allocation even for scalars

**Register spills**: None observed in any test kernel (up to 128 live ints). The compiler's constant propagation and dead-code elimination are aggressive at `-O3` on sm_90. Real spills would require non-constant-dependent code with >255 live registers.

### 5.3. Shared Memory Window Formula

The window base computation is a two-instruction pattern observed universally across all kernels with `__shared__` usage:

```sass
S2UR  UR5, SR_CgaCtaId        ; UR5 = CTA identifier
UMOV  UR4, 0x400              ; UR4 = 0x400 (window granularity constant)
ULEA  UR4, UR5, UR4, 0x18     ; UR4 = (UR5 << 24) | 0x400
```

#### 5.3.1. Derivation of 16 MB Per CTA

The `ULEA` (Uniform Logic/Arithmetic Extended) instruction on sm_90 computes:
```
ULEA R, A, B, shift → R = (A << shift) + B
```

Substituting the actual operands:
- `A = UR5 = SR_CgaCtaId`
- `B = UR4 = 0x400`
- `shift = 0x18 = 24`

```
UR4 = (CgaCtaId << 24) + 0x400
```

The stride between adjacent CTA IDs is `1 << 24 = 16,777,216 bytes = 16 MB`. Therefore each CTA occupies a **16 MB window** in the generic address space. The `0x400` (1024) is a constant offset applied to all windows — likely the minimum shared memory allocation granularity in bytes.

#### 5.3.2. Confirming Evidence from `test_shared_window_granularity.cu`

The additional probe kernels provide multi-angle confirmation:

| Test | Finding |
|------|---------|
| `k_no_shared` | No `__shared__` → **no ULEA/UMOV at all** |
| `k_single_block` | Even with gridDim=1 (CgaCtaId always 0), ULEA still appears with `0x400`/`0x18` |
| `k_two_shared_arrays` | Two `__shared__` arrays → **two ULEA** instructions. `arr1` uses `ULEA(Id, 0x400, 0x18)`, `arr2` uses `ULEA(Id, 0x800, 0x18)` — the second array's base is offset by the first array's size (1 KB = 0x400). The shift `0x18` is **identical** for both. |
| `k_shared_max_48KB` / `k_shared_mid_32KB` / `k_shared_mid_16KB` | Three kernels with 16KB, 32KB, and 48KB shared — **instruction-for-instruction identical SASS**. `UMOV 0x400` and `ULEA 0x18` unchanged at the 48KB boundary. |
| `k_use_blockidx` | `SR_CTAID.X` (blockIdx) is loaded **separately** from `SR_CgaCtaId` — blockIdx is used only for data values, not for the shared window address. |
| `k_export_all_generic_ptrs` | `SR_SWINHI` is read via `S2UR` and used as the upper 32 bits of generic shared pointers (a fixed space tag). CTA window identity comes from ULEA (via `SR_CgaCtaId`) for STS/LDS; SR_SWINHI provides the shared-space tag for generic pointer construction. |
| `k_builtin_vs_asm_cvta` | When passing generic pointers to `__noinline__` functions, the compiler uses regular-register `LEA` instead of `ULEA` — same formula `(CgaCtaId << 24) + 0x400`, but stored in `R*` registers for the calling convention instead of `UR*` registers for STS/LDS. Both `__cvta_generic_to_shared()` builtin and asm `cvta.to.shared` produce identical `STS [Rx]` patterns in the callee. |

#### 5.3.3. Summary

```
SharedMemoryWindowBase(CTA_id) = (CgaCtaId << 24) + 0x400
                                 └────16 MB/CTA───┘ └─1 KB base─┘
```

This formula is invariant across:
- All shared memory sizes (64 B to 48 KB)
- Static and dynamic (`extern __shared__`) shared memory
- Single-block and multi-block kernels
- Single array and multi-array configurations
- Uniform register path (ULEA for direct STS/LDS) and regular register path (LEA for generic pointer passing)

The actual physical shared memory allocation is a runtime parameter — the hardware virtualizes the window and enforces bounds.

#### 5.3.4. What the 0x400 Offset Is NOT: mbarrier State Storage

A natural hypothesis is that the reserved 0x000–0x3FF region stores hardware `mbarrier` (asynchronous barrier) state entries. This was tested via `test_mbarrier_shared_reserved.cu`, which declares `cuda::barrier<cuda::thread_scope_block>` objects (compiler-recognized mbarrier state) before user `__shared__` arrays and observes the resulting ULEA offsets:

| Kernel | Barrier count | First barrier offset | First user array offset |
|--------|:---:|------|------|
| `k_t2_baseline` | 0 | — | **0x400** |
| `k_t2_one_barrier` | 1 | **0x400** | **0x408** |
| `k_t2_three_barriers` | 3 | **0x400** | **0x418** |
| `k_t2_barrier_between_arrays` | 1 (between arr1/arr2) | **0x500** | **0x400** (arr1), **0x508** (arr2) |

The compiler treats `cuda::barrier` state identically to any other `__shared__` variable — it is laid out sequentially starting from offset 0x400 and pushes subsequent variables proportionally. The mbarrier state is **not** placed in the reserved 0x000–0x3FF region.

This disproves the specific mbarrier-storage hypothesis. The 0x400 offset remains an architectural constant whose consumer (if any) is not observable via CUDA C++ variable declarations — it may be a hardware-defined alignment gap, a `__syncthreads()` scratch area, or simply a fixed window base with no logical consumer in the user-visible shared memory layout.

### 5.4. Dynamic Shared Memory

`extern __shared__` kernels produce **instruction-for-instruction identical** SASS to equivalent static shared kernels. The compiler treats dynamic shared memory identically — the `ULEA`/`UMOV` pattern is unchanged, and no additional constant bank references are added.

The runtime (via `cudaLaunchKernel`) configures the shared memory window size; the hardware enforces access bounds. This is an elegant design: the compiled code is agnostic to the actual allocation size.

### 5.5. Special Register Catalog

| Register | Width | Read via | Purpose |
|----------|-------|----------|---------|
| `SR_TID.X` | 32 | `S2R` | Thread index within block |
| `SR_CTAID.X` | 32 | `S2R` | Block index within grid |
| `SR_CgaCtaId` | 32 | `S2R`, `S2UR` | Cooperative Group Array ID (per-CTA) |
| `SR_SWINHI` | 32 | `S2R`, `S2UR` | Fixed shared-space tag — upper 32 bits of generic shared pointers. Does NOT encode CTA identity (runtime-confirmed: constant across all blocks). |

**`S2R` vs `S2UR`**:
- `S2R` (Special to Register) → general-purpose `R*` register (per-thread)
- `S2UR` (Special to Uniform Register) → `UR*` register (warp-uniform, optimized for broadcast)

The compiler promotes `SR_SWINHI` to `S2UR` when all threads in a warp share the same CTA (the common case), enabling uniform-register `STS`/`LDS` addressing.

---

## 6. `CVTA` Is Not a Hardware Instruction

All CVTA PTX variants lower to basic operations — there is no `CVTA` opcode in sm_90 SASS:

| PTX Operation | SASS Implementation | Cost |
|---------------|---------------------|------|
| `cvta.to.global` | Identity NO-OP | 0 cycles (pass-through) |
| `cvta.global` | Identity NO-OP | 0 cycles |
| `cvta.to.shared` | Identity NO-OP | 0 cycles (low 32 bits used directly) |
| `cvta.shared` | `S2R SR_SWINHI` + MOV | 1 special register read |
| `cvta.to.local` | `IADD3` subtract window base | 1 integer subtraction |
| `cvta.local` | `IADD3` / `IADD3.X` add window base | 1-2 integer additions |

The pointer format was intentionally designed so that the most common operations (shared dereference, global dereference) require zero conversion overhead.

### 6.1. Pointer Surgery Resilience

From `test_generic_ptr_surgery.sass`: even when the generic pointer's bits are surgically modified (upper byte cleared, bit 32 toggled, upper 32 replaced with constant), **ptxas continues to emit STS** (space-specific store). The compiler does not track pointer provenance through inline assembly — it trusts the PTX `st.shared.u32` directive. The hardware's `STS` instruction routes to the current CTA's shared window regardless of the pointer bits.

---

## 7. Compiler Strategy Under Generic Addressing

The existence of a unified generic address space is not purely a hardware abstraction — it fundamentally shapes **compiler code generation strategy** at multiple levels. This section catalogs the compiler behaviors forced by the need to convert between space-specific and generic addresses.

### 7.1. The Root Cause: No Address-Space Qualifiers on Parameters

CUDA C++ does not allow address-space qualifiers on function parameters:

```cpp
__device__ void foo(__shared__ int *p);    // ❌ not valid CUDA C++
__device__ void foo(int *p);               // ✅ p is a generic pointer
```

Any `int *` parameter to a `__device__` function is implicitly a **generic** 64-bit pointer. This means every call site that passes a shared, local, or global pointer into a `__device__` function **must** first convert the pointer into generic form. The compiler inserts the appropriate `cvta.*` instruction at each call site.

### 7.2. Call-Site cvta Injection

When a kernel calls a `__noinline__` function with a shared pointer:

```cpp
__shared__ int smem[256];
foo(&smem[tid]);   // compiler inserts cvta.shared before the call
```

The compiler generates:
```sass
S2R  R9, SR_SWINHI        ; construct generic shared pointer
MOV  R6, offset             ;  {R9,R6} = 64-bit generic ptr
CALL.REL.NOINC foo_addr     ; pass generic ptr in registers
```

For local pointers:
```sass
IADD3   R0, P0, R1, UR4, RZ       ; cvta.local: generic = stack + window
IADD3.X R2, RZ, UR5, RZ, P0, !PT
CALL.REL.NOINC foo_addr
```

For global pointers, no conversion is needed (identity), so the call site is unchanged.

### 7.3. Inter-Procedural Optimization (IPO)

From `01-shared-ptr-analysis` (which ran in the same toolchain), the compiler applies aggressive IPO when it has whole-program visibility of all call sites:

**Case 1 — single address space caller**: If `__noinline__ foo(int *p)` is called exclusively with shared pointers, the compiler can **specialize** the callee: it inserts `cvta.to.shared` inside `foo` and uses `STS`/`LDS` directly, avoiding generic `ST.E`/`LD.E` overhead.

**Case 2 — pure read function**: If `foo(int *p) { return *p; }`, the compiler eliminates the pointer entirely — it performs the load at the call site with the correct space-specific instruction (`LDS`/`LDG`/`LDL`) and passes only the **value** to the callee:
```sass
; Call site with shared pointer:
LDS  R8, [offset+UR4]       ; ld.shared at call site
st.param.b32 [param0], R8    ; pass VALUE, not pointer
call foo                     ; foo receives an int, not a pointer
```

The callee degenerates to `return param0;` — the address space problem is sidestepped entirely.

**Case 3 — multiple address space callers**: If `foo` is called with shared, global, and local pointers, the compiler cannot specialize and must use generic `ST.E`/`LD.E` in a single shared function body.

### 7.4. Generic vs. Space-Specific Instruction Selection

The compiler's instruction selection depends on what it can prove about the pointer's origin:

| Compiler Knowledge | Generated SASS | Reason |
|---|---|---|
| Pointer is *certainly* shared (inlined, or IPO'd single caller) | `STS`/`LDS` | Zero-overhead shared access |
| Pointer is *certainly* global | `STG`/`LDG` via descriptor | Zero-overhead global access |
| Pointer is *certainly* local | `STL`/`LDL` | Zero-overhead local access |
| Pointer provenance is **unknown** (multi-space callers, opaque function boundary) | `ST.E`/`LD.E` | Hardware-dispatch at runtime |

The `ST.E`/`LD.E` fallback is a deliberate performance compromise: it avoids bloating the binary with per-space clones while accepting the hardware dispatch overhead. On sm_90, `ST.E` adds one pointer state check versus `STS`, but the hardware is heavily optimized for this path.

### 7.5. No Per-Call-Site Function Cloning

A key non-behavior: the compiler does **not** automatically generate multiple clones of a `__noinline__` function specialized for each address space. From `01-shared-ptr-analysis`:

```
generic_write (called with shared, global, and local pointers)
→ ONE function body using ST.E, shared by all three call sites
```

The three call sites all use the same `CALL.REL.NOINC` target. The compiler could in principle clone `generic_write` into `generic_write_shared` (using `STS`), `generic_write_global` (using `STG`), and `generic_write_local` (using `STL`), but it chooses not to. This is likely a code-size optimization — the overhead of `ST.E` dispatch is acceptable compared to code duplication.

**Exception**: If the programmer explicitly templates the function on address space, the template instantiation naturally produces per-space copies. This is an opt-in optimization pattern.

### 7.6. Uniform Register Promotion

When all threads in a warp share the same CTA (the common case), the compiler promotes `SR_SWINHI` reads from per-thread registers to **uniform registers**:

```sass
; Per-thread path (warp-divergent offset):
S2R  R5, SR_SWINHI        ; duplicated across all 32 threads

; Uniform path (warp-uniform offset):
S2UR UR7, SR_SWINHI        ; read once, broadcast to all threads in warp
```

This reduces register pressure and enables `STS [UR4], R5` — a uniform-register indexed store that executes once per warp rather than once per thread. The compiler chooses the uniform path when it can prove that the shared offset is warp-uniform (e.g., all threads access the same `smem[0]`).

### 7.7. cvta Fusion at the Instruction Level

When `cvta.to.shared` is immediately followed by `st.shared` (or `ld.shared`), the two operations **fuse** into a single `STS`/`LDS`:

```sass
; Without fusion (logical PTX):
cvta.to.shared.u64 tmp, gen_ptr
cvt.u32.u64 addr, tmp
st.shared.u32 [addr], val

; With fusion (actual SASS):
MOV  R7, R2            ; low 32 bits = offset (cvta is NO-OP)
STS  [R7], R0           ; single shared store
```

The `cvta.to.shared` + `cvt.u32.u64` pair collapses to a single `MOV`, and `STS` directly consumes the result. This fusion is why shared memory dereference through generic pointers is zero-overhead on the hot path: the "conversion" does not exist at the SASS level.

The same fusion applies to `cvta.to.local` + `STL`/`LDL`, where the subtraction collapses into the addressing mode:
```sass
IADD3 R3, R6, -UR4, RZ    ; cvta.to.local subtraction
STL   [R3], R2              ; local store
; These remain two instructions (subtract + store), not fully fused,
; but the subtract is a single-cycle integer operation.
```

### 7.8. Summary: Compiler Decision Table

| Scenario | Compiler Action | Performance |
|---|---|---|
| Direct shared access (`smem[tid] = x`) | `STS [Rx+UR4], Ry` | Zero overhead |
| Direct global access (`gmem[tid] = x`) | `STG.E desc[UR][R.64], Ry` | Zero overhead |
| `__noinline__` call with shared ptr, single caller | IPO: specialize callee with `STS` | Zero overhead |
| `__noinline__` call with shared ptr, multiple spaces | Generic `ST.E` in shared callee body | +1 state check |
| Pure-read `__noinline__` with ptr param | IPO: load at call site, pass value | Zero overhead (ptr eliminated) |
| `cvta.shared` + `st.shared` adjacent | Fuse to single `STS` | Zero overhead |
| `cvta.to.shared` from asm then `st.shared` | `MOV` + `STS` (no-op conversion) | 1 MOV overhead |
| Warp-uniform shared access | `S2UR` + `STS [UR]` (uniform path) | Reduced register pressure |
| Warp-divergent shared access | `S2R` + `STS [R]` (per-thread path) | Full register usage |

---

## 8. Key Architectural Inferences

### 8.1. Why Cross-CTA Shared Access Is Not Possible

The `STS`/`LDS` instruction microarchitecture on sm_90 does not accept a 64-bit generic address — it uses a 32-bit offset **implicitly scoped** to the current CTA's shared memory window (indexed by `CgaCtaId`). The upper 32 bits of the generic pointer are for ABI-level pointer identity only; the hardware ignores them for shared memory access.

### 8.2. Why Local Memory Is Per-Thread

`STL`/`LDL` instructions use a 32-bit offset **implicitly scoped** to the current thread's local memory storage. The window base is configured by uniform constants (`c[0x20:0x24]`), and the stack pointer `R1` from `c[0x28]` is **uniform across all threads** (runtime-confirmed on sm_87). Thread isolation arises entirely at the `STL`/`LDL` hardware level — these instructions unconditionally route to the executing thread's local storage, independent of the address register values. Since `R1` and the window base are both uniform, every thread computes the same generic local pointer for a given variable; the hardware isolates at the point of access, not the address.

### 8.3. The `QSPC.E.G` Dual-Check Design

The `isspacep.global` dual check (tag bits OR range check) suggests that sm_90 supports legacy global pointers that may not carry explicit space tag bits. Any pointer below the threshold `c[0x0][0xd0]` is considered global regardless of tags — this provides backward compatibility with older pointer formats and memory allocated via `cudaMalloc` (which returns pointers in the low address range).

---

## 9. Test File Inventory

```
03-generic-address-space/
├── scripts/compile.sh
├── src/
│   ├── test_isspacep_const_fold.cu       — Phase 1.1: QSPC.E.* instruction discovery
│   ├── test_generic_ptr_surgery.cu       — Phase 1.2: pointer bit modification resilience
│   ├── test_window_sizes.cu              — Phase 1.3: window size sweep (16 kernels)
│   ├── test_local_cross_thread_ptr.cu    — Phase 2.1: cross-thread local visibility
│   ├── test_shared_cross_cta_ptr.cu      — Phase 2.2: cross-CTA shared visibility
│   ├── test_warp_uniform_vs_divergent.cu — Phase 2.3: UR* vs R* register paths
│   ├── test_kernel_param_map.cu          — Phase 3.1: constant bank layout (10 kernels)
│   ├── test_stack_frame_vary.cu          — Phase 3.2: stack frame sizing (12 kernels)
│   ├── test_register_pressure_spill.cu   — Phase 3.3: register pressure (8 kernels)
│   ├── test_shared_window_granularity.cu — Phase 1.3 supp.: ULEA confirmation (7 kernels)
│   ├── test_shared_window_granularity.cu — Phase 1.3 supp.: ULEA confirmation (7 kernels)
│   ├── test_mbarrier_shared_reserved.cu   — Phase 1.3 supp.: 0x400 reservation test (6 kernels)
│   ├── test_dynamic_shared_config.cu     — Phase 3.4: dynamic vs static shared (6 kernels)
└── build/
    ├── ptx/
    ├── sass/
    └── cubin/
```

**Total**: 12 source files, ~80 kernel variants across all tests.

---

## 10. Open Questions

1. **Actual `c[0x0][0xd0]` threshold value**: The 64-bit constant at `c[0x0][0xd0]` used for the global-space range check cannot be observed statically — it's a runtime value. Its exact value would confirm the global address space range bound in sm_90.

2. **SR_SWINHI encoding** — **RESOLVED**: Runtime probe on sm_87 confirms `SR_SWINHI` is a **fixed shared-space tag** (value 0x0001ffff) that does not vary across blocks. It marks the pointer as belonging to shared space but does **not** encode CTA identity. Per-CTA window isolation comes from `SR_CgaCtaId` in the ULEA computation, not from `SR_SWINHI`. Whether the tag value is the same on sm_90 remains unconfirmed (cannot be tested without sm_90 hardware).

3. **`QSPC.E.C` (constant space)**: No `cvta.const` test was successful (ptxas rejected the operand types). The constant space query instruction (`QSPC.E.C` or similar) may exist but remains unverified.

4. **Generic `LD.E`/`ST.E` dispatch mechanism**: When the compiler falls back to generic load/store (as seen in `01-shared-ptr-analysis`), the hardware dispatches based on pointer tags. The internal dispatch logic (tag decoder, window lookup, bounds check) cannot be observed from binary disassembly.

5. **Cross-CTA access via manually-tagged generic pointers**: If a 64-bit generic pointer is manually constructed with a foreign CTA's `SR_SWINHI` value in the upper 32 bits and dereferenced via `ST.E`/`LD.E` (generic store/load), would the hardware:
    - Route to the target CTA's shared memory on the same SM?
    - Route to a different SM's shared memory across the interconnect?
    - Fault due to ownership mismatch between the executing CTA's `CgaCtaId` and the pointer's embedded window ID?
    
    This cannot be determined from static SASS disassembly. Resolving it requires runtime testing on physical hardware. The architectural implication is significant: if the hardware *does* allow same-SM cross-CTA access via `ST.E`, it means the generic address space provides a flat view of all shared memory windows without CTA-level protection in generic load/store path — only `STS`/`LDS` enforce the implicit isolation.

6. **Consumer of the 0x400 reserved region**: The fixed 0x400 (1 KB) offset at the base of each CTA's shared memory window is invariant on sm_90 but its consumer remains unknown. `test_mbarrier_shared_reserved.cu` disproved the mbarrier-storage hypothesis. **Runtime evidence from sm_87 shows shared variables start at offset 0** — the reserved region is present on sm_90 and absent on sm_87, but cannot be assumed to be exclusive to sm_90 without testing other architectures. Possible explanations: hardware alignment, `__syncthreads()` scratch space, or a fixed architectural gap.

---

## 11. Runtime Verification (sm_87 Jetson Orin)

Three runtime probes were executed on Jetson Orin (sm_87) to complement the offline SASS analysis (sm_90). Key findings:

### 11.1. Local Memory — R1 Is Uniform, STL/LDL Hardware Isolation

> **Probe**: `runtime-probes/01-local-memory/probe_local_memory.cu`

| Observation | Result |
|---|---|
| R1 across 256 threads | **Uniform** (identical) |
| `gen_lo`, `gen_hi`, `local_addr` | **Uniform** |
| T1 reading via T0's generic local ptr | Returns T1's **own** value |
| Window base (`gen_lo - local_addr`) | Uniform |
| **Conclusion** | R1 is not per-thread. Thread isolation for local memory is enforced at the `STL`/`LDL` hardware level, not through per-thread address differentiation. |

These findings corrected the initial static-analysis assumption (see Section 3.2) that `c[0x0][0x28]` delivers a per-thread value — it is uniform, and per-thread isolation occurs at the instruction level.

### 11.2. CTA Identity — SR_SWINHI Is a Fixed Tag

> **Probe**: `runtime-probes/02-cta-identity/probe_cta_identity.cu`

| Observation | Result |
|---|---|
| `SR_SWINHI` across 8 blocks | **Constant** (0x0001ffff) |
| `gen_lo` (cvta.shared low 32b) | Constant across blocks (includes window base) |
| `arr[0]` shared offset (`__cvta_generic_to_shared`) | **0x00000000** (no reserved 0x400 on sm_87) |
| `arr[0]` value verification (each block writes `bid+tid`) | Value == `blockIdx.x` for all blocks → **CTA isolation confirmed** |
| **Conclusion** | `SR_SWINHI` is a fixed shared-space tag, not a per-CTA identifier. CTA window isolation comes from `SR_CgaCtaId` in the ULEA computation, not from `SR_SWINHI`. |

The 0x400 reserved region observed on sm_90 is absent on sm_87 — user shared variables start at offset 0.

### 11.3. Address Conversion — Builtin vs. Inline Asm

> **Probe**: `runtime-probes/03-cross-cta-generic/probe_cross_cta_generic.cu`

| Observation | Result |
|---|---|
| `__cvta_generic_to_shared(&arr[8])` builtin | Correct (0x20) |
| asm `cvta.to.shared.u64` → `cvt.u32.u64` | **Incorrect** — upper bits polluted (e.g., 0xb1000020) |
| asm `st.shared.u32 [offset]` with a valid offset | Works correctly |
| Cross-CTA generic pointer test | **Not feasible on sm_87** — `cvta.shared` produces identical 64-bit pointers for all blocks |
| **Conclusion** | On sm_87, inline asm `cvta.to.shared.u64` produces incorrect output. Use `__cvta_generic_to_shared()` builtin instead. `st.shared.u32` via asm is fine when given a valid offset. |

---

**Overall**: Runtime evidence on sm_87 resolved the SR_SWINHI encoding question (Open Question 2, now closed), confirmed that R1 is uniform, and revealed an inline asm quirk specific to cvta.to.shared on this platform. The core architectural model from SASS analysis — window-based addressing, per-CTA/per-thread hardware isolation, fixed space tags — holds across both sm_90 (SASS) and sm_87 (runtime).
