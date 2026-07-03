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
   - 3.4. [Constant Memory — Revised](#34-constant-memory--revised)
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
- Shared memory is allocated in 16 MB virtual address strides per CTA (`1 << 24`); actual usable shared memory is limited by hardware (ptxas defaults to 48 KB static, up to ~227 KB dynamic on sm_90)
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

Judgment mechanism: The upper 32 bits of a shared generic pointer carry `SR_SWINHI`, a **fixed shared-space tag** — runtime evidence on sm_87 confirms this is a constant value (~0x0001ffff on that architecture) that does NOT vary per CTA. The per-CTA window isolation is provided by `SR_CgaCtaId` in the ULEA computation, not by `SR_SWINHI`. The shared space occupies a well-defined, contiguous region in the high portion of the generic address space — a 16 MB virtual address stride per CTA. QSPC.E.S likely compares the upper bits against the hardware's known shared-window range boundaries in a single cycle. No additional constant-bank lookup is needed because the shared window boundaries are fixed by the architecture (number of CTAs × 16 MB), not by the kernel.

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
64-bit Generic Address Space (sm_87 runtime-verified)
══════════════════════════════════════
 ↑ 0xFFFF_FFFF_FFFF_FFFF
 │  ┌──────────────────────┐
 │  │  TAGGED REGION       │  ← QSPC.E.S / QSPC.E.L: tag ~0x0001ffff
 │  │  Shared | Local      │
 │  │  + Constant (0x04)   │  ← QSPC.E.C: tag 0x00000004
 │  ├──────────────────────┤  ← c[0x0][0xd0] threshold
 │  │  GLOBAL              │  ← tag 0x00000002 OR ptr < threshold
 │  │  identity-mapped     │
 ↓ 0x0000_0000_0000_0000
```

All three spaces occupy **non-overlapping, architecturally defined ranges**. The QSPC hardware instructions exploit this by performing a single-cycle range comparison on the pointer's upper bits. The only exception is `QSPC.E.G`'s range fallback, which exists solely to handle legacy untagged pointers in the low address range.

### 2.2. 64-bit Generic Pointer Structure

```
┌────────────────────┬────────────────────────────────┐
│  Upper 32 bits     │  Lower 32 bits                 │
│  (Window / Tag)    │  (Address within space)        │
├────────────────────┼────────────────────────────────┤
│  Global: identity  │  physical address              │
│         (no tag)   │                                │
│  Shared: SR_SWINHI │  full shared virtual address   │
│         (fixed tag)│  (incl. CgaCtaId in upper bits)│
│  Local:  window    │  stack offset                  │
│         base       │                                │
│  Const:  c[0xd0]   │  const byte offset             │
│         ×2         │                                │
└────────────────────┴────────────────────────────────┘
```

| Address Space | Upper 32 bits source | Lower 32 bits | cvta.to (extract) | cvta.from (construct) |
|---------------|---------------------|---------------|-------------------|----------------------|
| **Global** | same as lower 32 (identity) | physical address | NO-OP (identity) | NO-OP (identity) |
| **Shared** | `SR_SWINHI` (fixed shared-space tag) | full shared virtual address: `(CgaCtaId << 24) + 0x400 + off` (compiler-computed via ULEA) | NO-OP `MOV` — low 32 bits used directly as shared address | `S2R SR_SWINHI` + `MOV` (input shared address passed through to low 32) |
| **Local** | `c[0x0][0x24]:c[0x0][0x20]` window base | stack frame offset | `IADD3 ptr - UR_base` subtraction | `IADD3 ptr + UR_base` addition |
| **Constant** | `c[0x0][0xd0]` half-base (upper 32) | constant byte offset | `UIADD3 -UR_base` subtraction with borrow | `UIADD3 ×2` + offset |

> **Note on shared generic pointer composition**: In normal compiler-generated code, the lower 32 bits of a shared generic pointer are the **full shared virtual address** — not a bare offset. This address is computed by the compiler via `ULEA(CgaCtaId << 24) + 0x400 + element_offset` and includes `CgaCtaId` in its upper bits. The earlier observation of low-32 bits being a pure offset (e.g. `0`) only applies to the artificial inline-asm path (`cvta.shared.u64(0)`) used in `test_shared_cross_cta_ptr.cu`. See §3.1 for the SASS evidence.

### 2.3. Address Space Conversion Rules

The following table consolidates all space ↔ generic conversion rules, SASS implementations, and runtime cost for each address space on sm_90.

#### Space → Generic (cvta.from)

Constructing a 64-bit generic pointer from a space-specific address:

| Space | PTX | SASS Implementation | Cost |
|-------|-----|--------------------|------|
| **Global** | `cvta.global.u64` | Identity NO-OP — same bit pattern | 0 |
| **Shared** | `cvta.shared.u64` | `S2R SR_SWINHI` (fixed tag → upper 32) + `MOV` offset → lower 32 | 1 special register read |
| **Local** | `cvta.local.u64` | `IADD3 R0, P0, R1, UR4, RZ` (R1 + c[0x20] → lower 32); `IADD3.X R2, RZ, UR5, RZ, P0` (c[0x24] + carry → upper 32) | 2 integer ALU ops |
| **Constant** | `cvta.const.u64` | `UIADD3` double base + offset (`2 * c[0x0][0xd0] + off`) | 2-3 integer ALU ops |

#### Generic → Space (cvta.to)

Extracting a space-specific address from a 64-bit generic pointer:

| Space | PTX | SASS Implementation | Cost |
|-------|-----|--------------------|------|
| **Global** | `cvta.to.global.u64` | Identity NO-OP | 0 |
| **Shared** | `cvta.to.shared.u64` | Identity NO-OP — low 32 bits used directly as shared offset | 0 |
| **Local** | `cvta.to.local.u64` | `IADD3 R3, R6, -UR4, RZ` (gen_lo - c[0x20] → stack offset) | 1 integer ALU op |
| **Constant** | `cvta.to.const.u64` | `UIADD3` subtract base (`gen_ptr - c[0x0][0xd0]` with borrow) | 2 integer ALU ops |

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
CONSTANT offset ◄───────────────────► 2 * c[0x0][0xd0] + offset
 c[bank][off]     UIADD3 ×2 add/sub     (64-bit generic)
```

> **Note**: On sm_87 (Jetson Orin, runtime-verified), inline asm `cvta.to.shared.u64` produces incorrect output (upper bits polluted). Use `__cvta_generic_to_shared()` builtin instead. `st.shared.u32` via asm works correctly with a valid offset.
>
> † **Constant cvta restriction**: `cvta.const` / `cvta.to.const` are documented in the PTX ISA (since 3.1) and require sm_20+. However, the PTX ISA notes: *"The current implementation does not allow generic pointers to const space variables in programs that contain pointers to constant buffers passed as kernel parameters."* Our test triggered this restriction; the instructions may work in kernels without parameter-buffer pointers. Whether an SASS-level constant-to-generic conversion exists on sm_90 hardware remains unverified.

---

## 3. Window Layout in Generic Address Space

### 3.1. Shared Memory Window

Shared memory uses a **windowed virtual address scheme**:

```sass
S2UR   UR5, SR_CgaCtaId        ; UR5 = per-CTA ID
UMOV   UR4, 0x400              ; UR4 = 0x400 (window base offset constant)
ULEA   UR4, UR5, UR4, 0x18     ; UR4 = (UR5 << 24) | UR4  → shared window base
```

- **Virtual address stride**: 16 MB per CTA (`1 << 24` bytes), encoded by the ULEA shift of `0x18`
- **Window base offset on sm_90**: `0x400` = 1024 (absent on sm_87 where shared variables start at offset 0)
- **UR4 (uniform register)** becomes the shared memory window base address used in `STS [Rx+UR4]` / `LDS [Rx+UR4]`

This pattern is **identical across all shared memory sizes** (64B to 48KB) — confirming the window is a virtual construct, not physically sized by the allocation.

`SR_SWINHI` provides the upper 32 bits of the shared generic pointer — a **fixed shared-space tag**, not a per-CTA identifier. Runtime evidence on sm_87 confirms this is a constant value (~0x0001ffff) that does not vary across blocks:

In normal compiler-generated code, constructing a generic shared pointer preserves the **full shared virtual address** (already containing `CgaCtaId` in its upper bits) in the lower 32:

```sass
; Shared address already computed via ULEA:
;   R4 = (CgaCtaId << 24) + 0x400 + element_offset
; Now convert to generic 64-bit pointer:
S2R   R5, SR_SWINHI     ; upper 32 bits = fixed shared-space tag
MOV   R6, R4             ; lower 32 bits = full shared address (INCLUDES CgaCtaId!)
; → 64-bit generic shared pointer = (R5, R6)
```

The lower 32 bits of the resulting generic pointer **do encode `CgaCtaId`** (in the high bits of the 32-bit shared virtual address), because the compiler's `LEA`/`ULEA` already embedded it when computing `&smem[offset]`. This is the normal path used by standard CUDA C++ code (`cvta.shared.u64(&smem[tid])`).

The earlier observation of a "pure offset" (e.g. `low32 = 0`) came from the artificial inline-asm path `cvta.shared.u64(0)` in `test_shared_cross_cta_ptr.cu`, where a raw integer was fed directly as an argument — this does not represent normal compiler code generation. See `test_cgactaid_generic_shared.cu` (added 2026-07-03) for the definitive SASS evidence of both paths side by side.

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

### 3.4. Constant Memory — Revised

> **2026-07-01 Correction**: Contrary to earlier analysis, `cvta.const` **works** on sm_90. The prior claim was based on `cvta_all.cu` triggering a documented PTX ISA restriction (constant buffer pointers in kernel parameters). Re-testing without kernel parameters produced valid SASS, revealing the constant-to-generic conversion.

Constant memory participates in the generic address space with its own conversion formula, distinct from shared/local/global:

**`cvta.const.u64`** (constant → generic):
```sass
LDC.64  UR4, c[0x0][0xd0]           ; UR4:UR5 = constant window half-base
; For offset 0:
UIADD3   UR6, UP0, UR4, UR4, URZ     ; UR6 = 2 * UR4
UIADD3.X UR4, UR5, UR5, URZ, UP0     ; UR4 = 2 * UR5 + carry

; For offset N:
UIADD3   UR6, UP0, UR4, N, URZ       ; UR6 = UR4 + N
UIADD3   UR4, UP1, UR4, UR6, URZ     ; UR4 = 2*UR4 + N
```

**`cvta.to.const.u64`** (generic → constant):
```sass
UIADD3   UR6, UP0, -UR4, gen_val, URZ  ; UR6 = gen_val - UR4
UIADD3.X UR4, URZ, ~UR5, URZ, UP0      ; borrow propagation via complement
```

**Formula** (runtime-verified on sm_87):
```
generic_const_ptr = 2 * c[0x0][0xd0] + byte_offset
const_byte_offset  = gen_ptr - 2 * c[0x0][0xd0]

Verified: gen(c_const_data2[0]) - gen(c_const_data[0]) = 256 (exact byte offset)
          gen(c_const_data[0]) / 2 = 0x00000002_053a0000  (= c[0xd0] at runtime)
```

The **multiplication by 2** is unique to constant memory — unlike shared/local where `cvta` is a NO-OP or simple window-base addition. The doubling likely reflects the constant cache's 16-byte entry encoding (two 8-byte entries per line). `c[0x0][0xd0]` serves as the constant segment's half-base in the internal constant addressing model.

Within the constant space, `__constant__` variables are laid out sequentially from `c[0x0][0xd0]` upward. The internal address of each variable is `c[0x0][0xd0] + var_offset`, with byte offsets derived from declaration order.

Space-specific access via `LDC c[bank][Rx]` remains the primary access path — the compiler prefers it when the target bank is statically known. `cvta.const` appears when a constant address must be passed as a generic `int *` through a function boundary, though IPO often eliminates this case.

### 3.5. Unified Window Layout Diagram

> **Note on ordering**: Runtime evidence from sm_87 (Jetson Orin) confirms the generic address space layout from highest to lowest: **Shared → Local → Constant → Global**. Shared and Local occupy the same tagged region (upper 32 bits ~0x0001ffff), differentiated only by their lower 32-bit window base. Constant and Global each have distinct tags (0x00000004 and 0x00000002). The `c[0x0][0xd0]` threshold (0x00000002_053a0000 at runtime) cleanly separates the tagged region from Global.

```
64-bit Generic Virtual Address Space
══════════════════════════════════════════════════════════════
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  TAGGED REGION (upper-32 tag distinguishes from global)    │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SHARED MEMORY — per-CTA, 16 MB stride each           │  │
│  │  sm_87 tag: ~0x0001ffff   lo range: ~0x02000000       │  │
│  │  generic = (SR_SWINHI << 32) | offset                │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LOCAL MEMORY — per-thread                           │  │
│  │  sm_87 tag: ~0x0001ffff (same region as Shared)      │  │
│  │           lo range: ~0xfe000000                      │  │
│  │  generic = c[0x24]:c[0x20] + R1 + frame_off          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CONSTANT MEMORY — per-kernel, read-only             │  │
│  │  sm_87 tag: 0x00000004    lo range: ~0x0a740000      │  │
│  │  generic = 2 × c[0x0][0xd0] + byte_off               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
├────────────────────────────────────────────────────────────┤  ← c[0x0][0xd0] threshold
│                                                            │
│  GLOBAL MEMORY                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  sm_87 tag: 0x00000002 or identity-mapped            │  │
│  │  generic = physical address (identity)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

### 3.6. Generic Address Space from a Single CTA's Perspective

This section synthesizes the above window layout into a coherent picture: what does the full 64-bit generic address space look like from the viewpoint of a single CTA executing on an SM?

#### 3.6.1. Why the CTA Must Compute Its Own Shared Base

The shared memory window base is not a compile-time constant — it depends on the CTA's runtime identity:

```sass
S2UR  UR5, SR_CgaCtaId        ; UR5 = CTA identifier (assigned by hardware scheduler)
UMOV  UR4, 0x400              ; UR4 = 0x400 (window base offset constant)
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
High addresses
┌────────────────────────────────────────────┐
│  SHARED MEMORY — THIS CTA ONLY (16 MB stride)     │ ← highest
│  tag ~0x0001ffff, lo ~0x3c000000                   │
│  generic = (SR_SWINHI << 32) | offset             │
├────────────────────────────────────────────┤
│  LOCAL MEMORY WINDOWS (per-thread)               │
│  tag ~0x0001ffff (shared tag!), lo ~0x38000000     │
│  Base: c[0x24]:c[0x20]; Offset: R1 from c[0x28]  │
├────────────────────────────────────────────┤
│  CONSTANT MEMORY (per-kernel, read-only)          │
│  tag 0x00000004, generic = 2×c[0xd0] + off       │
├────────────────────────────────────────────┤  ← c[0x0][0xd0] threshold
│  GLOBAL MEMORY (all CTAs, all threads)            │ ← lowest
│  tag 0x00000002, identity-mapped                  │
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
| Shared | `SR_SWINHI` (fixed space tag) | Yes (`QSPC.E.S`) | **No** — `STS`/`LDS` decode CTA routing from the 32-bit virtual address (which already encodes `CgaCtaId` via ULEA); the upper-32 tag is discarded |
| Local | `c[0x24]` (window base upper) | Yes (`QSPC.E.L`) | **No** — `STL`/`LDL` use computed stack offset, window is implicit |
| Generic `ST.E`/`LD.E` | All 64 bits matter | Dispatch source | Hardware decodes tags to route to correct space |

This is the central tension in the design: **the generic pointer carries a space tag (which region of the address space it belongs to), but the space-specific hardware instructions (`STS`, `LDS`, `STL`, `LDL`) ignore the upper-32 tag bits entirely.** The window routing is determined by the *executing context* (current CTA for shared, current thread for local), not by the upper-32 tag.

However, for shared memory specifically, **the lower 32 bits of a compiler-constructed generic pointer already contain the full shared virtual address** — which includes `CgaCtaId` in its upper bits (via `ULEA(CgaCtaId << 24) + 0x400 + offset`). This means `CgaCtaId` **is** encoded in the generic pointer's low 32 bits, and `cvta.to.shared` passes it through to `STS`/`LDS` unchanged. The isolation is therefore two-layered: the address already targets the correct CTA (via `CgaCtaId` in the virtual address), and the hardware additionally enforces CTA scoping. The earlier claim that "CgaCtaId is not in the generic pointer" was based on the artificial `cvta.shared.u64(0)` pattern used in `test_shared_cross_cta_ptr.cu`, which passes a raw integer rather than a compiler-computed shared address. See §3.1 for the corrected SASS evidence.

The tags serve three purposes:
1. **ABI correctness**: Functions receiving `int *` can call `__isspacep_shared(p)` to determine what space the pointer targets — compiled to `QSPC.E.S`
2. **Generic load/store dispatch**: `LD.E`/`ST.E` read the tags to route to the correct memory subsystem
3. **Pointer provenance tracking in PTX**: The compiler can constant-fold `isspacep` when it statically knows the pointer's origin

But for the hot path (`STS`/`LDS`/`STL`/`LDL`), the tags are stripped, making dereference zero-overhead.

#### 3.6.5. Can a CTA Access Another CTA's Shared Memory via Generic Pointers?

> **2026-07-03 Correction**: This section has been substantially revised following the discovery that in normal compiler-generated code, the low 32 bits of a shared generic pointer contain the **full shared virtual address** (including `CgaCtaId` from the ULEA computation) — see the corrected §2.2, §3.1, and `test_cgactaid_generic_shared.cu`. The earlier analysis in this section assumed a bare offset without CgaCtaId, which only applies to the artificial `cvta.shared.u64(0)` asm path used in `test_shared_cross_cta_ptr.cu`.

This question decomposes into two scenarios:

**Scenario A: Using `STS`/`LDS` (space-specific instructions).**

When the compiler constructs a shared generic pointer normally (`cvta.shared.u64(&smem[tid])`), the low 32 bits encode the full shared virtual address of the constructing CTA: `(CgaCtaId << 24) + 0x400 + offset`. `cvta.to.shared` passes this address through unchanged, and `STS`/`LDS` uses it as-is.

If CTA 1 receives a generic shared pointer constructed by CTA 0 (via global memory or any other channel), the low 32 bits will contain CTA 0's virtual address range. When CTA 1 executes `STS [extracted_address]`, the 32-bit operand targets CTA 0's shared window at the address level. **Whether the SM hardware actually routes the access to CTA 0's shared memory — or detects a CTA-ownership mismatch and faults — is not answerable from static SASS disassembly.** The instruction stream contains no runtime `CgaCtaId` comparison; any enforcement would be at the hardware microarchitectural level.

**Scenario B: Using `ST.E`/`LD.E` (generic load/store).**

`ST.E`/`LD.E` receive the full 64-bit tagged pointer and dispatch based on the tags. If a pointer carries the shared-space tag (`SR_SWINHI`) in the upper 32 bits and a foreign CTA's virtual address in the lower 32 bits, the hardware would decode the tags and route to the shared memory subsystem with a foreign CTA's address. As with Scenario A, whether the hardware permits this depends on its protection model — not observable from offline disassembly.

**Conclusion**: The 32-bit virtual address embedded in a normal shared generic pointer's low 32 bits explicitly encodes CTA identity via `CgaCtaId`. At the instruction/address level, this means `STS`/`LDS` from a foreign CTA would receive an address targeting a different CTA's shared window. Whether the hardware actually permits the access depends on runtime protection checks that SASS disassembly cannot reveal. The issue was masked in earlier analysis because `test_shared_cross_cta_ptr.cu` used the artificial constructor `cvta.shared.u64(0)` which does not encode any CTA identity (low32 = 0), making all CTAs appear to share the same virtual address 0.

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

**Would `LD.E`/`ST.E` (generic load/store) change this?** No. The analysis in §3.6.5 draws a crucial contrast between shared and local memory: for **shared memory**, generic pointers from different CTAs carry different window bases (computed from `CgaCtaId`), so `ST.E`/`LD.E` *theoretically* could route to a foreign CTA's shared storage if the hardware permits it. For **local memory**, however, `cvta.local.u64` uses only uniform operands (`R1`, `c[0x20]`, `c[0x24]` — all identical across threads in the same warp), producing **bit-identical generic pointers** for every thread. This means even the generic load/store path has **no bit-level information** to distinguish which thread's local storage to target — the pointer value is the same for all threads. The thread isolation for local memory is therefore a *stronger* architectural guarantee than for shared memory: it holds regardless of whether the dereference path is `STL`/`LDL` (space-specific, isolation by instruction scoping) or `ST.E`/`LD.E` (generic, isolation by the impossibility of pointer differentiation).

### 4.2. Shared Memory: Architecturally NOT cross-CTA accessible

**Evidence** — from `test_shared_cross_cta_ptr.sass` (which uses the artificial `cvta.shared.u64(0)` inline-asm path; in normal compiler-generated code the low 32 already contains the full shared virtual address including `CgaCtaId` — see §3.1):

When CTA 0 constructs a generic shared pointer and CTA 1 attempts to dereference it:
1. `cvta.to.shared` is a **complete NO-OP** in SASS — the low 32 bits are used directly:
   ```sass
   MOV  R7, R2        ; raw address from generic pointer's low 32
   STS  [R7], R0      ; store to shared at that address
   ```
2. The upper 32 bits (containing the fixed shared-space tag from `SR_SWINHI`) are **discarded**
3. `STS`/`LDS` instructions route to the executing CTA's shared memory window — the virtual address in low 32 already targets the **current** CTA (it was constructed by the **current** CTA's ULEA), so cross-CTA isolation is preserved regardless
4. No `SR_CgaCtaId` or `SR_SWINHI` is consulted in the dereference callee at runtime (the address was baked in at construction time)

**Result**: The empirical result from `test_shared_cross_cta_ptr` showed that CTA 1 ended up accessing **CTA 1's own shared memory** — but this was because the test used the artificial constructor `make_shared_generic_ptr(0)`, which produces a generic pointer with `low32 = 0` (no `CgaCtaId` encoded). With such a pointer, `STS [0]` from CTA 1 naturally targets CTA 1's own shared window (virtual address 0 falls within the executing CTA's window range).

However, with a **normal compiler-constructed** generic shared pointer (where `low32 = (CgaCtaId_CTA0 << 24) + 0x400 + offset`), the 32-bit virtual address passed to `STS`/`LDS` explicitly encodes CTA 0's identity. If CTA 1 were to execute `STS` with that address, the hardware would see a virtual address within CTA 0's shared window. Whether the SM hardware actually permits this cross-CTA access — or enforces CTA-level ownership checks that cause a fault — is **an open question that cannot be resolved from static SASS disassembly**. The same question applies symmetrically to the `ST.E`/`LD.E` generic load/store path, as discussed in §3.6.5, Scenario B.

The core architectural insight that survives this correction: the SASS **instruction stream** contains no runtime checks on `SR_CgaCtaId` or `SR_SWINHI` in the dereference path — the callee simply uses the raw 32-bit address from the generic pointer. Isolation, if enforced, happens at the hardware memory-subsystem level.

### 4.3. `SR_SWINHI` — Constructor vs. Dereferencer

`SR_SWINHI` is a **fixed shared-space tag**, not a per-CTA identifier (runtime-confirmed on sm_87: constant value ~0x0001ffff across all blocks). It is read **only when constructing** a generic shared pointer and is never consulted during dereference:

Artificial asm-constructed pointer (from `test_shared_cross_cta_ptr` — note: `offset=0` is an artefact of the asm test, not normal compiler behavior):
```sass
; CONSTRUCTOR (make_shared_generic_ptr, artificial asm path):
S2R  R4, SR_SWINHI         ; read fixed shared-space tag
MOV  R2, RZ                  ; low32 = 0 (artificial; real code would use full address)
MOV  R3, R4                  ; return (low32=0, upper=fixed tag)

; DEREFERENCER (write_shared_via_generic):
MOV  R7, R2                  ; take the low 32 bits → full shared address
STS  [R7], R0                ; address already encodes CgaCtaId if from normal path
```

Normal compiler-generated code (from `test_cgactaid_generic_shared.cu`):
```sass
; CONSTRUCTOR (make_generic_shared, normal path):
S2R  R0, SR_SWINHI         ; hi32 ← fixed shared-space tag
MOV  R6, R8                  ; lo32 ← full shared address (incl. CgaCtaId from ULEA)
MOV  R7, R0                  ; hi32 ← SR_SWINHI
; return {R6=address, R7=SR_SWINHI}

; DEREFERENCER (cvta_to_shared + store):
MOV  R18, R14                ; cvta.to.shared: NO-OP — low32 passed through
STS  [R18], R11              ; write to shared at the full virtual address
```

### 4.4. Visibility Matrix

| Accessor \ Target | Thread 0 Local | Thread N Local | CTA 0 Shared | CTA N Shared | Global |
|--------------------|---------------|---------------|-------------|-------------|--------|
| **Thread 0** | RW (own) | Access to own local at offset (not target's) | RW | Access to own CTA's shared | RW |
| **Thread N** | Access to own local at offset | RW (own) | RW | Access to own CTA's shared | RW |
| **CTA 0** | N/A | N/A | RW | Access to own CTA's shared | RW |
| **CTA N** | N/A | N/A | Access to own CTA's shared | RW | RW |

The generic pointer's upper 32 bits carry a fixed space tag (e.g. `SR_SWINHI` for shared), which the space-specific dereference hardware (`STS`/`LDS`/`STL`/`LDL`) **discards**. For shared memory, the CTA identity is instead encoded in the **lower 32 bits** of the virtual address (via `CgaCtaId` from the ULEA computation), which is baked into the generic pointer at construction time and used as-is by `STS`/`LDS`. For local memory, the hardware scoping (`STL`/`LDL`) provides thread isolation independent of the pointer value — since `R1` (stack pointer from `c[0x0][0x28]`) and the window base are both uniform, a generic local pointer encodes the same stack offset for all threads; isolation arises from hardware-level enforcement, not from pointer differentiation.

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
UMOV  UR4, 0x400              ; UR4 = 0x400 (window base offset constant)
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

The stride between adjacent CTA IDs is `1 << 24 = 16,777,216 bytes = 16 MB`. Therefore each CTA maps to a **16 MB virtual address stride** in the generic address space. The actual usable shared memory is limited by hardware (e.g., ptxas defaults to 48 KB static; sm_90 supports up to ~227 KB dynamic). The `0x400` offset is sm_90-specific (absent on sm_87).

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
                                 └──16 MB stride/CTA───┘ └─1 KB base offset (sm_90)─┘
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

The `STS`/`LDS` instruction microarchitecture on sm_90 does not accept a 64-bit generic address — it uses a 32-bit virtual address. In **compiler-generated code**, this 32-bit address already encodes the CTA identity: the compiler's `ULEA(CgaCtaId << 24) + 0x400 + offset` embeds `CgaCtaId` in the address's upper bits before the address is ever used with `STS`/`LDS`. The upper 32 bits (the `SR_SWINHI` space tag) of the generic pointer are for ABI-level pointer identity only; the hardware ignores them for shared memory access.

Since the 32-bit address is constructed at pointer-creation time (before the pointer is passed across function boundaries), any pointer that arrives at a callee already targets the **constructing** CTA's shared window. The hardware additionally enforces that `STS`/`LDS` routes to the **executing** CTA's physical shared memory — this provides a second layer of isolation. The earlier characterisation of the address as a "bare offset without CgaCtaId" applied only to the artificial `cvta.shared.u64(0)` inline-asm test pattern in `test_shared_cross_cta_ptr`.

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
│   ├── test_cgactaid_generic_shared.cu   — Phase 2.2 supp.: CgaCtaId encoding in generic shared ptrs (2 constructors, side-by-side)
└── build/
    ├── ptx/
    ├── sass/
    └── cubin/
```

**Total**: 13 source files, ~80 kernel variants across all tests.

---

## 10. Open Questions

1. **Actual `c[0x0][0xd0]` threshold value**: The 64-bit constant at `c[0x0][0xd0]` used for the global-space range check cannot be observed statically — it's a runtime value. Its exact value would confirm the global address space range bound in sm_90.

2. **SR_SWINHI encoding** — **RESOLVED**: Runtime probe on sm_87 confirms `SR_SWINHI` is a **fixed shared-space tag** (~0x0001ffff) that does not vary across blocks. It marks the pointer as belonging to shared space but does **not** encode CTA identity. Per-CTA window isolation comes from `SR_CgaCtaId` in the ULEA computation, not from `SR_SWINHI`. Whether the tag value is the same on sm_90 remains unconfirmed (cannot be tested without sm_90 hardware).

3. **`QSPC.E.C` (constant space)**: `cvta.const` / `cvta.to.const` are confirmed working in parameter-free kernels (see Section 3.4 revision) with SASS formula `2 * c[0x0][0xd0] + offset`. However, the constant-space query instruction (`QSPC.E.C`) remains untested — verifying it requires running a kernel with `isspacep.const` on actual hardware.

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
| `SR_SWINHI` across 8 blocks | **Constant** (~0x0001ffff) |
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
