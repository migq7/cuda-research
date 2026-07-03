# sm_90 泛型地址空间设计 — 完整分析

**日期**: 2026-06-27
**工具链**: CUDA 12.9 (NVCC Build: V12.9.86)
**目标架构**: sm_90 (Hopper)
**基于**: `03-generic-address-space/` 测试套件（10 个探测内核）

---

## 目录

1. [概述](#1-概述)
2. [泛型指针的位布局](#2-泛型指针的位布局)
   - 2.1. [空间识别：`QSPC.E.*` 指令](#21-空间识别qspce-指令)
   - 2.2. [64 位泛型指针结构](#22-64-位泛型指针结构)
   - 2.3. [地址空间转换规则](#23-地址空间转换规则)
3. [泛型地址空间中的窗口布局](#3-泛型地址空间中的窗口布局)
   - 3.1. [Shared Memory 窗口](#31-shared-memory-窗口)
   - 3.2. [Local Memory 窗口](#32-local-memory-窗口)
   - 3.3. [Global Memory](#33-global-memory)
   - 3.4. [Constant Memory — 修订](#34-constant-memory--修订)
   - 3.5. [统一窗口布局图](#35-统一窗口布局图)
   - 3.6. [单个 CTA 视角下的泛型地址空间](#36-单个-cta-视角下的泛型地址空间)
4. [跨线程与跨 CTA 的可见性](#4-跨线程与跨-cta-的可见性)
   - 4.1. [Local Memory：架构上不可跨线程访问](#41-local-memory架构上不可跨线程访问)
   - 4.2. [Shared Memory：架构上不可跨 CTA 访问](#42-shared-memory架构上不可跨-cta-访问)
   - 4.3. [`SR_SWINHI` — 构造函数 vs 解引用函数](#43-sr_swinhi--构造函数-vs-解引用函数)
   - 4.4. [可见性矩阵](#44-可见性矩阵)
5. [运行时资源管理](#5-运行时资源管理)
   - 5.1. [常量 Bank 布局](#51-常量-bank-布局)
   - 5.2. [栈帧分配](#52-栈帧分配)
   - 5.3. [Shared Memory 窗口公式](#53-shared-memory-窗口公式)
   - 5.4. [动态 Shared Memory](#54-动态-shared-memory)
   - 5.5. [特殊寄存器目录](#55-特殊寄存器目录)
6. [`CVTA` 并非硬件指令](#6-cvta-并非硬件指令)
   - 6.1. [指针修改的鲁棒性](#61-指针修改的鲁棒性)
7. [泛型寻址下的编译器策略](#7-泛型寻址下的编译器策略)
   - 7.1. [根因：函数参数不允许地址空间限定符](#71-根因函数参数不允许地址空间限定符)
   - 7.2. [调用点的 cvta 注入](#72-调用点的-cvta-注入)
   - 7.3. [过程间优化（IPO）](#73-过程间优化ipo)
   - 7.4. [泛型与空间特定指令的选择](#74-泛型与空间特定指令的选择)
   - 7.5. [不生成逐调用点函数克隆](#75-不生成逐调用点函数克隆)
   - 7.6. [Uniform 寄存器提升](#76-uniform-寄存器提升)
   - 7.7. [指令级 cvta 融合](#77-指令级-cvta-融合)
   - 7.8. [总结：编译器决策表](#78-总结编译器决策表)
8. [关键架构推论](#8-关键架构推论)
   - 8.1. [为什么跨 CTA Shared 访问不可能](#81-为什么跨-cta-shared-访问不可能)
   - 8.2. [为什么 Local Memory 是每线程的](#82-为什么-local-memory-是每线程的)
   - 8.3. [`QSPC.E.G` 双重检查设计](#83-qspceg-双重检查设计)
9. [测试文件清单](#9-测试文件清单)
10. [待解决问题](#10-待解决问题)
11. [运行时验证（sm_87 Jetson Orin）](#11-运行时验证sm_87-jetson-orin)

---

## 1. 概述

sm_90 的泛型地址空间采用**带标记的 64 位指针方案**：高 32 位编码地址空间标识符（窗口 ID），低 32 位编码该空间内的偏移量。硬件使用专用的指令族（`QSPC.E.*`）来测试指针标记，并通过每个 CTA/每个线程的窗口寄存器将各个空间映射到虚拟地址空间中对应的窗口。

核心发现：
- **3 种不同空间**：shared、local、global — 每种都有独特的标记编码
- **不存在 `CVTA` SASS 指令** — 所有转换都是算术运算（ADD/SUB/MOV）或寄存器读取
- 通过泛型指针进行**跨线程 local 访问和跨 CTA shared 访问在架构上是不可能的**
- shared memory 以每 CTA 16 MB 的虚拟地址跨距分配（`1 << 24`）；实际可用 shared memory 受硬件限制（ptxas 默认为 48 KB 静态，sm_90 上动态最高约 227 KB）
- 运行时配置使用常量 bank 0 存放 ABI 常量，bank 4 存放内核参数

---

## 2. 泛型指针的位布局

### 2.1. 空间识别：`QSPC.E.*` 指令

PTX 的 `isspacep.*` 内建函数被编译为专用的硬件空间查询指令：

| PTX 内建函数 | SASS 指令 | 操作 |
|-------------|----------|------|
| `isspacep.shared` | `QSPC.E.S P0, RZ, [Rptr]` | 单次标记位测试 — 若为 shared 空间则置位谓词 |
| `isspacep.local` | `QSPC.E.L P0, RZ, [Rptr]` | 单次标记位测试 — 若为 local 空间则置位谓词 |
| `isspacep.global` | `QSPC.E.G` + 64 位范围检查 | 标记测试 **OR** `(ptr >= threshold)` — 双重条件 |

**关键发现（`isspacep.global`）**：SASS 生成了**两个**测试，通过 OR 组合：
1. `QSPC.E.G` — 标记位测试
2. `ISETP.GE.U32` / `ISETP.GE.U32.AND.EX` — 对来自 `c[0x0][0xd0]` 的阈值进行 64 位无符号比较

这种双重检查意味着 global 地址既可以通过标记位识别，也可以通过在特定虚拟地址范围内来识别。任一条件满足即视为 global 指针。

来自 `test_isspacep_const_fold.sass`：
```sass
/*0db0*/  QSPC.E.G P0, RZ, [R18]                  ; 标记测试
/*0dc0*/  ULDC.64 UR16, c[0x0][0xd0]              ; 阈值常量
/*0de0*/  ISETP.GE.U32.AND P2, PT, R18, UR16, PT  ; 范围检查（低 32 位）
/*0e00*/  ISETP.GE.U32.AND.EX P2, PT, R19, UR17, PT, P2  ; 范围检查（高 32 位）
/*0e20*/  PLOP3.LUT P0, PT, P0, P1, PT, 0x20, 0x0  ; P0 = tag OR range
```

#### 2.1.1. QSPC 的产生条件

`QSPC.E.*` 指令**不会**在普通指针解引用、`cvta` 操作或泛型指针跨函数边界传递时自动出现。它**仅在**程序员显式请求空间查询时生成，有两种等价路径：

1. **C++ builtin**（无需 `#include` — 编译器内置识别）：
   ```cpp
   if (__isspacep_shared(p))  { /* ... */ }   // → isspacep.shared → QSPC.E.S
   if (__isspacep_global(p))  { /* ... */ }   // → isspacep.global → QSPC.E.G
   if (__isspacep_local(p))   { /* ... */ }   // → isspacep.local  → QSPC.E.L
   ```

2. **内联 PTX 汇编**：
   ```cpp
   asm("isspacep.shared p, %1; selp.u32 %0, 1, 0, p;" ...);
   ```

Builtin 与内联 asm 等价 — builtin 被编译器展开为相同的 PTX `isspacep.*` 指令。

| 代码模式 | 产生的 SASS | 是否 QSPC？ |
|---|---|:---:|
| `int *p = &smem[tid]; *p = val;` | `STS [Rx+UR4], Ry` | 否 |
| `__noinline__` 函数接收泛型指针后解引用 | `ST.E desc[...], Ry` | 否 |
| `cvta.shared.u64` 构造泛型 shared 指针 | `S2R SR_SWINHI` + `MOV` | 否 |
| `cvta.to.shared.u64` 提取 shared 偏移 | `MOV`（NO-OP，取低 32 位） | 否 |
| 泛型指针跨函数调用边界传递 | `ST.E`/`LD.E`（不检查空间） | 否 |
| **`__isspacep_shared(p)`** | **`QSPC.E.S`** | **是** |
| **`__isspacep_global(p)`** | **`QSPC.E.G`** + 范围检查 | **是** |
| **`__isspacep_local(p)`** | **`QSPC.E.L`** | **是** |

编译器不会在地址转换、指针传递或泛型 load/store 路径中自动插入 QSPC。这也解释了 `QSPC.E.G` 双重检查的非对称性：`QSPC.E.S` 和 `QSPC.E.L` 测试的是编译器在指针构造时有意设置的标记位，单次标记测试即可。`QSPC.E.G` 额外检查地址范围，是因为 global 指针可能来自 `cudaMalloc`（低位地址，没有任何标记位），但仍需被识别为 global。

#### 2.1.2. 判断原理 — QSPC 如何确定地址空间

所有三种 QSPC 变体都从寄存器对中读取**完整的 64 位指针**，并将其与硬件已知的泛型地址空间布局进行比对。操作数 `[R18]` 在 `QSPC.E.S P0, RZ, [R18]` 中表示 64 位寄存器对 `{R18, R19}` — SASS 将连续寄存器视为一个 64 位值。

**QSPC.E.S — "此指针是否在 shared 空间范围内？"**

```sass
QSPC.E.S P0, RZ, [R18]    ; P0 = 1  若 {R18,R19} ∈ shared 空间
```

判断机制：shared 泛型指针的高 32 位携带 `SR_SWINHI`，一个**固定的 shared 空间标记**——sm_87 上的运行时证据确认这是一个常量值（~0x0001ffff），不随 CTA 变化。每 CTA 的窗口隔离由 ULEA 计算中的 `SR_CgaCtaId` 提供，而非 `SR_SWINHI`。Shared 空间在泛型地址空间的高地址段占据明确定义的连续区域 — 每 CTA 16 MB 虚拟地址跨距。QSPC.E.S 很可能在单个周期内将高位比特与硬件已知的 shared 窗口范围边界进行比较。无需额外的常量 bank 查表，因为 shared 窗口边界由架构固定（CTA 数量 × 16 MB），与内核无关。

**QSPC.E.L — "此指针是否在 local 空间范围内？"**

```sass
QSPC.E.L P0, RZ, [R18]    ; P0 = 1  若 {R18,R19} ∈ local 空间
```

判断机制：原理相同 — local 泛型指针的高 32 位携带 local 窗口基址（`c[0x24]` 高位，加上 `c[0x20]` 与栈偏移相加的进位）。Local 空间占据另一个连续的地址范围。QSPC.E.L 在运行时不查常量 bank，仅检查高位比特是否落在 local 空间边界内。

与 shared 类似，local 空间边界也是架构固定的范围，因此硬件可以将成员判定嵌入 QSPC 指令微架构内的单次范围比较操作中。

**QSPC.E.G — "此指针是否在 global 空间范围内？"**

```sass
QSPC.E.G P0, RZ, [R18]                   ; 步骤 1：标记测试
ULDC.64 UR16, c[0x0][0xd0]               ; 步骤 2：加载阈值
ISETP.GE.U32.AND P2, PT, R18, UR16, PT     ; 步骤 2a：低 32 >= 阈值？
ISETP.GE.U32.AND.EX P2, PT, R19, UR17, PT, P2  ; 步骤 2b：高 32 >= 阈值？（带进位扩展）
PLOP3.LUT P0, PT, P0, P1, PT, 0x20, 0x0    ; P0 = 标记测试 OR 范围测试
```

Global 是唯一需要**两步判断**的空间，因为指针可以通过两种不同方式成为 global：

| 情况 | 原因 | 由什么检查 |
|------|------|----------|
| 指针被显式标记为 global（编译器在构造时设置了 global 标记位） | 正常路径 — 三种空间均使用标记 | `QSPC.E.G` 标记测试 |
| 指针来自 `cudaMalloc` 或主机端分配，返回无标记的低位地址 | 向后兼容 — 旧式指针无空间标记 | `ISETP.GE.U32.*` 对阈值 `c[0xd0]` 的范围检查 |

阈值 `c[0x0][0xd0]` 是一个 64 位运行时常量。任何低于此阈值的指针，无论其标记位如何，都被视为 global。这保持了与泛型地址出现之前时代（Kepler 及更早）的兼容性，彼时所有指针 de facto 都是 global。`PLOP3.LUT` 的 OR 组合确保任一条件满足即可。

**底层原理 — 平坦 64 位空间中的非重叠范围**

```
64 位泛型地址空间（sm_87 运行时验证）
══════════════════════════════════════
 ↑ 0xFFFF_FFFF_FFFF_FFFF
 │  ┌──────────────────────┐
 │  │  带标记的区域          │  ← QSPC.E.S / QSPC.E.L: 标记 ~0x0001ffff
 │  │  Shared | Local      │
 │  │  + Constant (0x04)   │  ← QSPC.E.C: 标记 0x00000004
 │  ├──────────────────────┤  ← c[0x0][0xd0] 阈值
 │  │  GLOBAL              │  ← 标记 0x00000002 OR ptr < 阈值
 │  │  (恒等映射)           │
 ↓ 0x0000_0000_0000_0000
```

所有三种空间占据**非重叠、由架构定义的地址范围**。QSPC 硬件指令利用这一点，对指针高位进行单周期范围比较。唯一的例外是 `QSPC.E.G` 的范围回退，它仅存在于处理低位地址中无标记的旧式指针。

### 2.2. 64 位泛型指针结构

```
┌────────────────────┬────────────────────────────────┐
│  高 32 位           │  低 32 位                       │
│  (窗口 / 标记)      │  (空间内地址)                    │
├────────────────────┼────────────────────────────────┤
│  Global: 恒等映射   │  物理地址                       │
│         (无标记)    │                                │
│  Shared: SR_SWINHI │  完整 shared 虚拟地址            │
│         (固定标签)  │  (高位含 CgaCtaId)              │
│  Local:  窗口基址   │  栈偏移                         │
│  Const:  c[0xd0]   │  const 字节偏移                  │
│         ×2          │                                │
└────────────────────┴────────────────────────────────┘
```

| 地址空间 | 高 32 位来源 | 低 32 位 | cvta.to（提取） | cvta.from（构造） |
|---------|------------|---------|---------------|-----------------|
| **Global** | 与低 32 位相同（恒等） | 物理地址 | NO-OP（恒等） | NO-OP（恒等） |
| **Shared** | `SR_SWINHI`（固定 shared 空间标记） | 完整 shared 虚拟地址: `(CgaCtaId << 24) + 0x400 + off`（编译器通过 ULEA 计算） | NO-OP `MOV` — 低 32 位直接用作 shared 地址 | `S2R SR_SWINHI` + `MOV`（输入的 shared 地址原样传入低 32 位） |
| **Local** | `c[0x0][0x24]:c[0x0][0x20]` 窗口基址 | 栈帧偏移 | `IADD3 ptr - UR_base` 减法 | `IADD3 ptr + UR_base` 加法 |
| **Constant** | `c[0x0][0xd0]` 半基址（高 32 位） | const 字节偏移 | `UIADD3 -UR_base` 减法（含借位） | `UIADD3 ×2` + 偏移 |

> **关于 shared 泛型指针构成的说明**：在编译器正常生成的代码中，shared 泛型指针的低 32 位是**完整 shared 虚拟地址**——而非裸偏移量。该地址由编译器通过 `ULEA(CgaCtaId << 24) + 0x400 + 元素偏移` 计算，其高位包含 `CgaCtaId`。早期观察到的"纯偏移量"（如 `0`）仅适用于 `test_shared_cross_cta_ptr.cu` 中使用的人造内联 asm 路径（`cvta.shared.u64(0)`）。详见 §3.1 的 SASS 证据。

### 2.3. 地址空间转换规则

下表汇总了 sm_90 上每种地址空间与泛型之间的所有转换规则、SASS 实现及运行时开销。

#### 空间 → 泛型（cvta.from）

从空间特定地址构造 64 位泛型指针：

| 空间 | PTX | SASS 实现 | 开销 |
|------|-----|----------|------|
| **Global** | `cvta.global.u64` | 恒等 NO-OP — 相同比特模式 | 0 |
| **Shared** | `cvta.shared.u64` | `S2R SR_SWINHI`（固定标记 → 高 32）+ `MOV` 偏移 → 低 32 | 1 次特殊寄存器读取 |
| **Local** | `cvta.local.u64` | `IADD3 R0, P0, R1, UR4, RZ`（R1 + c[0x20] → 低 32）；`IADD3.X R2, RZ, UR5, RZ, P0`（c[0x24] + 进位 → 高 32） | 2 次整数 ALU 操作 |
| **Constant** | `cvta.const.u64` | `UIADD3` 倍乘基址 + 偏移（`2 * c[0x0][0xd0] + off`） | 2-3 次整数 ALU 操作 |

#### 泛型 → 空间（cvta.to）

从 64 位泛型指针提取空间特定地址：

| 空间 | PTX | SASS 实现 | 开销 |
|------|-----|----------|------|
| **Global** | `cvta.to.global.u64` | 恒等 NO-OP | 0 |
| **Shared** | `cvta.to.shared.u64` | 恒等 NO-OP — 低 32 位直接用作 shared 偏移 | 0 |
| **Local** | `cvta.to.local.u64` | `IADD3 R3, R6, -UR4, RZ`（gen_lo - c[0x20] → 栈偏移） | 1 次整数 ALU 操作 |
| **Constant** | `cvta.to.const.u64` | `UIADD3` 减基址（`gen_ptr - c[0x0][0xd0]` 含借位传播） | 2 次整数 ALU 操作 |

#### 空间特定 Load/Store（不涉及泛型指针）

当编译器静态知晓地址空间时，会完全绕过泛型指针：

| 操作 | 空间 | SASS |
|------|------|------|
| Load | Global | `LDG.E desc[UR][Rx.64], Ry` |
| Load | Shared | `LDS Ry, [Rx+UR4]` |
| Load | Local | `LDL Ry, [Rx]` |
| Load | Constant | `LDC Ry, c[bank][Rx]` |
| Store | Global | `STG.E desc[UR][Rx.64], Ry` |
| Store | Shared | `STS [Rx+UR4], Ry` |
| Store | Local | `STL [Rx], Ry` |

#### 泛型 Load/Store（运行时空间调度）

当编译器**无法**静态确定地址空间时，回退到泛型路径，由硬件根据指针标记在运行时调度：

| 操作 | SASS |
|------|------|
| Load | `LD.E desc[UR][Rx.64], Ry` |
| Store | `ST.E desc[UR][Rx.64], Ry` |

#### 汇总图

```
                        cvta.shared   cvta.to.shared
SHARED 偏移 ◄───────────────────► (SR_SWINHI << 32) | offset
 (32-bit)       NO-OP / S2R          (64-bit 泛型)

                        cvta.local    cvta.to.local
LOCAL  偏移 ◄───────────────────► window_base + offset
 (32-bit)       IADD3 add/sub         (64-bit 泛型)

                        cvta.global   cvta.to.global
GLOBAL 地址 ◄──────────────────────► 恒等
 (64-bit)         NO-OP / NO-OP       (64-bit 泛型)

                        cvta.const    cvta.to.const
                        cvta.const    cvta.to.const
CONSTANT 偏移 ◄───────────────────► 2 * c[0x0][0xd0] + offset
 c[bank][off]     UIADD3 ×2 add/sub     (64-bit 泛型)
```

> **注意**：在 sm_87（Jetson Orin，运行时已验证）上，内联 asm 的 `cvta.to.shared.u64` 产生不正确输出（高位被污染）。请使用 `__cvta_generic_to_shared()` builtin 替代。给定有效偏移量时，asm `st.shared.u32` 工作正常。
>
> † **Constant cvta 限制**：`cvta.const` / `cvta.to.const` 在 PTX ISA 中有文档记载（自 PTX 3.1 起），要求 sm_20+。然而，PTX ISA 注明：*"当前实现不允许在包含指向常量缓冲区的指针作为内核参数的程序中，使用泛型指针指向 const 空间变量。"* 我们的测试触发了此限制；这些指令可能在不带参数缓冲指针的内核中正常工作。sm_90 硬件上是否存在 SASS 级别的常量到泛型转换仍未验证。

---

## 3. 泛型地址空间中的窗口布局

### 3.1. Shared Memory 窗口

Shared memory 采用**窗口化虚拟地址方案**：

```sass
S2UR   UR5, SR_CgaCtaId        ; UR5 = 每 CTA 的 ID
UMOV   UR4, 0x400              ; UR4 = 0x400（窗口基址偏移常量）
ULEA   UR4, UR5, UR4, 0x18     ; UR4 = (UR5 << 24) | UR4  → shared 窗口基址
```

- **虚拟地址跨距**：每 CTA 16 MB（`1 << 24` 字节），由 ULEA 位移量 `0x18` 编码
- **sm_90 上的窗口基址偏移**：`0x400` = 1024（sm_87 上不存在，shared 变量从偏移 0 开始）
- **UR4（uniform 寄存器）** 成为 shared memory 窗口基址，用于 `STS [Rx+UR4]` / `LDS [Rx+UR4]`

该模式在**所有 shared memory 大小**（64B 到 48KB）下**完全一致** — 确认窗口是虚拟构造，其实际大小并非由分配大小物理决定。

`SR_SWINHI` 提供 shared 泛型指针的高 32 位——一个**固定的 shared 空间标记**，不是每 CTA 标识符。sm_87 上的运行时证据确认这是一个常量值（~0x0001ffff），不随 CTA 变化：

在编译器正常生成的代码中，构造泛型 shared 指针时将**完整 shared 虚拟地址**（高位已含 `CgaCtaId`）保留在低 32 位：

```sass
; shared 地址已通过 ULEA 计算完毕：
;   R4 = (CgaCtaId << 24) + 0x400 + 元素偏移
; 现在转换为 64 位泛型指针：
S2R   R5, SR_SWINHI     ; 高 32 位 = 固定 shared 空间标记
MOV   R6, R4             ; 低 32 位 = 完整 shared 地址（含 CgaCtaId!）
; → 64 位泛型 shared 指针 = (R5, R6)
```

泛型指针的低 32 位**确实编码了 `CgaCtaId`**（在 32 位 shared 虚拟地址的高位），因为编译器在计算 `&smem[偏移]` 时已经通过 `LEA`/`ULEA` 将其嵌入。这是标准 CUDA C++ 代码的正常路径（`cvta.shared.u64(&smem[tid])`）。

早期观察到的"纯偏移量"（如 `low32 = 0`）来自于 `test_shared_cross_cta_ptr.cu` 中的人造内联 asm 路径 `cvta.shared.u64(0)`，其中将裸整数裸传给 cvta——这并不代表正常的编译器代码生成。参见 `test_cgactaid_generic_shared.cu`（2026-07-03 新增）获取两种路径并排对比的 SASS 证据。

### 3.2. Local Memory 窗口

Local memory 使用每线程隔离模型，窗口基址为来自 bank 0 的 64 位常量：

```sass
LDC   R1, c[0x0][0x28]       ; 初始栈指针（所有线程统一）
ULDC  UR4, c[0x0][0x20]      ; local 窗口基址（低 32 位）
ULDC  UR5, c[0x0][0x24]      ; local 窗口基址（高 32 位）
VIADD R1, R1, -frame_size    ; 分配栈帧（可选）
```

64 位 local 窗口基址 `c[0x0][0x24]:c[0x0][0x20]` 是一个**uniform 常量** — 对内核中所有线程相同。来自 `c[0x0][0x28]` 的栈指针 `R1` 同样在**所有线程中统一**（经 sm_87 运行时测试确认——256 线程全部收到相同的 `R1` 值）。

> **运行时证据**：在 Jetson Orin（sm_87）上以 256 线程进行的运行时探测确认，`R1`（以及 `gen_lo = R1 + UR4`、`gen_hi`、第一个局部变量的栈相对 `local_addr`）在所有线程中均不变。Local memory 的线程隔离**不是**源于每线程 R1 值——它完全由 `STL`/`LDL` 硬件级别强制执行。

**`cvta.local.u64`**（local→泛型）：
```sass
IADD3   R0, P0, R1, UR4, RZ          ; gen_lo = R1 + window_base_lo  (uniform: 两个操作数均为 uniform)
IADD3.X R2, RZ, UR5, RZ, P0, !PT     ; gen_hi = window_base_hi + carry
```

**`cvta.to.local.u64`**（泛型→local）：
```sass
IADD3   R3, R6, -UR4, RZ             ; addr = gen_lo - window_base_lo
STL     [R3], R2                       ; local store（硬件限定到当前线程）
; 或
LDL     R5, [R5]                       ; local load（硬件限定到当前线程）
```

减法 `gen_lo - UR4` 剥离窗口基址，恢复栈相对偏移量。由于 `gen_lo` 和 `UR4` 均为 uniform，每个线程为同一局部变量计算相同的栈相对地址。然而，`STL`/`LDL` 硬件无条件限定到执行线程——一个线程解引用另一个线程的泛型 local 指针时，访问的是该偏移量下**自己**的存储，而非目标线程的。隔离在指令层面，而非地址层面。

### 3.3. Global Memory

Global memory **没有窗口转换** — 64 位泛型指针即为物理虚拟地址：

```sass
; cvta.global.u64 → IADD3/IADD3.X（与 global 窗口基址进行 64 位加法）
LDC   R0, c[0x0][0x20]       ; global 窗口基址（低 32 位）
ULDC  UR4, c[0x0][0x24]      ; global 窗口基址（高 32 位）
IADD3 R0, P0, R0, R4, RZ     ; 指针加窗口基址
IADD3.X R11, R5, UR4, RZ, P0, !PT

; cvta.to.global.u64 → NO-OP（恒等）
HFMA2.MMA R7, -RZ, RZ, 0, 0 ; NOP 占位
; 指针原封不动通过
```

当输入已经是 global 指针时，`cvta.global` 与 `cvta.to.global` 完全一致（均为 NO-OP）。

### 3.4. Constant Memory — 修订

> **2026-07-01 更正**：与早期分析相反，`cvta.const` 在 sm_90 上**可以工作**。此前的判断基于 `cvta_all.cu` 触发了一条 PTX ISA 文档注明的实现限制（内核参数中包含常量缓冲区指针）。在无内核参数的情况下重新测试，产生了有效的 SASS，揭示了常量到泛型转换。

常量内存参与泛型地址空间，但使用自己独特的转换公式，不同于 shared/local/global：

**`cvta.const.u64`**（常量 → 泛型）：
```sass
LDC.64  UR4, c[0x0][0xd0]           ; UR4:UR5 = 常量窗口半基址
; 偏移量为 0 时：
UIADD3   UR6, UP0, UR4, UR4, URZ     ; UR6 = 2 * UR4
UIADD3.X UR4, UR5, UR5, URZ, UP0     ; UR4 = 2 * UR5 + carry

; 偏移量为 N 时：
UIADD3   UR6, UP0, UR4, N, URZ       ; UR6 = UR4 + N
UIADD3   UR4, UP1, UR4, UR6, URZ     ; UR4 = 2*UR4 + N
```

**`cvta.to.const.u64`**（泛型 → 常量）：
```sass
UIADD3   UR6, UP0, -UR4, gen_val, URZ  ; UR6 = gen_val - UR4
UIADD3.X UR4, URZ, ~UR5, URZ, UP0      ; 按位取反传播借位
```

**公式**（sm_87 运行时已验证）：
```
generic_const_ptr = 2 * c[0x0][0xd0] + byte_offset
const_byte_offset  = gen_ptr - 2 * c[0x0][0xd0]

已验证：gen(c_const_data2[0]) - gen(c_const_data[0]) = 256（精确字节偏移）
          gen(c_const_data[0]) / 2 = 0x00000002_053a0000  （= 运行时的 c[0xd0]）
```

**乘 2** 是常量内存独有的——不同于 shared/local 的 NO-OP 或简单窗口基址加法。倍乘可能与常量缓存的 16 字节条目编码有关（每行两个 8 字节条目）。`c[0x0][0xd0]` 在内部常量寻址模型中充当常量段的半基址。

在常量空间内部，`__constant__` 变量从 `c[0x0][0xd0]` 向上顺序排列。每个变量的内地址为 `c[0x0][0xd0] + var_offset`，其中字节偏移由声明顺序决定。

通过 `LDC c[bank][Rx]` 的空间特定访问仍是主要访问路径——当目标 bank 静态已知时编译器优先使用它。`cvta.const` 在常量地址必须作为泛型 `int *` 跨函数边界传递时出现，但 IPO 往往会消除这种情况。

### 3.5. 统一窗口布局图

> **关于排序的说明**：sm_87（Jetson Orin）运行时证据确认泛型地址空间从高到低的排序为：**Shared → Local → Constant → Global**。Shared 和 Local 占据同一标记区域（高 32 位 ~0x0001ffff），仅通过低 32 位窗口基址区分。Constant 和 Global 各有独立标记（0x00000004 和 0x00000002）。`c[0x0][0xd0]` 阈值（运行时为 0x00000002_053a0000）将带标记区域与 Global 清晰分隔。

```
64 位泛型虚拟地址空间
══════════════════════════════════════════════════════════════
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  带标记区域（高位标记与 global 区分）                        │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SHARED MEMORY — 每 CTA, 各 16 MB 虚拟跨距              │  │
│  │  sm_87 标记: ~0x0001ffff   低位范围: ~0x02000000      │  │
│  │  generic = (SR_SWINHI << 32) | offset                │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LOCAL MEMORY — 每线程                               │  │
│  │  sm_87 标记: ~0x0001ffff（与 Shared 同一区域）         │  │
│  │           低位范围: ~0xfe000000                      │  │
│  │  generic = c[0x24]:c[0x20] + R1 + frame_off          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CONSTANT MEMORY — 每核函, 只读                      │  │
│  │  sm_87 标记: 0x00000004   低位范围: ~0x0a740000      │  │
│  │  generic = 2 × c[0x0][0xd0] + byte_off               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
├────────────────────────────────────────────────────────────┤  ← c[0x0][0xd0] 阈值
│                                                            │
│  GLOBAL MEMORY                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  sm_87 标记: 0x00000002 或恒等映射                    │  │
│  │  generic = 物理地址 (恒等)                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

### 3.6. 单个 CTA 视角下的泛型地址空间

本节将上述窗口布局综合为一幅连贯的图景：从单个正在 SM 上执行的 CTA 的角度来看，完整的 64 位泛型地址空间是什么样的？

#### 3.6.1. 为什么 CTA 必须自己计算 Shared 基址

Shared memory 窗口基址不是编译期常量 — 它取决于 CTA 的运行时标识：

```sass
S2UR  UR5, SR_CgaCtaId        ; UR5 = CTA 标识符（由硬件调度器分配）
UMOV  UR4, 0x400              ; UR4 = 0x400（窗口基址偏移常量）
ULEA  UR4, UR5, UR4, 0x18     ; UR4 = (CgaCtaId << 24) | 0x400
```

`SR_CgaCtaId` 是一个**只读硬件特殊寄存器**。其值由硬件调度器在 CTA 被发射到 SM 时动态分配。CTA 在编译期不知道自己的 ID — ID 取决于它被分配到哪个 SM、CTA 的分发顺序以及同一 SM 上共有多少个 CTA。因此：

- CTA **必须**在运行时读取 `SR_CgaCtaId` 来计算自己的窗口基址
- 窗口基址 `UR4` 存储在 **uniform 寄存器**（warp 内所有线程共享）中，因为同一 warp 的所有线程属于同一个 CTA
- 计算结果 `UR4 = (CgaCtaId << 24) | 0x400` 为该 CTA 的 shared memory 提供了一个唯一的 16 MB 虚拟地址范围

每 CTA 的窗口隔离由 `ULEA` 通过 `SR_CgaCtaId` 处理 — `SR_SWINHI`（见上文）是固定标记，不是 CTA 标识符。

#### 3.6.2. 单个 CTA 看到的地址空间

从单个 CTA 的视角来看，64 位泛型地址空间划分为以下区域：

```
高地址
┌────────────────────────────────────────────┐
│  SHARED MEMORY — 仅限本 CTA（16 MB 跨距）            │ ← 最高
│  标记 ~0x0001ffff, 低位 ~0x3c000000               │
│  generic = (SR_SWINHI << 32) | offset            │
├────────────────────────────────────────────┤
│  LOCAL MEMORY 窗口（每线程）                       │
│  标记 ~0x0001ffff（与共享相同！），低位 ~0x38000000   │
│  Base: c[0x24]:c[0x20]; Offset: R1（来自 c[0x28]）│
├────────────────────────────────────────────┤
│  CONSTANT MEMORY（每核函, 只读）                    │
│  标记 0x00000004, generic = 2×c[0xd0] + off      │
├────────────────────────────────────────────┤  ← c[0x0][0xd0] 阈值
│  GLOBAL MEMORY（所有 CTA、所有线程）                 │ ← 最低
│  标记 0x00000002, 恒等映射                         │
└────────────────────────────────────────────┘
低地址 (0x0)
```

#### 3.6.3. 各空间访问规则总结

| 操作 | 地址操作数 | 寻址空间 | 跨 CTA？ | 跨线程？ |
|------|-----------|----------|---------|---------|
| `LDG`/`STG` via desc | 32b 偏移 + 64b 描述符 | Global | 是 | 是 |
| `LDS`/`STS` | 32b 偏移（+ uniform 基址 UR4） | **本 CTA 的 shared** | 否 | 是（CTA 内） |
| `LDL`/`STL` | 32b 偏移（栈相对） | **本线程的 local** | 否 | 否 |
| `LDC` | bank# + 偏移 | Constant | 所有相同 | 所有相同 |
| `LD.E`/`ST.E`（泛型） | 64b 标记指针 | 硬件根据指针标记分派 | 未知† | 未知† |

> † `LD.E`/`ST.E` 是否允许通过手动标记的泛型指针进行跨 CTA 或跨线程访问，取决于硬件保护逻辑，无法从静态 SASS 中观察。详见下方 3.6.5 节。

#### 3.6.4. 指针标记的使用与被忽略

64 位泛型指针的高 32 位携带标记信息：

| 空间 | 高 32 位 | `QSPC.E.*` 是否使用？ | 空间特定 ld/st 是否使用？ |
|------|----------|---------------------|-------------------------|
| Global | 0 或与低 32 位相同（恒等） | 是 (`QSPC.E.G`) | 不需要（LDG/STG 使用描述符） |
| Shared | `SR_SWINHI`（固定标记） | 是 (`QSPC.E.S`) | **否** — `STS`/`LDS` 通过 32 位虚拟地址解码 CTA 路由（该地址已通过 ULEA 编码了 `CgaCtaId`）；高 32 位标记被丢弃 |
| Local | `c[0x24]`（窗口基址高位） | 是 (`QSPC.E.L`) | **否** — `STL`/`LDL` 使用计算后的栈偏移，窗口是隐式的 |
| 泛型 `ST.E`/`LD.E` | 全部 64 位均有意义 | 分派依据 | 硬件解码标记以路由到正确空间 |

这是整个设计的核心矛盾：**泛型指针携带了完整的空间标签（属于地址空间的哪个区域），但空间特定硬件指令（`STS`、`LDS`、`STL`、`LDL`）完全忽略高 32 位标记**。窗口路由由*执行上下文*（shared 为当前 CTA、local 为当前线程）决定，不由高 32 位标记决定。

然而，对于 shared memory，**编译器构造的泛型指针的低 32 位已包含完整 shared 虚拟地址**——其高位通过 `ULEA(CgaCtaId << 24) + 0x400 + offset` 包含了 `CgaCtaId`。这意味着 `CgaCtaId` **确实**编码在泛型指针的低 32 位中，`cvta.to.shared` 将其原封不动传给 `STS`/`LDS`。因此隔离是双层的：地址本身已指向正确的 CTA（虚拟地址中含 `CgaCtaId`），硬件额外施加 CTA 限定。早期"指针中不含 CgaCtaId"的说法是基于 `test_shared_cross_cta_ptr.cu` 中使用的人造 `cvta.shared.u64(0)` 模式。详见 §3.1 更正后的 SASS 证据。

标记位服务于三个目的：
1. **ABI 正确性**：接收 `int *` 的函数可调用 `__isspacep_shared(p)` 来判断指针所属的空间 — 编译为 `QSPC.E.S`
2. **泛型 load/store 分派**：`LD.E`/`ST.E` 读取标记以路由到正确的内存子系统
3. **PTX 中的指针来源追踪**：当编译器静态知道指针来源时，可常量折叠 `isspacep`

但在热路径（`STS`/`LDS`/`STL`/`LDL`）上，标记被剥离，使解引用零开销。

#### 3.6.5. CTA 能否通过泛型指针访问其他 CTA 的 Shared Memory？

> **2026-07-03 更正**：本节经过大幅修订。发现编译器正常生成代码中，shared 泛型指针的低 32 位包含**完整 shared 虚拟地址**（通过 ULEA 计算包含 `CgaCtaId`）——参见更正后的 §2.2、§3.1 及 `test_cgactaid_generic_shared.cu`。早期分析假定不含 CgaCtaId 的裸偏移量，仅适用于 `test_shared_cross_cta_ptr.cu` 中使用的人造 `cvta.shared.u64(0)` asm 路径。

这个问题可以分解为两种场景：

**场景 A：使用 `STS`/`LDS`（空间特定指令）。**

当编译器正常构造 shared 泛型指针时（`cvta.shared.u64(&smem[tid])`），低 32 位编码了构造 CTA 的完整 shared 虚拟地址：`(CgaCtaId << 24) + 0x400 + offset`。`cvta.to.shared` 将该地址原封不动传递，`STS`/`LDS` 按原样使用。

如果 CTA 1 接收到 CTA 0 构造的泛型 shared 指针（通过全局内存或其他通道），低 32 位包含 CTA 0 的虚拟地址范围。当 CTA 1 执行 `STS [extracted_address]` 时，32 位操作数在地址层面指向 CTA 0 的 shared 窗口。**SM 硬件是实际将访问路由到 CTA 0 的 shared memory——还是检测到 CTA 归属不匹配并触发 fault——无法仅凭静态 SASS 反汇编回答。**指令流中没有任何运行时 `CgaCtaId` 比较；任何强制措施都在硬件微架构层面。

**场景 B：使用 `ST.E`/`LD.E`（泛型 load/store）。**

`ST.E`/`LD.E` 接收完整的 64 位标记指针并根据标记分派。如果指针的高 32 位携带 shared 空间标记（`SR_SWINHI`），低 32 位携带其他 CTA 的虚拟地址，硬件会解码标记并将包含其他 CTA 地址的请求路由到 shared 内存子系统。与场景 A 一样，硬件是否允许取决于其保护模型——无法从离线反汇编中观察。

**结论**：正常 shared 泛型指针低 32 位中嵌入的 32 位虚拟地址通过 `CgaCtaId` 显式编码了 CTA 身份。在指令/地址层面，这意味着其他 CTA 执行 `STS`/`LDS` 时会收到指向不同 CTA shared 窗口的地址。硬件是否实际允许访问取决于 SASS 反汇编无法揭示的运行时保护检查。此问题在早期分析中被掩盖，因为 `test_shared_cross_cta_ptr.cu` 使用了人造构造函数 `cvta.shared.u64(0)`，不编码任何 CTA 身份（low32 = 0），使所有 CTA 看起来共享同一个虚拟地址 0。

---

## 4. 跨线程与跨 CTA 的可见性

### 4.1. Local Memory：架构上不可跨线程访问

**证据** — 来自 `test_local_cross_thread_ptr.sass`：

当线程 0 构造一个指向其 local 变量的泛型指针，线程 1 尝试解引用时：
1. 泛型指针的**高 32 位在到达被调用函数之前被剥离**（仅从 shared memory 加载低 32 位：`LDS R6, [UR4]`）
2. 被调用函数中的 `cvta.to.local` 是对线程 uniform 常量的**纯减法**：`IADD3 R3, R6, -UR4, RZ`，其中 `UR4 = c[0x0][0x20]`
3. 结果地址用于 `STL [R3]` / `LDL [R5]` — 该指令**始终访问执行线程自己的 local memory**
4. 被调用函数中没有引用任何 `SR_TID` 或每线程寄存器

**结论**：线程 1 解引用线程 0 的泛型 local 指针时，只会访问**线程 1 自己的 local memory** 中指针编码的偏移量位置。硬件没有机制将 `STL`/`LDL` 重定向到另一个线程的 local memory 窗口。

**如果改用 `LD.E`/`ST.E`（泛型 load/store）会改变结果吗？** 不会。§3.6.5 的分析揭示了 local 与 shared 之间的关键差异：对于 **shared memory**，不同 CTA 的泛型指针携带不同的窗口基址（由 `CgaCtaId` 计算得来），因此 `ST.E`/`LD.E` 在理论*可能*路由到其他 CTA 的 shared 存储（如果硬件允许的话）。但对于 **local memory**，`cvta.local.u64` 仅使用 uniform 操作数（`R1`、`c[0x20]`、`c[0x24]` — 同一 warp 内所有线程完全相同），为每个线程产生**比特级完全相同的泛型指针**。这意味着即使走泛型 load/store 路径，硬件也**没有任何比特级信息**可以区分应该访问哪个线程的 local 存储 — 指针值对所有线程都一样。因此 local memory 的线程隔离是比 shared memory *更强的*架构保证：无论解引用路径是 `STL`/`LDL`（空间特定，通过指令级限定实现隔离）还是 `ST.E`/`LD.E`（泛型，因指针无法区分线程而天然隔离），结果都一样。

### 4.2. Shared Memory：架构上不可跨 CTA 访问

**证据** — 来自 `test_shared_cross_cta_ptr.sass`（该测试使用人造的 `cvta.shared.u64(0)` 内联 asm 路径；编译器正常生成的代码中低 32 位已含完整 shared 虚拟地址包括 `CgaCtaId`——详见 §3.1）：

当 CTA 0 构造一个泛型 shared 指针，CTA 1 尝试解引用时：
1. `cvta.to.shared` 在 SASS 中是**完全 NO-OP** — 直接使用低 32 位：
   ```sass
   MOV  R7, R2        ; 来自泛型指针低 32 位的原始地址
   STS  [R7], R0      ; 存储到该地址的 shared 位置
   ```
2. 高 32 位（包含固定 shared 空间标记 `SR_SWINHI`）被**丢弃**
3. `STS`/`LDS` 指令路由到执行 CTA 的 shared memory 窗口——低 32 位中的虚拟地址本来就指向**当前**CTA（由当前 CTA 的 ULEA 构造），因此跨 CTA 隔离无论如何都能保证
4. 解引用被调用函数中运行时没有引用任何 `SR_CgaCtaId` 或 `SR_SWINHI`（地址在构造时已固化）

**结论**：`test_shared_cross_cta_ptr` 的实证结果显示 CTA 1 最终访问了**CTA 1 自己的 shared memory**——但这是因为该测试使用了人造构造函数 `make_shared_generic_ptr(0)`，其产生的泛型指针 `low32 = 0`（不含 `CgaCtaId` 编码）。使用这种指针时，CTA 1 的 `STS [0]` 自然落在执行 CTA 自己的 shared 窗口范围内（虚拟地址 0 在任意 CTA 的窗口内）。

然而，对于**编译器正常构造**的泛型 shared 指针（`low32 = (CgaCtaId_CTA0 << 24) + 0x400 + offset`），传给 `STS`/`LDS` 的 32 位虚拟地址明确编码了 CTA 0 的身份。如果 CTA 1 用该地址执行 `STS`，硬件会看到一个落在 CTA 0 shared 窗口内的虚拟地址。SM 硬件是*允许*这种跨 CTA 访问，还是*施加* CTA 级别的所有权检查并触发 fault——这是**一个无法仅凭静态 SASS 反汇编回答的开放问题**。同一问题对称地适用于 `ST.E`/`LD.E` 泛型 load/store 路径，如 §3.6.5 场景 B 的讨论。

经过本次修正后依然成立的核心架构洞见：解引用路径的 SASS **指令流**中对 `SR_CgaCtaId` 或 `SR_SWINHI` 没有任何运行时检查——被调用函数只是单纯使用泛型指针中的原始 32 位地址。隔离（若存在）由硬件内存子系统层面强制执行。

### 4.3. `SR_SWINHI` — 构造函数 vs 解引用函数

`SR_SWINHI` 是一个**固定的 shared 空间标记**，不是每 CTA 标识符（运行时在 sm_87 上确认：所有 block 中均为常量 ~0x0001ffff）。它仅在构造泛型 shared 指针时读取，解引用时从不查阅：

人造 asm 构造的指针（来自 `test_shared_cross_cta_ptr`——注意：`offset=0` 是 asm 测试的产物，不代表正常编译器行为）：
```sass
; 构造函数 (make_shared_generic_ptr, 人造 asm 路径):
S2R  R4, SR_SWINHI         ; 读取固定的 shared 空间标记
MOV  R2, RZ                  ; low32 = 0 (人造的; 正常代码会使用完整地址)
MOV  R3, R4                  ; 返回 (low32=0, upper=固定标记)

; 解引用函数 (write_shared_via_generic):
MOV  R7, R2                  ; 取低 32 位 → 完整 shared 地址
STS  [R7], R0                ; 地址已编码 CgaCtaId（如果来自正常路径）
```

编译器正常生成的代码（来自 `test_cgactaid_generic_shared.cu`）：
```sass
; 构造函数 (make_generic_shared, 正常路径):
S2R  R0, SR_SWINHI         ; hi32 ← 固定 shared 空间标记
MOV  R6, R8                  ; lo32 ← 完整 shared 地址 (通过 ULEA 含 CgaCtaId)
MOV  R7, R0                  ; hi32 ← SR_SWINHI
; 返回 {R6=地址, R7=SR_SWINHI}

; 解引用函数 (cvta_to_shared + 存储):
MOV  R18, R14                ; cvta.to.shared: NO-OP — 低 32 位直通
STS  [R18], R11              ; 向该完整虚拟地址写入 shared
```

### 4.4. 可见性矩阵

| 访问者 \ 目标 | 线程 0 Local | 线程 N Local | CTA 0 Shared | CTA N Shared | Global |
|--------------|-------------|-------------|-------------|-------------|--------|
| **线程 0** | RW（自己的） | 访问自己 local 对应偏移 | RW | 访问自己 CTA 的 shared | RW |
| **线程 N** | 访问自己 local 对应偏移 | RW（自己的） | RW | 访问自己 CTA 的 shared | RW |
| **CTA 0** | N/A | N/A | RW | 访问自己 CTA 的 shared | RW |
| **CTA N** | N/A | N/A | 访问自己 CTA 的 shared | RW | RW |

泛型指针的高 32 位携带固定空间标记（如 shared 的 `SR_SWINHI`），空间特定解引用硬件（`STS`/`LDS`/`STL`/`LDL`）**丢弃它们**。对于 shared memory，CTA 身份编码在**低 32 位**虚拟地址中（通过 ULEA 计算中的 `CgaCtaId`），在指针构造时固定下来并被 `STS`/`LDS` 直接使用。对于 local memory，硬件限定（`STL`/`LDL`）提供与指针值无关的线程隔离——由于 `R1`（来自 `c[0x0][0x28]` 的栈指针）和窗口基址均为 uniform，泛型 local 指针对所有线程编码相同的栈偏移；隔离来自硬件级别强制，而非指针差异。

---

## 5. 运行时资源管理

### 5.1. 常量 Bank 布局

NVIDIA GPU 将常量内存划分为独立寻址的 **bank**，每个 bank 为 64 KB。语法 `c[bank][offset]` 标识一个特定的 bank（0–7 或更多）及其 64 KB 区域内的字节偏移量。在整个仓库的所有项目中观察到了三个 bank：

| Bank | 内容 | 证据 |
|------|------|------|
| `c[0x0]` | 内核 ABI 上下文 + 用户参数 | 每个内核（通过 `test_kernel_param_map.cu` 系统扫描） |
| `c[0x3]` | `__constant__` 变量数据 | `02-cvta-analysis/cvta_all.cu`：`__constant__ int cdata[64]` → `LDC R10, c[0x3][R10]` |
| `c[0x4]` | `__device__` 全局变量地址 | `k1`–`k10`：`g_dump` 的地址 → `LDC.64 R*, c[0x4][RZ]` |

Bank `c[0x1]` 和 `c[0x2]` 在所有受测内核中从未被引用 — 它们可能为空，或在 bank 0 的参数空间耗尽时作为溢出保留。

#### Bank 0 — 内部布局

Bank 0 是主要 bank。其低半部分（偏移量 0x000–0x20F）保存**执行上下文头** — 描述内核资源环境的运行时计算值。高半部分（偏移量 0x210+）保存**用户内核参数**。

```
c[0x0] Bank（64 KB）
┌──────────────────────────────────────────┐
│ 0x000: 隐式参数 (blockDim.x 等)             │  执行
│ 0x020: local/global 窗口基址 (低 32)          │  上下文头
│ 0x024: local/global 窗口基址 (高 32)          │  (运行时
│ 0x028: 栈指针 R1（uniform）                   │   计算)
│  ...   (未映射区域 — 配置数据？)                │
│ 0x0D0: global 地址空间阈值                    │
│  ...                                         │
│ 0x208: 全局内存描述符基址                      │
├──────────────────────────────────────────┤
│ 0x210: 用户内核参数槽位 0 (arg0)               │  用户
│ 0x218: 用户内核参数槽位 1 (arg1)               │  内核
│  ...   (每个槽位 0x8, 按顺序递增)                │  参数
└──────────────────────────────────────────┘
```

执行上下文头**不是**用户参数列表的扩展 — 其值来自 launch configuration（`blockDim`、网格维度）和运行时资源计算（每线程栈大小、local/shared 内存分配、窗口基址）。用户参数从 0x210 开始，构成 bank 的尾部。

**ABI / 运行时常量（固定偏移量）**：

| 偏移量 | 宽度 | 内容 |
|--------|------|------|
| `c[0x0][0x00]` | 32-bit | `blockDim.x`（网格维度） |
| `c[0x0][0x20]` | 32-bit | Local/global 窗口基址（低 32 位） |
| `c[0x0][0x24]` | 32-bit | Local/global 窗口基址（高 32 位） |
| `c[0x0][0x28]` | 32-bit | 初始栈指针（R1）— 所有线程统一 — **每个内核都存在** |
| `c[0x0][0xd0]` | 64-bit | Global 地址空间范围阈值（用于 `QSPC.E.G` 回退检查） |
| `c[0x0][0x208]` | 64-bit | Global memory 描述符基址（用于 `STG.E desc[UR][...]`） |

**内核参数槽位（随参数数量变化）**：

内核参数从偏移量 `0x210` 开始，每个槽位递增 `0x8`：

| 内核 | 参数 | 偏移量 | 类型 |
|------|------|--------|------|
| `k1(int a)` | a | `0x210` | 32-bit int |
| `k2(int a, int b)` | a, b | `0x210` | 64-bit 对（两个 int 打包） |
| `k3(int *a, int b)` | a | `0x210` | 64-bit ptr |
| | b | `0x218` | 32-bit int |
| `k5(int*, int*, int*)` | ptr0, ptr1, ptr2 | `0x210`, `0x218`, `0x220` | 3×64-bit ptr |
| `k7(int*,..., int*)` | 6 ptrs | `0x210`–`0x238` | 6×64-bit ptr |

每个 64 位指针或 32 位标量占用一个 `0x8` 对齐的槽位。模式从 `0x210` 顺序递增。

#### 槽位分配规则

全局内存描述符位于 `c[0x0][0x208]`（一个固定的 ABI 偏移量，以 `ULDC.64` 加载，并用作 `STG.E desc[UR][Rx.64], Ry` 中的 `desc[UR]` 操作数）。与描述符不同，用户参数槽位从 `c[0x0][0x210]` 开始，根据参数类型以 32 位（`ULDC`）或 64 位（`LDC.64`）访问。

**打包规则**：两个连续的 32 位标量参数可以**打包**到单个 8 字节槽位中。`k2(int a, int b)` 只引用 `c[0x0][0x210]`（以 `LDC.64` 加载）— 第二个 32 位标量不占用独立槽位。64 位指针始终独占一个完整槽位，并强制下一个参数从下一个 8 字节边界开始：

```
k2(int a, int b):            | a(4B) + b(4B) |          ← 打包在槽位 0 (0x210)
k3(int *a, int b):           | a(8B ptr)      | b(4B)    ← 槽位 0 (0x210) + 槽位 1 (0x218)
k4(int *a, int *b):          | a(8B ptr)      | b(8B)    ← 槽位 0 + 槽位 1
```

#### Bank 3 — `__constant__` 变量存储

在 `02-cvta-analysis/cvta_all.cu` 中观察到，`__constant__ int cdata[64]` 通过 `LDC R10, c[0x3][R10]` 访问。编译器计算 bank 3 内的字节偏移量（使用 `LOP3.LUT` 计算 `cdata_offset & 0xFC | 0xC0`），并以此偏移量发出 `LDC` 指令。Bank 3 保存模块中声明的所有 `__constant__` 变量。

#### Bank 4 — `__device__` 全局变量地址

```sass
LDC.64  R*, c[0x4][RZ]     ; __device__ 全局变量的地址
```

`c[0x4][0x0]` 保存指向内核输出目标的指针 — 即内核写入的 `__device__` 全局变量（如 `g_dump`、`g_storage`）的地址。偏移量始终为 `RZ`（零寄存器），意味着该值位于 bank 4 的起始位置。这与 bank 0 的参数槽位相互独立。

### 5.2. 栈帧分配

栈帧在内核入口处通过以下指令分配：
```sass
LDC   R1, c[0x0][0x28]      ; 加载初始栈指针
VIADD R1, R1, immediate      ; 分配帧 (R1 -= frame_size)
```

**帧大小规则**（来自 `test_stack_frame_vary.sass` 和 `test_window_sizes.sass`）：

| 内核 | 局部变量 | VIADD | 帧大小 |
|------|----------|-------|--------|
| `kf_empty` | 无 | 无 | 0 |
| `kf_1_int` 至 `kf_1024_int` | 被优化消除 | 无 | 0 |
| `kf_local_ptr_dump` | 2 个 int（地址被获取） | `0xfffffff8` (-8) | 8 字节 |
| `k_combined_1KB_256B` | `int arr[64]` | `0xffffff00` (-256) | 256 字节 |
| `k_local_1KB` | `int arr[256]` | `0xfffffc00` (-1024) | 1024 字节 |
| `k_local_4KB` | `int arr[1024]` | `0xfffff000` (-4096) | 4096 字节 |

关键观察：
- 可保留在寄存器中的标量局部变量 → **无栈帧**
- 未被优化消除的数组 → 帧 = `sizeof(array)` 并向上对齐
- 地址被获取的局部变量（如 `&a` 传递给函数）→ 即使标量也强制栈分配

**寄存器溢出**：在所有测试内核中（最多 128 个活跃 int）均未观察到。在 sm_90 上使用 `-O3` 时，编译器的常量传播和死代码消除非常激进。真正的溢出需要非常量依赖的代码且超过 255 个活跃寄存器。

### 5.3. Shared Memory 窗口公式

窗口基址计算是一个两指令模式，在所有使用 `__shared__` 的内核中都可观察到：

```sass
S2UR  UR5, SR_CgaCtaId        ; UR5 = CTA 标识符
UMOV  UR4, 0x400              ; UR4 = 0x400（窗口基址偏移常量）
ULEA  UR4, UR5, UR4, 0x18     ; UR4 = (UR5 << 24) | 0x400
```

#### 5.3.1. 16 MB / CTA 的推导过程

sm_90 上的 `ULEA`（Uniform Logic/Arithmetic Extended）指令的计算公式为：
```
ULEA R, A, B, shift → R = (A << shift) + B
```

代入实际操作数：
- `A = UR5 = SR_CgaCtaId`
- `B = UR4 = 0x400`
- `shift = 0x18 = 24`

```
UR4 = (CgaCtaId << 24) + 0x400
```

相邻 CTA ID 之间的跨度为 `1 << 24 = 16,777,216 字节 = 16 MB`。因此每个 CTA 在泛型地址空间中映射到一个 **16 MB 虚拟地址跨距**。实际可用 shared memory 受硬件限制（如 ptxas 默认为 48 KB 静态；sm_90 支持最高约 227 KB 动态）。`0x400` 偏移量是 sm_90 特有的（sm_87 上不存在）。

#### 5.3.2. 来自 `test_shared_window_granularity.cu` 的确认证据

新增的探测内核从多个角度提供了确认：

| 测试 | 发现 |
|------|------|
| `k_no_shared` | 无 `__shared__` → **完全无 ULEA/UMOV** |
| `k_single_block` | 即使 gridDim=1（CgaCtaId 始终为 0），ULEA 仍以 `0x400`/`0x18` 出现 |
| `k_two_shared_arrays` | 两个 `__shared__` 数组 → **两个 ULEA** 指令。`arr1` 使用 `ULEA(Id, 0x400, 0x18)`，`arr2` 使用 `ULEA(Id, 0x800, 0x18)` — 第二个数组的基址偏移了第一个数组的大小（1 KB = 0x400）。位移量 `0x18` 两者**完全相同**。 |
| `k_shared_max_48KB` / `k_shared_mid_32KB` / `k_shared_mid_16KB` | 三个内核，分别 16KB、32KB 和 48KB shared — **逐条指令完全相同的 SASS**。在 48KB 边界处 `UMOV 0x400` 和 `ULEA 0x18` 未变。 |
| `k_use_blockidx` | `SR_CTAID.X`（blockIdx）**与 `SR_CgaCtaId` 分开读取** — blockIdx 仅用于数据值，不用于 shared 窗口地址。 |
| `k_export_all_generic_ptrs` | `SR_SWINHI` 通过 `S2UR` 读取并用作泛型 shared 指针的高 32 位（固定空间标记）。CTA 窗口标识通过 ULEA 计算（用于 STS/LDS），而 SR_SWINHI 提供 shared 空间的标记位（用于 generic pointer construction）。 |
| `k_builtin_vs_asm_cvta` | 向 `__noinline__` 函数传递泛型指针时，编译器使用常规寄存器 `LEA` 而非 `ULEA` — 相同的公式 `(CgaCtaId << 24) + 0x400`，但为满足调用约定存储在 `R*` 而非 `UR*` 寄存器中。`__cvta_generic_to_shared()` builtin 和 asm `cvta.to.shared` 在 callee 中产生相同的 `STS [Rx]` 模式。 |

#### 5.3.3. 总结

```
SharedMemoryWindowBase(CTA_id) = (CgaCtaId << 24) + 0x400
                                 └──16 MB 跨距/CTA───┘ └─1 KB 基址偏移 (sm_90)─┘
```

此公式在以下情况中保持不变：
- 所有 shared memory 大小（64 B 到 48 KB）
- 静态和动态（`extern __shared__`）shared memory
- 单 block 和多 block 内核
- 单数组和多数组配置
- Uniform 寄存器路径（ULEA 用于直接 STS/LDS）和常规寄存器路径（LEA 用于泛型指针传递）

实际的物理 shared memory 分配是运行时参数 — 硬件虚拟化窗口并执行边界检查。

#### 5.3.4. 0x400 偏移量不是什么：mbarrier 状态存储

一个自然的假说是保留的 0x000–0x3FF 区域存储硬件 `mbarrier`（异步 barrier）状态条目。通过 `test_mbarrier_shared_reserved.cu` 进行了测试，该测试在用户 `__shared__` 数组之前声明 `cuda::barrier<cuda::thread_scope_block>` 对象（编译器识别的 mbarrier 状态），并观察产生的 ULEA 偏移量：

| 内核 | barrier 数量 | 首个 barrier 偏移 | 首个用户数组偏移 |
|------|:---:|------|------|
| `k_t2_baseline` | 0 | — | **0x400** |
| `k_t2_one_barrier` | 1 | **0x400** | **0x408** |
| `k_t2_three_barriers` | 3 | **0x400** | **0x418** |
| `k_t2_barrier_between_arrays` | 1（在 arr1/arr2 之间） | **0x500** | **0x400**（arr1），**0x508**（arr2） |

编译器将 `cuda::barrier` 状态与任何其他 `__shared__` 变量等同处理 — 从偏移量 0x400 开始按顺序布局，并将后续变量按比例后推。mbarrier 状态**没有**被放入保留的 0x000–0x3FF 区域。

这推翻了 mbarrier 存储假说。0x400 偏移量仍然是一个架构常量，其消费者（如果有）无法通过 CUDA C++ 变量声明观察到 — 它可能是硬件定义的对齐空洞、`__syncthreads()` 暂存区域，或者仅仅是一个在用户可见的 shared memory 布局中没有逻辑消费者的固定窗口基址。

### 5.4. 动态 Shared Memory

`extern __shared__` 内核产生的 SASS 与等效的静态 shared 内核**逐条指令完全相同**。编译器对动态 shared memory 的处理与静态完全一致 — `ULEA`/`UMOV` 模式不变，且不添加额外的常量 bank 引用。

运行时（通过 `cudaLaunchKernel`）配置 shared memory 窗口大小；硬件执行访问边界检查。这是一个优雅的设计：编译后的代码与实际分配大小无关。

### 5.5. 特殊寄存器目录

| 寄存器 | 宽度 | 读取方式 | 用途 |
|--------|------|----------|------|
| `SR_TID.X` | 32 | `S2R` | 块内线程索引 |
| `SR_CTAID.X` | 32 | `S2R` | 网格内块索引 |
| `SR_CgaCtaId` | 32 | `S2R`, `S2UR` | Cooperative Group Array ID（每 CTA） |
| `SR_SWINHI` | 32 | `S2R`, `S2UR` | 固定 shared 空间标记 — 泛型 shared 指针的高 32 位。不编码 CTA 身份（运行时确认：所有 block 中均为常量）。 |

**`S2R` vs `S2UR`**：
- `S2R`（Special to Register）→ 通用 `R*` 寄存器（每线程）
- `S2UR`（Special to Uniform Register）→ `UR*` 寄存器（warp-uniform，针对广播优化）

当 warp 中所有线程共享同一 CTA 时（常见情况），编译器将 `SR_SWINHI` 提升为 `S2UR`，从而可以使用 uniform 寄存器的 `STS`/`LDS` 寻址。

---

## 6. `CVTA` 并非硬件指令

所有 CVTA PTX 变体都被编译为基础操作 — sm_90 SASS 中不存在 `CVTA` 操作码：

| PTX 操作 | SASS 实现 | 开销 |
|----------|----------|------|
| `cvta.to.global` | 恒等 NO-OP | 0 周期（直通） |
| `cvta.global` | 恒等 NO-OP | 0 周期 |
| `cvta.to.shared` | 恒等 NO-OP | 0 周期（直接使用低 32 位） |
| `cvta.shared` | `S2R SR_SWINHI` + MOV | 1 次特殊寄存器读取 |
| `cvta.to.local` | `IADD3` 减窗口基址 | 1 次整数减法 |
| `cvta.local` | `IADD3` / `IADD3.X` 加窗口基址 | 1–2 次整数加法 |

指针格式被精心设计，使最常见的操作（shared 解引用、global 解引用）需要零转换开销。

### 6.1. 指针修改的鲁棒性

来自 `test_generic_ptr_surgery.sass`：即使泛型指针的比特被直接修改（高字节清零、bit 32 翻转、高 32 位替换为常量），**ptxas 仍然生成 STS**（空间特定存储）。编译器不会通过内联汇编追踪指针来源 — 它信任 PTX `st.shared.u32` 指令。硬件的 `STS` 指令无视指针比特，总是路由到当前 CTA 的 shared 窗口。

---

## 7. 泛型寻址下的编译器策略

统一泛型地址空间的存在不仅是硬件抽象 — 它在多个层面从根本上塑造了**编译器的代码生成策略**。本节梳理了地址空间转换需求对编译器行为的各种影响。

### 7.1. 根因：函数参数不允许地址空间限定符

CUDA C++ 不允许在函数参数上使用地址空间限定符：

```cpp
__device__ void foo(__shared__ int *p);    // ❌ 非法的 CUDA C++
__device__ void foo(int *p);               // ✅ p 是泛型指针
```

任何 `__device__` 函数的 `int *` 参数都隐式为**泛型** 64 位指针。这意味着每个将 shared、local 或 global 指针传入 `__device__` 函数的调用点，都**必须**先将指针转换为泛型形式。编译器在每个调用点插入对应的 `cvta.*` 指令。

### 7.2. 调用点的 cvta 注入

当内核以 shared 指针调用 `__noinline__` 函数时：

```cpp
__shared__ int smem[256];
foo(&smem[tid]);   // 编译器在此处插入 cvta.shared 再调用
```

编译器生成：
```sass
S2R  R9, SR_SWINHI        ; 构造泛型 shared 指针
MOV  R6, offset             ;  {R9,R6} = 64 位泛型指针
CALL.REL.NOINC foo_addr     ; 通过寄存器传递泛型指针
```

对于 local 指针：
```sass
IADD3   R0, P0, R1, UR4, RZ       ; cvta.local: 泛型 = 栈 + 窗口
IADD3.X R2, RZ, UR5, RZ, P0, !PT
CALL.REL.NOINC foo_addr
```

对于 global 指针，无需转换（恒等），调用点不变。

### 7.3. 过程间优化（IPO）

来自 `01-shared-ptr-analysis`（同一工具链下运行），当编译器对所有调用点具有全程序可见性时，会应用激进的 IPO：

**场景 1 — 单一地址空间调用者**：若 `__noinline__ foo(int *p)` 仅被 shared 指针调用，编译器可将被调用函数**特化**：在 `foo` 内部插入 `cvta.to.shared` 并直接使用 `STS`/`LDS`，避免泛型 `ST.E`/`LD.E` 的开销。

**场景 2 — 纯读取函数**：若 `foo(int *p) { return *p; }`，编译器完全消除指针 — 在调用点以正确的空间特定指令（`LDS`/`LDG`/`LDL`）完成 load，仅将**值**传给被调用函数：
```sass
; 以 shared 指针调用处：
LDS  R8, [offset+UR4]       ; 在调用点执行 ld.shared
st.param.b32 [param0], R8    ; 传值，不传指针
call foo                     ; foo 接收的是 int，不是指针
```

被调用函数退化为 `return param0;` — 地址空间问题被完全绕过。

**场景 3 — 多地址空间调用者**：若 `foo` 被 shared、global 和 local 指针调用，编译器无法特化，必须在单一的共享函数体内使用泛型 `ST.E`/`LD.E`。

### 7.4. 泛型与空间特定指令的选择

编译器的指令选择取决于它能证明什么关于指针的来源：

| 编译器已知信息 | 生成的 SASS | 原因 |
|---|---|---|
| 指针*确定*是 shared（内联，或 IPO 的单一调用者） | `STS`/`LDS` | 零开销 shared 访问 |
| 指针*确定*是 global | `STG`/`LDG`（通过描述符） | 零开销 global 访问 |
| 指针*确定*是 local | `STL`/`LDL` | 零开销 local 访问 |
| 指针来源**未知**（多空间调用者、跨函数边界） | `ST.E`/`LD.E` | 硬件运行时调度的 |

`ST.E`/`LD.E` 回退是一种刻意的性能折衷：避免为每个空间生成函数克隆而增大二进制体积，同时接受硬件调度的开销。在 sm_90 上，`ST.E` 相比 `STS` 多一次指针状态检查，但硬件对此路径高度优化。

### 7.5. 不生成逐调用点函数克隆

一个关键的非行为：编译器**不会**自动为每个地址空间生成 `__noinline__` 函数的多个专用克隆。来自 `01-shared-ptr-analysis`：

```
generic_write（被 shared、global 和 local 指针调用）
→ 仅一个函数体，使用 ST.E，三个调用点共享
```

三个调用点使用同一个 `CALL.REL.NOINC` 目标。编译器原则上可将 `generic_write` 克隆为 `generic_write_shared`（用 `STS`）、`generic_write_global`（用 `STG`）和 `generic_write_local`（用 `STL`），但它选择不这样做。这很可能是代码体积优化 — `ST.E` 调度的开销相比代码复制是可接受的。

**例外**：若程序员显式按地址空间模板化函数，模板实例化自然产生每空间副本。这是一种可选的优化模式。

### 7.6. Uniform 寄存器提升

当 warp 内所有线程共享同一 CTA（常见情况），编译器将 `SR_SWINHI` 的读取从每线程寄存器提升为 **uniform 寄存器**：

```sass
; 每线程路径（warp-divergent 偏移）：
S2R  R5, SR_SWINHI        ; 所有 32 线程各执行一次

; Uniform 路径（warp-uniform 偏移）：
S2UR UR7, SR_SWINHI        ; 读一次，广播给 warp 内所有线程
```

这减少了寄存器压力，并可使用 `STS [UR4], R5` — 基于 uniform 寄存器的索引 store，每个 warp 执行一次而非每个线程一次。当编译器能证明 shared 偏移量是 warp-uniform（如所有线程访问同一个 `smem[0]`）时，选择 uniform 路径。

### 7.7. 指令级 cvta 融合

当 `cvta.to.shared` 紧接 `st.shared`（或 `ld.shared`）时，两者**融合**为单条 `STS`/`LDS`：

```sass
; 未融合（逻辑 PTX）：
cvta.to.shared.u64 tmp, gen_ptr
cvt.u32.u64 addr, tmp
st.shared.u32 [addr], val

; 融合后（实际 SASS）：
MOV  R7, R2            ; 低 32 位 = 偏移量（cvta 是 NO-OP）
STS  [R7], R0           ; 单条 shared store
```

`cvta.to.shared` + `cvt.u32.u64` 对折叠为单条 `MOV`，`STS` 直接消费结果。这一融合使得泛型指针的 shared memory 解引用在热路径上零开销："转换"在 SASS 层面不存在。

相同的融合也适用于 `cvta.to.local` + `STL`/`LDL`，此时减法折叠到寻址模式中：
```sass
IADD3 R3, R6, -UR4, RZ    ; cvta.to.local 减法
STL   [R3], R2              ; local store
; 两者保持两条指令（减法 + store），未完全融合，
; 但减法仅为单周期整数操作。
```

### 7.8. 总结：编译器决策表

| 场景 | 编译器动作 | 性能 |
|---|---|---|
| 直接 shared 访问（`smem[tid] = x`） | `STS [Rx+UR4], Ry` | 零开销 |
| 直接 global 访问（`gmem[tid] = x`） | `STG.E desc[UR][R.64], Ry` | 零开销 |
| `__noinline__` 调用，shared 指针，单一调用者 | IPO：用 `STS` 特化被调用函数 | 零开销 |
| `__noinline__` 调用，shared 指针，多空间 | 共享函数体中泛型 `ST.E` | +1 状态检查 |
| 纯读取 `__noinline__`，指针参数 | IPO：调用点 load，传值 | 零开销（指针消除） |
| `cvta.shared` + `st.shared` 相邻 | 融合为单条 `STS` | 零开销 |
| 来自 asm 的 `cvta.to.shared` 然后 `st.shared` | `MOV` + `STS`（无操作转换） | 1 MOV 开销 |
| Warp-uniform shared 访问 | `S2UR` + `STS [UR]`（uniform 路径） | 降低寄存器压力 |
| Warp-divergent shared 访问 | `S2R` + `STS [R]`（每线程路径） | 使用全部寄存器 |

---

## 8. 关键架构推论

### 8.1. 为什么跨 CTA Shared 访问不可能

sm_90 上的 `STS`/`LDS` 指令微架构不接受 64 位泛型地址——它使用 32 位虚拟地址。在**编译器生成的代码**中，这个 32 位地址已经编码了 CTA 身份：编译器的 `ULEA(CgaCtaId << 24) + 0x400 + offset` 在地址被 `STS`/`LDS` 使用之前，就将 `CgaCtaId` 嵌入了地址的高位。泛型指针的高 32 位（`SR_SWINHI` 空间标记）仅用于 ABI 级别的指针标识；硬件在 shared memory 访问时忽略它们。

由于 32 位地址在指针构造时（指针跨函数边界传递之前）就已计算完毕，任何到达被调用函数的指针已经指向**构造该指针的** CTA 的 shared 窗口。硬件额外强制 `STS`/`LDS` 路由到**执行** CTA 的物理 shared memory——这提供了第二层隔离。早期将地址描述为"不含 CgaCtaId 的裸偏移量"仅适用于 `test_shared_cross_cta_ptr` 中人造的 `cvta.shared.u64(0)` 内联 asm 测试模式。

### 8.2. 为什么 Local Memory 是每线程的

`STL`/`LDL` 指令使用 32 位偏移量，**隐式限定**在当前线程的 local memory 存储。窗口基址由 uniform 常量（`c[0x20:0x24]`）配置，来自 `c[0x28]` 的栈指针 `R1` 在**所有线程中统一**（sm_87 运行时已确认）。线程隔离完全在 `STL`/`LDL` 硬件层面实现——这些指令无条件路由到执行线程的 local 存储，与地址寄存器值无关。由于 `R1` 和窗口基址均为 uniform，每个线程为同一变量计算相同的泛型 local 指针；硬件在访问点进行隔离，而非在地址上。

### 8.3. `QSPC.E.G` 双重检查设计

`isspacep.global` 的双重检查（标记比特 OR 范围检查）表明 sm_90 支持可能不携带显式空间标记比特的旧式 global 指针。任何低于阈值 `c[0x0][0xd0]` 的指针都被视为 global，无论标记如何 — 这提供了与旧指针格式以及通过 `cudaMalloc` 分配的（低位地址范围内的）内存的向后兼容性。

---

## 9. 测试文件清单

```
03-generic-address-space/
├── scripts/compile.sh
├── src/
│   ├── test_isspacep_const_fold.cu       — 阶段 1.1：QSPC.E.* 指令发现
│   ├── test_generic_ptr_surgery.cu       — 阶段 1.2：指针位修改鲁棒性
│   ├── test_window_sizes.cu              — 阶段 1.3：窗口大小扫描（16 个内核）
│   ├── test_local_cross_thread_ptr.cu    — 阶段 2.1：跨线程 local 可见性
│   ├── test_shared_cross_cta_ptr.cu      — 阶段 2.2：跨 CTA shared 可见性
│   ├── test_warp_uniform_vs_divergent.cu — 阶段 2.3：UR* vs R* 寄存器路径
│   ├── test_kernel_param_map.cu          — 阶段 3.1：常量 bank 布局（10 个内核）
│   ├── test_stack_frame_vary.cu          — 阶段 3.2：栈帧大小变化（12 个内核）
│   ├── test_register_pressure_spill.cu   — 阶段 3.3：寄存器压力（8 个内核）
│   ├── test_shared_window_granularity.cu — 阶段 1.3 补充：ULEA 确认（7 个内核）
│   ├── test_mbarrier_shared_reserved.cu   — 阶段 1.3 补充：0x400 保留区域测试（6 个内核）
│   ├── test_dynamic_shared_config.cu     — 阶段 3.4：动态 vs 静态 shared（6 个内核）
│   ├── test_cgactaid_generic_shared.cu   — 阶段 2.2 补充：CgaCtaId 在泛型 shared 指针中的编码（2 种构造方式并排对比）
└── build/
    ├── ptx/
    ├── sass/
    └── cubin/
```

**总计**：13 个源文件，所有测试中共约 80 个内核变体。

---

## 10. 待解决问题

1. **`c[0x0][0xd0]` 阈值的实际值**：用于 global 空间范围检查的 64 位常量位于 `c[0x0][0xd0]`，无法通过静态方式观察 — 它是运行时值。其确切值将确认 sm_90 的 global 地址空间范围边界。

2. **`SR_SWINHI` 编码** — **已解决**：sm_87 运行时探针确认 `SR_SWINHI` 是**固定的 shared 空间标记**（~0x0001ffff），不随 block 变化。它标记指针属于 shared 空间，但**不**编码 CTA 身份。每 CTA 的窗口隔离来自 ULEA 计算中的 `SR_CgaCtaId`，而非 `SR_SWINHI`。该标记值在 sm_90 上是否相同尚未确认（无 sm_90 硬件无法测试）。

3. **`QSPC.E.C`（constant 空间）**：`cvta.const` / `cvta.to.const` 在无内核参数的内核中确认可用（见 3.4 节修订），SASS 显示 `2 * c[0x0][0xd0] + off` 的转换公式。但常量空间查询指令（`QSPC.E.C`）仍未测试——需要实际运行含有 `isspacep.const` 的内核才能验证。

4. **泛型 `LD.E`/`ST.E` 分派机制**：当编译器回退到泛型 load/store 时（如 `01-shared-ptr-analysis` 中所见），硬件根据指针标记进行分派。内部分派逻辑（标记解码器、窗口查找、边界检查）无法从二进制反汇编中观察。

5. **通过手动构造的泛型指针进行跨 CTA 访问**：如果手动构造一个 64 位泛型指针，将高 32 位设为其他 CTA 的 `SR_SWINHI` 值，并通过 `ST.E`/`LD.E`（泛型 store/load）解引用，硬件会如何行为？
    - 路由到同一 SM 上目标 CTA 的 shared memory？
    - 跨互联路由到不同 SM 的 shared memory？
    - 因执行 CTA 的 `CgaCtaId` 与指针嵌入的窗口 ID 不匹配而 fault？

    无法通过静态 SASS 反汇编确定，需要在物理硬件上进行运行时测试才能解答。该问题的架构意义重大：如果硬件*允许*通过 `ST.E` 进行同一 SM 内的跨 CTA 访问，则意味着泛型地址空间在泛型 load/store 路径中提供了所有 shared memory 窗口的平坦视图，不设 CTA 级别的保护 — 只有 `STS`/`LDS` 强制执行隐式隔离。

6. **0x400 保留区域的消费者**：sm_90 上每个 CTA 的 shared memory 窗口基址中固定的 0x400（1 KB）偏移量已被确认不变，但其消费者仍然未知。`test_mbarrier_shared_reserved.cu` 推翻了 mbarrier 存储假说。**sm_87 运行时证据显示 shared 变量从偏移 0 开始** — 保留区域在 sm_90 上存在、在 sm_87 上缺失，但未测试其他架构前不能认定是 sm_90 独有的。

---

## 11. 运行时验证（sm_87 Jetson Orin）

在 Jetson Orin（sm_87）上执行了三个运行时探针，以补充离线 SASS 分析（sm_90）。关键发现如下：

### 11.1. Local Memory — R1 统一，STL/LDL 硬件隔离

> **探针**：`runtime-probes/01-local-memory/probe_local_memory.cu`

| 观测 | 结果 |
|---|---|
| R1 跨 256 线程 | **统一**（相同） |
| `gen_lo`、`gen_hi`、`local_addr` | **统一** |
| T1 用 T0 的泛型 local 指针读取 | 返回 T1 **自己**的值 |
| 窗口基址（`gen_lo - local_addr`） | 统一 |
| **结论** | R1 不是 per-thread。线程隔离在 `STL`/`LDL` 硬件级别实现，而非通过 per-thread 地址差异。 |

该发现纠正了静态分析的初始假设（见 3.2 节）——`c[0x0][0x28]` 不是 per-thread 值，而是统一的；每线程隔离发生在指令级别。

### 11.2. CTA 标识 — SR_SWINHI 是固定标记

> **探针**：`runtime-probes/02-cta-identity/probe_cta_identity.cu`

| 观测 | 结果 |
|---|---|
| `SR_SWINHI` 跨 8 个 block | **固定**（~0x0001ffff） |
| `gen_lo`（cvta.shared 低 32b） | 跨 block 固定（包含窗口基址） |
| `arr[0]` shared offset（`__cvta_generic_to_shared`） | **0x00000000**（sm_87 无 0x400 reserved 区域） |
| `arr[0]` 值验证（每 block 写入 `bid+tid`） | 值 == `blockIdx.x` 对所有 block → **CTA 隔离确认** |
| **结论** | `SR_SWINHI` 是固定 shared 空间标记，不是每 CTA 标识符。CTA 窗口隔离来自 ULEA 计算中的 `SR_CgaCtaId`，而非 `SR_SWINHI`。 |

在 sm_90 上观察到的 0x400 保留区域在 sm_87 上不存在——用户 shared 变量从偏移量 0 开始。

### 11.3. 地址转换 — Builtin vs 内联 asm

> **探针**：`runtime-probes/03-cross-cta-generic/probe_cross_cta_generic.cu`

| 观测 | 结果 |
|---|---|
| `__cvta_generic_to_shared(&arr[8])` builtin | 正确（0x20） |
| asm `cvta.to.shared.u64` → `cvt.u32.u64` | **错误** — 高位被污染（如 0xb1000020） |
| asm `st.shared.u32 [offset]` 给定有效偏移量 | 正常 |
| 跨 CTA 泛型指针测试 | **sm_87 上不可行** — `cvta.shared` 为所有 block 产生相同的 64 位指针 |
| **结论** | 在 sm_87 上，内联 asm 的 `cvta.to.shared.u64` 产生不正确输出。应使用 `__cvta_generic_to_shared()` builtin 替代。给定有效偏移量时，asm 的 `st.shared.u32` 正常。 |

---

**总结**：sm_87 运行时证据解决了 SR_SWINHI 编码问题（待解决问题 2，现已关闭），确认了 R1 是统一的，并揭示了该平台上 cvta.to.shared 的内联 asm 特有行为。SASS 分析的核心架构模型——基于窗口的寻址、每 CTA/每线程的硬件隔离、固定的空间标记——在 sm_90（SASS）和 sm_87（运行时）之间均成立。
