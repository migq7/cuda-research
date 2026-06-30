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
3. [泛型地址空间中的窗口布局](#3-泛型地址空间中的窗口布局)
   - 3.1. [Shared Memory 窗口](#31-shared-memory-窗口)
   - 3.2. [Local Memory 窗口](#32-local-memory-窗口)
   - 3.3. [Global Memory](#33-global-memory)
   - 3.4. [Constant Memory：不在泛型地址空间中](#34-constant-memory不在泛型地址空间中)
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

---

## 1. 概述

sm_90 的泛型地址空间采用**带标记的 64 位指针方案**：高 32 位编码地址空间标识符（窗口 ID），低 32 位编码该空间内的偏移量。硬件使用专用的指令族（`QSPC.E.*`）来测试指针标记，并通过每个 CTA/每个线程的窗口寄存器将各个空间映射到虚拟地址空间中对应的窗口。

核心发现：
- **3 种不同空间**：shared、local、global — 每种都有独特的标记编码
- **不存在 `CVTA` SASS 指令** — 所有转换都是算术运算（ADD/SUB/MOV）或寄存器读取
- 通过泛型指针进行**跨线程 local 访问和跨 CTA shared 访问在架构上是不可能的**
- shared memory 窗口粒度为每个 CTA 固定 16 MB（`1 << 24`）
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

判断机制：shared 泛型指针的高 32 位携带 `SR_SWINHI`，该值通过 `(CgaCtaId << 24) | 0x400` 编码了 CTA 的窗口 ID。Shared 空间在泛型地址空间的高地址段占据明确定义的连续区域 — 每个 CTA 一个 16 MB 窗口。QSPC.E.S 很可能在单个周期内将高位比特与硬件已知的 shared 窗口范围边界进行比较。无需额外的常量 bank 查表，因为 shared 窗口边界由架构固定（CTA 数量 × 16 MB），与内核无关。

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
64 位泛型地址空间
══════════════════════════════════════
 ↑ 0xFFFF_FFFF_FFFF_FFFF
 │  ┌──────────────────────┐
 │  │  带标记的区域          │  ← QSPC.E.S / QSPC.E.L: 单次范围检查
 │  │  (shared + local,     │
 │  │   互不重叠)            │
 │  ├──────────────────────┤  ← c[0x0][0xd0] 阈值（运行时）
 │  │  GLOBAL              │  ← QSPC.E.G 标记测试 OR ptr < 阈值
 │  │  (恒等映射)           │
 ↓ 0x0000_0000_0000_0000
```

所有三种空间占据**非重叠、由架构定义的地址范围**。QSPC 硬件指令利用这一点，对指针高位进行单周期范围比较。唯一的例外是 `QSPC.E.G` 的范围回退，它仅存在于处理低位地址中无标记的旧式指针。

### 2.2. 64 位泛型指针结构

```
┌────────────────────┬────────────────────┐
│  高 32 位           │  低 32 位           │
│  (窗口 / 标记)      │  (偏移量)           │
├────────────────────┼────────────────────┤
│  Global: 恒等映射   │  物理地址           │
│         (无标记)    │                    │
│  Shared: SR_SWINHI │  shared 偏移       │
│  Local:  窗口基址   │  栈偏移             │
└────────────────────┴────────────────────┘
```

| 地址空间 | 高 32 位来源 | 低 32 位 | cvta.to（提取） | cvta.from（构造） |
|---------|------------|---------|---------------|-----------------|
| **Global** | 与低 32 位相同（恒等） | 物理地址 | NO-OP（恒等） | NO-OP（恒等） |
| **Shared** | `SR_SWINHI`（每 CTA 窗口寄存器） | shared memory 偏移 | NO-OP `MOV` — 直接使用低 32 位 | `S2R SR_SWINHI` + MOV |
| **Local** | `c[0x0][0x24]:c[0x0][0x20]` 窗口基址 | 栈帧偏移 | `IADD3 ptr - UR_base` 减法 | `IADD3 ptr + UR_base` 加法 |

---

## 3. 泛型地址空间中的窗口布局

### 3.1. Shared Memory 窗口

Shared memory 采用**窗口化虚拟地址方案**：

```sass
S2UR   UR5, SR_CgaCtaId        ; UR5 = 每 CTA 的 ID
UMOV   UR4, 0x400              ; UR4 = 0x400（窗口粒度常量）
ULEA   UR4, UR5, UR4, 0x18     ; UR4 = (UR5 << 24) | UR4  → shared 窗口基址
```

- **每个 CTA 的窗口大小**：16 MB（`1 << 24` 字节），由 ULEA 位移量 `0x18` 编码
- **窗口粒度**：`0x400` = 1024（可能为 shared memory 最小分配粒度的字节数）
- **UR4（uniform 寄存器）** 成为 shared memory 窗口基址，用于 `STS [Rx+UR4]` / `LDS [Rx+UR4]`

该模式在**所有 shared memory 大小**（64B 到 48KB）下**完全一致** — 确认窗口是虚拟构造，其实际大小并非由分配大小物理决定。

`SR_SWINHI` 提供 shared 泛型指针的高 32 位：
```sass
S2R   R5, SR_SWINHI     ; 高 32 位 = shared 窗口高寄存器
MOV   R6, R4             ; 低 32 位 = shared 偏移
; → 64 位泛型 shared 指针 = (R5, R6)
```

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

### 3.4. Constant Memory：不在泛型地址空间中

常量内存**不参与** 64 位泛型指针寻址方案。与 shared、local、global 不同——它们都通过 `cvta.*` 转换将地址映射到统一泛型空间——常量内存使用独立的 `<bank, 偏移量>` 寻址模式，从不经过 64 位泛型指针。

**空间特定访问**使用 `LDC` 指令配合 bank 索引操作数：

```sass
LDC R10, c[0x3][R10]       ; bank 3, 字节偏移量来自 R10
```

`c[0x3]` 不是一个泛型地址范围 — 它是直接编码在 sass 操作码中的独立寻址模式。该指令告诉硬件"从常量 bank 3 的字节偏移 R10 处读取"，完全绕过了 `LD.E`/`ST.E` 用于 shared/local/global 的泛型指针分派逻辑。

**sm_90 上不存在常量到泛型转换的证据**：

1. **`cvta.const` 和 `cvta.to.const` 被 ptxas 拒绝** — 在 `cvta_all.cu` 中尝试使用产生了编译错误。如果 PTX 层面没有 cvta.const，就无从降级到 SASS。

2. **IPO 消除了跨函数边界的常量指针** — 当 `read_const_ptr(const int *p)` 以 `__constant__` 实参调用时，编译器在**调用点**完成 `LDC c[0x3][...]`，只将值（一个 32 位 int）传入，而非指针。被调用函数在 PTX 中的签名为 `.param .b32`（值），而非 `.param .b64`（泛型指针）：
   ```ptx
   .func (.param .b32 func_retval0) _Z14read_const_ptrPKi(
       .param .b32 _Z14read_const_ptrPKi_param_0   ; ← .b32 值, 不是 .b64 泛型指针
   )
   ```

3. **PTX 对常量使用直接符号寻址** — `mov.u64 %rd19, cdata;` 直接引用常量符号，不涉及 `cvta`：
   ```ptx
   .const .align 4 .b8 cdata[256];
   mov.u64 %rd19, cdata;
   ```

4. **`QSPC.E.C`（常量空间查询）无法测试** — 由于 ptxas 拒绝 `cvta.const`，无法构造一个标记为"常量"的 64 位泛型指针来测试 `isspacep.const`。该指令可能存在，也可能不存在。

**架构原因**：常量内存在根本上不同于其他空间：
- 它是**只读的**，对内核中所有线程**统一可见**
- 它通过专用的缓存路径（常量缓存）访问，与 L1/shared memory 子系统分离
- 与 shared/local 指针不同，它们必须通过泛型 `int *` 函数参数流动，而常量数据很少以指针方式传递 — 通常直接按名称访问
- `LDC` 中嵌入的 bank 索引已提供充分的寻址信息 — 不需要 64 位虚拟地址

```
Shared / Local / Global 空间               Constant 空间
═════════════════════════                  ════════════
  通过泛型 64 位指针访问                      通过 LDC c[bank][offset] 访问
  cvta.* 在空间与泛型之间转换                   sm_90 无 cvta.const
  QSPC.E.{S,G,L} 用于空间检测                 QSPC.E.C 未验证
  STS/LDS/STG 是空间特定的                    LDC 是空间特定的
```

### 3.5. 统一窗口布局图

> **关于排序的说明**：Global 已确认占据低位地址（低于阈值 `c[0x0][0xd0]`）。Shared 和 local 均占据带标记的高位地址范围，但两者的**相对垂直顺序无法**从静态 SASS 中确定 — 图中将它们放在同一个标记区域中，不隐含特定的上下顺序。

```
64 位泛型虚拟地址空间 (sm_90)
══════════════════════════════════════════════════════════════
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  带标记的高位地址区域                                       │
│  （内部确切顺序未知 — shared 和 local                      │
│   在此区域内占据互不重叠的子范围）                          │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SHARED MEMORY 窗口 (每 CTA, 各 16 MB)               │  │
│  │  标记来源: SR_SWINHI                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LOCAL MEMORY 窗口 (每线程)                          │  │
│  │  标记来源: c[0x24]:c[0x20] + R1 栈指针               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
├────────────────────────────────────────────────────────────┤  ← c[0x0][0xd0] 阈值
│                                                            │
│  GLOBAL MEMORY                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  恒等映射的 64 位地址空间                              │  │
│  │  (无窗口转换 — 直接物理寻址)                           │  │
│  │  同时捕获无标记的旧式指针 (cudaMalloc)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘

CONSTANT MEMORY — 独立地址空间, 不在泛型指针范围内
┌──────┐ ┌──────┐ ┌──────┐
│Bank 0│ │Bank 1│ │Bank N│  (按 c[bank][offset] 索引)
└──────┘ └──────┘ └──────┘
```

---

### 3.6. 单个 CTA 视角下的泛型地址空间

本节将上述窗口布局综合为一幅连贯的图景：从单个正在 SM 上执行的 CTA 的角度来看，完整的 64 位泛型地址空间是什么样的？

#### 3.6.1. 为什么 CTA 必须自己计算 Shared 基址

Shared memory 窗口基址不是编译期常量 — 它取决于 CTA 的运行时标识：

```sass
S2UR  UR5, SR_CgaCtaId        ; UR5 = CTA 标识符（由硬件调度器分配）
UMOV  UR4, 0x400              ; UR4 = 0x400（窗口粒度常量）
ULEA  UR4, UR5, UR4, 0x18     ; UR4 = (CgaCtaId << 24) | 0x400
```

`SR_CgaCtaId` 是一个**只读硬件特殊寄存器**。其值由硬件调度器在 CTA 被发射到 SM 时动态分配。CTA 在编译期不知道自己的 ID — ID 取决于它被分配到哪个 SM、CTA 的分发顺序以及同一 SM 上共有多少个 CTA。因此：

- CTA **必须**在运行时读取 `SR_CgaCtaId` 来计算自己的窗口基址
- 窗口基址 `UR4` 存储在 **uniform 寄存器**（warp 内所有线程共享）中，因为同一 warp 的所有线程属于同一个 CTA
- 计算结果 `UR4 = (CgaCtaId << 24) | 0x400` 为该 CTA 的 shared memory 提供了一个唯一的 16 MB 虚拟地址范围

同样的机制也适用于 `SR_SWINHI` — 这个 shared 窗口高寄存器提供泛型 shared 指针的高 32 位，它也是一个运行时分配的寄存器，编码了 CTA 的身份。

#### 3.6.2. 单个 CTA 看到的地址空间

从单个 CTA 的视角来看，64 位泛型地址空间划分为以下区域：

```
高地址（带标记区域 — shared 与 local，互不重叠，
        内部顺序无法从静态 SASS 确定）
┌────────────────────────────────────────────┐
│  SHARED MEMORY — 仅限本 CTA（16 MB 窗口）            │
│  Base = (CgaCtaId << 24) | 0x400                 │
│  Generic ptr upper = SR_SWINHI                    │
│  STS/LDS: 隐式限定在本 CTA                         │
├────────────────────────────────────────────┤
│  LOCAL MEMORY 窗口（每线程独立，对其他线程不透明）                │
│  Base: c[0x24]:c[0x20] (uniform 常量)              │
│  Offset: R1 栈指针来自 c[0x28]（所有线程统一）          │
│  STL/LDL: 隐式限定在当前线程                           │
├────────────────────────────────────────────┤  ← c[0x0][0xd0] 阈值
│  GLOBAL MEMORY（所有 CTA、所有线程共享）                    │
│  恒等映射的 64 位地址空间                                │
├────────────────────────────────────────────┤
│  CONSTANT MEMORY（独立空间，按 bank 索引）                  │
│  Bank 0 = ABI + 内核参数, Bank 4 = 内核参数数据                │
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
| Shared | `SR_SWINHI`（CTA 编码） | 是 (`QSPC.E.S`) | **否** — `STS`/`LDS` 仅使用 32 位偏移，窗口是隐式的 |
| Local | `c[0x24]`（窗口基址高位） | 是 (`QSPC.E.L`) | **否** — `STL`/`LDL` 使用计算后的栈偏移，窗口是隐式的 |
| 泛型 `ST.E`/`LD.E` | 全部 64 位均有意义 | 分派依据 | 硬件解码标记以路由到正确空间 |

这是整个设计的核心矛盾：**泛型指针携带了完整的空间和所有者身份信息，但空间特定硬件指令（`STS`、`LDS`、`STL`、`LDL`）完全忽略标记位**。窗口由*执行上下文*（shared 为当前 CTA、local 为当前线程）决定，而非指针本身。

标记位服务于三个目的：
1. **ABI 正确性**：接收 `int *` 的函数可调用 `__isspacep_shared(p)` 来判断指针所属的空间 — 编译为 `QSPC.E.S`
2. **泛型 load/store 分派**：`LD.E`/`ST.E` 读取标记以路由到正确的内存子系统
3. **PTX 中的指针来源追踪**：当编译器静态知道指针来源时，可常量折叠 `isspacep`

但在热路径（`STS`/`LDS`/`STL`/`LDL`）上，标记被剥离，使解引用零开销。

#### 3.6.5. 通过修改指针比特，CTA 能否访问其他 CTA 的 Shared Memory？

这个问题可以分解为两种场景：

**场景 A：使用 `STS`/`LDS`（空间特定指令）。**

不可能。`STS [R7], R0` 只接收一个 32 位偏移量 — 硬件无条件地将此路由到当前 CTA 的 shared 窗口。`cvta.to.shared` 在 SASS 中是 NO-OP：指令只是取 64 位输入值的低 32 位作为偏移量。不存在指令层面的机制将 `STS`/`LDS` 重定向到另一个 CTA。

**场景 B：使用 `ST.E`/`LD.E`（泛型 load/store）。**

在指令层面是*可能*的，因为 `ST.E` 接收完整的 64 位标记指针并根据标记分派。如果手动构造一个高 32 位为另一个 CTA 的 `SR_SWINHI` 的 64 位值，并将其传递给 `ST.E`，硬件会解码标记并看到"这指向 CTA-K 的 shared 空间"。

硬件是*允许*这种跨 CTA 访问还是*触发 fault*，取决于 GPU 的保护模型，无法从静态 SASS 反汇编中观察到。硬件拥有检测违规所需的所有信息（执行 CTA 的 `CgaCtaId` 在 `SR_CgaCtaId` 中 vs 指针内嵌的 `SR_SWINHI`），但它是否真正检查是具体实现决定的。

然而，**编译器的行为**提供了一个强烈的提示：当 `__noinline__` 函数接收到可能指向多个空间的泛型指针时，编译器生成 `ST.E`/`LD.E`（而非 `STS`/`LDS`）。这表明 NVIDIA 的意图是 `ST.E`/`LD.E` 应作为处理来源未知的指针的正确方式 — 且硬件级别的分派被信任为正确路由。但编译器从未*故意*构造跨 CTA 的泛型指针，因此这条路径在正常代码中不会被执行。

**结论**：`SR_CgaCtaId` 是只读的，用户代码无法修改。仅修改泛型指针的高位比特无法绕过 `STS`/`LDS` 的隔离（它们忽略这些位）。`ST.E`/`LD.E` 是否强制 CTA 级别的保护是硬件实现细节，无法通过离线反汇编回答。

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

### 4.2. Shared Memory：架构上不可跨 CTA 访问

**证据** — 来自 `test_shared_cross_cta_ptr.sass`：

当 CTA 0 构造一个泛型 shared 指针，CTA 1 尝试解引用时：
1. `cvta.to.shared` 在 SASS 中是**完全 NO-OP** — 直接使用低 32 位：
   ```sass
   MOV  R7, R2        ; 来自泛型指针的原始偏移量
   STS  [R7], R0      ; 存储到 shared[R7]
   ```
2. 高 32 位（包含窗口 ID = `SR_SWINHI`）被**丢弃**
3. `STS`/`LDS` 指令仅使用偏移量 — shared memory 窗口由硬件根据**执行 CTA 的 `CgaCtaId`** 隐式确定
4. 解引用被调用函数中没有使用 `SR_CgaCtaId` 或 `SR_SWINHI`

**结论**：CTA 1 解引用 CTA 0 的泛型 shared 指针时，访问的是**CTA 1 自己的 shared memory** 中指针编码的偏移量位置。硬件将 `STS`/`LDS` 路由到当前 CTA 的窗口，无论指针的来源。

### 4.3. `SR_SWINHI` — 构造函数 vs 解引用函数

`SR_SWINHI` 寄存器**仅在构造**泛型 shared 指针时使用，解引用时不用：
```sass
; 构造函数 (make_shared_generic_ptr):
S2R  R4, SR_SWINHI         ; 读取窗口高比特
MOV  R2, RZ                  ; offset = 0
MOV  R3, R4                  ; 返回 (offset=0, upper=SR_SWINHI)

; 解引用函数 (write_shared_via_generic):
MOV  R7, R2                  ; 仅取低 32 位 → 偏移量
STS  [R7], R0                ; 硬件根据 CgaCtaId 确定窗口
```

### 4.4. 可见性矩阵

| 访问者 \ 目标 | 线程 0 Local | 线程 N Local | CTA 0 Shared | CTA N Shared | Global |
|--------------|-------------|-------------|-------------|-------------|--------|
| **线程 0** | RW（自己的） | 访问自己 local 对应偏移 | RW | 访问自己 CTA 的 shared | RW |
| **线程 N** | 访问自己 local 对应偏移 | RW（自己的） | RW | 访问自己 CTA 的 shared | RW |
| **CTA 0** | N/A | N/A | RW | 访问自己 CTA 的 shared | RW |
| **CTA N** | N/A | N/A | 访问自己 CTA 的 shared | RW | RW |

泛型指针的高 32 位编码了窗口标识，但**解引用硬件在空间特定指令（STS/LDS/STL/LDL）中忽略它们**。窗口由执行上下文决定：shared 为当前 CTA 的窗口；local 为当前线程的存储——由 STL/LDL 硬件级别强制。由于 `R1`（来自 `c[0x0][0x28]` 的栈指针）和窗口基址均为 uniform，泛型 local 指针对所有线程编码相同的栈偏移；`STL`/`LDL` 隔离来自硬件限定，而非指针差异。

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
UMOV  UR4, 0x400              ; UR4 = 0x400（窗口粒度常量）
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

相邻 CTA ID 之间的跨度为 `1 << 24 = 16,777,216 字节 = 16 MB`。因此每个 CTA 在泛型地址空间中占据一个 **16 MB 窗口**。`0x400`（1024）是应用于所有窗口的常数偏移量 — 很可能对应硬件 shared memory 最小分配粒度的字节数。

#### 5.3.2. 来自 `test_shared_window_granularity.cu` 的确认证据

新增的探测内核从多个角度提供了确认：

| 测试 | 发现 |
|------|------|
| `k_no_shared` | 无 `__shared__` → **完全无 ULEA/UMOV** |
| `k_single_block` | 即使 gridDim=1（CgaCtaId 始终为 0），ULEA 仍以 `0x400`/`0x18` 出现 |
| `k_two_shared_arrays` | 两个 `__shared__` 数组 → **两个 ULEA** 指令。`arr1` 使用 `ULEA(Id, 0x400, 0x18)`，`arr2` 使用 `ULEA(Id, 0x800, 0x18)` — 第二个数组的基址偏移了第一个数组的大小（1 KB = 0x400）。位移量 `0x18` 两者**完全相同**。 |
| `k_shared_max_48KB` / `k_shared_mid_32KB` / `k_shared_mid_16KB` | 三个内核，分别 16KB、32KB 和 48KB shared — **逐条指令完全相同的 SASS**。在 48KB 边界处 `UMOV 0x400` 和 `ULEA 0x18` 未变。 |
| `k_use_blockidx` | `SR_CTAID.X`（blockIdx）**与 `SR_CgaCtaId` 分开读取** — blockIdx 仅用于数据值，不用于 shared 窗口地址。 |
| `k_export_all_generic_ptrs` | `SR_SWINHI` 通过 `S2UR` 读取并用作泛型 shared 指针的高 32 位。相同的 CTA 窗口标识通过两条路径可用：ULEA 计算（用于 STS/LDS）和 SR_SWINHI 读取（用于泛型指针构造）。 |
| `k_builtin_vs_asm_cvta` | 向 `__noinline__` 函数传递泛型指针时，编译器使用常规寄存器 `LEA` 而非 `ULEA` — 相同的公式 `(CgaCtaId << 24) + 0x400`，但为满足调用约定存储在 `R*` 而非 `UR*` 寄存器中。`__cvta_generic_to_shared()` builtin 和 asm `cvta.to.shared` 在 callee 中产生相同的 `STS [Rx]` 模式。 |

#### 5.3.3. 总结

```
SharedMemoryWindowBase(CTA_id) = (CgaCtaId << 24) + 0x400
                                 └────16 MB/CTA───┘ └─1 KB 基础─┘
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
| `SR_SWINHI` | 32 | `S2R`, `S2UR` | Shared memory 窗口高比特（在泛型地址中编码 CTA 标识） |

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

sm_90 上的 `STS`/`LDS` 指令微架构不接受 64 位泛型地址 — 它使用 32 位偏移量，**隐式限定**在当前 CTA 的 shared memory 窗口（由 `CgaCtaId` 索引）。泛型指针的高 32 位仅用于 ABI 级别的指针标识；硬件在 shared memory 访问时忽略它们。

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
└── build/
    ├── ptx/
    ├── sass/
    └── cubin/
```

**总计**：12 个源文件，所有测试中共约 80 个内核变体。

---

## 10. 待解决问题

1. **`c[0x0][0xd0]` 阈值的实际值**：用于 global 空间范围检查的 64 位常量位于 `c[0x0][0xd0]`，无法通过静态方式观察 — 它是运行时值。其确切值将确认 sm_90 的 global 地址空间范围边界。

2. **`SR_SWINHI` 编码**：`SR_SWINHI` 的内部结构 — 它是否仅编码 `CgaCtaId`，还是也包含 SM ID、分区 ID 或其他调度信息 — 无法仅从 SASS 确定。

3. **`QSPC.E.C`（constant 空间）**：`cvta.const` 测试未成功（ptxas 拒绝操作数类型）。constant 空间查询指令（`QSPC.E.C` 或类似）可能存在但未经验证。

4. **泛型 `LD.E`/`ST.E` 分派机制**：当编译器回退到泛型 load/store 时（如 `01-shared-ptr-analysis` 中所见），硬件根据指针标记进行分派。内部分派逻辑（标记解码器、窗口查找、边界检查）无法从二进制反汇编中观察。

5. **通过手动构造的泛型指针进行跨 CTA 访问**：如果手动构造一个 64 位泛型指针，将高 32 位设为其他 CTA 的 `SR_SWINHI` 值，并通过 `ST.E`/`LD.E`（泛型 store/load）解引用，硬件会如何行为？
    - 路由到同一 SM 上目标 CTA 的 shared memory？
    - 跨互联路由到不同 SM 的 shared memory？
    - 因执行 CTA 的 `CgaCtaId` 与指针嵌入的窗口 ID 不匹配而 fault？

    无法通过静态 SASS 反汇编确定，需要在物理硬件上进行运行时测试才能解答。该问题的架构意义重大：如果硬件*允许*通过 `ST.E` 进行同一 SM 内的跨 CTA 访问，则意味着泛型地址空间在泛型 load/store 路径中提供了所有 shared memory 窗口的平坦视图，不设 CTA 级别的保护 — 只有 `STS`/`LDS` 强制执行隐式隔离。

6. **0x400 保留区域的消费者**：每个 CTA 的 shared memory 窗口基址中固定的 0x400（1 KB）偏移量已被确认不变，但其消费者仍然未知。`test_mbarrier_shared_reserved.cu` 推翻了 `cuda::barrier` 的 mbarrier 状态存储于此的假说 — 编译器将 barrier 状态视为从 0x400 起步的普通用户 `__shared__` 变量。可能的解释包括：`__syncthreads()` 暂存空间、硬件对齐要求、或无逻辑消费者的固定架构空洞。
