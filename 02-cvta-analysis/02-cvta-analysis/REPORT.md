# sm_90 `cvta` 指令 SASS 实现分析报告

**日期**: 2026-06-24
**工具链**: CUDA 12.9 (NVCC Build: V12.9.86)
**目标架构**: sm_90 (Hopper)

## 核心结论

**sm_90 SASS 中不存在专用的 `CVTA` 硬件指令。** 所有地址空间转换（cvta）都是通过基础操作实现的：寄存器 MOV、特殊寄存器读取（`SR_SWINHI`）、整数加法/减法，或直接消去（no-op）。

---

## 各 cvta 变体 SASS 实现

### 1. `cvta.to.global.u64` — 泛型→global

**实现**: **完全 NO-OP**

```
SASS:
/*0490*/   HFMA2.MMA R7, -RZ, RZ, 0, 0 ;    // R7=0 (NOP)
/*04a0*/   RET.REL.NODEC R6 0x0 ;            // 原值通过
```

Global 地址空间是泛型指针的默认空间——泛型指针和 global 指针在 sm_90 上是**同一 bit pattern**。函数返回输入值，不做任何转换。

---

### 2. `cvta.global.u64` — global→泛型

**实现**: **完全 NO-OP**

```
SASS:
/*03c0*/   IMAD.MOV.U32 R7, RZ, RZ, 0x0 ;    // R7=0
/*03d0*/   RET.REL.NODEC R6 0x0 ;            // 原值通过
```

与 cvta.to.global 一样是 identity 操作。

---

### 3. `cvta.to.shared.u64` — 泛型→shared

**实现**: **MOV（取低 32 位）**

```
SASS:
/*04b0*/   MOV R6, R8 ;                      // R6 = 泛型指针低32位 = shared偏移
/*04c0*/   IMAD.MOV.U32 R8, RZ, RZ, R10 ;
/*04d0*/   IMAD.MOV.U32 R9, RZ, RZ, 0x0 ;
```

泛型指针的低 32 位直接就是 shared memory 偏移量。`cvta.to.shared` 仅仅是一次 `MOV`。

**扩展观察**：当 `cvta.to.shared` 后紧接着 shared 访存时，编译器和硬件会融合：
- PTX: `cvta.to.shared + st.shared` → SASS: 单条 `STS [Rx], Ry`
- 融合后的 `STS` 直接接受 32-bit 偏移量作为地址

---

### 4. `cvta.shared.u64` — shared→泛型

**实现**: **`S2R SR_SWINHI` + MOV（读特殊寄存器）**

```
SASS:
/*03e0*/   S2R R6, SR_SWINHI ;               // ← 关键：读 Shared Window High 寄存器
/*03f0*/   MOV R7, 0x0 ;
/*0400*/   MOV R8, R4 ;                      // R8 = shared 偏移 (低32位)
/*0410*/   MOV R9, R6 ;                      // R9 = SWINHI (高32位)
/*0430*/   RET.REL.NODEC R6 0x0 ;            // 返回 (R8,R9) = 64-bit 泛型指针
```

**`SR_SWINHI`** 是一个硬件特殊寄存器，其中包含了当前 CTA 在 shared memory 窗口中的高位编码信息。泛型指针通过 `(SWINHI << 32) | shared_offset` 的方式构建。

在 kernel 调用点，编译器用以下方式生成 shared base：
```
S2UR UR5, SR_CgaCtaId ;           // CTA ID
UMOV UR4, 0x400 ;                  // 每 CTA shared mem 大小
ULEA UR4, UR5, UR4, 0x18 ;        // 通过 ULEA 编码 shared 基址
STS [R4+UR4], Rx ;                 // 直接使用 encoded 地址
```

---

### 5. `cvta.to.local.u64` — 泛型→local

**实现**: **IADD3 / IMAD.IADD（整数减法去除窗口偏移）**

```
SASS:
/*0440*/   LDC R3, c[0x0][0x20] ;            // R3 = local 窗口偏移常量
/*0460*/   IMAD.IADD R3, R8, 0x1, -R3 ;      // R3 = 泛型指针 + 1 - 窗口偏移
/*0480*/   RET.REL.NODEC R8 0x0 ;            // 返回 (R3, 0)
```

常量 `c[0x0][0x20]` 是 local memory 窗口的基础偏移。`cvta.to.local` 通过**减法**去除这个偏移，恢复原始的栈帧地址。

当此操作后接 local 访存时，也会融合：
```
VIADD R3, R3, -UR4 ;               // 去除窗口偏移
STL [R3], R2 ;                     // 单条 local store
```

---

### 6. `cvta.local.u64` — local→泛型

**实现**: **IMAD.IADD（整数加法添加窗口偏移）**

```
SASS:
/*0350*/   LDC R3, c[0x0][0x20] ;            // R3 = local 窗口偏移常量
/*0370*/   IMAD.IADD R3, R3, 0x1, R8 ;       // R3 = 窗口偏移 + 1 + 栈指针
/*0390*/   RET.REL.NODEC R8 0x0 ;            // 返回 (R3, 0)
```

与 cvta.to.local 互逆：通过**加法**将 local 窗口偏移注入泛型指针。

---

## 实现机制总览

```
┌──────────────────────────────────────────────────────────────────┐
│                     CVTA 实现层次                                  │
├──────────────────────┬───────────────────────────────────────────┤
│ 地址空间              │ SASS 实现方式                              │
├──────────────────────┼───────────────────────────────────────────┤
│ global ↔ generic     │ NO-OP（同一 bit pattern）                   │
│ shared ↔ generic     │ SR_SWINHI 特殊寄存器 + MOV                  │
│ local  ↔ generic     │ 整数 ADD/SUB + 常量 c[0x20]                │
│ const  ↔ generic     │ （ptxas 报操作数类型错，未能测试）            │
└──────────────────────┴───────────────────────────────────────────┘
```

### 为什么不需要专用 CVTA 指令？

sm_90 的 64-bit 泛型指针由以下部分构成：

| 地址空间 | 高 32 位编码 | 低 32 位 |
|---------|-------------|---------|
| global  | 标准 64-bit 地址高位 | 地址低位 |
| shared  | `SR_SWINHI` (硬件窗口寄存器) | 32-bit 偏移量 |
| local   | 0（经窗口常量调整） | 栈内偏移量 |

由于泛型指针的格式设计使得转换操作极其简单：
- **提取低位**就是共享/局部偏移
- **高位**由硬件寄存器或常量直接提供
- **global** 不需要任何转换

因此不需要专门的转换指令。

---

## 测试文件

```
src/
├── cvta_to_shared.cu          # cvta.to.shared (generic→shared)
├── cvta_to_global.cu          # cvta.to.global (generic→global)
├── cvta_shared_to_generic.cu  # cvta.shared (shared→generic)
├── cvta_local.cu              # cvta.local / cvta.to.local
└── cvta_all.cu                # 综合对比
```

编译: `bash scripts/compile.sh`
产物: `build/ptx/` (PTX) + `build/sass/` (SASS) + `build/cubin/`



同一个 SM core 上的所有 warp，看到的 generic 空间是不是等价的