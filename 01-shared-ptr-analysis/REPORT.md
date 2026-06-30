# CUDA `__device__` 函数 Shared Memory Pointer 参数分析报告

**日期**: 2026-06-24
**工具链**: CUDA 12.9 (NVCC Build: V12.9.86)
**目标架构**: sm_90 (Hopper)
**分析目标**: 当 `__device__` 函数声明为 `__noinline__` 且参数列表中有指向 shared memory 的指针时，编译器如何决定生成何种 ld/st 指令。

---

## 背景

CUDA 函数参数**不允许**使用 `__shared__` 关键字指定地址空间类型。按常规理解，`__device__` 函数在非内联情况下无法确定应该按哪种方式（shared / global / local / const scope）访问这个指针，也就无法决定生成什么样的指令。

NVIDIA GPU 对不同地址空间有专门的访存指令：

| 地址空间 | PTX 指令 | SASS 指令 |
|---------|---------|----------|
| Global  | `ld.global` / `st.global` | `LDG` / `STG` |
| Shared  | `ld.shared` / `st.shared` | `LDS` / `STS` |
| Local   | `ld.local` / `st.local`  | `LDL` / `STL` |
| Generic | `ld` / `st` (无限定符)   | `LD.E` / `ST.E` |

其中泛型指令在运行时根据指针高位 bit 解析实际地址空间（sm_30+ 引入的 Unified Memory Addressing）。

---

## 测试用例

| 文件 | 测试目的 |
|------|---------|
| `src/test_baseline.cu` | `__noinline__` 函数仅从 shared memory 调用，对比 inline vs noinline |
| `src/test_multiple_spaces.cu` | 同一 `__noinline__` 函数被 shared / global / local 三种地址空间调用 |
| `src/test_template.cu` | 模板方式、`__restrict__`、内联 asm 等替代方案 |
| `src/test_ptr_intrinsics.cu` | 使用 `__isShared()` 等 intrinsic 做运行时地址空间判断 |

---

## 核心发现

### 发现 1：全程序分析 → 地址空间特化

当编译器通过全程序分析发现 `__noinline__` 函数**仅被一种地址空间**的指针调用时，会在函数内部先通过 `cvta.to.shared.u64` 提取 shared memory 偏移量，然后生成**地址空间专用指令**。

**证据** — `test_baseline.cu` 中的 `write_ptr_noinline`（仅被 shared memory 指针调用）：

**PTX**:
```ptx
.func _Z18write_ptr_noinlinePii(
    .param .b64 _Z18write_ptr_noinlinePii_param_0,   // 64-bit 泛型指针
    .param .b32 _Z18write_ptr_noinlinePii_param_1,
)
{
    ld.param.u64         %rd1, [_Z18write_ptr_noinlinePii_param_0];
    ld.param.u32         %r1,  [_Z18write_ptr_noinlinePii_param_1];
    cvta.to.shared.u64   %tmp, %rd1;       // 泛型→shared 地址转换
    cvt.u32.u64          %r2,  %tmp;
    st.shared.u32        [%r2], %r1;       // ✅ 地址空间专用 store
    ret;
}
```

**SASS**:
```sass
/*02d0*/   STS [R8], R11 ;                  // ✅ 专用 shared memory store
```

> 📌 **关键**: 编译器利用全程序可见性，推断出该函数的所有调用点都传入 shared memory 指针，于是生成 `st.shared` / `STS` 而非泛型指令。

---

### 发现 2：多地址空间调用 → 泛型寻址

当同一个 `__noinline__` 函数被**多种地址空间**（shared、global、local）的指针调用时，编译器无法特化到单一地址空间，只能生成**泛型访存指令**，由硬件在运行时解析地址空间。

**证据** — `test_multiple_spaces.cu` 中的 `generic_write`（被 `kernel_shared`、`kernel_global`、`kernel_local` 三个 kernel 分别用 shared、global、local 指针调用）：

**PTX**:
```ptx
.func _Z13generic_writePii(
    .param .b64 _Z13generic_writePii_param_0,
    .param .b32 _Z13generic_writePii_param_1,
)
{
    ld.param.u64  %rd1, [_Z13generic_writePii_param_0];
    ld.param.u32  %r1,  [_Z13generic_writePii_param_1];
    st.u32        [%rd1], %r1;              // ⚠️ 泛型 store，无地址空间限定符
    ret;
}
```

**SASS** — 三个 kernel 共享同一实现：
```sass
/* kernel_shared 中的 generic_write @ 0x260 */
/*0260*/   IMAD.MOV.U32 R5, RZ, RZ, 0x0 ;
/*0270*/   ULDC.64 UR6, c[0x0][0x208] ;
/*0280*/   ST.E desc[UR6][R2.64], R7 ;       // ⚠️ 泛型 store (ST.E)
/*0290*/   RET.REL.NODEC R4 0x0 ;

/* kernel_global 中的 generic_write @ 0x160 */
/*0160*/   MOV R5, 0x0 ;
/*0170*/   ULDC.64 UR4, c[0x0][0x208] ;
/*0180*/   ST.E desc[UR4][R2.64], R7 ;       // ⚠️ 泛型 store (ST.E)
/*0190*/   RET.REL.NODEC R4 0x0 ;
```

> 📌 **关键**: 不生成多个克隆版本，三个调用点共享同一个 `ST.E` 实现。硬件根据指针高位 bit 决定实际访问 shared / global / local。

---

### 发现 3：纯读取函数的过程间优化（IPO）

对于 `return *p;` 这种纯读取的 `__noinline__` 函数，编译器通过**过程间优化**完全消除了指针解引用：

**PTX** — `generic_read`:
```ptx
.func (.param .b32 func_retval0) _Z12generic_readPi(
    .param .b32 _Z12generic_readPi_param_0   // ⚠️ .b32 值，而非 .b64 指针！
)
{
    ld.param.u32  %r1, [_Z12generic_readPi_param_0];
    st.param.b32  [func_retval0+0], %r1;
    ret;
}
```

调用点（kernel_shared 中以 shared memory 指针调用 `generic_read`）：
```ptx
ld.shared.u32  %r8, [%r2];              // ← 在调用点用 LDS 完成 load
st.param.b32   [param0+0], %r8;         // ← 传递值，不传指针
call.uni (retval0), _Z12generic_readPi, (param0);
```

> 📌 **关键**: 编译器在调用点用正确的地址空间指令（LDS / LDG / LDL）完成加载，只将**值**传给函数。函数退化为 identity 操作。这个问题被巧妙地绕过去了。

---

### 发现 4：`__isShared()` 等 intrinsic 被优化掉

`test_ptr_intrinsics.cu` 中使用 `__isShared(p)` 做运行时分支的 `read_with_dispatch` 函数，在编译后**完全等同于** `read_generic_only`：两个函数都被 IPO 优化为 identity，分支被消除。

---

### 发现 5：SASS 层面的函数嵌入

虽然 PTX 中 `__noinline__` 函数表现为独立的 `.func` 条目并通过 `call.uni` 调用，但在 SASS 层面它们通过 `CALL.REL.NOINC` 嵌入在 kernel 的代码段中，不显示为独立的 `Function :` 条目。`cuobjdump` 仅列出 kernel 入口函数。

---

## 编译器行为总结

| 编译器行为 | 是否发生 | 条件 |
|-----------|---------|------|
| 生成泛型 ld/st (LD.E/ST.E) | ✅ 是 | 多地址空间调用同一函数 |
| 全程序分析 + 地址空间特化 (LDS/STS) | ✅ 是 | 单一地址空间调用 |
| 过程间优化消除指针 (传值) | ✅ 是 | 纯读取函数 |
| 生成 per-call-site 克隆 | ❌ 否 | 不自动生成多版本 |
| 忽略 `__noinline__` 强制内联 | ❌ 否 | 尊重 noinline 语义 |

---

## 底层机制：CUDA 泛型寻址（Generic Addressing）

自 sm_30 (Kepler) 起，CUDA 引入统一地址空间模型：

- **64-bit 泛型指针**的高位 bit 编码了地址空间类型（shared / global / local / constant）
- `cvta.shared.u64` — 将 shared memory 基址包装为泛型指针（标记 shared 空间）
- `cvta.to.shared.u64` — 从泛型指针中提取 shared memory 偏移
- 泛型 `ld`/`st` 指令（PTX）→ `LD.E`/`ST.E`（SASS）在运行时检查指针高位，自动分发到正确物理存储

**泛型寻址的开销**：在 sm_90 上，`ST.E` 比 `STS` 多一次指针状态检查，但硬件对此高度优化。在性能敏感的 kernel 中，如果需要确保使用 dedicated 指令，应让函数仅被一种地址空间调用，或使用内联汇编。

---

## 文件结构

```
.
├── README.md                     # 本文件
├── REPORT.md                     # 详细分析报告
├── src/                          # 测试源码
│   ├── test_baseline.cu
│   ├── test_multiple_spaces.cu
│   ├── test_template.cu
│   └── test_ptr_intrinsics.cu
├── scripts/
│   └── compile.sh                # 编译脚本
└── build/                        # 编译产物
    ├── ptx/                      # PTX 虚拟 ISA
    ├── sass/                     # SASS 机器码反汇编
    └── cubin/                    # 原始 cubin 文件
```
