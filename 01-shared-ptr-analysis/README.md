# CUDA `__device__` Shared Memory Pointer 分析项目

分析 NVIDIA 编译器（CUDA 12.9）在 sm_90 架构下，如何处理非内联 `__device__` 函数参数中的 shared memory 指针。

## 快速开始

```bash
# 编译所有测试用例
bash scripts/compile.sh

# 查看 PTX 指令（虚拟 ISA）
grep -n 'ld\.\|st\.' build/ptx/*.ptx

# 查看 SASS 指令（真实机器码）
grep -n 'LDS\|LDG\|STS\|STG\|ST\.E' build/sass/*.sass
```

## 目录结构

```
.
├── README.md                     # 入口
├── REPORT.md                     # 完整分析报告
├── src/                          # 测试源码
│   ├── test_baseline.cu          # 基础对比：inline vs noinline
│   ├── test_multiple_spaces.cu   # 多地址空间调用同一函数
│   ├── test_template.cu          # 模板/asm 替代方案
│   └── test_ptr_intrinsics.cu    # __isShared() intrinsic 测试
├── scripts/
│   └── compile.sh                # 编译脚本
└── build/                        # 编译产物
    ├── ptx/                      # PTX 虚拟 ISA
    ├── sass/                     # SASS 机器码反汇编
    └── cubin/                    # 原始 cubin
```

## 主要结论

1. **全程序分析 + 特化**：若函数仅被一种地址空间调用 → 生成专用指令（`STS`/`LDS`）
2. **泛型寻址**：若被多种地址空间调用 → 生成泛型 `ST.E`/`LD.E`，运行时解析
3. **过程间优化**：纯读取函数在调用点完成加载，只传值给函数
4. **不会**自动生成 per-call-site 克隆
