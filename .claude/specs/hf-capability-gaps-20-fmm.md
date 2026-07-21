---
title: "HF capability gaps 20 — 快速多极方法:点电荷 FMM 算子核(开边界)"
slug: hf-capability-gaps-20-fmm
status: approved
chain: hf-capability-gaps
link: 20/21
created: 2026-07-20
---

# 快速多极方法:点电荷 FMM 算子核(开边界)

## Summary

在链环 19 创建的顶层 `longrange/` facade 簇内实装(用户裁定:真实实现,非仅设计)经典
Greengard-Rokhlin FMM 的算子核,作用对象为开边界点电荷集:实固体谐波多极展开 M_lm 与局部
展开 L_lm(P2M/M2P/L2P,建于链环 02 上提的 `core/real_harmonics.h` 之上)、三个平移算子
M2M/M2L/L2L(实固体谐波加法定理形式),以及自适应八叉树(叶容量为成员,标准 well-separated
判据:≥1 个格胞宽度间隔;近场直和 + 远场 M2L 游走),提供能量与指定点电势求值。整个簇保持
backend 无关、Component/Frame/MPI 无关——这是 PME 的开边界对偶物,也是链环 21(CFMM 高斯
分布扩展 + HF 接线)的地基。验收仪器是误差界本身:对随机与晶格点电荷集,逐阶 p 误差表相对
O(N²) 直和呈几何衰减,至文档标注的 p 达 ≤1e-10;N 对 4N 的墙钟比在 regression 中宽松钉住
近线性标度。

## Domain basis

- 原始 FMM 与一致误差界 ~ (1/(c−1))(1/c)^p(c 为间隔比):Greengard & Rokhlin,
  *J. Comput. Phys.* **73**, 325 (1987),DOI 10.1016/0021-9991(87)90140-9。
- 实固体谐波形式的 M2M/M2L/L2L 平移与高效实现:White & Head-Gordon,
  *J. Chem. Phys.* **101**, 6593 (1994),DOI 10.1063/1.468354。
- CFMM(链环 21 的目标,本链环仅引作路线):White, Johnson, Gill, Head-Gordon,
  *Chem. Phys. Lett.* **230**, 8 (1994),DOI 10.1016/0009-2614(94)01128-1。
- 动机引文:Frank Jensen, *Introduction to Computational Chemistry*,线性标度/SCF 瓶颈节
  (库仑远场用 FMM 压缩是线性标度的推荐路径)。
- 数学不变量(编码为单元测试):M2M 与 L2L 平移在阶 p 内**精确**(1e-12);仅 M2L 引入截断
  误差,且随 p 按上述几何因子衰减。核为无量纲库仑势 Σ q_i q_j / r_ij;e²/Å → eV 的单位换算
  (`core/units.h`)由链环 21 的 HF 接线方负责,Doxygen 中显式标注。系数一律 `AccReal`(double)。

## Design

- **命名空间** `ATV::longrange`,与 `core/`/`facade/` 同层的顶层 facade 簇(链环 19 的家);
  头文件只依赖 `core/`(`real_harmonics.h`、`vec3.h`、`precision.h`)——不 include 任何
  `cpu/`/`cuda/`、Component、Frame、MPI(验收 grep 门禁)。类 + 校验构造函数,无工厂函数。
- **`multipole_expansion.h`**:`MultipoleExpansion{int order, Vec3 center}` 与
  `LocalExpansion{int order, Vec3 center}`(构造校验 order ≥ 0,否则 throw;系数向量长
  (p+1)²)。成员:P2M `accumulate(q, pos)`(正则实固体谐波 R_lm 经 `real_solid_harmonics`
  的笛卡尔多项式展开求值)、M2P/L2P `evaluate(point)`(非正则 I_lm = R_lm/r^(2l+1) 规约,
  Helgaker 约定)。
- **`fmm_operators.h`**:算子类 `M2M` / `M2L` / `L2L`,构造函数按 order 预计算实加法定理
  系数表(White & Head-Gordon 1994 实数形式),`apply(src, dst)` 零分配(build/launch 精神)。
- **`fmm_tree.h`**:`FmmTree{int order, int max_per_leaf}`(构造校验两者 ≥ 1)。求值入口
  `energy(positions, charges)` 与 `potentials(positions, charges, out)`,输入为裸 span/vector
  (Frame-free)。自适应八叉树罩包围立方体;交互列表 = 父邻居子胞中同层非相邻者(标准
  well-separated,≥1 胞宽);上行 P2M+M2M、下行 M2L+L2L、叶层 L2P + 近场直和;开边界,无周期像。
- **私有工作区**:树结点、系数缓冲全部为 `FmmTree` 成员,不进任何公共数据结构(Frame 纯净
  规则的 facade 版)。

## Files to create or modify

- include/atomiverse/longrange/multipole_expansion.h (new)
- src/longrange/multipole_expansion.cpp (new)
- include/atomiverse/longrange/fmm_operators.h (new)
- src/longrange/fmm_operators.cpp (new)
- include/atomiverse/longrange/fmm_tree.h (new)
- src/longrange/fmm_tree.cpp (new)
- tests/longrange/test_multipole_expansion.cpp (new)、tests/longrange/test_fmm_operators.cpp (new)、
  tests/longrange/test_fmm_tree.cpp (new)
- tests/CMakeLists.txt、CMakeLists.txt(登记 src/longrange 无条件编译源与三个单测)
- regressions/longrange/fmm_point_charges/reg_fmm_point_charges.cpp (new)、同目录 REFERENCE.md (new)
- regressions/CMakeLists.txt
- docs/tutorial/ch19-hf-large-systems.md(链环 19 创建,本链环增节)

## Tasks

- [ ] Write failing tests for multipole/local expansions(tests/longrange/test_multipole_expansion.cpp,
      登记 tests/CMakeLists.txt):单点电荷 P2M+M2P 对 1/r 逐阶几何收敛;l=0 单极精确;L2P 对
      well-separated 源的收敛;order<0 构造 throw
- [ ] Implement MultipoleExpansion/LocalExpansion in include/atomiverse/longrange/multipole_expansion.h
      + src/longrange/multipole_expansion.cpp;CMakeLists.txt 登记(带单位说明的 Doxygen)
- [ ] Write failing tests for translation operators(tests/longrange/test_fmm_operators.cpp):
      M2M 平移后 M2P 与直接对父中心 P2M 一致(1e-12,阶内精确);L2L 同式精确;M2L+L2P 在固定
      间隔比 c 下逐阶几何衰减且不劣于 (1/(c−1))(1/c)^p 量级
- [ ] Implement M2M/M2L/L2L in include/atomiverse/longrange/fmm_operators.h + src/longrange/fmm_operators.cpp
      (构造期预计算系数表;apply 零分配)
- [ ] Write failing tests for FmmTree(tests/longrange/test_fmm_tree.cpp,秒级:N≤512、p≤8):
      随机点集能量/电势 vs O(N²) 直和相对误差 <1e-6;交互列表逐对满足 ≥1 胞宽间隔;叶容量约束;
      退化例(N=1、N=2、全部落入单叶 → 纯直和)
- [ ] Implement FmmTree in include/atomiverse/longrange/fmm_tree.h + src/longrange/fmm_tree.cpp
- [ ] Add regression regressions/longrange/fmm_point_charges/(随机 + 晶格点电荷集:逐阶 p 误差表
      几何衰减至文档 p 处 ≤1e-10;N vs 4N 墙钟比宽松钉住近 O(N);REFERENCE.md 记录 Greengard-Rokhlin
      与 White-Head-Gordon DOI、误差界、文档 p;登记 regressions/CMakeLists.txt,≤60 s)
- [ ] Extend docs/tutorial/ch19-hf-large-systems.md:新增「长程库仑 II:快速多极方法」节
      (多极压缩远场的层级叙事、M2M/M2L/L2L 游走、为何是 PME 的开边界对偶;误差-阶数收敛**表格**
      ——标量进表格、禁用条形图;molplot 标度曲线图)
- [ ] Run full check + test suite

## Testing strategy

- 快乐路径:展开求值逐阶收敛、M2M/L2L 阶内精确恒等式、树能量/电势 vs 直和——全部秒级单元测试
  (tests/ 严禁 E2E,无计时、无外部引擎)。
- 边界:order<0 / max_per_leaf<1 构造 throw(无回退值);N=1(零能量)、N=2(纯近场)、单叶树;
  共线/共面晶格点集(八叉树退化分支)。
- 领域验证(science.required):(a) M2L 截断误差随 p 的几何衰减与 Greengard-Rokhlin 界一致
  (单元测试 + regression 误差表);(b) regression 对 O(N²) 直和 oracle 的 ≤1e-10 收敛与近线性
  标度烟检(标度只钉在 regression,不进 tests/)。

## Out of scope

- CFMM 高斯分布扩展、well-separatedness 的 extent 修正、HF/J-矩阵接线与单位换算(链环 21)。
- 周期边界(PME/Ewald 已覆盖;FMM 保持开边界)、力/梯度(链环 21 按需)、MPI/CUDA 并行化。
- benchmarks/ 性能剖析(标度烟检只在 regression 宽松钉住)。
