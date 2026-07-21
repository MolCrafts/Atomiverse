---
title: "HF capability gaps 02 — 球谐 AO(5d/7f):cart2sph 变换与 SCF AO 维度切换"
slug: hf-capability-gaps-02-cart2sph
status: code-complete
chain: hf-capability-gaps
link: 02/21
created: 2026-07-20
---

# 球谐 AO(5d/7f):cart2sph 变换与 SCF AO 维度切换

## Summary

为 Gaussian SCF 域引入球谐 AO:把 `cpu/gpw/internal/real_harmonics.h` 上提为 `core/real_harmonics.h`
(scf 是第二个调用者,hoist-on-second-caller;镜像 `core/gto.h` 先例),在其上生成逐壳层 cart→sph
系数矩阵(d 6×5、f 10×7),并把变换施加在**积分壳层块级**(PySCF/libint 的做法,绝不在 Fock/MO 上)。
SCF 的 AO 维度随之切换:def2 系基组按定义使用球谐(H2O/def2-SVP 的 n_basis 由 25 → 24),
6-31G* 按定义保持笛卡尔 6d — 通过 basis_table 中的逐基组 cart/sph 约定标志区分。本链环改变
后续所有积分/Fock/梯度/性质所见的 AO 维度,因此必须最先落地(科学家裁定)。验收 oracle:
球谐 S 等于 PySCF `gto.mole.cart2sph_coeff` 对笛卡尔 S 的三明治(1e-12);RHF/def2-SVP 能量
与 PySCF `cart=False` 钉死值一致(01-env 协议)。

## Domain basis

- 闭式 c2s 系数(至 h 壳层):Schlegel & Frisch, *Int. J. Quantum Chem.* **54**, 83 (1995),
  DOI 10.1002/qua.560540202。树内 `real_harmonics` 生成公式采用其修正重述:
  Ribaldone & Desmarais, *J. Chem. Phys.* **163**, 074102 (2025), arXiv:2412.16733, Eq. (7)。
- 实固体谐波与 Cartesian↔spherical 理论:Helgaker, Jørgensen, Olsen,
  *Molecular Electronic-Structure Theory* (Wiley, 2000),第 6 章(solid harmonics)。
- 维度计数:笛卡尔 (l+1)(l+2)/2 维,球谐 2l+1 维;6d = 5d ⊕ 1×s(r² 污染分量),10f = 7f ⊕ 3×p。
- **已知陷阱(必须编码为单元测试)**:逐分量单位归一的笛卡尔 d 中,d_xx 与 d_xy 的归一化常数差
  √((2lx−1)!!(2ly−1)!!(2lz−1)!!) 比率(d 为 √3),c2s 系数必须吸收该比率 — 判据:单位归一 d 壳层
  自重叠经 c2s 三明治后为 5×5 单位阵(1e-12)。系数无量纲;能量内部 Hartree,写回 Frame 为 eV。

## Design

- **上提**:`ATV::cpu::gpw::{HarmTerm, RealHarmonic, sphere_monomial_integral, real_solid_harmonics,
  real_spherical_harmonics_normalized}` → `ATV::` 顶层(`core/real_harmonics.h`),实现文件移入
  `src/core/`(无条件编译 — 现在 GPW 与 HF 共用,不再挂在 `ATV_BUILD_GPW` 源列表下;
  CMakeLists.txt:464 的 GPW 门源列表迁出)。
  gpw 消费者(`gpw_solver.cpp`、`gth_integrals.cpp`)仅改 include 与命名空间;数值零变化。
- **新符号** `cpu::scf::internal::CartToSph`:构造函数按 l 从 `real_solid_harmonics(l)` 的笛卡尔展开
  生成系数矩阵(构造中吸收笛卡尔双阶乘归一化比率;无工厂函数),成员提供 `matrix(l)` 与
  壳层块变换 `block_1e` / 壳层四元组变换 `block_eri`。m 排序对齐 PySCF `cart2sph_coeff`。
- **接线点(壳层块级)**:`one_electron.cpp` 的 S/T/V 逐壳层对装配处做 Cᵀ·block·C;
  `two_electron.cpp` 直接 Fock 的 Rys 壳层四元组消化前变换四个指标;`integrals.cpp` 的显式 ERI
  张量装配同点接线。MINAO 猜测投影、DIIS、密度构建只读 `n_basis`,维度无关,自动跟随。
- **约定门**:`basis_data::AngularConvention{Cartesian, Spherical}` 加入 `basis_table.h`;
  `basis_set.h` 增查询(与 `basis_set_name` 同型):STO-3G/6-31G/6-31G* → Cartesian(6-31G* 按定义
  6d),Def2SVP/Def2TZVP → Spherical。`ShellList` 按约定给出每壳层 AO 数(2l+1 或 (l+1)(l+2)/2)。
- **梯度护栏**:本链环内球谐基的解析梯度以清晰 `std::runtime_error` 拒绝(后续链环接管),
  防止静默输出错误力。

### 实现审计注(2026-07-20,impl 期偏差记录)

1. **Fock/ERI 路径改基桥接而非壳块级变换**:spec 起草时树上还是壳块结构;
   slater 链 04-methods 落地后 UHF/UKS Fock 与 dense ERI 均为 AO 级(无壳块缝)。
   实现改为:pack 级三明治(S/T/V/Hcore/ERI)+ `HamiltonianBuilder::step` 内
   c2s 桥接(P→cart 引擎→算子回 sph;线性精确,O(n²)/迭代)。验收判据不受
   影响(回归与 PySCF 差 4e-13 Ha)。
2. **c2s 常数表格化**:`real_solid_harmonics()` 契约明言 m 序不定,无法直接
   生成约定钉死的变换;改为在 cart2sph.cpp 表格化 PySCF/libcint 约定常数
   (l≤3),测试以 1e-12 对 PySCF 钉死;pack 内再按引擎自身 S_cart 自归一
   (引擎收缩 AO 非单位归一)。
3. **MINAO 初猜在球谐下回退 Hcore**:`minao_supported` 对球谐 pack 返回
   false(交叉重叠机器是笛卡尔的);收敛终点不变,仅迭代数略增;球谐 MINAO
   归 07-guess 环。

## Files to create or modify

- include/atomiverse/core/real_harmonics.h (new)
- src/core/real_harmonics.cpp (new)
- include/atomiverse/cpu/gpw/internal/real_harmonics.h(删除,内容上提)
- src/cpu/gpw/internal/real_harmonics.cpp(删除,内容上提)
- src/cpu/gpw/gpw_solver.cpp、src/cpu/gpw/internal/gth_integrals.cpp(改指 core)
- include/atomiverse/cpu/scf/internal/cart2sph.h (new)
- src/cpu/scf/internal/cart2sph.cpp (new)
- include/atomiverse/cpu/scf/basis_data/basis_table.h、include/atomiverse/cpu/scf/basis_set.h、
  src/cpu/scf/basis_set.cpp(约定标志 + ShellList 维度)
- src/cpu/scf/one_electron.cpp、src/cpu/scf/two_electron.cpp、src/cpu/scf/integrals.cpp(壳层块 c2s)
- src/cpu/scf/gradient.cpp(球谐 guard-throw)
- tests/core/test_real_harmonics.cpp(new,自 tests/cpu/gpw/test_cpu_gpw_real_harmonics.cpp 迁移)
- tests/cpu/scf/test_cart2sph.cpp (new)、tests/cpu/scf/test_spherical_basis.cpp (new)
- regressions/scf/h2o_rhf_def2svp_sph/reg_h2o_rhf_def2svp_sph.cpp (new)、同目录 REFERENCE.md (new)
- CMakeLists.txt、tests/CMakeLists.txt、regressions/CMakeLists.txt
- docs/tutorial/ch04-basis-sets.md

## Tasks

- [x] Write failing tests for hoisted core real-harmonics(tests/core/test_real_harmonics.cpp,
      迁移改写 tests/cpu/gpw/test_cpu_gpw_real_harmonics.cpp,include 指向 core)
- [x] Implement hoist:core/real_harmonics.h + src/core/real_harmonics.cpp 无条件编译;改指
      gpw_solver.cpp 与 gth_integrals.cpp;删除 gpw 副本;更新 CMakeLists.txt、tests/CMakeLists.txt
- [x] Write failing tests for CartToSph(tests/cpu/scf/test_cart2sph.cpp):d 6×5 / f 10×7 形状,
      与 PySCF cart2sph_coeff 钉死系数 1e-12 一致,单位归一 d 壳层三明治恒等(双阶乘归一化陷阱)
- [x] Implement CartToSph in include/atomiverse/cpu/scf/internal/cart2sph.h + src/cpu/scf/internal/cart2sph.cpp
- [x] Write failing tests for spherical AO dimension(tests/cpu/scf/test_spherical_basis.cpp):
      H2O/def2-SVP n_basis==24,6-31G* 维持笛卡尔现值;球谐 S == 笛卡尔 S 三明治(1e-12,钉死
      fixture);球谐基解析梯度 throw
- [x] Implement spherical AO dimension:basis_table.h AngularConvention + basis_set.h/.cpp 逐基组
      标志与 ShellList 维度;one_electron.cpp 壳层块 c2s;two_electron.cpp / integrals.cpp 四元组
      c2s;gradient.cpp guard-throw
- [x] Add regression regressions/scf/h2o_rhf_def2svp_sph/(RHF/def2-SVP vs PySCF cart=False 钉死值,
      REFERENCE.md 记录 01-env oracle 出处,登记 regressions/CMakeLists.txt,≤60 s)
- [x] Extend docs/tutorial/ch04-basis-sets.md:新增「笛卡尔与球谐高斯基」节(实固体谐波推导、
      6d 的 s 污染分量、维度计数、Atomiverse 上手、章末 PySCF 对比表行 — 表格标量,不用条形图)
- [x] Run full check + test suite

## Testing strategy

- 快乐路径:core real-harmonics 迁移后原 gpw 断言全部保持;CartToSph 形状与钉死系数;
  H2O/def2-SVP 球谐 S 三明治 1e-12;n_basis 24。全部为秒级单元测试(tests/ 无 E2E)。
- 边界:l=0/1 时 c2s 为恒等(1×1、3×3);不支持的 l>3 抛异常(无回退值);6-31G* 门保持笛卡尔;
  球谐基梯度 guard-throw 断言。
- 领域验证(science.required):(a) 双阶乘归一化陷阱三明治恒等测试;(b) regression
  reg_h2o_rhf_def2svp_sph 对 PySCF `cart=False` 外部钉死能量(oracle 协议来自链环 01-env;
  PySCF 是外部 oracle,非构建/测试期依赖)。

## Out of scope

- 球谐基的解析梯度 / 积分导数 c2s(本链环 guard-throw,后续链环接管)。
- CUDA SCF 路径(能量/密度)、MP2/FCI 的球谐适配之外的性质模块;g 及更高角动量。
- gpw/pw 域改用 CartToSph(其投影和按加法定理与 m 排序无关,维持现状);benchmarks/ 平价扫描
  (属 benchmarks/scf/validation,后续链环)。

<!-- simplify: janitor pass applied 2026-07-20 — 1 red (doxygen block collision) + 5 yellow + 8 green fixed inline (reference-binding bridge, O(n^3) comment, static_assert kMaxL, de-spec-ID comments, test rename test_real_harmonics); deferred: bridge GEMM avoidance → optimizer, IntegralPack-surface judgment → architect. -->
