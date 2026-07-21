---
title: External point-charge embedding and uniform electric field in the core Hamiltonian
slug: hf-capability-gaps-15-embed
status: code-complete
chain: hf-capability-gaps
link: 15/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# 外部点电荷嵌入与均匀电场(核哈密顿量环境项)

## Summary

为具体平均场方法(`RHF`/`UHF`/`RKS`/`UKS`)加入两类静电环境:外部点电荷嵌入(QM/MM 静电嵌入的 QM 侧)与均匀外电场。两者都是单电子 Fock 贡献,属 SCF 私有工作区(`cpu/scf/internal/`),在 `drive_scf_core_` 的积分装配序幕中并入私有 Hcore 与经典能量项;用户面只有两个方法成员:`rhf.point_charges = {...}`(普通 vector,无 Config 包)与 `rhf.external_field = {Ex,Ey,Ez}`。交叉验证锚点:有限场偶极 dE/dF 与 link 03 解析偶极一致;点电荷嵌入 H₂O 能量对外部钉定的 PySCF `qmmm.mm_charge` 值一致。

## Domain basis

- 点电荷嵌入:h_μν += −Σ_C q_C⟨μ|1/|r−R_C||ν⟩;经典项 E += Σ_A Σ_C Z_A q_C/|R_A−R_C| + Σ_{C<C'} q_C q_C'/|R_C−R_C'|。积分即"位于 R_C 的单位电荷核吸引积分"——复用现有核吸引原语(link 04 ESP 势核的同一机器)。Obara & Saika, J. Chem. Phys. 84, 3963 (1986), DOI 10.1063/1.450106;静电嵌入:Warshel & Levitt, J. Mol. Biol. 103, 227 (1976), DOI 10.1016/0022-2836(76)90311-9。
- 均匀电场:h_μν += +F·⟨μ|(r−R₀)|ν⟩(电子电荷 −1 a.u.),核项 E += −Σ_A Z_A F·(R_A−R₀);R₀ = 核电荷中心,**必须与 link 03 偶极积分同一原点约定**(以测试固化)。一阶恒等式 E(F) = E(0) − μ·F ⇒ 中心差分 dE/dF 在 F→0 时等于 −μ(解析)。
- 单位:成员 position [Å](内部经 `kAngstrom2Bohr` 转 Bohr)、charge [e]、field [a.u.];能量内部 Hartree。
- **有限场变分坍缩(必须以测试+教程固化)**:有限场中能量无下界;弥散基下 SCF 可"电离"。场强须保持小(~1e-4 a.u.)。

## Design

- `internal::PointChargeEmbedding`(新,`internal/point_charges.h`):构造自 `const std::vector<PointCharge>&`(构造时 Å→Bohr,校验有限值);成员函数 `core_matrix(const ShellList&)`、`nuclear_energy(const AtomList&)`、`self_energy()`。类+构造函数,**无 `build_*` 自由函数**。
- `internal::ExternalField`(新,`internal/external_field.h`):构造自 `(field[3] a.u.], const AtomList&)`,构造时算核电荷中心 R₀;`core_matrix(const ShellList&)` 用 link 03 的偶极积分;`nuclear_energy(const AtomList&)`。
- `MeanFieldMethod<D>` 新成员(触碰 scf-slater-determinant-03-unbag 的成员面,故 external-depends):`struct PointCharge { double x, y, z; double q; };`(POD,定义于 `mean_field_method.h`,非 Config 包)、`std::vector<PointCharge> point_charges;`、`std::array<double,3> external_field{0,0,0};`。公共头**不** include 新 internal 头(实现文件内 include)。
- 接线:`drive_scf_core_` 序幕在 `IntegralPack` 一电子装配后、`HamiltonianBuilder::prepare` 前,把两个环境矩阵加入私有 Hcore 工作区,经典项加入 `v_nuc_`——Fock 循环(`hamiltonian_builder`)无感知,e_total 自动含环境项。(设计偏差说明:worldview 说接线 hamiltonian_builder,但现树 Hcore 装配在 `IntegralPack`/`drive_scf_core_` 序幕,单点接入、Fock 循环零改动,意图相同、表面更小。)
- 生命周期:环境对象是每次 `drive_scf_` 的局部量,随几何重建;Frame 上不存任何环境 scratch(Frame 纯净规则)。
- 梯度:含环境项的解析力尚未推导——`forces()` 在环境非空时抛 `std::runtime_error`(无静默错误力;后续 link 处理)。

## Files to create or modify

- include/atomiverse/cpu/scf/internal/point_charges.h (new)
- src/cpu/scf/internal/point_charges.cpp (new)
- include/atomiverse/cpu/scf/internal/external_field.h (new)
- src/cpu/scf/internal/external_field.cpp (new)
- include/atomiverse/cpu/scf/mean_field_method.h
- src/cpu/scf/mean_field_method.cpp
- tests/cpu/scf/test_embedding.cpp (new)
- tests/CMakeLists.txt
- regressions/scf/h2o_embedding_field/reg_h2o_embedding_field.cpp (new)
- regressions/CMakeLists.txt
- docs/tutorial/ch20-hf-properties-environment.md(由本链早前 link 创建;若尚未落地则新建章骨架后加节)

## Tasks

- [x] Write failing tests for embedding and external field (tests/cpu/scf/test_embedding.cpp,注册进 tests/CMakeLists.txt):单电荷核矩阵 == Z=q 的核吸引矩阵(机器恒等);经典核-电荷/电荷-电荷手算和;ExternalField 原点 == link 03 偶极原点(非对称分子);小场能移 E(F)−E(0) ≈ −μ·F;环境非空时 forces() 抛异常
- [x] Implement internal::PointChargeEmbedding in include/atomiverse/cpu/scf/internal/point_charges.h + src/cpu/scf/internal/point_charges.cpp(复用核吸引原语;构造函数校验,无 build_*)
- [x] Implement internal::ExternalField in include/atomiverse/cpu/scf/internal/external_field.h + src/cpu/scf/internal/external_field.cpp(link 03 偶极积分 + 核电荷中心原点)
- [x] Implement 成员 point_charges / external_field 于 include/atomiverse/cpu/scf/mean_field_method.h 并在 src/cpu/scf/mean_field_method.cpp 的 drive_scf_core_ 序幕接线(Hcore += 环境矩阵;v_nuc_ += 经典项;环境非空 + forces 路径抛异常)
- [x] Add Doxygen(@brief/@param 带单位/@return)到所有新符号,含坍缩风险警告
- [x] Write regression regressions/scf/h2o_embedding_field/reg_h2o_embedding_field.cpp(注册进 regressions/CMakeLists.txt):H₂O/STO-3G,F=1e-4 a.u. 中心差分偶极 vs 解析偶极 ≤1e-5 a.u.;点电荷嵌入能量 vs 外部钉定 PySCF qmmm.mm_charge ≤1e-8 Ha(MM–MM 常数项在断言中显式手算处理);≤60 s
- [x] Write docs/tutorial/ch20-hf-properties-environment.md 的「外场与点电荷嵌入」节:QM/MM 嵌入叙事(环境如何进 Fock 算符)、有限场 vs 解析偶极对照表、坍缩风险的诚实警告、章末 PySCF 标量表行
- [x] Run full check + test suite

## Testing strategy

- 快乐路径(单元,秒级):机器恒等(点电荷矩阵 vs Z=q 核吸引)、经典能量手算和、原点约定钉死、小场线性响应符号与量级。
- 边界:空 point_charges / 零场 == 完全不扰动(逐元素恒等);非有限输入构造即抛;forces()+环境抛异常。
- 领域验证(regression,文献/外部 oracle):有限场偶极恒等式 ≤1e-5 a.u.(E(F)=E0−μ·F 推导);嵌入 H₂O vs PySCF ≤1e-8 Ha。
- 教程含坍缩风险演示(叙述+小场纪律),不做长时间演示计算。

## Out of scope

- 含点电荷/外场的解析梯度与 QM/MM 力(后续 link;本 link 抛异常拦截)。
- 极化嵌入、周期性嵌入、Ewald 修正的 MM 电荷;CUDA 路径;UHF/UKS 之外的开壳特化(成员在基类,自动继承,但 oracle 只钉 RHF)。
- 有限场超极化率 / 高阶响应;弥散基坍缩的自动防护(只文档化纪律)。
