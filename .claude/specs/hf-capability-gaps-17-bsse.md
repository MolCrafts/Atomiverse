---
title: BSSE counterpoise correction (ghost basis + 5-SCF orchestration)
slug: hf-capability-gaps-17-bsse
status: code-complete
chain: hf-capability-gaps
link: 17/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# BSSE counterpoise 校正(ghost 基组 + 5-SCF 编排)

## Summary
为高斯基 SCF 引入 ghost 原子支持并提供 Boys–Bernardi counterpoise(CP)校正:ghost 原子只贡献基函数,不贡献核电荷与电子;`Counterpoise` 编排类对二聚体执行 5 次收敛 SCF(二聚体、A 单体、B 单体、A 在二聚体基组中(B ghost)、B 在二聚体基组中(A ghost)),返回 CP 校正的相互作用能及各分量。纯编排——不新增 Driver、不新增 Component、不注册 Plan;用户面只有具体方法成员 `ghost_atoms` 与消费者工具类 `Counterpoise`。

## Domain basis
- CP 配方(Boys & Bernardi, *Mol. Phys.* **19**, 553 (1970), DOI 10.1080/00268977000101561):
  ΔE_CP = E_AB^{AB} − E_A^{AB}(B ghost) − E_B^{AB}(A ghost);**符号约定(锁定)**:bsse = (E_A^{AB} − E_A^A) + (E_B^{AB} − E_B^B) ≤ 0(单体在更大的二聚体基组中能量必然降低),|bsse| 即基组重叠误差幅度——头文件 Doxygen、测试与教程三处同此约定。
- Ghost 原子:基函数进入 AO 空间,但核吸引 V 与核排斥 V_nuc 中 Z→0,且不计电子。
- 单位:内部 Hartree;`run(const Frame&)` 常几何入口,不写 Frame meta。
- 诚实教学(编入文档与测试注释,Jensen《Introduction to Computational Chemistry》基组/BSSE 节):小基组下 CP 常**过度校正**;多体簇 site–site 与 full-cluster CP 不一致(非 size-consistent);单体形变校正是独立项,本 spec 不含。
- Oracle:水二聚体 HF/6-31G*,对照外部固定的 PySCF ghost 前缀结果(`ghost:O` 等),1e-6 Ha;CP 校正量级应落在文献 mHa 级。

## Design
- **Frame 约定裁决(RULE):ghost 标记不进 Frame。** molrs `Element` 禁止负 Z,且 Frame 纯净性禁止方法专属列 —— ghost 是**方法成员** `std::vector<int> ghost_atoms`(`MeanFieldMethod<D>` 上,局部原子下标),与 `basis`/`charge` 同级。不新增任何 `kAtom_*` 列。
- `AtomList`(私有 workspace)构造子增加 ghost 下标参数:ghost 原子 `atoms[i].Z = 0`(V、V_nuc、dV_nuc、电子计数自动正确——已核实 integrals/integral_derivatives 全部经 `atoms[i].Z`),新增并行列 `std::vector<int> basis_z`(真实元素,供基组查表;实原子二者相等)。越界下标抛异常(no-fallback)。
- `ShellList` 查表从 `atoms[a].Z` 改读 `basis_z[a]`(basis_set.cpp:111/136);`IntegralPack` 构造子线程化传递 ghost 列表。`ghost_atoms` 非空且方法非纯 HF(KS 格点按 Z 取原子半径)→ 抛异常。
- `cpu::scf::Counterpoise`(新,公开,header-only 模板):构造子取两个单体下标集;`template <typename Method> CounterpoiseResult run(const Frame& dimer, Method& m)` 借用用户已配置的方法对象,临时改写/恢复其 `ghost_atoms`,单体单独计算用从二聚体 Frame 复制行构建的子 Frame(公开 Frame API)。`CounterpoiseResult` 含 `e_dimer, e_a, e_b, e_a_ghost, e_b_ghost, e_int_cp, e_int_raw, bsse`(均 Hartree)。无 Component 继承、无 `build_*`/`make_*`。
- 生命周期:`Counterpoise` 不持有 Frame/方法所有权;5 次 SCF 串行同线程。

## Files to create or modify
- include/atomiverse/cpu/scf/internal/basis_workspace.h — `AtomList` ghost 参数 + `basis_z`
- src/cpu/scf/basis_set.cpp — `AtomList`/`ShellList` 实现
- include/atomiverse/cpu/scf/internal/integrals.h — `IntegralPack` 构造子 ghost 参数
- src/cpu/scf/integrals.cpp — 传递 ghost 列表
- include/atomiverse/cpu/scf/mean_field_method.h — 成员 `ghost_atoms` + doxygen
- src/cpu/scf/mean_field_method.cpp — ghost 传入 IntegralPack;非 HF 守卫
- include/atomiverse/cpu/scf/counterpoise.h (new) — 编排类 + 结果结构
- tests/cpu/scf/test_ghost_basis.cpp (new) — workspace 级单元测试
- tests/cpu/scf/test_counterpoise.cpp (new) — He–He STO-3G 编排恒等式
- tests/CMakeLists.txt — 注册两个新测试
- regressions/scf/h2o_dimer_cp_hf631gs/reg_h2o_dimer_cp_hf631gs.cpp (new) + REFERENCE.md (new)
- regressions/CMakeLists.txt — 注册回归用例
- docs/tutorial/ch04-basis-sets.md — "BSSE 与 counterpoise"收尾节

## Tasks
- [x] Write failing tests for ghost-basis workspace (tests/cpu/scf/test_ghost_basis.cpp;含 tests/CMakeLists.txt 注册):ghost 后 `n_electrons` 不含其 Z、`V_nuc` 不含其对、`n_basis` 仍含其壳层、V 矩阵无其核项、越界下标抛异常
- [x] Implement ghost support in AtomList/ShellList(internal/basis_workspace.h + src/cpu/scf/basis_set.cpp:Z→0、`basis_z` 查表)
- [x] Implement ghost threading:IntegralPack 参数(internal/integrals.h + src/cpu/scf/integrals.cpp)与 `MeanFieldMethod::ghost_atoms` 成员 + 非 HF 守卫(mean_field_method.h + mean_field_method.cpp),doxygen 带单位
- [x] Write failing tests for Counterpoise orchestration (tests/cpu/scf/test_counterpoise.cpp):He–He/STO-3G 恒等式 `e_int_cp == e_dimer − e_a_ghost − e_b_ghost`、`bsse ≤ 0`(锁定约定)且 |bsse| 有限、run 后方法 `ghost_atoms` 恢复原值
- [x] Implement cpu::scf::Counterpoise in include/atomiverse/cpu/scf/counterpoise.h(构造子取单体下标集;5-SCF `run`;doxygen 编入符号约定与过度校正/size-consistency/形变校正三条 pitfall)
- [x] Add regression regressions/scf/h2o_dimer_cp_hf631gs/(reg_h2o_dimer_cp_hf631gs.cpp + REFERENCE.md + regressions/CMakeLists.txt):水二聚体 HF/6-31G* 对 PySCF ghost 定标值 1e-6 Ha,≤60 s,仅公开头文件
- [x] Extend docs/tutorial/ch04-basis-sets.md:"BSSE 与 counterpoise"一节(有限基组"借邻居函数"直觉、5-SCF 配方、过度校正诚实段、水二聚体动手示例、章末 PySCF 对照表标量行)
- [x] Run full check + test suite

## Testing strategy
- 快乐路径:`test_ghost_basis` 在 workspace 层验证 ghost 语义(不跑 SCF 循环);`test_counterpoise` 用最小 He–He/STO-3G 验证编排恒等式与状态恢复,秒级。
- 边界:ghost 下标越界抛异常;`ghost_atoms` 非空 + RKS/UKS 抛异常;空 ghost 列表 == 现状(回归不变量:既有 SCF 测试全绿)。
- 领域验证(science.required):回归用例对外部固定 PySCF 值 1e-6 Ha,CP 校正量级为文献 mHa 级;REFERENCE.md 引 Boys & Bernardi 1970 DOI 与 PySCF 定标命令。

## Out of scope
- CP 梯度/几何优化、单体形变(relaxation)校正、多体簇 site–site CP。
- 带电单体(`Counterpoise` 子计算取中性;方法 `charge` 仅作用于二聚体运行)。
- KS(RKS/UKS)与 CUDA/半经验/PW/GPW 的 ghost 支持(守卫抛异常,后续链节)。
- `SCFSolver` 债务面改动 —— 新 API 只落在具体方法成员面上。
