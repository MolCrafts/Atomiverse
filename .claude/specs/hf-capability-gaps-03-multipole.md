---
title: "HF capability gaps 03 — 多极矩单电子积分内核 + 偶极/四极矩观测器"
slug: hf-capability-gaps-03-multipole
status: code-complete
chain: hf-capability-gaps
link: 03/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# 多极矩单电子积分内核 + 偶极/四极矩观测器

## Summary
为 Gaussian SCF 增加多极矩单电子积分内核与分子多极矩观测器。在 `cpu/scf/internal/one_electron.h` 中以移位中心 Obara–Saika 递推实现 ⟨μ|(r−C)^e|ν⟩ 类积分(链环 15 外场、链环 18 X2C 性质将复用同一内核);新建 `cpu/observer/multipole.h` 的 `MultipoleObserver` 组件,消费 `MeanFieldOutputs`(共享的方法后视图;自 scf-slater-determinant-03-unbag 起其 `result` 为 `shared_ptr<const SlaterDeterminant>`),计算偶极矩 μ = Σ Z_A R_A − Tr(P r) 与无迹 Buckingham 四极矩,原点约定为核电荷中心并随数值一并报告。验收基准:RHF/6-31G* 水分子偶极矩与外部钉定的 PySCF `mf.dip_moment()` 逐分量吻合到 1e-6 a.u.(HF 高估 ~2.2 D vs 实验 1.85 D)。同时撰写教程第 20 章"分子多极矩"一节。

## Domain basis
- 移位中心 Obara–Saika 递推:多极矩积分 ⟨μ|(x−C_x)^e|ν⟩ 由重叠积分的 OS 递推增加一条多极指标升阶关系得到。Obara & Saika, *J. Chem. Phys.* **84**, 3963 (1986), DOI 10.1063/1.450106。
- 期望值与偶极矩:μ = Σ_A Z_A R_A − Tr(P·r),Szabo & Ostlund, *Modern Quantum Chemistry*, properties 一节(单行列式期望值;⟨Φ|Ô|Φ⟩ = Tr(P·O) 仅对单电子算符成立 — 与 `SlaterDeterminant::expect` 的文档约定一致)。
- 无迹 Buckingham 四极矩:Θ_ij = ½ Σ q(3 r_i r_j − r² δ_ij) 之核−电子和,Tr Θ = 0。
- 原点依赖(带电/有偶极体系):原点平移 d 时 μ′ = μ − q_tot·d;四极矩按偶极项平移(Θ′ 含 μ 与 d 的交叉项)。中性分子偶极矩原点无关——两者都编码为单元测试。
- 单位:内部 a.u.(e·Bohr / e·Bohr²),报告 Debye 与 Buckingham(换算入 `core/units.h`);Frame meta 以 Debye/Buckingham 存储,原点以 Å 存储。
- 平移恒等式(内核级):M¹_x(C′) = M¹_x(C) + (C_x − C′_x)·S。

## Design
- **内核**(与现有一电子 builder 同型的 internal 实现,非公共 API;注意 scf-slater-determinant-04-methods 收编 `build_*` 自由函数并开 `build_*` 通配门禁——若本环晚于 04-methods 落地,内核以所属类成员方法落地(如 ShellList/积分工作区的成员),不得新增 `build_*` 自由函数):
  - 偶极矩积分:`(shells, origin_bohr)` → 3 个 (n×n) 矩阵 ⟨μ|(r−C)_i|ν⟩;
  - 二阶矩积分:`(shells, origin_bohr)` → 6 个 (n×n) 矩阵 ⟨μ|(r−C)_i(r−C)_j|ν⟩。
  依赖链环 02(cart2sph)确定最终 AO 维度:积分按笛卡尔壳块生成后走与 S/T/V 相同的球谐变换,维度与 P 一致。
- **观测器** `ATV::cpu::MultipoleObserver`:继承 `Observer` kind(Role = FrameImmutable|Deferred),构造函数取 `scf::MeanFieldOutputs`(librarian 裁定:**禁止**扩展 IntegralPack 债务面;不得仿照 mulliken.h 绑定 IntegralPack)。`launch` 时:由 Frame 的 Z/坐标求核电荷中心为原点 → 由 handle 的 shells 重建多极积分 → 自 `MeanFieldOutputs` 的 `SlaterDeterminant` 按自旋道取密度并求和 P = P^α + P^β(受限单道 ×2 由 determinant 派生),电子部分单电子期望值优先经 `SlaterDeterminant::expect(O)` → 组装 μ 与无迹 Θ → 写 `frame.meta`:新 meta 键 `kDipole`(MetaF64x3, Debye)、`kQuadrupole`(MetaF64x6, Buckingham)、`kMultipoleOrigin`(MetaF64x3, Å);访问器 `dipole_au()` / `dipole_debye()` / `quadrupole_buckingham()` / `origin_angstrom()`。无默认绑定,用户挂到宿主驱动的 `end_of_step`(同 `OrbitalInfoObserver` 用法)。积分为观测器私有暂存,不上 Frame。
- **教程**:`docs/tutorial/ch20-hf-properties-environment.md`(骨架由链环 01 创建;若尚未落地则本链环建骨架)新增"分子多极矩"一节:期望值推导、原点依赖推导、Atomiverse 实操(molvis 在分子上渲染偶极矢量)、章末 PySCF 对照表行(标量入表,禁止条形图)。

## Files to create or modify
- `include/atomiverse/cpu/scf/internal/one_electron.h` — 多极积分声明
- `src/cpu/scf/one_electron.cpp` — 移位中心 OS 递推实现
- `include/atomiverse/cpu/observer/multipole.h` (new) — `MultipoleObserver`
- `src/cpu/observer/multipole.cpp` (new) — 观测器实现
- `include/atomiverse/core/frame_schema.h` — `kDipole` / `kQuadrupole` / `kMultipoleOrigin`
- `tests/cpu/scf/test_multipole_integrals.cpp` (new) — 内核单元测试
- `tests/cpu/observer/test_multipole.cpp` (new) — 观测器单元测试
- `tests/CMakeLists.txt` — 注册两个测试
- `regressions/scf/h2o_rhf_dipole/reg_h2o_rhf_dipole.cpp` (new) — e2e 回归
- `regressions/scf/h2o_rhf_dipole/REFERENCE.md` (new) — 钉定值与引文
- `regressions/CMakeLists.txt` — 注册回归
- `docs/tutorial/ch20-hf-properties-environment.md` — "分子多极矩"一节

## Tasks
- [x] Write failing tests for 多极积分内核 (`tests/cpu/scf/test_multipole_integrals.cpp`,注册进 `tests/CMakeLists.txt`):两 s 型 Gaussian 的 ⟨s|(x−C_x)|s⟩ 闭式值、平移恒等式 M¹(C′)=M¹(C)+(C−C′)S、矩阵对称性、二阶矩对 p 壳的解析值
- [x] Implement 偶极/二阶矩积分内核(移位中心 Obara–Saika)in `include/atomiverse/cpu/scf/internal/one_electron.h` + `src/cpu/scf/one_electron.cpp`,含带单位的 Doxygen(命名遵守 04-methods 后的 `build_*` 门禁形状)
- [x] Write failing tests for `MultipoleObserver` (`tests/cpu/observer/test_multipole.cpp`,注册进 `tests/CMakeLists.txt`,合成 `MeanFieldOutputs`——手工构造 `SlaterDeterminant`,不跑 SCF):单原子 μ=0、原点=核电荷中心、Tr Θ=0、带电体系 μ′=μ−q_tot·d 与四极矩偶极平移项、meta 键写入
- [x] Implement `MultipoleObserver` in `include/atomiverse/cpu/observer/multipole.h` + `src/cpu/observer/multipole.cpp`(Observer kind,FrameImmutable|Deferred,消费 `MeanFieldOutputs` 的 `SlaterDeterminant`,零 IntegralPack/SCFResult 引用),并在 `include/atomiverse/core/frame_schema.h` 添加三个 meta 键
- [x] Add 回归 `regressions/scf/h2o_rhf_dipole/`(`reg_h2o_rhf_dipole.cpp` + `REFERENCE.md`,注册进 `regressions/CMakeLists.txt`):库外消费者,RHF/6-31G* H₂O,偶极分量 vs 外部钉定 PySCF `mf.dip_moment()` 至 1e-6 a.u.,REFERENCE.md 记录 PySCF 版本/输入/钉定值并引 Szabo–Ostlund,≤60 s
- [x] Write `docs/tutorial/ch20-hf-properties-environment.md` 的"分子多极矩"一节:期望值与原点依赖推导、Atomiverse 实操 + molvis 偶极矢量渲染、章末 PySCF 对照表行
- [x] Run full check + test suite(含 no-factory / stage-typing scrub 与 regressions 目标)

## Testing strategy
- **happy path(单元,秒级)**:内核闭式值与平移恒等式;观测器在合成 `SlaterDeterminant` 上的 μ/Θ 组装与 meta 写入。tests/ 内**不跑任何完整 SCF**。
- **边界**:带电体系原点平移 μ′=μ−q_tot·d;四极矩的偶极平移项;中性分子偶极原点无关;Tr Θ=0(数值 <1e-12);单原子孤立体系 μ=0。
- **领域验证(regressions/)**:H₂O RHF/6-31G* 偶极 vs 钉定 PySCF 值 1e-6 a.u.(HF ~2.2 D 类,高估 vs 1.85 D 实验)——PySCF 仅为外部 oracle,不成为依赖。
- 观测器测试构造合成 `MeanFieldOutputs`(手工 `SlaterDeterminant` + 单/双壳 ShellList),隔离验证观测器本身。

## Out of scope
- 外场耦合(−μ·E 进 Fock)— 链环 15;X2C 图像变化修正的性质算符 — 链环 18(两者复用本内核)。
- 八极及更高阶矩;CUDA 后端多极矩;UHF/UKS 自旋分解多极(P^α+P^β 已覆盖总密度)。
- mulliken.h 从 IntegralPack 迁走(scf-slater-determinant 链 03-unbag 已覆盖其迁移,不在本链环顺手做)。
- molvis 本体功能开发(教程仅调用现有渲染)。

<!-- simplify: inline self-check 2026-07-20 — new files only (kernel class + observer + tests + regression + docs); meta insert-or-set documented; MeanFieldOutputs gained the c2s member (spherical consumers); no dead code. Deviation noted in ac-003 audit (observer-level translation test pins the origin-co-moving convention). -->
