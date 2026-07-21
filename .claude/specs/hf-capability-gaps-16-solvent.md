---
title: C-PCM implicit solvation at the SCF Fock seam
slug: hf-capability-gaps-16-solvent
status: code-complete
chain: hf-capability-gaps
link: 16/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# C-PCM implicit solvation at the SCF Fock seam

## Summary

为 Gaussian-basis 平均场方法(RHF/UHF/RKS/UKS,经共享基类 `MeanFieldMethod`)加入导体类隐式溶剂
C-PCM:由原子中心球(Bondi 半径 × 缩放系数)构成空腔,用现有 Lebedev 网格离散为表面 tesserae,
每次 SCF 迭代在 fock 阶段由当前密度解导体方程得表面表观电荷,并将反应场贡献加回 Fock 矩阵——
与 link 15 的静态 Hcore 嵌入不同,溶剂电荷随密度自洽。用户面 API 仅为三个方法成员
(`solvent_eps` 默认 0=关闭、`solvent_radii_scale`、`solvent_x`),Frame 进出不变。
本 link 只含静电部分单点能;解析梯度、表面平滑(SWIG)、非静电项与 ddCOSMO 线性标度求解器均诚实排除
(scope 裁定:本环 C-PCM only;ddCOSMO 为线性标度后续路线)。

## Domain basis

- 导体方程(tesserae 点电荷离散):`S q = −f(ε) V`,其中 `V_k = Σ_A Z_A/|R_A−s_k| − Σ_μν P_μν ⟨μ|1/|r−s_k||ν⟩`
  为溶质在 tessera `s_k` 处的总静电势;`S_kl = 1/|s_k−s_l|`,对角 `S_kk = 1.0694·√(4π/a_k)`
  (Klamt–Schüürmann),`a_k` 为 tessera 面积。单位:内部 Hartree/Bohr,对外 eV。
- 缩放因子 `f(ε) = (ε−1)/(ε+x)`:**x=0 为 C-PCM(Barone–Cossi,PySCF 兼容,默认);x=0.5 为原始
  COSMO(Klamt)**。约定分歧在低 ε 时改变结果,必须在文档与 Doxygen 中写明。
- 反应场 Fock 贡献:`F_μν += Σ_k q_k ⟨μ|1/|r−s_k||ν⟩`(复用 link 15 的点势积分机制,即
  `internal::one_electron` 任意中心核吸引积分)。静电溶剂化能 `ΔG_el = ½ Σ_k q_k V_k`。
- 单球解析 oracle(单元测试):Born 离子 `ΔG = −½ f(ε) Q²/R`(Born, Z. Phys. 1, 45 (1920))。
- Refs: Barone & Cossi, J. Phys. Chem. A 102, 1995 (1998), DOI 10.1021/jp9716997 (C-PCM);
  Klamt & Schüürmann, J. Chem. Soc. Perkin Trans. 2 (1993) 799, DOI 10.1039/P29930000799 (COSMO);
  Cancès, Maday & Stamm, J. Chem. Phys. 139, 054111 (2013), DOI 10.1063/1.4816767(ddCOSMO,
  仅作为标度路线引用,本 link 不实现); Jensen, *Introduction to Computational Chemistry*,
  连续介质溶剂化一章(叙事基调)。

## Design

- `internal::Pcm`(`cpu/scf/internal/pcm.h`,SCF 私有工作区,不继承 Component,不出现在公共 API):
  - 构造子接收原子几何 + 半径缩放 + 每球 Lebedev 阶(默认 302 点,与 PySCF 匹配);Bondi 半径为
    头文件内静态表,不支持的 Z 直接 throw(无 fallback,无解析器,无工厂函数)。
  - `prepare(shells)`:生成 tesserae(复用 `lebedev_data.h` 网格;埋没点按"点在他球内即弃"硬开关剔除
    ——空腔面不连续导致能量对几何不连续,文档写明,平滑不做);组装并分解 S 矩阵;预缓存每个
    tessera 的 AO 势矩阵 `⟨μ|1/|r−s_k||ν⟩`(一次/几何,之后每迭代仅做收缩)。
  - `update(P_total) → {V_pcm 矩阵, ΔG_el}`:算 V、解 `S q = −f(ε)V`、收缩回 AO。
- `MeanFieldMethod<D>` 新成员(成员即参数,无 Config 袋):`double solvent_eps = 0`(0=关)、
  `double solvent_radii_scale = 1.2`、`double solvent_x = 0.0`;私有 `std::unique_ptr<internal::Pcm>`。
- Fock 缝接线:`FockStage` 增加可选非拥有 `internal::Pcm*`(null=关)与 `double& e_solvent` 输出引用,
  在 `scf::Phase::fock` 阶段每迭代调用 `update`,把 `V_pcm` 加到两个自旋通道(开壳层用总密度
  P=Pα+Pβ),故 UHF/UKS 免费获得。能量装配(`mean_field_method.cpp`)加 `e_solvent_`;
  结果面增 `double e_solvent`(Hartree;落点跟随 scf-slater-determinant-03-unbag 后的结果形态)。
  `solvent_eps==0` 路径逐位不变。
- `forces()` 在 `solvent_eps > 0` 时 throw(解析 PCM 梯度为后续 link)。

## Files to create or modify

- `include/atomiverse/cpu/scf/internal/pcm.h` (new)
- `src/cpu/scf/pcm.cpp` (new)
- `CMakeLists.txt`(注册 pcm.cpp,随 `ATV_BUILD_HF`)
- `include/atomiverse/cpu/scf/mean_field_method.h`
- `src/cpu/scf/mean_field_method.cpp`
- `include/atomiverse/cpu/scf/stages/fock_stage.h`
- `src/cpu/scf/stages/fock_stage.cpp`
- `include/atomiverse/cpu/scf/scf_result.h`(e_solvent 结果字段;03-unbag 后跟随新结果面)
- `tests/cpu/scf/test_pcm.cpp` (new)
- `tests/CMakeLists.txt`
- `regressions/scf/h2o_rhf_631gs_cpcm/REFERENCE.md` (new)
- `regressions/scf/h2o_rhf_631gs_cpcm/reg_h2o_rhf_631gs_cpcm.cpp` (new)
- `regressions/CMakeLists.txt`
- `docs/tutorial/ch20-hf-properties-environment.md`(由 link 15 创建;若尚未落地则本 link 新建章壳)

## Tasks

- [x] Write failing tests for `internal::Pcm`:空腔面积/埋点剔除、f(ε) 两约定取值、Born 离子
      `ΔG=−½f(ε)Q²/R` 1% 内、Σq=−f(ε)Q(`tests/cpu/scf/test_pcm.cpp` (new),注册进 `tests/CMakeLists.txt`)
- [x] Implement `internal::Pcm` in `include/atomiverse/cpu/scf/internal/pcm.h` (new) +
      `src/cpu/scf/internal/pcm.cpp` (new),注册进根 `CMakeLists.txt`
- [x] Write failing tests for 方法成员与 fock 缝自洽:H2O/STO-3G `solvent_eps=78.4` 收敛且
      `e_solvent<0`、总能低于气相;`solvent_eps=0` 与气相逐位一致;`forces()` 开溶剂时 throw
      (扩展 `tests/cpu/scf/test_pcm.cpp`)
- [x] Implement 成员 + 接线 in `include/atomiverse/cpu/scf/mean_field_method.h`、
      `src/cpu/scf/mean_field_method.cpp`、`include/atomiverse/cpu/scf/stages/fock_stage.h`、
      `src/cpu/scf/stages/fock_stage.cpp` (e_solvent on MeanFieldMethod accessor; no scf_result.h)
- [x] Verify against 外部钉定的 PySCF `solvent.PCM`(C-PCM 设置逐项匹配):
      `regressions/scf/h2o_rhf_631gs_cpcm/` (new),`REFERENCE.md` 钉定版本/ε/半径/网格与数值,
      注册进 `regressions/CMakeLists.txt`
- [x] Add 教程「隐式溶剂」节 in `docs/tutorial/ch20-hf-properties-environment.md`(导体叙事、空腔
      是什么、x 约定分歧、诚实声明:非静电项不含、表面不连续未平滑;动手算溶剂化能 + molvis
      空腔/表面电荷渲染 + 章末 PySCF 标量对照表)并为全部新公共符号补 Doxygen(含单位)
- [x] Run full check + test suite

## Testing strategy

- Happy path:Born 单球解析值(1%);H2O/STO-3G C-PCM 收敛、`e_solvent<0`(秒级,单元层不做
  6-31G* 大例)。
- Edge cases:`solvent_eps=0` 逐位等于气相;不支持元素的 Bondi 查表 throw;`forces()` 开溶剂 throw;
  `solvent_x=0.5` 与 `0.0` 在低 ε(如 ε=2)给出可分辨差异(约定测试)。
- Domain validation(regressions/,≤60 s):H2O HF/6-31G*,ε=78.4,ΔG_el 对外部钉定 PySCF C-PCM
  值 |Δ| ≤ 0.1 kcal/mol,量级落在 −6…−8 kcal/mol 类。

## Out of scope

- ddCOSMO 线性标度求解器(引用为后续路线,DOI 10.1063/1.4816767)。
- 解析 PCM 核梯度(本 link `forces()` throw);数值梯度留给使用方。
- 表面开关平滑(SWIG/York–Karplus)——空腔不连续如实写入文档。
- 非静电项(空穴化、色散、排斥)——文档诚实声明不含。
- CUDA 路径、周期性边界下的 PCM、`SCFSolver` 债务面的任何扩展。
