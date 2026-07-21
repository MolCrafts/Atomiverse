---
title: CPHF 线性响应与解析 HF Hessian(振动频率)
slug: hf-capability-gaps-13-cphf
status: code-complete
chain: hf-capability-gaps
link: 13/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# CPHF 线性响应与解析 HF Hessian(振动频率)

## Summary

实现耦合微扰 Hartree-Fock(CPHF)线性响应求解器与 RHF 解析二阶导数(核坐标 Hessian)。CPHF 方程 Σ_bj A_ai,bj U^x_bj = −B^x_ai 的左端算符直接复用链环 08 的 `OrbitalHessian` σ-向量核——这是该共享内核的第三个配置(08 稳定性求本征值、09 SOSCF 做 Newton 步、本环做线性方程组 LHS);对 3N_atom 个右端做迭代(CG/DIIS 型)求解。解析 Hessian 以方法输出 `rhf.hessian(frame)` 暴露,与 `rhf.forces(frame)` 并列,返回既有 `vib::HessianResult`,直接喂入现成 vib 流水(质量加权→对角化→热化学)——不新增任何 driver,`vib::Hessian` 保持有限差分生产者身份,二者汇于同一 `HessianResult`(librarian 裁定)。用户可见结果:H₂O 的 HF/6-31G* 解析谐振频率,与自家 FD Hessian 及外部钉定的 PySCF `hessian.RHF` 均吻合到 ~1 cm⁻¹。

## Domain basis

闭壳层 RHF、实微扰(核位移),全部内部量取 Hartree/Bohr,输出按 `vib::HessianResult` 契约转 eV/Å²,频率 cm⁻¹:
- CPHF 方程(Pople-Krishnan-Schlegel-Binkley 1979):¹(A+B)·U^x = −B^x,其中 ¹(A+B)_ai,bj = δ_ab δ_ij(ε_a−ε_i) + 4(ai|bj) − (ab|ij) − (aj|ib),恰为 08 环 kernel 的 singlet σ₊ 通道。
- 非齐次项(经典 bug 源之一,必须逐项 FD 门禁):B^x_ai = F^{(x)}_ai − ε_i S^{(x)}_ai − Σ_kl S^{(x)}_kl [2(ai|kl) − (ak|il)],F^{(x)}/S^{(x)} 为 skeleton(骨架)导数 Fock/重叠矩阵的 MO 块。
- 解析 Hessian:E^{xy} = Tr[P h^{xy}] + ½ ΣΣ P P g^{xy} + V_nuc^{xy} − Tr[W S^{xy}] + 4 Σ_ai U^y_ai B^x-耦合项;W = 能量加权密度 2 Σ_i ε_i C_μi C_νi(第二个经典 bug 源)。
- **陷阱(编码为验收门)**:重叠导数/能量加权密度项与 B^x 非齐次项是历史高发错误;**第一道闸 = 解析 Hessian 对"解析梯度的有限差分"的自洽吻合**(两者均在自家实现内,频率 ~1 cm⁻¹),过闸后才对外部 oracle。
- ε、C、占据一律取自 `SlaterDeterminant`(external-depends scf-slater-determinant-03-unbag)。
- 参考:Pople, Krishnan, Schlegel, Binkley, Int. J. Quantum Chem. Symp. 13, 225 (1979), DOI 10.1002/qua.560160825;Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* 响应理论章;Pulay, Mol. Phys. 17, 197 (1969)(梯度,gradient.h 已引)。Oracle:H₂O HF/6-31G*(`Pople631Gs`)谐振频率,vs 自家 FD Hessian ~1 cm⁻¹,vs 外部运行钉定的 PySCF `hessian.RHF` 频率 ~1 cm⁻¹。

## Design

- `internal/integral_derivatives.h` 扩展(现有 `IntegralDerivPack` 之侧):二阶导数积分 d²S/d²T/d²V/d²V_nuc 与二阶 ERI 直接缩并,以及 B^x 所需的 skeleton F^{(x)}/S^{(x)} MO 块装配——延续该头"任意 l ≤ kMaxL、Hartree/Bohr"的既有约定,升角动量公式二次应用。私有工作区,不上 Frame。
- `cpu::scf::internal::CphfPack`(新,internal 头,mp2_pack.h 先例):构造函数接 `(const SlaterDeterminant&, const OrbitalHessian&, …skeleton 导数件)`(构造即接线,`cpu/scf` 禁 `build_*`,受 `no_factory_scrub.sh` 门禁);成员 `solve()` 对 3N 个 RHS 迭代求 U^x(σ₊-singlet 为算符,CG + 可选 DIIS 加速,小维度稠密回退经 `cpu/linalg`),成员 `hessian()` 装配 E^{xy}。
- 公共薄面即 `rhf.h`:`RHF::hessian(Frame&)` → `vib::HessianResult`(fresh SCF + 梯度工作区 + CPHF + 装配;`hessian` 域 eV/Å²、`n_atoms` 填好,频率字段留给 vib 件)。**不新建公共结果类型、不新建 driver**;本环 RHF-only(UHF Hessian 越界)。
- 回归中频率经既有 `vib::MassWeighting`/`HessianDiagonalizer`(必要时 `Thermochem`)从 `HessianResult` 得出——解析与 FD 两条生产线汇于同一消费端。
- 链环 12 的 DF/RI 仅作规模化的可选消费(备注级),本环不接线。

## Files to create or modify

- include/atomiverse/cpu/scf/internal/integral_derivatives.h — 二阶导数积分 + skeleton F^{(x)}/S^{(x)} 扩展
- src/cpu/scf/integral_derivatives.cpp — 同上实现
- include/atomiverse/cpu/scf/internal/cphf_pack.h (new) — `internal::CphfPack`
- src/cpu/scf/cphf_pack.cpp (new) — CPHF 迭代解 + Hessian 装配
- include/atomiverse/cpu/scf/rhf.h — `RHF::hessian(Frame&)` 声明(Doxygen 带单位)
- src/cpu/scf/rhf.cpp — `hessian()` 实现
- CMakeLists.txt — HF 源列表追加 cphf_pack.cpp
- tests/cpu/scf/test_cphf.cpp (new)
- tests/cpu/scf/test_analytic_hessian.cpp (new)
- tests/CMakeLists.txt — 注册两个单测
- regressions/scf/h2o_rhf_freq/reg_h2o_rhf_freq.cpp (new)
- regressions/scf/h2o_rhf_freq/REFERENCE.md (new)
- regressions/CMakeLists.txt — `atv_add_regression(scf h2o_rhf_freq)`(`ATV_BUILD_HF` 门内)
- docs/tutorial/ch20-hf-properties-environment.md — "解析 Hessian 与振动频率"一节

## Tasks

- [x] Write failing tests for 二阶/skeleton 导数积分 (tests/cpu/scf/test_cphf.cpp,注册 tests/CMakeLists.txt):HeH⁺/STO-3G 上 d²S/d²T/d²V vs 一阶导数的中心差分 ≤ 1e-7;MO 基 S^{(x)}/F^{(x)} vs S/F 的 FD ≤ 1e-6;B^x 平动零和
- [x] Implement 积分二阶导数与 skeleton F^{(x)}/S^{(x)} in include/atomiverse/cpu/scf/internal/integral_derivatives.h + src/cpu/scf/integral_derivatives.cpp
- [x] Write failing tests for CphfPack 求解 (test_cphf.cpp 追加):迭代 U^x vs 稠密 ¹(A+B) 直解 ≤ 1e-10;LHS 作用 == 08 环 `sigma_plus(v, singlet)` 逐位一致
- [x] Implement CphfPack in include/atomiverse/cpu/scf/internal/cphf_pack.h + src/cpu/scf/cphf_pack.cpp(CG/DIIS 迭代 + 稠密回退;登记 CMakeLists.txt;Doxygen 带单位)
- [x] Write failing tests for RHF::hessian (tests/cpu/scf/test_analytic_hessian.cpp,注册 tests/CMakeLists.txt):H₂/STO-3G 解析 Hessian vs 解析力的中心差分逐元 ≤ 1e-6 eV/Å²(第一道闸的微型体现);对称性 H=Hᵀ;行平动零和 Σ_j H_ij ≈ 0
- [x] Implement RHF::hessian in include/atomiverse/cpu/scf/rhf.h + src/cpu/scf/rhf.cpp(返回 vib::HessianResult,eV/Å²,n_atoms 填好)
- [x] Verify against H₂O oracle:regressions/scf/h2o_rhf_freq(6-31G*;第一道闸解析-vs-自家-FD 频率 ≤ 1 cm⁻¹,后对 REFERENCE.md 钉定的 PySCF hessian.RHF 频率 ≤ 1 cm⁻¹;频率经既有 vib 件产出;注册 regressions/CMakeLists.txt,≤ 60 s)
- [x] Write docs/tutorial/ch20-hf-properties-environment.md 之"解析 Hessian 与振动频率"节:为何二阶导数需要轨道响应(U^x)、稳定性内核的第三次复用叙事、H₂O 频率上手、molvis 简正模可视化(molplot 辅助),章末 PySCF 频率对照表(标量入表)
- [x] Run full check + test suite

## 实现审计注记(2026-07-21)

- **Hessian 响应项的等价改写**:装配式用约定无关的
  `E^{xy} += Tr[P^y F^{(x)}] − Tr[W^y S^{(x)}]`(P^y/W^y 为密度与能量加权
  密度响应,W = ½ P F P 的导数)替代教科书的 `4 Σ_ai U^y_ai B^x_ai + occ-occ
  伴随项`。两式代数等价;所选形式对 occ-occ 旋转规范不敏感(只有对称组合
  U_ji + U_ij = −S^{(y)}_ij 进入),从构造上规避"响应耦合因子"这一经典
  bug 源。因子由 FD 门禁钉死:H₂ 解析-vs-FD ≤1e-6 eV/Å²、密度响应
  ∂P/∂x-vs-FD ≤1e-6(约定无关的独立符号/因子检验)、H₂O 频率 vs 自家 FD
  0.0092 cm⁻¹ / vs PySCF 0.0003 cm⁻¹。
- **l+2 内部升角动量**:一阶导数缓冲已含 +1 裕量;二阶只需两处
  native_integrals.cpp 常量扩容(`kMaxKineticCells` 25→36、`kMDEIDim`
  kDerivL+1→kMaxLOrbital+3),f 壳(l=3)+2=5 由此覆盖;nuclear R 表
  (dim 9,Ltot≤8)经逐项核算无需扩。二阶 ERI 不经 eri_prim——
  integral_derivatives.cpp 自带共享 R 表(至 Ltot+2)+ 升角 Hermite E 表
  内核。
- **d²ERI 缩并**:每原始四元组一次 R 表,45 个独立二阶元素经 bra/ket
  部分缩并(Gket/Hbra/K 中间量)装配,D 中心行列由平移不变性补全;
  四指标二阶张量从不落地。OpenMP 逐 canonical block pair + Schwarz 筛。
- **skeleton dF 的 2e 部分**:全序 block 四元组枚举(无 8 重对称计数),
  O(n_block⁴) 无 Schwarz 筛——Hessian 装配尺度每次调用仅一遍,微型/回归
  体系代价可忽略;规模化留给 DF/RI 消费(链环 12 备注级)。
- **超出 spec 列表新增一项门禁**:`DensityResponseMatchesFdOfConvergedDensity`
  (HeH⁺,∂P/∂x 解析 vs 收敛密度的 5 点 FD)——约定无关地钉 CPHF 符号与
  因子,置于任何 Hessian 装配之前。
- **回归中 vib 消费端的接线**:`MassWeighting`/`HessianDiagonalizer` 以
  公开组件身份手工 launch(finalize 步 = 6N);`ws.r0`(平衡几何)需由
  调用方填——FD 生产线由 `ReferenceGeometry` 填,解析生产线自填。

## Testing strategy

- **快乐路径(单测,秒级,微型体系)**:HeH⁺/H₂/STO-3G——导数积分逐项 FD、CPHF 迭代 vs 稠密、σ-向量复用一致性、微型解析-vs-FD Hessian、对称与平动和规则。禁止在 tests/ 放 H₂O/6-31G* 全套收敛。
- **边界**:n_virt = 0(He/STO-3G)调 hessian() throw(no-fallback);CPHF 不收敛在预算内 throw 带迭代数;UHF 无 hessian 成员(编译面不存在,越界即编译错)。
- **领域验证(regressions/,唯一 e2e 之家)**:`reg_h2o_rhf_freq` 走公开 API(`cpu::scf::RHF` + `Pople631Gs`),先过第一道闸(解析 vs 自家 FD ≤ 1 cm⁻¹,双方均在库内),再对 REFERENCE.md 钉定的 PySCF 值(外部一次运行,数值入库)≤ 1 cm⁻¹;≤ 60 s。
- **架构门禁**:`no_factory_scrub.sh`、Doxygen 约定、既有 `test_scf_solver`/梯度测试全绿证明默认路径零扰动。

## Out of scope

- UHF/RKS/UKS Hessian(XC 二阶核、开壳层 CPHF 属后续环);电场微扰 CPHF(极化率)与 TDHF(14 环)。
- 链环 12 DF/RI 在 CPHF 中的实际接线(仅备注可选消费);CUDA 后端。
- 新 driver 或 `vib::Hessian` 的任何改动——FD 生产线原样保留;IR 强度(偶极导数)。
- 遗留 `SCFSolver`/`SCFConfig` 债务面的任何扩展。
