---
title: Orbital Hessian A/B kernel + Seeger-Pople stability analysis
slug: hf-capability-gaps-08-stability
status: code-complete
chain: hf-capability-gaps
link: 08/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# Orbital Hessian A/B kernel + Seeger-Pople stability analysis

## Summary

实现整条链复用次数最高的对象——MO 基轨道 Hessian A/B 核(matrix-free σ-向量形式 σ = (A±B)v,附小体系显式稠密构建),并在其上落地 Seeger-Pople 稳定性分析:internal(RHF→RHF,单重态块)与 external(RHF→UHF,三重态块)通道的最低本征值与不稳定标志,经 `rhf.stability()` 方法成员暴露。external 不稳定时给出沿不稳定模式旋转的破缺对称初猜,衔接既有 `scf_guess.h` 桥入 UHF。同一 kernel 是 09-SOSCF、13-CPHF、14-TDHF 的公共底座:此环节实现一次、经稳定性分析验证。用户可见结果:拉伸 H₂ 的 RHF 被正确判定为 external 不稳定,沿模式重收敛的 UHF 能量低于 RHF。

## Domain basis

自旋轨道形式(实轨道,Hartree):
- A_ai,bj = δ_ab δ_ij (ε_a − ε_i) + ⟨aj||ib⟩, B_ai,bj = ⟨ab||ij⟩。
- 闭壳层空间轨道分块(化学家记号 (pq|rs),来自 `build_eri_dense` 的 AO 张量做 O(N⁵) 四分之一变换):
  - 单重态:¹A = δδ(ε_a−ε_i) + 2(ai|jb) − (ij|ab); ¹B = 2(ai|bj) − (aj|bi)
  - 三重态:³A = δδ(ε_a−ε_i) − (ij|ab); ³B = −(aj|bi)
- 实轨道 RHF 的 Seeger-Pople 分类:internal 稳定 ⇔ ¹(A+B) 最低本征值 ≥ 0;external(triplet)稳定 ⇔ ³(A+B) 最低本征值 ≥ 0。(A−B) 块服务 complex 通道与 13/14 的 RPA 结构,kernel 必须同时提供 σ∓,但本环节只以 A+B 两通道验收。
- **陷阱(必须编码为测试)**:internal 与 external 是不同的块——internal 稳定的 RHF 完全可以 external 不稳定,这正是 Seeger-Pople 分类的内容。
- ε、C、f 一律取自 `SlaterDeterminant`(外部依赖 scf-slater-determinant-03-unbag:`MeanFieldOutputs::result = shared_ptr<const SlaterDeterminant>`)。
- 参考:Seeger & Pople, J. Chem. Phys. 66, 3045 (1977), DOI 10.1063/1.434318;Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory*, ch. 10;Szabo & Ostlund §3.8.7(H₂ 解离的 UHF)。Oracle:拉伸 H₂(R = 1.5 Å,STO-3G)RHF 有负的 external Hessian 本征值;与外部运行的 PySCF `mf.stability()` 钉定本征值比对;沿模式的 UHF 能量低于 RHF。单位:本征值与能量 Hartree,几何 Å。

## Design

- `cpu::scf::internal::OrbitalHessian`(新,internal 头):构造函数接 `(const SlaterDeterminant&, const xt::xtensor<double,4>& eri_ao)` 并一次性完成 occ-virt MO-ERI 变换(构造函数接线、成员计算;`cpu/scf` 禁 `build_*` 自由函数,受 `no_factory_scrub.sh` 门禁)。成员:`sigma_plus(v, ch)` / `sigma_minus(v, ch)`(ch ∈ {singlet, triplet},matrix-free,09/13/14 的迭代求解器直接复用)、`dense_plus(ch)` / `dense_minus(ch)`(n_ov×n_ov 显式构建,测试与小体系回退)、`n_occ()/n_virt()/n_ov()`。AO 维数约定沿用链 02。
- `cpu::scf::internal::StabilityAnalysis`(新):构造接 `const OrbitalHessian&`;`lowest(ch)` 求 (A+B) 通道最低根——n_ov 小时经 `cpu/linalg/eigensolve.h` 稠密路径,否则 matrix-free Davidson 最低根(子空间对角化仍走 linalg)。
- `cpu::scf::StabilityResult`(公共,声明于 `rhf.h`):`{internal_eigenvalue, external_eigenvalue [Ha], internal_stable, external_stable, RealMatrix rotated_C}`——`rotated_C` 为沿 external 模式旋转后的系数(桥接 `scf_guess.h` 的破缺对称机制),用作 UHF 初猜。
- `RHF::stability()`(方法成员,run 之后调用):从 determinant + 私有 shell 工作区重建稠密 ERI,装配上述两类并返回 `StabilityResult`;未 run 先调 throw(no-fallback)。
- 生命周期:全部为方法私有工作区,不上 Frame,不新增公共中间类型。

### 实现审计注记(2026-07-21)

- **ac-002 的措辞与物理**:¹A 对角元的严格值是 Δε + 2(ai|ai) − (ii|aa),
  不是裸的 ε_a − ε_i。单测按"element-wise 精确公式"验证(测试内独立做
  AO→MO 四重循环变换),并单独断言 A(0,0) − (2(ai|ai) − (ii|aa)) == Δε
  到 1e-12——δδ(ε_a−ε_i) 项即以此形式钉定。
- **kernel 附加成员**:`diagonal_plus(ch)`(Davidson 预条件子;09/13/14 的
  迭代求解器同样要用),超出 spec 成员清单但属 sigma 接口的自然配套。
- **稠密 ERI 重建路径**:`IntegralPack` 增公共成员 `eri_working()`
  (工作维稠密 ERI:Cartesian `eri_dense` + 球谐三明治),球谐三明治按
  "第二个调用者才提取"规则从构造函数提为私有静态 `eri_to_spherical`。
  `MeanFieldMethod` 增 **protected** `integral_workspace_()` 只读访问器
  (RHF::stability() 从私有 shell 工作区重建;不新增公共面)。
- **StabilityAnalysis 可测性**:公开 `lowest_dense(ch)` / `lowest_davidson(ch)`
  与 `dense_threshold` 成员(缺省 512),Davidson==稠密 的平价单测据此
  强制走两条路径;`lowest(ch)` 按 n_ov 自动分派。
- **rotated_C → UHF 桥**:走既有公共机制——α 密度由 rotated_C 占据列构成、
  β 用未旋转行列式密度,经 `SCFInitialGuess` 传入 `uhf.run(frame, &guess)`;
  旋转量取 `apply_homo_lumo_mixing` 的缺省 20°(kRotationTheta),
  U = exp(K) 泰勒级数(‖K‖ ≤ 0.35,机器精度收敛)。未发明新公共 API。
- **单测几何**:r_e 用 acceptance 约束的 0.735 Å(prompt 的 0.7414 Å 亦跑了
  oracle:internal +1.1280718048 / external +0.4029165720 Ha,记录于此)。
- **单测无 SCF 循环**:H₂/STO-3G 收敛行列式由对称性闭式构造
  (φ_g/φ_u 由 S₁₂ 定,ε 来自对称密度上的**一次** Fock 构建,该密度即
  自洽解),Davidson 平价用确定性 LCG 合成体系(n_ov = 32)。

## Files to create or modify

- include/atomiverse/cpu/scf/internal/orbital_hessian.h (new)
- src/cpu/scf/orbital_hessian.cpp (new)
- include/atomiverse/cpu/scf/internal/stability.h (new)
- src/cpu/scf/stability.cpp (new)
- include/atomiverse/cpu/scf/rhf.h — `StabilityResult` + `RHF::stability()`
- src/cpu/scf/rhf.cpp — `stability()` 实现
- CMakeLists.txt — HF 源列表追加两个新 .cpp
- tests/cpu/scf/test_orbital_hessian.cpp (new)
- tests/cpu/scf/test_stability.cpp (new)
- tests/CMakeLists.txt — 注册两个单测
- regressions/scf/h2_rhf_stability/reg_h2_rhf_stability.cpp (new)
- regressions/scf/h2_rhf_stability/REFERENCE.md (new)
- regressions/CMakeLists.txt — `atv_add_regression(scf h2_rhf_stability)`
- docs/tutorial/ch03-open-shell.md — 新增 "SCF Stability Analysis(SCF 解的稳定性)" 节

## Tasks

- [x] Write failing tests for OrbitalHessian (tests/cpu/scf/test_orbital_hessian.cpp,注册 tests/CMakeLists.txt):H₂/STO-3G 上 ¹A 对角元 = ε_a−ε_i;随机 v 的 dense_±·v 与 sigma_± 一致到 1e-12;singlet/triplet 块互异
- [x] Implement OrbitalHessian in include/atomiverse/cpu/scf/internal/orbital_hessian.h + src/cpu/scf/orbital_hessian.cpp(CMakeLists.txt HF 源列表;Doxygen 带单位)
- [x] Write failing tests for StabilityAnalysis 与 RHF::stability() (tests/cpu/scf/test_stability.cpp):r_e 处双通道 ≥ 0;R = 1.5 Å 处 internal ≥ 0 而 external < 0(Seeger-Pople 陷阱);Davidson 最低根 == 稠密最低根到 1e-10
- [x] Implement StabilityAnalysis in include/atomiverse/cpu/scf/internal/stability.h + src/cpu/scf/stability.cpp(Davidson + linalg 稠密回退)
- [x] Implement RHF::stability() + StabilityResult in include/atomiverse/cpu/scf/rhf.h / src/cpu/scf/rhf.cpp(rotated_C 破缺对称初猜,桥 scf_guess.h;未 run 先调 throw)
- [x] Verify against 拉伸-H₂ oracle:regressions/scf/h2_rhf_stability(reg cpp + REFERENCE.md 钉 PySCF `mf.stability()` 本征值与文献 UHF 解离行为,注册 regressions/CMakeLists.txt,≤ 60 s)
- [x] Extend docs/tutorial/ch03-open-shell.md:"SCF Stability Analysis(SCF 解的稳定性)" 节——RHF 何时破缺、本征值含义、internal vs external,拉伸-H₂ 上手示例与 RHF vs UHF 能量-R 曲线(molplot),章末 PySCF 最低本征值对照表(标量入表)
- [x] Run full check + test suite

## Testing strategy

- **Happy path(单测,秒级,H₂/STO-3G n=2)**:对角元、dense-vs-sigma 一致性、r_e 处双通道稳定、Davidson==稠密。
- **边界**:未 run 调 stability() throw;singlet 与 triplet 块数值互异;n_virt = 0 体系(He/STO-3G)返回空通道语义明确(throw,no-fallback)。
- **领域验证(regression,唯一 e2e 之家)**:R = 1.5 Å H₂/STO-3G——external 最低本征值 < 0 且与 REFERENCE.md 钉定的 PySCF 值一致(1e-6 Ha);以 rotated_C 种子的 UHF 能量低于 RHF ≥ 1e-4 Ha(Szabo-Ostlund §3.8.7 行为)。禁止在 tests/ 放任何完整 SCF 收敛回归。
- **架构门禁**:`no_factory_scrub.sh` 与 Doxygen 约定随 full suite 生效。

## Out of scope

- SOSCF(09)、CPHF(13)、TDHF/激发能(14)——只消费本 kernel,不在此实现。
- UHF/UKS 出发的稳定性、complex(A−B)通道的验收、DFT 核贡献(RKS/UKS Hessian)。
- direct-SCF/shell-pair 路径的 MO 变换(本环节尺度用 in-core 稠密 ERI 即可);CUDA 后端。
- 遗留 `SCFSolver`/`SCFConfig` 债务面的任何扩展。
