---
title: "HF capability gaps 14 — TDHF/RPA 与 TDA 激发能(A/B kernel 的第四种配置)"
slug: hf-capability-gaps-14-tdhf
status: code-complete
chain: hf-capability-gaps
link: 14/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# TDHF/RPA 与 TDA 激发能(A/B kernel 的第四种配置)

## Summary
在链环 08 的轨道 Hessian A/B kernel 之上落地激发态:TDHF/RPA 本征问题 [[A,B],[−B,−A]](X;Y)=ω(X;Y) 经约化对称形式 (A+B)(A−B) 求解(matrix-free σ-向量直接复用 `internal::OrbitalHessian` 的 `sigma_±`),TDA 丢弃 B 得 Hermitian A X = ωX(HF 参考下即 CIS),Davidson 求最低若干根。公共薄表面为方法成员 `rhf.tdhf(n_states, channel)` / `rhf.tda(n_states, channel)`(singlet 与 triplet 两通道),返回激发能 [eV]、跃迁偶极 [a.u.](经链环 03 的多极积分内核)与振子强度。**陷阱内建为前置检查**:参考态 triplet 不稳定时完整 RPA 恰好给出虚 ω——`tdhf()` 之前必跑链环 08 的稳定性诊断,参考不稳定则 throw(附诊断信息),TDA 在同一参考上保持实数。用户可见结果:甲醛 HF/6-31G* 的 n→π* 激发能与外部钉定的 PySCF 逐根吻合。

## Domain basis
- RPA/TDHF:[[A,B],[−B,−A]](X;Y)=ω(X;Y);实轨道下经 Hermitian 约化 (A−B)^{1/2}(A+B)(A−B)^{1/2} Z = ω²Z 求解(小体系稠密;大体系以 σ_± = (A±B)v 的 matrix-free 乘积 + Davidson 求 ω² 最低根)。该约化仅在 (A−B)、(A+B) 正定时良定——正是链环 08 稳定性条件;³(A+B) 出现负本征值 ⇒ 三重态通道 ω² < 0 ⇒ 虚 ω(triplet instability)。Szabo & Ostlund §7;Dreuw & Head-Gordon, Chem. Rev. 105, 4009 (2005), DOI 10.1021/cr0505627。
- TDA:置 B=0 ⇒ A X = ωX(Hermitian;HF 参考 = CIS),不稳定参考上仍为实谱。Hirata & Head-Gordon, Chem. Phys. Lett. 314, 291 (1999), DOI 10.1016/S0009-2614(99)01149-5。
- 闭壳层分块沿用链环 08:¹A/¹B(singlet)、³A/³B(triplet);对每个最低根有 ω_RPA ≤ ω_TDA(变分性质,编码为测试)。
- 跃迁偶极(长度规范):μ_0n = √2 Σ_ia (X+Y)_ia ⟨φ_i|r|φ_a⟩(singlet;triplet 自旋禁阻 ⇒ μ=0, f=0);AO 偶极矩阵来自链环 03 的移位中心 OS 内核,occ-virt MO 变换取自 `SlaterDeterminant` 的 C(外部依赖 scf-slater-determinant-03-unbag)。振子强度 f = (2/3)·ω[a.u.]·|μ_0n|²[a.u.]。
- 单位:内部 Hartree/a.u.;结果报告 eV(换算 `core/units.h`)与 a.u. 偶极。
- Oracle:甲醛 HF/6-31G* 的 n→π*(TDA=CIS 与 TDHF)激发能 vs 外部运行钉定的 PySCF `tdscf.TDA`/`tdscf.TDHF` 至 1e-4 eV;π→π* 亮态振子强度交叉核对。

## Design
- `cpu::scf::internal::TDHFPack`(新,internal 头,mp2_pack 同型):构造函数接 `(const OrbitalHessian&, Channel)`(构造接线,无 `build_*` 自由函数,受 `no_factory_scrub.sh` 门禁)。成员:`tda(n_states)`——A 上 Davidson 最低根(小 n_ov 走 `cpu/linalg` 稠密回退),返回 {ω, X};`rpa(n_states)`——约化对称形式,matrix-free σ_+·σ_− 复合乘积 + Davidson 求 ω²,回代得归一化 (X,Y)(X^TX−Y^TY=1);ω² < 0(数值上 < −阈)则 throw(no-fallback)。全部为私有工作区,不上 Frame。
- `cpu::scf::ExcitedStatesResult`(公共,新头 `tdhf.h`,mp2.h 同型的结果-only 头):`{std::vector<double> energy_ev; xt::xtensor<double,2> transition_dipole_au; std::vector<double> oscillator_strength; Channel channel; bool tda;}`。
- `RHF::tda(n_states, channel)` / `RHF::tdhf(n_states, channel)`(方法成员,声明于 `rhf.h`,与 `stability()` 并列):未 run 先调 throw;`tdhf()` 先跑 `stability()`,相应通道不稳定即 throw(消息含最低本征值与"改用 tda() 或先沿模式重收敛"指引);由 determinant + 私有 shell 工作区重建稠密 ERI → `OrbitalHessian` → `TDHFPack`;singlet 通道再经链环 03 偶极内核装配跃迁偶极与 f,triplet 通道置零。
- 生命周期:每次调用重建(几何可能已动);不新增公共中间类型,不扩展 `SCFSolver`/`IntegralPack` 债务面。

## Files to create or modify
- include/atomiverse/cpu/scf/internal/tdhf_pack.h (new) — `TDHFPack`
- src/cpu/scf/tdhf.cpp (new) — pack 实现 + 方法薄表面实现
- include/atomiverse/cpu/scf/tdhf.h (new) — `ExcitedStatesResult`(结果-only 公共头)
- include/atomiverse/cpu/scf/rhf.h — 声明 `RHF::tda()` / `RHF::tdhf()`
- src/cpu/scf/rhf.cpp — 方法入口接线(稳定性前置检查)
- CMakeLists.txt — HF 源列表追加 tdhf.cpp
- tests/cpu/scf/test_tdhf_pack.cpp (new)
- tests/cpu/scf/test_tdhf_method.cpp (new)
- tests/CMakeLists.txt — 注册两个单测
- regressions/scf/ch2o_tdhf/reg_ch2o_tdhf.cpp (new)
- regressions/scf/ch2o_tdhf/REFERENCE.md (new)
- regressions/CMakeLists.txt — `atv_add_regression(scf ch2o_tdhf)`
- docs/tutorial/ch20-hf-properties-environment.md — "激发态:TDHF/RPA 与 TDA" 节(骨架由链环 01/03 创建)

## Tasks
- [x] Write failing tests for TDHFPack (tests/cpu/scf/test_tdhf_pack.cpp,注册 tests/CMakeLists.txt):H₂/STO-3G 上 TDA 稠密根 == A 的本征值;约化形式 RPA 根与完整 2n_ov×2n_ov [[A,B],[−B,−A]] 非对称稠密本征值一致到 1e-10;每个最低根 ω_RPA ≤ ω_TDA;Davidson == 稠密到 1e-8;构造 triplet-不稳定 kernel 时 `rpa()` throw 而 `tda()` 实数
- [x] Implement TDHFPack in include/atomiverse/cpu/scf/internal/tdhf_pack.h + src/cpu/scf/tdhf.cpp(matrix-free σ 复用 OrbitalHessian;CMakeLists.txt HF 源列表;Doxygen 带单位)
- [x] Write failing tests for RHF::tda()/tdhf() 薄表面 (tests/cpu/scf/test_tdhf_method.cpp,注册 tests/CMakeLists.txt):未 run 先调 throw;拉伸 H₂(R=1.5 Å,STO-3G,链环 08 判定 external 不稳定)上 `tdhf(…, triplet)` throw 且消息含稳定性诊断、`tda()` 返回实激发能;singlet 跃迁偶极非零而 triplet 通道 μ=0、f=0;f 与 (2/3)ω|μ|² 恒等式到 1e-12;energy_ev 单位换算正确
- [x] Implement ExcitedStatesResult + RHF::tda()/tdhf() in include/atomiverse/cpu/scf/tdhf.h、include/atomiverse/cpu/scf/rhf.h、src/cpu/scf/rhf.cpp(稳定性前置检查;跃迁偶极经链环 03 偶极内核;Doxygen 带单位)
- [x] Verify against 甲醛 oracle:regressions/scf/ch2o_tdhf(reg cpp 为库外消费者 + REFERENCE.md 钉外部运行 PySCF tdscf.TDA/tdscf.TDHF 各前 3 个 singlet 根与 n→π* 归属、PySCF 版本与输入,注册 regressions/CMakeLists.txt,≤ 60 s)
- [x] Extend docs/tutorial/ch20-hf-properties-environment.md:"激发态:TDHF/RPA 与 TDA" 节——响应叙事(极化率的极点、A/B kernel 的第四次生命)、TDA-vs-RPA 的诚实对比(triplet instability 回链 ch03 稳定性节)、甲醛上手 + molvis 轨道/跃迁可视化、章末 PySCF 激发能对照表(标量入表,禁止条形图)
- [x] Run full check + test suite(含 no-factory / stage-typing scrub 与 regressions 目标)

## 实现审计注记(2026-07-21)

- **方法薄表面落在 tdhf.cpp,rhf.cpp 未动**:文件清单原写 rhf.cpp 承载
  "方法入口接线",实际 `RHF::tda()/tdhf()` 连同稳定性前置检查一起实现在
  src/cpu/scf/tdhf.cpp(与 TDHFPack 同 TU;spec 同页 "tdhf.cpp — pack 实现
  + 方法薄表面实现" 即此含义),rhf.h 只加声明。
- **公共通道枚举**:`internal::Channel` 不可出现在公共头,故 tdhf.h 新增
  公共 `ExcitationChannel{singlet, triplet}`,tdhf.cpp 内映射到 internal。
- **RPA matrix-free 路线**:采用 spec 建议里的 Stratmann-Scuseria-Frisch
  约化子空间方案(JCP 109, 8218 (1998)):单一正交子空间 V,σ± 投影出
  P/Q,(A−B) 仅在 m 维子空间内本征分解,Q^{1/2}PQ^{1/2}z=ω²z,残差
  r₁=(A+B)(X+Y)−ω(X−Y)、r₂=(A−B)(X−Y)−ω(X+Y) 双向扩展;确定性种子
  (diagonal_plus 最小元坐标向量)。
- **单测合成 kernel 的 ERI 缩放 0.03**(链环 08 原型为 0.1):0.1 缩放的
  合成 kernel 本身就是 RPA-不稳定的(完整超矩阵出现复本征值,约化形式
  正确地 throw)——RPA 一致性测试需要一个双通道稳定的 kernel,故加了
  scale 参数;0.1 版行为反而被 TripletUnstable 测试覆盖(经拉伸 H₂)。
- **PySCF 钉定命令用 nstates=8, conv_tol=1e-10**:nstates=3 且
  conv_tol=1e-12 时 PySCF 自身的 TDHF Davidson 会跳过 9.71 eV 根(其
  求解器 root-homing 伪影);前 3 根数值在收敛成功的各设置间 <1e-12 Ha
  一致,REFERENCE.md 已记录。
- ch20 §20.4 教程内容已写入(响应叙事/内核四配置/TDA-vs-RPA 与 triplet-instability 回链 ch03/甲醛上手 + molvis/章末 PySCF 对照 8 行);ac-007 按验收流程保持 pending 等 human review。

## Testing strategy
- **happy path(单元,秒级)**:H₂/STO-3G(n_ov=1)与合成小 kernel 上,TDA/RPA 稠密与 matrix-free/Davidson 全链一致;约化形式 vs 完整非对称 RPA 矩阵;f 恒等式与单位换算。tests/ 内不放甲醛级 e2e。
- **边界(即陷阱)**:triplet-不稳定参考——`tdhf()` throw(前置稳定性检查)、`tda()` 保持实数;未 run 先调 throw;n_states 超出 n_ov 维度 throw(no-fallback);triplet 通道 f≡0。
- **领域验证(regressions/,唯一 e2e 之家)**:甲醛 HF/6-31G*——TDA 与 TDHF 各根 vs REFERENCE.md 钉定 PySCF 值 1e-4 eV;每根 ω_RPA ≤ ω_TDA;π→π* 亮态振子强度 vs PySCF 1e-3。PySCF 仅为外部 oracle,不成为依赖。
- **架构门禁**:`no_factory_scrub.sh`、Doxygen 约定随 full suite 生效。

## Out of scope
- TDDFT(RKS/UKS 核的 A/B 扩展)、UHF 参考的 TDHF、自旋翻转 TDA;解析激发态梯度/非绝热耦合。
- 速度规范跃迁偶极、双光子性质;core 激发窗口化(CVS)。
- 沿不稳定模式自动重收敛(链环 08 已交付 rotated_C,本环节只 throw + 指引)。
- 遗留 `SCFSolver`/`SCFConfig`/`IntegralPack` 债务面的任何扩展;CUDA 后端。
