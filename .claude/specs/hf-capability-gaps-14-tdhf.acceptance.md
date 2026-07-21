---
slug: hf-capability-gaps-14-tdhf
criteria:
  - id: ac-001
    summary: TDHFPack 约化 RPA 与完整非对称 RPA 及稠密 TDA 一致
    type: code
    pass_when: |
      tests/cpu/scf/test_tdhf_pack.cpp 通过:H₂/STO-3G 上 TDA 根 == A 本征值;
      约化 (A+B)(A−B) 形式的 RPA 根与完整 [[A,B],[−B,−A]] 稠密本征值一致到
      1e-10 Ha;Davidson 最低根 == 稠密到 1e-8;每个最低根 ω_RPA ≤ ω_TDA。
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_tdhf_pack 8/8 通过(~1 ms 总):TdaDenseRootEqualsAEigenvalueH2
      (两通道 ω_TDA == A(0,0) 到 1e-14);RpaReducedFormMatchesFullNonsymmetric
      H2/Synthetic(n=6,no=2,n_ov=8;全部 8 根 vs 测试内独立构造的 16×16
      [[A,B],[−B,−A]] xt::linalg::eig 谱,两通道逐根 ≤1e-10 Ha);
      DavidsonMatchesDense(n_ov=12 合成 kernel,dense_threshold=0 强制
      matrix-free,tda/rpa 各 3 根两通道 ≤1e-8);RpaBelowTdaPerLowestRoot
      (H2 两通道 + 合成 3 根逐根 ω_RPA ≤ ω_TDA)。
  - id: ac-002
    summary: TDHFPack 为构造接线的 internal pack,matrix-free σ 复用 OrbitalHessian
    type: code
    pass_when: |
      include/atomiverse/cpu/scf/internal/tdhf_pack.h 存在;TDHFPack 由构造函数
      接 (OrbitalHessian&, Channel);grep 无新增 build_*/make_* 自由函数,
      no_factory_scrub.sh 通过;公共头不含 IntegralPack/SCFConfig 引用。
    status: verified
    last_checked: 2026-07-21
    note: |
      include/atomiverse/cpu/scf/internal/tdhf_pack.h:TDHFPack ctor
      (const OrbitalHessian&, Channel),成员 dense_threshold(StabilityAnalysis
      同型);σ 全部经 hessian_.sigma_plus/sigma_minus/diagonal_plus,RPA
      matrix-free 为 Stratmann-Scuseria-Frisch 约化子空间((A−B) 仅在子空间
      本征分解)。test_no_factory_scrub Passed(3.7 s);公共头 tdhf.h 无
      IntegralPack/SCFConfig 引用(仅 <vector> + xtensor)。
  - id: ac-003
    summary: triplet-instability 陷阱:tdhf 前置稳定性检查 throw,tda 保持实数
    type: code
    pass_when: |
      tests/cpu/scf/test_tdhf_method.cpp 通过:拉伸 H₂ (R=1.5 Å, STO-3G) 上
      RHF::tdhf(…, triplet) 抛异常且消息含稳定性诊断;同一参考上 RHF::tda()
      返回全实激发能;合成不稳定 kernel 上 TDHFPack::rpa() 亦 throw。
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_tdhf_method.StretchedH2TripletTdhfThrowsWithDiagnosticTdaReal:
      R=1.5 Å H2/STO-3G 收敛 RHF 上 tdhf(1, triplet) 抛 runtime_error,消息含
      "unstable"、"lowest"(最低本征值 −0.209 Ha)与 "tda()" 指引;同一参考
      tda(1, triplet) 返回有限实 energy_ev。test_cpu_scf_tdhf_pack.
      TripletUnstableKernelRpaThrowsTdaReal:闭式拉伸-H2 kernel 上 rpa(1)
      抛 runtime_error(消息含负 ω² [Ha²] 数值)、tda(1) 实数。
  - id: ac-004
    summary: 公共薄表面:eV 激发能、跃迁偶极、振子强度、双通道、未 run throw
    type: code
    pass_when: |
      ExcitedStatesResult 含 energy_ev / transition_dipole_au /
      oscillator_strength / channel;未 run 先调 tda()/tdhf() throw;
      singlet 偶极经链环 03 内核装配、triplet 通道 μ=0 且 f=0;
      f == (2/3)·ω·|μ|² (a.u.) 到 1e-12(单测断言)。
    status: verified
    last_checked: 2026-07-21
    note: |
      tdhf.h:ExcitedStatesResult{energy_ev, transition_dipole_au[n×3],
      oscillator_strength, channel, tda} + 公共 ExcitationChannel。
      CallBeforeRunThrows(tda/tdhf 未 run 抛 runtime_error);
      SingletDipoleNonzeroTripletZero(H2 σg→σu |μ_z|>0.1 a.u.,triplet
      通道 μ≡0、f≡0,tda 与 tdhf 皆然);OscillatorStrengthIdentity
      (f == (2/3)·ω[Ha]·|μ|² ≤1e-12,TDA+TDHF);
      EnergyEvConversionMatchesKernelRoot(energy_ev == ω_Ha·kHartreeInEv
      ≤1e-10 eV,比值 27.211386±1e-4)。偶极经 internal/one_electron.h
      MultipoleIntegrals(链环 03 内核)+ c2s 三明治。
  - id: ac-005
    summary: 甲醛回归:TDA 与 TDHF 激发能 vs 钉定 PySCF 至 1e-4 eV
    type: scientific
    pass_when: |
      regressions/scf/ch2o_tdhf 以库外消费者身份跑 HF/6-31G* 甲醛,TDA 与
      TDHF 各前 3 个 singlet 根与 REFERENCE.md 钉定的 PySCF tdscf 值逐根差
      < 1e-4 eV,n→π* 归属一致;运行 ≤ 60 s;REFERENCE.md 记录 PySCF
      版本/输入并引用 Hirata 1999 与 Dreuw 2005 DOI。
    status: verified
    last_checked: 2026-07-21
    note: |
      reg_ch2o_tdhf PASS,7.7 s(/tmp cwd)。E_RHF=−113.8655751 vs 钉
      −113.865575118441 (≤1e-6 Ha)。TDA roots 4.6689702/9.9504661/10.3001674 eV,
      TDHF roots 4.4999396/9.7075259/9.7558617 eV — 各 vs PySCF 2.14.0 钉
      (Ha×kHartreeInEv) 偏差 ~3e-7 eV ≪ 1e-4 eV。root 1 = n→π*(HOMO n_O→π*,
      两理论 f=0 dark)归属一致;逐根 ω_RPA ≤ ω_TDA。REFERENCE.md 记 PySCF
      版本/几何/verbatim 命令 + Hirata 1999 与 Dreuw 2005 DOI。
  - id: ac-006
    summary: 亮态振子强度与 PySCF 交叉核对
    type: scientific
    pass_when: |
      同一回归中 π→π* 亮态的长度规范振子强度与 REFERENCE.md 钉定 PySCF 值
      之差 < 1e-3(无量纲),且 n→π* 根 f ≈ 0(< 1e-3)。
    status: verified
    last_checked: 2026-07-21
    note: |
      同一回归:π→π* 亮态(root 3,跃迁偶极沿 C=O 轴 z,|μ_z|²>0.99|μ|²)
      f_TDA=0.2165662 vs 钉 0.2165661719、f_TDHF=0.1795989 vs 钉
      0.1795988961(偏差 ~1e-8 ≪ 1e-3);n→π* root 1 f=0.0000000 < 1e-3。
  - id: ac-007
    summary: ch20 教程"激发态:TDHF/RPA 与 TDA"节完整
    type: docs
    pass_when: |
      docs/tutorial/ch20-hf-properties-environment.md 含该节:极化率极点叙事、
      A/B kernel 第四配置、TDA-vs-RPA 与 triplet-instability 回链 ch03、
      甲醛 hands-on(molvis 可视化)、章末 PySCF 激发能标量对照表(无条形图)。
    status: pending
  - id: ac-008
    summary: 全套构建与测试绿灯
    type: runtime
    pass_when: |
      -DATV_BUILD_HF=ON 配置下 cmake --build 无错;ctest 全绿(tests/ 秒级,
      新增两单测已注册);regressions 目标含 ch2o_tdhf 且通过。
    status: verified
    last_checked: 2026-07-21
    note: |
      -DATV_BUILD_HF=ON 全量 cmake --build 无错;ctest 207 项 202 Passed,
      仅剩基线 5 个预存 MPI 失败(test_mpi_halo_exchange_n{1,2,8},
      test_mpi_coul_energy_bruteforce_n{1,8})——与本链环前基线一致。
      新增 test_cpu_scf_tdhf_pack(8 用例 ~1 ms)与 test_cpu_scf_tdhf_method
      (6 用例 ~140 ms)已注册并通过;no_factory / stage-typing scrub 通过;
      15 个 SCF 回归(含新 ch2o_tdhf)全 PASS(0.3–7.9 s each)。
---

# Acceptance criteria

ac-001/002 钉住第四配置的数学与架构形状(约化 RPA 正确性、pack 无工厂);
ac-003 是本链环的领域陷阱(RPA 虚根 ⇔ triplet 不稳定,TDA 免疫);ac-004 钉公共
表面契约;ac-005/006 是甲醛 oracle 的科学验收;ac-007/008 收尾教程与门禁。
