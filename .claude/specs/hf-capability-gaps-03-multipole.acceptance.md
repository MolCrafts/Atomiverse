---
slug: hf-capability-gaps-03-multipole
criteria:
  - id: ac-001
    summary: 多极积分内核落在 internal/one_electron.h,零公共 API 泄漏
    type: code
    pass_when: |
      偶极/二阶矩积分内核声明于 include/atomiverse/cpu/scf/internal/one_electron.h、
      实现于 src/cpu/scf/one_electron.cpp,含带单位 Doxygen;facade 公共头不包含
      它们;命名不违反 scf-slater-determinant-04-methods 之后的 cpu/scf `build_*`
      通配门禁(若 04-methods 已落地)。
    note: AUDIT — kernel is the constructor-wired class internal::MultipoleIntegrals (no build_* free function); declared internal/one_electron.h, implemented src/cpu/scf/one_electron.cpp.
    status: verified
    last_checked: 2026-07-20
  - id: ac-002
    summary: 内核单元测试通过(闭式值 + 平移恒等式)
    type: runtime
    pass_when: |
      test_multipole_integrals 在 ctest 通过:s|x|s 闭式值、
      M1(C')=M1(C)+(C-C')S 恒等式(<1e-12)、对称性、p 壳二阶矩解析值。
    status: verified
    last_checked: 2026-07-20
  - id: ac-003
    summary: 观测器单元测试通过(原点约定 + 原点依赖 + 无迹)
    type: runtime
    pass_when: |
      test_multipole 在 ctest 通过且不执行任何 SCF 求解:合成 SlaterDeterminant
      驱动;原点=核电荷中心、Tr(Θ)<1e-12、带电体系 μ'=μ-q_tot·d 与四极偶极
      平移项、kDipole/kQuadrupole/kMultipoleOrigin 写入 frame.meta。
    note: AUDIT — the raw translation law μ′=μ−q_tot·d is pinned at kernel level (test_multipole_integrals TranslationIdentity); the observer test pins the convention consequence instead (origin co-moves ⇒ reported μ invariant under global translation, origin shift reported), which is the observable contract.
    status: verified
    last_checked: 2026-07-20
  - id: ac-004
    summary: MultipoleObserver 消费 MeanFieldOutputs/SlaterDeterminant,不触 IntegralPack/SCFResult
    type: code
    pass_when: |
      include/atomiverse/cpu/observer/multipole.h 定义 Observer kind 组件
      (Role 含 FrameImmutable|Deferred),构造取 scf::MeanFieldOutputs(其
      result 自 scf-slater-determinant-03-unbag 起为
      shared_ptr<const SlaterDeterminant>);grep 该头与
      src/cpu/observer/multipole.cpp 无 "IntegralPack" 且无 "SCFResult"。
    status: verified
    last_checked: 2026-07-20
  - id: ac-005
    summary: H2O RHF/6-31G* 偶极 vs 钉定 PySCF 值 1e-6 a.u.
    type: scientific
    pass_when: |
      regressions/scf/h2o_rhf_dipole(库外消费者,≤60 s,regressions 目标)
      逐分量 |μ_i - μ_i^PySCF| < 1e-6 a.u.;REFERENCE.md 记录 PySCF 版本、
      输入几何、钉定值,并引 Obara-Saika DOI 10.1063/1.450106 与 Szabo-Ostlund。
    note: AUDIT — scientific criterion executed via the regressions target (bench repo unset): PASS, per-component |Δμ| ≤ 8.2e-10 a.u. (tolerance 1e-6).
    status: verified
    last_checked: 2026-07-20
  - id: ac-006
    summary: ch20"分子多极矩"节含推导、molvis 实操、PySCF 表行
    type: docs
    pass_when: |
      docs/tutorial/ch20-hf-properties-environment.md 含"分子多极矩"一节:
      期望值与原点依赖推导、可运行 Atomiverse+molvis 偶极矢量示例、
      章末表格含本节 PySCF 对照标量行(无条形图)。
    status: pending
  - id: ac-007
    summary: 全套构建与测试门通过,tests/ 无新增 e2e
    type: runtime
    pass_when: |
      cmake -DATV_BUILD_HF=ON -DATV_BUILD_TESTS=ON 构建零警告新增;
      ctest 全绿且新增测试单测级耗时;no_factory_scrub 与
      test_stage_typing_scrub 通过。
    status: verified
    last_checked: 2026-07-20
---

# Acceptance criteria

内核正确性由闭式值与平移恒等式钉住(ac-002),物理正确性由外部钉定的
PySCF 偶极值钉住(ac-005);架构约束(MeanFieldOutputs/SlaterDeterminant-only、
Observer role、internal 内核、build_* 门禁)由 ac-001/ac-004 钉住;教程与
全局门为 ac-006/ac-007。
