---
slug: hf-capability-gaps-09-soscf
criteria:
  - id: ac-001
    summary: Stepper 单元件全过:幺正性/梯度/信赖域
    type: code
    pass_when: |
      tests/cpu/scf/test_soscf.cpp 注册进 ctest 并通过:C·exp(κ) 保
      C^T S C = I(≤1e-12);g_ai 与微型体系旋转能量有限差分一致
      (≤1e-6 Ha);信赖域把 ‖κ‖ 截到 soscf_trust;运行秒级。
    status: verified
    last_checked: 2026-07-21
    note: |
      test_soscf:13 个用例全绿,总耗时 ~50 ms。exp(κ) 正交性
      |UᵀU−I|max ≤ 1e-13;H₂/6-31G 上 C·exp(κ) 后 |CᵀSC−I|max ≤ 1e-12;
      g = 4F_ai(RHF 真导数约定)对旋转能量中心差分(h=1e-5)逐分量
      ≤ 1e-6 Ha(3 个 virt 分量全查);trust=1e-3 时 last_step_norm()
      == 1e-3(±1e-15)且旋转后仍 S-正交;H₂/STO-3G 对称驻点上
      step() 逐位不动 C。非反对称生成元/非方阵抛
      std::invalid_argument。
  - id: ac-002
    summary: CIAH 矩阵自由解 = 稠密增广矩阵参考解
    type: code
    pass_when: |
      同一测试内,小维度问题上 SOSCFStepper 的 (λ, κ) 与测试里显式构造的
      增广矩阵 [[0,g^T],[g,H]] 稠密本征解一致(κ 方向余弦 ≥ 1-1e-8);
      生产代码路径中 H 从不显式构造(σ-vector 调 08 环内核,代码审查确认)。
    status: verified
    last_checked: 2026-07-21
    note: |
      SoscfCiah.MatchesDenseAugmentedEigensolve:H₂/6-31G(n_rot=3,增广
      4×4)λ 一致 ≤1e-10、全向量方向余弦 ≥ 1−1e-8;稠密 H=4·dense_plus
      只在测试内构造。代码审查:src/cpu/scf/internal/soscf.cpp 生产路径
      RHF 只经 OrbitalHessian::sigma_plus/diagonal_plus(×4),UHF σ 由
      非限制性 MO-ERI occ-virt 块矩阵自由收缩(08 内核 RHF-only,审计
      注记 1);grep 确认 soscf.{h,cpp} 无 dense_plus/dense_minus。
  - id: ac-003
    summary: 升级成员接线,默认关断路径逐位不变
    type: code
    pass_when: |
      RHF/UHF 经继承获得 soscf/soscf_start/soscf_max_micro/soscf_trust
      成员(默认 false/0.25/12/0.4,Doxygen 带单位);soscf=false 时
      H₂/STO-3G 微型 SCF 能量与迭代轨迹和本环之前逐位相同
      (test_soscf 断言),既有 scf 单元测试全绿。
    status: verified
    last_checked: 2026-07-21
    note: |
      四成员落在 MeanFieldMethod<D>(mean_field_method.h,Doxygen 带
      单位与量纲),RHF/UHF/RKS/UKS 经 CRTP 继承;soscf=false 时
      MixStage/DiagonalizeStage 的新支路一律短路(stepper 指针为空、
      latch 恒 false),结构上逐位不变。
      SoscfDefaultOff.MicroH2Sto3gTrajectoryUnchanged:两次运行能量
      memcmp 逐位相等、迭代数相等,E = −1.11675930740 Ha 钉住(≤1e-9,
      与本环之前的默认路径一致)。既有 test_cpu_scf_solver /
      test_cpu_scf_uhf_solver / test_cpu_scf_accelerators /
      test_cpu_scf_stages 全绿;9 个既有 scf 回归
      (h2o_rhf_sto3g/def2svp_sph/dipole/charges/n2_ionization/
      diffuse_anion/ecp_heavy_atom/h2o_scf_convergence/h2_rhf_stability)
      全 PASS —— e2e 级默认路径零扰动证据。RKS 上开 soscf 抛
      invalid_argument(SoscfDefaultOff.SoscfOnKohnShamThrows)。
  - id: ac-004
    summary: 难例:CDIIS 振荡而 SOSCF 收敛,能量对 PySCF newton
    type: scientific
    pass_when: |
      reg_soscf_hard_case 中同一难例、同 max_iter 预算:CDIIS-alone 不收敛
      (断言 converged==false 或能量振荡),soscf=true 收敛且终能量与源码
      注释钉住的 PySCF mf.newton() 值(注明版本与命令)差 ≤ 1e-8 Ha;
      退出码 0,墙钟 ≤ 60 s。
    status: verified
    last_checked: 2026-07-21
    note: |
      难例 = 2.6× 拉伸 H₂O/def2-SVP UHF、Hcore 初猜、预算 60(07 环难例,
      spec 明示可沿用;拉伸 N₂ RHF 在引擎 CDIIS 下 13 步即收敛,无失速
      对比,见审计注记 4)。实测:CDIIS-alone 60 步 converged==false
      (在不稳定对称解 ~−75.357 附近振荡);soscf=true 同预算 13 步收敛,
      E = −75.71896632262 Ha,vs PySCF 2.14.0 mf.newton() 钉值
      −75.71896632262(|ΔE| < 5e-12 ≤ 1e-8),⟨S²⟩ = 1.9710 = 钉值
      (碎对称解镜像简并,⟨S²⟩+E 钉解身份)。命令逐字在 REFERENCE.md。
      退出码 0,墙钟 1.7 s ≤ 60 s。
  - id: ac-005
    summary: 迭代数与 PySCF newton 钉值可比
    type: scientific
    pass_when: |
      同一 regression 断言 SOSCF 宏迭代数 ≤ 2× 源码注释里钉住的
      PySCF newton 宏迭代数(RHF 与 UHF 难例各一行钉值)。
    status: verified
    last_checked: 2026-07-21
    note: |
      Newton 相宏迭代数经新增只读观测器 soscf_iterations() 断言(审计
      注记 5)。RHF 行:N₂ R=2.5 Å/6-31G Hcore——引擎 Newton 相 4 步
      ≤ 2×4(PySCF newton(hcore) 钉值 macro=4),E 差 2e-11,HOMO 差
      <1e-8(−0.30409370 vs −0.304093695;多解体系以 HOMO 钉盆),
      且总迭代 6 < CDIIS 13。UHF 行:引擎 Newton 相 9 步 ≤ 2×8
      (升级等价种子钉值:混合 Hcore 初猜+6 步 DIIS 预热后 mf.newton()
      = 8 宏;更冷/更浅种子下 PySCF newton 到不了碎对称极小——种子
      敏感性全记录于 REFERENCE.md;稳定性种子 4 宏一并记录)。
  - id: ac-006
    summary: 教程 ch18 二阶方法半章落地
    type: docs
    pass_when: |
      docs/tutorial/ch18-scf-techniques.md 含 "二阶方法" 一节:轨道旋转
      Newton-Raphson 推导、二次收敛论证、DIIS-早期/SOSCF-收尾配方叙事衔接
      07 环上半章;molplot DIIS vs SOSCF 收敛曲线;章末 PySCF 对照表行
      (迭代数 + 终能量,标量只进表)。
    status: pending
    last_checked: 2026-07-21
    note: |
      §18.4 已成文(18.4.1 轨道旋转与二阶展开、18.4.2 CIAH/σ-vector/
      信赖域、18.4.3 升级配方与两难例数据),molplot CDIIS vs
      DIIS→SOSCF 曲线用引擎实测逐迭代能量,章末对照表加 6 行标量
      (E/⟨S²⟩/Newton 宏迭代数)。待人工阅读 + molplot 产物确认后置
      verified。
  - id: ac-007
    summary: 全量构建 + 单元测试套件绿
    type: runtime
    pass_when: |
      -DATV_BUILD_HF=ON 配置下 cmake --build 无错误,ctest 全绿且 tests/
      仍为秒级单元层(难例回归只挂 regressions target,不进 ctest)。
    status: verified
    last_checked: 2026-07-21
    note: |
      全量 cmake --build 退出码 0。ctest 200 项:195 过,5 失败 ——
      恰为基线既有的 5 个 MPI 失败(test_mpi_halo_exchange_n{1,2,8},
      test_mpi_coul_energy_bruteforce_n{1,8}),与本环无关且与开工前
      基线逐项一致。test_soscf 在 ctest 内 <1 s;reg_soscf_hard_case
      只在 regressions target(1.7 s),未注册 ctest。
---

# Acceptance criteria

ac-001/002/003 由 `/mol:impl` 跑单元测试 + 代码审查验证(ac-002 的"H 不显式
构造"看 internal/soscf.h 只经 08 环 σ-vector 接口触 H);ac-004/005 经
regressions 目标验证,oracle 在实现时用外部 PySCF(mf.newton())钉死并写进
回归源码注释;ac-006 人工阅读 + molplot 产物存在;ac-007 为收尾门。
