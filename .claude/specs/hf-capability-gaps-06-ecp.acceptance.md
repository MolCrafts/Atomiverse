---
slug: hf-capability-gaps-06-ecp
criteria:
  - id: ac-001
    summary: ECP 表结构 + 查表类单元测试通过(钉 PySCF)
    type: code
    pass_when: |
      tests/cpu/scf/test_ecp.cpp 注册进 ctest 并通过:ecp_def2() 中 I/Au 的
      n_core、通道数与首末指数等于测试内钉住的 PySCF def2-ECP 值;
      EcpPotential 对表外元素与未知名字抛含支持项的 std::runtime_error;
      测试不跑任何完整 SCF,运行秒级。
    status: verified
    last_checked: 2026-07-21
    note: |
      test_ecp 全绿(8 tests, 1.4 s 总)。I:n_core=28、4 通道(s/p/d + 局域
      末位 l=3)、s 道首项 (40.015835, 49.994293)、末项 (4.884315, 0.320804)、
      局域首项 (19.458609, −21.84204);Au:n_core=60、局域 {(4.78982,
      30.4900889), (2.39491, 5.17107381)} — 均逐位对 PySCF 2.14.0
      gto.basis.load_ecp("def2-svp", El)。EcpPotential{O, "def2-ECP"} 与未知
      名字均抛含 "def2-ECP"/"Supported" 的 runtime_error。
  - id: ac-002
    summary: U 矩阵元(含高 l 投影道)钉 PySCF ECPscalar
    type: code
    pass_when: |
      EcpIntegrals 产出的若干 ⟨μ|U|ν⟩ 矩阵元(至少含一个 f 投影道高 l 项)
      与测试内钉住的 PySCF mol.intor('ECPscalar') 值差 ≤ 1e-10 Ha;
      实现为类成员,tests/architecture 与 04-methods 通配门下
      cpu/scf 无新增 build_* 自由函数。
    status: verified
    last_checked: 2026-07-21
    note: |
      两组探针共 27 个矩阵元钉死,断言 1e-10,实测 ≤3e-13(probe A:真实
      def2-ECP@I,15 元素含 f AO/d 投影道;probe B:合成 f(l=3)投影道 ECP,
      12 元素含 f_zzz|f_zzz = −0.06850139838965047)。注意:PySCF def2-ECP
      数据本身无 f 投影道(最高 d)— 高 l 钉值经 PySCF 用户自定义 ECP 外部
      钉定(见 spec 审计注记 1)。no_factory_scrub / cpu-scf build_* 通配门
      全绿(ctest architecture 层)。
  - id: ac-003
    summary: 价电子计数与 Z_val 接线正确,S/T/ERI 不动
    type: code
    pass_when: |
      单元测试断言:I₂ 的 IntegralPack(ecp="def2-ECP") n_electrons == 50、
      V_nuc 按 Z_val=25 计;同一几何 ecp 为空时 S/T 矩阵与现状逐元素相等;
      轻元素(全在表外)分子在 ecp="def2-ECP" 下 Hcore 与全电子路径一致。
    status: verified
    last_checked: 2026-07-21
    note: |
      EcpIntegralPack.I2ValenceWiring:n_electrons 106→50、n_occ=25、
      V_nuc = 625/R = 124.05692304013505(对 PySCF energy_nuc ≤1e-8,
      对解析式 ≤1e-12);S/T 与全电子路径 max|Δ| == 0.0(逐位),V 差 O(1)。
      LightMoleculeUnchangedAndUnknownNameThrows:H₂O + ecp="def2-ECP" 的
      Hcore/V_nuc/n_electrons 与空 ecp 逐位一致,has_ecp=false;
      未知名 "no-such" 抛 runtime_error。
  - id: ac-004
    summary: ECP↔x2c 逐元素互斥 THROW 门生效
    type: code
    pass_when: |
      设置 ecp="def2-ECP" 且 x2c=true 时,对含 ECP 元素(I)的 Frame 调 run()
      抛含元素名的 std::runtime_error;对全电子分子(H2O)同设置不抛
      (单元测试两向断言)。
    status: verified
    last_checked: 2026-07-21
    note: |
      EcpX2cGate.ThrowsForEcpElementPassesAllElectron:I₂ + x2c + ecp 于
      drive_scf_ 序幕抛 runtime_error,消息含 "I (Z=53)" 与 "x2c";
      全电子分子同设置 EXPECT_NO_THROW(门逐元素)。
  - id: ac-005
    summary: 生成器字节可复现(ecp_def2.h + def2_svp.h 重元素行)
    type: code
    pass_when: |
      按两表头 Regenerate 注释中的命令重跑 tools/gen_basis_data.py 后
      `git diff --stat include/atomiverse/cpu/scf/basis_data/` 为空,
      且 def2_svp.h 的 Z=37–86 行非空(test_ecp 抽查 I/Au 行)。
    status: verified
    last_checked: 2026-07-21
    note: |
      `gen_basis_data.py --check <path> --ecp def2-svp` 与
      `--check <path> def2-svp` 均报 byte-identical(生成两次,diff 空)。
      def2_svp.h 覆盖 Z=37–57 + 72–86(镧系 Ce–Lu 不属 def2 家族,PySCF 无
      行,保持空 — 与 ecp_def2.h 一致);test_ecp 钉 I 行(10 壳,首指数
      445.90489176)与 Au 行(12 壳,f 指数 0.72482)。
  - id: ac-006
    summary: I₂ 与 AuH RHF 能量对账 PySCF oracle
    type: scientific
    pass_when: |
      reg_ecp_heavy_atom 中 RHF/def2-SVP+def2-ECP 的 I₂ 与 AuH 总能量与
      源码注释里钉住的 PySCF(ecp='def2-svp',注明版本与命令)值差
      ≤ 1e-6 Ha,退出码 0,墙钟 ≤ 60 s,仅经公开 API(RHF 成员)驱动。
    status: verified
    last_checked: 2026-07-21
    note: |
      reg_ecp_heavy_atom PASS,墙钟 3.9 s:I₂ got=−593.316181880646 vs
      pin −593.3161818794323(|Δ|=1.2e-9);AuH got=−135.342567007924 vs
      pin −135.34256700791678(|Δ|=7.2e-12)。PySCF 2.14.0,conv_tol=1e-12,
      命令与几何在 REFERENCE.md。前置修复:operators.cpp f 壳 ao_pairs 栈
      溢出(spec 审计注记 2)— 修复前 AuH 偏 7.7e-6。旁证:既有
      reg_h2o_rhf_def2svp_sph(4.4e-13)与 reg_diffuse_anion 修复后仍 PASS。
  - id: ac-007
    summary: 教程第 4 章 "有效核势" 一节落地
    type: docs
    pass_when: |
      docs/tutorial/ch04-basis-sets.md 含 "有效核势" 一节:价电子近似动机、
      半定域形式 U_L + Σ(U_l−U_L)|lm⟩⟨lm|、价电子计数说明、I₂ 上手代码,
      以及章末含 I₂/AuH 行的 Atomiverse-vs-PySCF 标量对照表(无柱状图)。
    status: pending
  - id: ac-008
    summary: 全量构建 + 单元测试套件绿
    type: runtime
    pass_when: |
      -DATV_BUILD_HF=ON 配置下 cmake --build 无错误,ctest 全绿且 tests/
      仍为秒级单元层;回归只挂 regressions target,不进默认 ctest。
    status: verified
    last_checked: 2026-07-21
    note: |
      全量构建无错误;ctest 总 293 s,失败仅既有基线 5 项 MPI
      (test_mpi_halo_exchange_n{1,2,8}, test_mpi_coul_energy_bruteforce_
      n{1,8})— 与开工前基线一致,非本环引入。test_ecp 1.4 s。
      reg_ecp_heavy_atom 只挂 regressions target,未注册 ctest。
---

# Acceptance criteria

ac-001/002/003/004 由 `/mol:impl` 跑 `test_ecp` 验证(PySCF 钉值在实现时
外部运行一次并写进测试源码注释);ac-005 由实现者按 Regenerate 命令重跑
生成器后查 `git diff`;ac-006 经 regressions 目标验证;ac-007 人工阅读;
ac-008 为收尾门。
