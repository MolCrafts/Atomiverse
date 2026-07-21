---
slug: hf-capability-gaps-05-basis
criteria:
  - id: ac-001
    summary: 新表结构单元测试通过(壳数/AO 数钉 PySCF)
    type: code
    pass_when: |
      tests/cpu/scf/test_basis_tables.cpp 注册进 ctest 并通过:pople_631pgs()/
      aug_ccpvdz() 逐元素壳数、l 序列、AO 计数与首末指数抽查等于测试内钉住的
      PySCF mol._basis 值;测试不跑任何 SCF,运行秒级。
    status: verified
    last_checked: 2026-07-20
    note: ./build/tests/test_basis_tables 6/6 PASS,0.011 s 墙钟;H/O/Na/Cl(/Ar)
      壳数、l 序列、ncart/nsph、首末指数与首系数钉 PySCF 2.14.0
      gto.basis.load;aug-cc-pVDZ 按拆列后结构钉(O 9 壳、Cl 11 壳)。
  - id: ac-002
    summary: 四套既有表补齐 Z=11–18 且 sto3g 不动
    type: code
    pass_when: |
      pople_631g/pople_631gs/def2_svp/def2_tzvp 四表 Z=11..18 行 shells 非空
      (test_basis_tables 断言);git diff 显示 sto3g.h 零改动。
    status: verified
    last_checked: 2026-07-20
    note: ThirdPeriodRowsNonEmpty 断言六表 Z=11..18 非空 + Pople 三表 K/Ca
      非空,PASS;git status 只列四表 modified + 两新表 untracked,
      sto3g.h 不在 diff 中(逐字节未动)。
  - id: ac-003
    summary: 生成器字节可复现
    type: code
    pass_when: |
      按各表头 Regenerate 注释中的命令重跑 tools/gen_basis_data.py 后
      `git diff --stat include/atomiverse/cpu/scf/basis_data/` 为空
      (六个生成头全部逐字节复现)。
    status: verified
    last_checked: 2026-07-20
    note: 生成后对六个头各跑 `gen_basis_data.py --check <header> <name>`
      (venv python,pyscf 2.14.0)全部 "OK: byte-identical";--check 即
      第二次完整重生成 + 逐字节比对,等价于 git diff 为空。
  - id: ac-004
    summary: 枚举/分派/旗标注册完整,不支持 Z 抛错
    type: code
    pass_when: |
      BasisSet::Pople631pGs 与 BasisSet::AugCcPVDZ 可经 ShellList 构造;
      basis_set_name 返回 "6-31+G*"/"aug-cc-pVDZ";02 环旗标登记为
      cartesian-6d/spherical;对表外 Z 构造抛含基组名的 std::runtime_error
      (单元测试断言,无 default 回退分支)。
    status: verified
    last_checked: 2026-07-20
    note: ShellListConstructsForNewSets(Cl⁻:6-31+G* n_basis=23、
      aug-cc-pVDZ n_basis=29 笛卡尔)、NewEnumNamesAndConvention
      (名字 + Cartesian/Spherical)、UnsupportedZThrowsWithBasisName
      (He/6-31+G*、K/aug-cc-pVDZ 均抛含基组名的 runtime_error)全 PASS;
      basis_set_name/angular_convention/ShellList 三个 switch 均逐枚举列举。
  - id: ac-005
    summary: 阴离子/弥散能量对账 PySCF oracle
    type: scientific
    pass_when: |
      reg_diffuse_anion 中 Cl⁻/6-31+G* 与 H2O/aug-cc-pVDZ 的 RHF 总能量与
      源码注释里钉住的 PySCF 值(注明版本与命令)差 ≤ 1e-7 Ha,退出码 0,
      墙钟 ≤ 60 s。
    status: verified
    last_checked: 2026-07-20
    note: Cl⁻/6-31+G* got=-459.539661044387 |diff|=1.7e-13 Ha;
      H2O/aug-cc-pVDZ(球谐) got=-76.041393519988 |diff|=5.5e-13 Ha;
      exit 0,墙钟 1.4 s;钉值 PySCF 2.14.0 conv_tol=1e-12,命令逐字在
      REFERENCE.md 与源码头注释。
  - id: ac-006
    summary: OH⁻ 弥散束缚性翻转复现
    type: scientific
    pass_when: |
      同一 regression 内 OH⁻ 的 ε_HOMO 在 6-31G* 与 6-31+G* 下均与钉住的
      PySCF 同基组值一致(≤1e-6 Ha),且 6-31+G* 值严格更低并与 PySCF
      钉值同号(弥散恢复束缚)。
    status: verified
    last_checked: 2026-07-20
    note: ε_HOMO(6-31G*)=-0.037774390716(|Δ|=4.9e-10)、
      ε_HOMO(6-31+G*)=-0.103579759566(|Δ|=1.4e-8,≤1e-6),严格更低且
      同号(均束缚);实测"非束缚伪像"在 STO-3G(+0.251912,|Δ|=1.7e-11,
      亦已钉入回归)而非 spec 叙事预期的 6-31G* — 见 spec Design
      实现审计注记 1 与 REFERENCE.md 已知偏差节。
  - id: ac-007
    summary: 教程第 4 章弥散/第三周期扩写落地
    type: docs
    pass_when: |
      docs/tutorial/ch04-basis-sets.md 含 "极化、弥散与第三周期" 一节:
      基组命名法解码、OH⁻ 弥散演示、章末含 6-31+G*/aug-cc-pVDZ 行的
      Atomiverse-vs-PySCF 标量对照表,以及 molplot 基组收敛曲线(曲线可图,
      标量只进表)。
    status: pending
  - id: ac-008
    summary: 全量构建 + 单元测试套件绿
    type: runtime
    pass_when: |
      -DATV_BUILD_HF=ON 配置下 cmake --build 无错误,ctest 全绿且 tests/
      仍为秒级单元层(regression 只挂 regressions target,不进 ctest)。
    status: verified
    last_checked: 2026-07-20
    note: 全量 cmake --build 无错误;ctest 仅余 5 个文档化基线失败
      (test_mpi_halo_exchange_n{1,2,8}、test_mpi_coul_energy_bruteforce_n{1,8},
      本环之前即失败、与本环无关),其余全绿;reg_diffuse_anion 只挂
      regressions target,不进 ctest;既有五个 scf 回归对重生成表全 PASS。
---

# Acceptance criteria

ac-001/002/004 由 `/mol:impl` 跑单元测试与 grep 验证;ac-003 由实现者按
Regenerate 命令重跑生成器后查 `git diff`;ac-005/006 经 regressions 目标验证
(oracle 数值在实现时用外部 PySCF 钉死并写进回归源码注释);ac-007 人工阅读 +
molplot 产物存在;ac-008 为收尾门。
