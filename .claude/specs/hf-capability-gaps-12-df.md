---
title: RI-J / RI-JK 密度拟合 — def2-JKFIT 静态表、DF 工作区与成员驱动 J/K 接缝
slug: hf-capability-gaps-12-df
status: code-complete
chain: hf-capability-gaps
link: 12/21
external-depends: scf-slater-determinant-02-rehome
created: 2026-07-20
---

# RI-J / RI-JK 密度拟合 — def2-JKFIT 静态表、DF 工作区与成员驱动 J/K 接缝

## Summary
为 Gaussian-basis HF/KS 补上密度拟合(RI):以 Weigend def2 万能 JKFIT 辅助基(静态数据头,经链环 01 的 tools/gen_basis_data.py 生成)把 4-指标 ERI 压成 3-指标中间量,新增 `internal::DensityFitting` 工作区类(3-中心/2-中心积分、度规 Cholesky、`j_fitted`/`k_fitted` 成员),方法上以成员 `rhf.ri = RI::None|J|JK` 选路,`HamiltonianBuilder` 内成员驱动分派(该接缝将被链环 21 CFMM 复用)。交付后 H2O/def2-TZVP 的 RI-JK 能量与自家精确 ERI HF 差 ≤50 μHa,并与外部钉定的 PySCF `density_fit()` 一致到 1e-6 Ha;glycine/def2-SVP 的墙钟对照表(ours-RI / ours-exact / PySCF-DF)钉入 ch19 章末。同时写 docs/tutorial/ch19 的"密度拟合 RI"一节。

## Domain basis
- **Coulomb 度规稳健拟合**:(μν|λσ) ≈ Σ_PQ (μν|P)[V⁻¹]_PQ(Q|λσ),V_PQ=(P|Q)。Coulomb 度规下 J 的误差是拟合残差的**二阶小量**(匹配辅助基时 ~数十 μHa/原子)。Vahtras, Almlöf, Feyereisen, *Chem. Phys. Lett.* **213**, 514 (1993), DOI 10.1016/0009-2614(93)89151-7。
- **RI-J(γ_P 路线)**:γ_P = Σ_Q [V⁻¹]_PQ Σ_μν P_μν(μν|Q);J_μν = Σ_P (μν|P)γ_P。J 拟合精度讨论:Weigend, *PCCP* **8**, 1057 (2006), DOI 10.1039/b515623h。
- **RI-K(B 半变换路线)**:B^P_μi = Σ_Q [V^{-1/2}]_PQ Σ_ν C_νi(μν|Q);K_μν = Σ_{P,i} B^P_μi B^P_νi。Weigend, *PCCP* **4**, 4285 (2002), DOI 10.1039/B204199P。理论脉络:Helgaker-Jørgensen-Olsen §9。
- **单表决策(设计约束 + 教程内容)**:只收 JKFIT 一张表,不另设 JFIT——K 的误差比 J 大且**必须** JKFIT(JFIT 不够);JKFIT 兼任 J 拟合只是略贵而精度不降,一表两用。头文件与教程都要写明这一取舍。
- **陷阱(必须编码)**:轨道基/辅助基失配是标准用户错误——所用元素在 JKFIT 表无行时**抛异常**,无回退。
- **g 函数约束**:JKFIT 表含 l=4(g)壳,现行 `cart.h kMaxL=3`/`rys_eri.h kRysMaxL=3` 必须扩到 4。根预算不变:3c 最重 Ltot=10→6 根,2c (g|g) Ltot=8→5 根,均 ≤ `kRysMaxRoots=7`;本环无 DF 梯度,无导数 Ltot 增长。
- 单位:SCF 内部 Hartree;能量写回 Frame 为 eV(现状不变)。

## Design
- **前置**:链环 01(`tools/gen_basis_data.py`,若 dev 上缺则先恢复)与 05(基组管线);`RI` 枚举落 `scf_enums.h`(scf-slater-determinant-02-rehome 产物,未落地则本环 blocked)。
- **静态表**:`basis_data/def2_jkfit.h` (new),沿用 `basis_table.h` 约定(`inline const std::vector<AtomBasis>& def2_jkfit()`,Z 索引,空行=不支持);生成器扩展一次转录,不在构建/测试期读任何数据文件。`AuxBasis` 枚举(首值 `Def2JKFIT`)与 `BasisSet` 同住 `basis_set.h`;`basis_set.cpp` 加辅助 ShellList 构造路径(复用现表→shell 转换,无新自由函数)。
- **`internal::DensityFitting`**(new,类+构造函数,禁 `build_*` 自由函数——04-methods 通配门禁):构造取(轨道 shells、原子几何、`AuxBasis`);私有工作区建 (μν|P)(现有 `rys_eri_block` + 单位零指数 s 哑壳,系数 1 不归一化)、V_PQ、Cholesky→V^{-1/2}(线性相关时按 `lindep_threshold` 抛异常);成员 `j_fitted(P_total)`(γ_P 路线)与 `k_fitted(C_occ)`(B^P_μi 半变换,占据轨道取自方法持有的 SlaterDeterminant)。
- **接缝(成员驱动,链环 21 CFMM 复用)**:`HamiltonianBuilder` 增成员 `DensityFitting` 可选工作区;`prepare()` 增 ri/aux 离散参;`step()` 内 `ri != RI::None` 时 J/K 改走 `df_.j_fitted/k_fitted`,否则原路**位级不变**。`FockStage` 构造函数与 `MeanFieldMethod` 成员(`RI ri = RI::None`、`AuxBasis aux_basis = AuxBasis::Def2JKFIT`)逐层直传,无 Config 包。
- 单元测试只验单次调用的表值/积分/矩阵数学;跑到收敛的 SCF 归 regressions;速度归 benchmarks。

## Files to create or modify
- `tests/cpu/scf/test_density_fitting.cpp` (new)、`tests/CMakeLists.txt`
- `include/atomiverse/cpu/scf/cart.h`、`include/atomiverse/cpu/scf/rys_eri.h`、`src/cpu/scf/rys_eri.cpp`、`src/cpu/scf/native_integrals.cpp` — l=4 扩展
- `tools/gen_basis_data.py`(链环 01 产物,若缺先恢复)— JKFIT 发射
- `include/atomiverse/cpu/scf/basis_data/def2_jkfit.h` (new)
- `include/atomiverse/cpu/scf/basis_set.h`、`src/cpu/scf/basis_set.cpp` — `AuxBasis` + 辅助 shell 路径
- `include/atomiverse/cpu/scf/internal/density_fitting.h` (new)、`src/cpu/scf/internal/density_fitting.cpp` (new)、根 `CMakeLists.txt`(HF 源列表登记)
- `include/atomiverse/cpu/scf/scf_enums.h`(02-rehome 产物)— `RI` 枚举
- `include/atomiverse/cpu/scf/internal/hamiltonian_builder.h`、`src/cpu/scf/internal/hamiltonian_builder.cpp` — 成员分派接缝
- `include/atomiverse/cpu/scf/stages/fock_stage.h`、`src/cpu/scf/stages/fock_stage.cpp`、`include/atomiverse/cpu/scf/mean_field_method.h`、`src/cpu/scf/mean_field_method.cpp` — 成员直传
- `regressions/scf/h2o_ri_jk/reg_h2o_ri_jk.cpp` (new)、`regressions/scf/h2o_ri_jk/REFERENCE.md` (new)、regressions 构建登记
- `benchmarks/scf/validation/ri_speed/`(new:运行脚本 + README 方法学)
- `docs/tutorial/ch19-hf-large-systems.md`(链环 01 骨架;若缺先建)

## Tasks
- [x] Write failing tests for DensityFitting (tests/cpu/scf/test_density_fitting.cpp:JKFIT 表行对 Weigend 2002 发表值抽查、g 壳 (g|g)/(ss|g) 积分对外部钉定 PySCF 参考值 ≤1e-10、V_PQ 对称正定、H2/STO-3G 上 j_fitted/k_fitted 对稠密精确 J/K 的逐元素拟合误差上界、无 JKFIT 行元素抛异常;登记 tests/CMakeLists.txt)
- [x] Implement l=4 (g) per-shell extension in cart.h/rys_eri.h + src/cpu/scf/{rys_eri.cpp,native_integrals.cpp}(kMaxL/kRysMaxL 3→4、15 个 g 笛卡尔分量表;仅 2c/3c 路径需要,根预算不变并静态断言)
- [x] Extend tools/gen_basis_data.py and emit include/atomiverse/cpu/scf/basis_data/def2_jkfit.h;add AuxBasis to basis_set.h + aux ShellList path in basis_set.cpp(单表 JKFIT 取舍写入头文件 Doxygen)
- [x] Implement DensityFitting in include/atomiverse/cpu/scf/internal/density_fitting.h + src/cpu/scf/internal/density_fitting.cpp((μν|P) 经零指数 s 哑壳、V_PQ、Cholesky/V^{-1/2}、j_fitted/k_fitted;DOI 注释;登记根 CMakeLists.txt)
- [x] Wire RI enum in scf_enums.h + member-driven seam in internal/hamiltonian_builder.{h,cpp} + stages/fock_stage.{h,cpp} + mean_field_method.{h,cpp}(成员 ri/aux_basis 直传;默认 RI::None 位级不变;禁新 build_* 自由函数)
- [x] Implement regression regressions/scf/h2o_ri_jk/(H2O/def2-TZVP:RI-JK vs 自家精确 ERI ≤50 μHa,且对 REFERENCE.md 钉定的 PySCF mf.density_fit() 能量 ≤1e-6 Ha;≤60 s;登记 regressions 构建)
- [x] Implement speed sweep benchmarks/scf/validation/ri_speed/(glycine/def2-SVP 墙钟三列表 ours-RI-JK / ours-exact / PySCF-DF,README 记方法学与机器环境,一次跑完数字钉入 ch19)
- [x] Write "密度拟合 RI" section of docs/tutorial/ch19-hf-large-systems.md(Jensen SCF 瓶颈叙事:4-指标→3-指标为何压缩;μHa 误差预算诚实表;hands-on `rhf.ri = RI::JK`;章末 PySCF 精度表 + 速度表,标量表格不画柱状图)
- [x] Run full check + test suite

## 实现审计注记(2026-07-21)

- **k_fitted 的密度路线补充**:除谱面 `k_fitted(C_occ)`(B 半变换)外,
  Fock 接缝实际走新增的 `k_fitted_from_density(P)`(P 谱分解 →
  带符号 √|λ| 因子 → k_fitted)。原因:SCF 中途的密度(MINAO 猜测、
  damping、ASPC 外推)并非来自当前 SlaterDeterminant 的 C_occ,用方法
  持有的 C 会与实际密度失同步;谱分解路线对任意对称 P 精确,且与
  k_fitted(C_occ) 在单元测试里锁到 1e-11。
- **JKFIT 元素覆盖为轻元素**(H, C, N, O, F, Na–Ar):def2 重元素
  (Rb–Rn)的 JKFIT 行含 l≥5(h/i)壳,超出引擎 l=4 上限,故不发射;
  缺行元素抛异常(无回退),生成器注释与 def2_jkfit.h Doxygen 均说明。
- **RI 拒绝面**:ROHF、GHF、SOSCF、解析力 与 `ri != RI::None` 组合均
  显式抛异常(GHF 走稠密 ERI 旋量路径;SOSCF 的 CIAH 内部自建精确
  Fock 会与拟合循环失一致;DF 解析梯度是后续环)。level_shift 的
  de-shift 终末 Fock 重建改为透传 ri/aux(收敛的是拟合 Fock,ε 必须
  对同一算符对角化)。
- **4-中心路径保持 f 上限**:新增 `kMaxLOrbital = 3` /
  `kMaxNCartOrbital`,4-中心栈缓冲与 MD 递推表尺寸逐字节保持原状;
  `rys_eri_block` 公共入口对超根预算的四中心 g 夸特显式抛异常
  (单元测试 FourCenterGQuartetThrows)。
- **g 积分钉值容差为相对 1e-10**:引擎 cart 约定的 g 自归一 O(10²),
  绝对 1e-10 等价于亚 ULP;测试用 1e-10·max(1,|want|),注释记录引擎/
  PySCF 归一幅度比 s_g = 8.671834327921664 的推导与原始 PySCF 命令。
- **性能修正**:B 收缩必须先实体化 Wᵀ(惰性 transpose 视图会把
  xt::linalg::dot 推下 dgemm 快径,glycine 上 2.3 s → 30 ms);3c 建
  造带 Schwarz 预筛(1e-11)+ 融合 c2s/散射循环。基准:glycine/SVP
  ours-RI-JK 1.617 s < ours-exact 2.516 s(中位,8 线程);PySCF-DF
  0.104 s(差距为引擎整体速度,能量对齐 1e-9 Ha,README 已记)。
- **RI::J 的交换**:按设计走精确壳块 K(`ShellBlockPairList::k`,对
  密度线性);烟雾验证 RHF/UHF 的 RI::J 与 RI::JK 收敛且拟合误差
  ~5e-5 Ha 量级(OH 自由基 def2-SVP)。

## Testing strategy
- **快乐路径(单元,秒级)**:JKFIT 表值抽查(H/C/O 各一壳);g 壳 2c/3c 积分对钉定参考 ≤1e-10;V^{-1/2}·V·V^{-1/2}=I ≤1e-12;H2/STO-3G 上 ‖J_fit−J_exact‖∞ 与 ‖K_fit−K_exact‖∞ 落在文档化的拟合误差上界内。
- **边界**:无 JKFIT 行元素抛异常(信息含元素与辅助基名);度规近奇异抛异常;`ri = RI::None` 默认下现有 test_scf_solver / test_fock_stage 全部位级不变。
- **领域验证(regressions,非 tests/)**:H2O/def2-TZVP RI-JK 双门:vs 自家精确 ERI ≤50 μHa(二阶误差论断的直接检验)且 vs PySCF 钉定值 ≤1e-6 Ha。PySCF 只作外部 oracle 钉数,不作依赖。
- **性能(benchmarks,非 tests/)**:glycine/def2-SVP 墙钟表;方法学一次固化于 benchmarks/scf/validation/ri_speed/,数字钉入教程。

## Out of scope
- DF 解析梯度/力(RI 力是后续环);RI-MP2 / ao2mo 的 DF 化;CUDA 后端 DF。
- 独立 JFIT 表、衰减/局域度规拟合、全 ERI 的 Cholesky 分解替代路线。
- CFMM(链环 21,仅复用本环接缝);4-中心路径的 l=4 扩展(轨道基 kMaxL 仍为 f)。
