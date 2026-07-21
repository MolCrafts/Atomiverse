---
title: CFMM — 连续快速多极法加速 HF 库仑矩阵(J):延伸分类门面、CFMM-J 工作区与 J/K 接缝第三条路
slug: hf-capability-gaps-21-cfmm
status: approved
chain: hf-capability-gaps
link: 21/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# CFMM — 连续快速多极法加速 HF 库仑矩阵(J)

## Summary
链环 21(链尾)为 Gaussian-basis HF 补上线性标度库仑:在链环 20 的 FMM 树/算符之上,于顶层 `longrange/` 门面新增高斯分布**延伸分类**(White-Johnson-Gill-Head-Gordon 按延伸分支),新增 `internal::CfmmJ` 工作区类(壳对乘积分布携 P_μν 权重电荷、经 `core/real_harmonics` 的解析高斯多极 P2M、远场 M2L 游走、近场走现有 Schwarz 筛选的解析 ERI 直接路),以链环 12 建成的成员驱动 J/K 接缝接入第三条 J 路(`rhf.coulomb = Coulomb::Direct|RI|CFMM`,附 `cfmm_order`/`cfmm_threshold` 成员)。**硬门(用户+scientist 裁定)**:任何速度论断之前,CFMM J 必须先在文档化多极容差内逐元素复现 Schwarz 直接 J。K 不走 CFMM。诚实声明:PySCF **不含** FMM(其分子库仑是 libcint 直接 SCF+Schwarz,可选 DF),故验证基线是我们自家已被 PySCF 钉定的直接 J;CFMM 按 Jensen 的推荐,纯属可加选的线性标度选项。交付 C10H22 回归双门、烷烃系列交叉点表与 ch19 收官节"长程库仑 III:CFMM 加速 HF"。

## Domain basis
- **CFMM(按延伸分支)**:连续高斯分布不是点电荷——乘积分布(总指数 ζ=ζ_a+ζ_b)的延伸 r_ext ≈ √(2/ζ)·erfc⁻¹(ε_thr);两分布仅当中心距 R > r_ext(A)+r_ext(B)(良分离)才允许走多极,否则归近场解析 ERI。White, Johnson, Gill, Head-Gordon, *Chem. Phys. Lett.* **230**, 8 (1994), DOI 10.1016/0009-2614(94)01128-1。
- **FMM 骨架**:八叉树 P2M/M2M/M2L/L2L(链环 20 已建);Greengard, Rokhlin, *J. Comput. Phys.* **73**, 325 (1987), DOI 10.1016/0021-9991(87)90140-9。
- **高斯多极矩**:乘积高斯绕其中心的实立体谐波矩有闭式(s|s 的 0 阶矩即重叠电荷;高角动量经 Hermite 递推),复用 `core/real_harmonics`(链环 02 上提产物)。
- **推荐依据**:Jensen, *Introduction to Computational Chemistry*(线性标度一章)推荐 FMM 族处理大体系库仑;误差由截断阶 `cfmm_order`(L)与 `cfmm_threshold`(ε_thr)控制,教程如实给误差-参数表。
- **诚实基线(scientist 调查结论,写入教程)**:PySCF 无 FMM;其分子 J 为直接 SCF+Schwarz(可选 DF)。故正确性 oracle 是自家直接 J(该路已由链上 PySCF 回归钉定),不存在外部 FMM 数值可对。
- 单位:SCF 内部 Hartree;能量写回 Frame 为 eV(现状不变)。

## Design
- **前置**:链环 19(顶层 `longrange/` 门面)与链环 20(FMM 树 + M2M/M2L/L2L 算符,`longrange/` 内)未落地则本环 blocked;链环 12 的 `HamiltonianBuilder` 成员驱动 J/K 接缝为接入点;密度来自方法持有的 `SlaterDeterminant`(scf-slater-determinant-03-unbag)。
- **架构裁定(architect)**:cpu/scf 只 include 顶层 `longrange/` 门面头,**永不** include `cpu/longrange/` 的 Component 包装——CfmmJ 是方法私有工作区,不是 Plan 组件。
- **门面新增** `longrange/gaussian_extent.h`(new,header-only):`GaussianExtent{zeta, threshold}` 值类型(构造即算 r_ext,Doxygen 记公式与单位 Bohr)+ 良分离判据;叠加在链环 20 树上,不改树。
- **`internal::CfmmJ`**(new,类+构造函数,禁 `build_*` 自由函数——04-methods 门禁):构造取(壳对表、几何、`cfmm_order`、`cfmm_threshold`);`prepare()` 一次建树/分布/延伸分类与近远场配对表(远场 M2L 交互表与近场表互补,严禁双算);`j(P_total)` 每迭代:P2M(P_μν 加权)→ M2M/M2L/L2L 游走 → L2P 收远场 J,近场走现有 Schwarz 筛选解析 ERI 直接路。
- **接缝(第三条路)**:`scf_enums.h` 增 `Coulomb{Direct, RI, CFMM}`;`MeanFieldMethod` 增成员 `coulomb = Coulomb::Direct`、`cfmm_order = 10`、`cfmm_threshold = 1e-9`,经 `FockStage` 构造函数逐层直传入 `HamiltonianBuilder`(与链环 12 的 ri/aux 同型,无 Config 包)。默认 `Direct` 下现路**位级不变**;`Coulomb::CFMM` 与 `ri = RI::J/JK` 同设抛异常(J 路二选一);K 路永远 direct/RI,不受 coulomb 影响。
- 单元测试只验延伸公式/分类/单次 J 构建;收敛 SCF 归 regressions;墙钟归 benchmarks。

## Files to create or modify
- `tests/longrange/test_gaussian_extent.cpp` (new)、`tests/cpu/scf/test_cfmm_j.cpp` (new)、`tests/CMakeLists.txt`
- `include/atomiverse/longrange/gaussian_extent.h` (new)
- `include/atomiverse/cpu/scf/internal/cfmm_j.h` (new)、`src/cpu/scf/internal/cfmm_j.cpp` (new)、根 `CMakeLists.txt`(HF 源列表登记)
- `include/atomiverse/cpu/scf/scf_enums.h` — `Coulomb` 枚举
- `include/atomiverse/cpu/scf/internal/hamiltonian_builder.h`、`src/cpu/scf/internal/hamiltonian_builder.cpp` — 第三条 J 路分派
- `include/atomiverse/cpu/scf/stages/fock_stage.h`、`src/cpu/scf/stages/fock_stage.cpp`、`include/atomiverse/cpu/scf/mean_field_method.h`、`src/cpu/scf/mean_field_method.cpp` — 成员直传
- `regressions/scf/c10h22_cfmm_j/reg_c10h22_cfmm_j.cpp` (new)、`regressions/scf/c10h22_cfmm_j/REFERENCE.md` (new)、regressions 构建登记
- `benchmarks/scf/validation/cfmm_scaling/`(new:运行脚本 + README 方法学)
- `docs/tutorial/ch19-hf-large-systems.md` — 收官节

## Tasks
- [ ] Write failing tests for GaussianExtent and CfmmJ (tests/longrange/test_gaussian_extent.cpp:r_ext 对手算 erfc⁻¹ 值 ≤1e-12、良分离判据真值表;tests/cpu/scf/test_cfmm_j.cpp:s|s 分布 0 阶矩=重叠电荷 ≤1e-12、近/远场配对表互补无双算、H2O/STO-3G 单次 `j(P)` 对稠密直接 J 逐元素 ≤1e-8、CFMM 与 RI-J 同设抛异常;登记 tests/CMakeLists.txt)
- [ ] Implement GaussianExtent in include/atomiverse/longrange/gaussian_extent.h(header-only 值类型 + 良分离判据;WJGH 1994 DOI 注释;Doxygen 单位 Bohr)
- [ ] Implement CfmmJ in include/atomiverse/cpu/scf/internal/cfmm_j.h + src/cpu/scf/internal/cfmm_j.cpp(壳对分布 + 延伸分类、P2M 经 core/real_harmonics、链环 20 树上 M2M/M2L/L2L 游走、近场走现有 Schwarz 直接 ERI 路;只 include 顶层 longrange/ 门面头;登记根 CMakeLists.txt)
- [ ] Wire Coulomb enum in scf_enums.h + third J route in internal/hamiltonian_builder.{h,cpp} + stages/fock_stage.{h,cpp} + mean_field_method.{h,cpp}(成员 coulomb/cfmm_order/cfmm_threshold 直传;默认 Coulomb::Direct 位级不变;CFMM×RI-J 互斥抛异常;K 路不变;禁新 build_* 自由函数)
- [ ] Implement regression regressions/scf/c10h22_cfmm_j/(C10H22/STO-3G 双门:首迭代 ‖J_CFMM−J_direct‖_max ≤1e-8,且收敛 RHF 总能 CFMM-J vs 直接 ≤1e-7 Ha;REFERENCE.md 引 WJGH 1994 误差控制论断并声明基线为自家 PySCF 钉定直接路;≤60 s;登记 regressions 构建)
- [ ] Implement scaling sweep benchmarks/scf/validation/cfmm_scaling/(烷烃系列 CnH2n+2/STO-3G 的 J 构建墙钟两列 direct vs CFMM + 交叉点体系规模,README 记方法学/机器环境/cfmm_order/threshold,数字钉入 ch19)
- [ ] Write "长程库仑 III:CFMM 加速 HF" closing section of docs/tutorial/ch19-hf-large-systems.md(收束 Jensen 叙事:direct+Schwarz——诚实注明这正是 PySCF 的做法且 PySCF 无 FMM,引自家调查——→ RI → CFMM 线性标度;精度表 + 交叉点规模表,标量入表不画柱状图,molplot 标度曲线允许;hands-on `rhf.coulomb = Coulomb::CFMM`)
- [ ] Run full check + test suite

## Testing strategy
- **快乐路径(单元,秒级)**:延伸公式与分类判据的闭式抽查;s|s 矩守恒;近/远场互补;H2O/STO-3G 单次 J 构建 vs 稠密直接 ≤1e-8;单元测试均为单函数单调用,不跑收敛 SCF。
- **边界**:CFMM 与 RI-J 同设抛异常;`coulomb = Coulomb::Direct` 默认下 test_scf_solver / test_fock_stage 及链环 12 全部现测位级不变;`cfmm_threshold` 收紧一档(1e-9→1e-11)误差单调不增(单次 J,小体系)。
- **领域验证(regressions,非 tests/)**:C10H22/STO-3G 双门(J 矩阵 ≤1e-8、总能 ≤1e-7 Ha)——**硬门:先于并独立于一切速度论断**;oracle 为自家 Schwarz 直接 J(即链上已 PySCF 钉定之路),REFERENCE.md 说明 PySCF 无 FMM 故无外部 FMM 数可对。
- **性能(benchmarks,非 tests/)**:烷烃系列 J 构建墙钟与交叉点,方法学固化于 benchmarks/scf/validation/cfmm_scaling/,数字钉入教程。

## Out of scope
- CFMM-K(交换矩阵不走多极;K 保持 direct/RI);CFMM 解析梯度/力。
- 周期边界的 FMM/Ewald 耦合;CUDA 后端 CFMM;`cpu/longrange/` Component 包装的任何改动。
- 树/算符本体的改动(链环 20 产物只消费不修);RI 路(链环 12 产物只并列不改)。
- 教程 ch19 前节(链环 12/19/20 已写)重写——只加收官节。
