---
title: HF 能力补全 · 09 二阶 SCF — CIAH 增广 Hessian 轨道旋转(SOSCF)
slug: hf-capability-gaps-09-soscf
status: code-complete
chain: hf-capability-gaps
link: 09/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# HF 能力补全 · 09 二阶 SCF(SOSCF / CIAH)

## Summary

给 Gaussian SCF 增加二阶收敛路径:以 08 环交付的轨道 Hessian σ-vector 内核为 Newton 步算子,按 CIAH(co-iterative augmented Hessian,Sun 2016)解增广 Hessian 特征问题得到轨道旋转 κ,信赖域限步后以 C ← C·exp(κ) 指数更新轨道。接线走既有五阶段(mix/diagonalize 阶段各加一条 SOSCF 支路——"哪些组件填充阶段",不加新 Driver),升级策略是方法成员(`rhf.soscf = true`、`soscf_start` 阈值):DIIS 家族(07 环)负责早期迭代,SOSCF 收尾或救难。RHF 与 UHF 至少可用。用户可见成果:CDIIS 单独振荡不收敛的难例(如拉伸 N₂)一开 `soscf` 即二次收敛,终能量对上 PySCF `mf.newton()`;教程 ch18 补上"二阶方法"下半章。

## Domain basis

- CIAH:Sun, *arXiv:1610.08423* (2016) —— 增广 Hessian 特征问题 [[0, g^T],[g, H]](1; κ) = λ(1; κ) 与 σ-vector 共迭代解法(H 永不显式构造),信赖域 + 微迭代结构照此实现。
- 二阶/信赖域 SCF 理论:Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* (Wiley, 2000), ch. 10 —— 轨道旋转参数化 E(κ)、梯度 g_ai = F_ai(occ-virt Fock 块,自旋分辨)、二次收敛条件。
- 单位:g、H、能量均为 Hartree;κ 无量纲(反对称旋转参数);exp(κ) 保幺正(C^T S C = I 不变量)。
- Oracle:难例终能量与迭代数对外部运行的 PySCF `mf.newton()` 钉死(|ΔE| ≤ 1e-8 Ha;钉值时注明 PySCF 版本与命令)。

## Design

新内部类 `internal::SOSCFStepper`(`internal/soscf.h`,不继承 Component,遵循 `DIISMixer` 先例:internal/ 工作区类 + stages/ 接线):构造器收维度与自旋道数,持有 Davidson 子空间、信赖域半径等私有状态;`step(...)` 收 MO 系数/Fock/S,产出 κ 与旋转后 C。内部依次:g_ai 提取(RHF 单道,UHF α+β 拼接旋转向量)→ CIAH 增广 Hessian Davidson(σ-vector 调 08 环内核,矩阵自由)→ 信赖域缩步(‖κ‖ 超界即按比例收缩,能量上行则缩域重试)→ exp(κ) 幺正更新。升级策略为 `MeanFieldMethod` 成员(非 Config 包):`bool soscf = false`、`double soscf_start = 0.25`(DIIS 误差 ∞-范数低于阈值即切换;亦即"DIIS 早期、SOSCF 收尾"标准配方)、`int soscf_max_micro = 12`、`double soscf_trust = 0.4`。接线:`ensure_driver_built_` 把成员与升级标志传入 MixStage(SOSCF 激活时跳过 DIIS 外推)与 DiagonalizeStage(激活时以 SOSCFStepper 旋转更新 C_alpha_/C_beta_ 代替本征求解;收敛后补一次终末对角化出正则轨道能)。03-unbag 落地后结果面即方法持有的 `SlaterDeterminant`,旋转作用于其接线的 C 迭代态,接缝不变。依赖:07 环(混合器升级接缝 + ch18 上半章)、08 环(σ-vector 内核)先行。

### 实现审计注记(2026-07-21)

1. **UHF Hessian 作用在 SOSCFStepper 内部实现**:08 环 `OrbitalHessian`
   构造器显式拒绝非限制性行列式(RHF-only)。UHF 的 σ-vector 按
   H^σσ = 2[δδΔε + 2(ai|bj) − (ab|ij) − (aj|bi)]、H^αβ = 4(aα iα|bβ jβ)
   在 soscf.cpp 内从非限制性 MO-ERI occ-virt 块直接收缩(同一
   quarter-transform 模式,`mo_transform_`),H 仍从不显式组装。RHF 走
   08 内核(H = 4·¹(A+B),g = 4F_ai;两者对旋转能量的真导数,系数由
   有限差分单测钉住)。
2. **激活门槛 iter ≥ 1(对"soscf_start ≤ 0 即首迭代升级"的修正)**:
   第 0 迭代时密度型初猜(MINAO / 外部密度)与序幕 Hcore 对角化产生的
   系数描述的是两个不同状态;在该不一致点做 Newton 旋转实测发散
   ~7 Hartree(MINAO + soscf 扫描全灭)。故激活一律要求 iter ≥ 1
   (第 0 迭代先做一次 Roothaan 对角化同步 C 与 P);soscf_start ≤ 0
   语义改为"最早可激活迭代(第 1 迭代)即升级"。单测
   `SoscfMixStage.ActivatesAtFirstEligibleIterationWhenStartNonPositive`
   钉住。
3. **半正则化**:SOSCF 宏迭代内先对 occ-occ / virt-virt Fock 块做块对角
   化(密度不变),使 08 内核基于 ε 的 σ-vector 在非收敛迭代点严格成立
   (否则需全 F_oo/F_vv 块的修正项)。SOSCF 迭代期间 eig 为半正则值;
   收敛后复用 de-shift 尾声块做终末对角化,行列式携带正则 ε。
4. **难例选型**:拉伸 N₂ R=2.5 Å/6-31G RHF 在本引擎 CDIIS 下 13 步即收
   敛(PySCF DIIS 9 步)——无失速对比,且该几何 RHF 多解(minao 与
   hcore 各落一盆)。按 spec 的回退许可,失速对比由 07 环难例
   (2.6× H₂O/def2-SVP UHF,CDIIS 60 步预算内振荡)承担;N₂ 保留为
   RHF 钉值行(能量/HOMO/迭代数 vs mf.newton(),HOMO 为解盆指纹)。
5. **新增公共观测器 `soscf_iterations()`**:与 PySCF newton 宏迭代数可
   比的量是 Newton 相的 Fock 构建数,而非含 DIIS 前段的总迭代数;方法
   面加了这一只读诊断成员函数(ac-005 的断言主体)。UHF 难例的迭代数
   钉值采用"升级等价种子"(混合 Hcore 初猜 + 6 步 DIIS 预热后
   mf.newton(),8 宏)——更冷/更浅的种子下 PySCF newton 根本到不了
   碎对称极小(REFERENCE.md 记录了整套种子敏感性),故这是唯一诚实的
   同协议对照;稳定性种子的 4 宏钉值一并记录。
6. **exp(κ) 取缩放-平方 Taylor**(spec 允许本征分解或 scaling-squaring
   二选一);正交性 ≤1e-13、C^T S C = I ≤1e-12 由单测钉住。
7. **soscf 限 HF**:RKS/UKS 上开 soscf 抛 `std::invalid_argument`
   (XC 二阶核不在本环;拒绝而非静默走错曲面)。

## Files to create or modify

- `include/atomiverse/cpu/scf/internal/soscf.h` (new) — `internal::SOSCFStepper` 声明
- `src/cpu/scf/internal/soscf.cpp` (new) — CIAH Davidson / 信赖域 / exp(κ) 实现
- `include/atomiverse/cpu/scf/stages/mix_stage.h` — SOSCF 激活标志:跳过 DIIS 支路
- `src/cpu/scf/stages/mix_stage.cpp` — 同上实现
- `include/atomiverse/cpu/scf/stages/diagonalize_stage.h` — SOSCF 旋转更新支路(header-only)
- `include/atomiverse/cpu/scf/mean_field_method.h` — `soscf`/`soscf_start`/`soscf_max_micro`/`soscf_trust` 成员 + Doxygen
- `src/cpu/scf/mean_field_method.cpp` — `ensure_driver_built_` 接线 + 升级判据 + 终末对角化
- `CMakeLists.txt` — 登记 `src/cpu/scf/internal/soscf.cpp`
- `tests/cpu/scf/test_soscf.cpp` (new) — Stepper 各件单元测试
- `tests/CMakeLists.txt` — 注册 `test_soscf`
- `regressions/scf/soscf_hard_case/reg_soscf_hard_case.cpp` (new) — 难例 vs PySCF newton 回归
- `regressions/CMakeLists.txt` — `atv_add_regression(scf soscf_hard_case)`(`ATV_BUILD_HF` 门内)
- `docs/tutorial/ch18-scf-techniques.md` — "二阶方法" 下半章(07 环创建该文件)

## Tasks

- [x] Write failing tests tests/cpu/scf/test_soscf.cpp(exp(κ) 幺正性 C^T S C=I;g_ai 对微型体系旋转能量的有限差分;CIAH 矩阵自由解 vs 测试内显式小增广矩阵稠密本征解一致;信赖域限步;soscf=false 默认路径 H₂/STO-3G 微型 SCF 逐位不变)并注册 tests/CMakeLists.txt
- [x] Implement internal::SOSCFStepper in internal/soscf.h + src/cpu/scf/internal/soscf.cpp(g_ai 提取、CIAH Davidson 调 08 环 σ-vector、信赖域、exp(κ);Doxygen 带单位)并登记 CMakeLists.txt
- [x] Implement escalation members + wiring in mean_field_method.h / mean_field_method.cpp(四个 soscf 成员、DIIS 误差阈值切换、收敛后终末对角化)
- [x] Implement SOSCF stage paths in stages/mix_stage.h/.cpp 与 stages/diagonalize_stage.h(激活时 mix 跳 DIIS、diagonalize 走旋转更新;RHF 与 UHF 两道)
- [x] Add regression regressions/scf/soscf_hard_case(CDIIS-alone 振荡而 soscf=true 收敛;终能量 vs 钉住的 PySCF mf.newton() ≤1e-8 Ha;迭代数与钉值可比;≤60 s)并注册 regressions/CMakeLists.txt
- [x] Extend docs/tutorial/ch18-scf-techniques.md 二阶方法半章(轨道旋转上的 Newton-Raphson、为何二次收敛、何时划算;molplot DIIS vs SOSCF 收敛曲线;章末 PySCF 对照表行,标量进表)
- [x] Run full check + test suite

## Testing strategy

- **快乐路径(单元,秒级)**:`test_soscf` 逐件测 Stepper——幺正性、梯度块有限差分、小维度 CIAH vs 稠密增广矩阵(H 只在测试里显式构造)、信赖域截断;微型 H₂/STO-3G 验证默认关断路径逐位不变。
- **边界**:κ→0 时 exp(κ)=I、step 为 no-op;‖κ‖ 恰在信赖域边界;UHF 拼接向量的 α/β 维度错配抛错(无回退);`soscf_start ≤ 0` 即首迭代升级。
- **领域验证(regressions/,≤60 s)**:`reg_soscf_hard_case` 以公开 API(`cpu::scf::RHF`/`UHF` + soscf 成员)跑难例(拉伸 N₂,R=2.5 Å,6-31G;若 07 环难例更佳则沿用),断言 CDIIS-alone 同预算不收敛、SOSCF 收敛且能量/迭代数对 PySCF newton 钉值。
- **回归安全网**:既有 `test_scf_solver`/`test_uhf_solver`/`test_accelerators` 全绿证明默认路径零扰动。

## Out of scope

- RKS/UKS 的 SOSCF 升级(XC 二阶核进 σ-vector 属后续环;本环 RHF/UHF)。
- 08 环 σ-vector 内核本身及其正确性(08 环职责;本环只消费其接口)。
- ADIIS/EDIIS 早期混合器(07 环职责);SOSCF 与 ASPC 预测器的组合调优。
- CUDA SCF 路径;稳定性分析(internal instability 跟随波)与自动扰动重启。
