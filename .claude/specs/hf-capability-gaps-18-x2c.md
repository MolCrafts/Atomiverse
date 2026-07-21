---
title: HF 能力补全 · 18 标量相对论 x2c-1e — 精确二分量核哈密顿替换
slug: hf-capability-gaps-18-x2c
status: code-complete
chain: hf-capability-gaps
link: 18/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# HF 能力补全 · 18 标量相对论 x2c-1e

## Summary

为具体平均场方法(`RHF`/`UHF`/`RKS`/`UKS`)接通 06 环在 `MeanFieldMethod` 上保留的 `bool x2c = false;` 成员:无自旋(spin-free)一电子精确二分量(x2c-1e)。只替换一电子核哈密顿——新一电子积分核 pVp(W = ⟨σ·pVσ·p⟩ 的无自旋部分)加入 one_electron 家族(04 环收编后的 `IntegralPack` 私有成员形态,禁 `build_*` 自由函数),SCF 私有工作区类 `internal::X2cWorkspace` 在动能平衡基里组装 {V, T, W} 块的一电子 Dirac 矩阵、解退耦矩阵 X、做一步 FW 型变换得 h^X2C;`x2c=true` 时序幕里 Hcore ← h^X2C,S/正交化/ERI/Fock 循环逐字节不动(ERI 保持非相对论,即 x2c-1e 范围)。两条诚实线一并落地:ECP↔x2c 互斥 THROW 门(06 环已落)在本环两个方向都被测试钉死;picture-change 误差(x2c 轨道下偶极等期望值需变换算符)以文档警告 + 教程坦白,属性变换本身 out of scope。用户可见成果:Ar/def2-TZVP 的 x2c-1e RHF 能量与外部钉死的 PySCF `x2c1e()` 对账到 1e-5 Ha,相对论能量降低符号与幅值正确。

## Domain basis

- 精确二分量一步退耦:Kutzelnigg, Liu, *J. Chem. Phys.* **123**, 241102 (2005), DOI 10.1063/1.2137315;工作方程综述:Peng, Reiher, *Theor. Chem. Acc.* **131**, 1081 (2012), DOI 10.1007/s00214-011-1081-y。
- 动能平衡基无自旋修正 Dirac 方程(原子单位):h = [[V, T], [T, W/(4c²) − T]],度规 M = [[S, 0], [0, T/(2c²)]];取 n 个正能解得 X = C_S C_L⁻¹;S̃ = S + X†TX/(2c²),重整化 R = S^{−1/2}(S^{−1/2}S̃S^{−1/2})^{−1/2}S^{1/2};h^X2C = R†(V + TX + X†T + X†(W/(4c²)−T)X)R。教科书锚点:Reiher, Wolf, *Relativistic Quantum Chemistry*;Jensen, *Introduction to Computational Chemistry* 相对论一章。
- W_μν = ⟨∇χ_μ|V̂|∇χ_ν⟩,V̂ = −Σ_A Z_A/|r−R_A|(与 PySCF `int1e_pnucp` 同约定);由核吸引 OS 原语经高斯导数 l±1 关系装配(导数形式)。光速常量 kLightSpeedAU = 137.03599967994(取 PySCF `lib.param.LIGHT_SPEED` 同值;1e-5 Ha 对账下敏感度 ~1e-8 Ha,可忽略)。
- **收缩基裁定(必须钉死)**:退耦在收缩 AO 基内做;PySCF `x2c1e()` 默认 `xuncontract=True`,oracle 必须显式 `xuncontract=False` 外部运行钉值,否则 1e-5 Ha 对不上。uncontract 精化 out of scope。
- **picture change 陷阱(必须文档化)**:x2c 轨道下的期望值(偶极、ESP、多极,link 03/04 面)若用未变换算符即含 PCE;本环只警告不变换。
- 验证靶(oracle):RHF Ar 原子 def2-TZVP(05 环后 Z≤18 全电子覆盖已核实;Kr/HBr 不在表内故不选),vs 外部 PySCF `scf.RHF(mol).x2c1e()`(`xuncontract=False`,钉值注明版本与命令)|ΔE| ≤ 1e-5 Ha;且 E_x2c − E_nonrel < 0,幅值(Ar 约 −1.8 Ha 量级)与 PySCF 钉死的 Δ 相差 ≤ 5%。单位:内部 Hartree/Bohr。

## Design

- **积分面**:`IntegralPack` 私有静态成员 `pvp(shells, atoms)`(声明 internal/integrals.h,定义 src/cpu/scf/one_electron.cpp,与 04 环 overlap/kinetic/nuclear_attraction 同形同址),OS 递推导数形式,输出对称 (n×n) W 矩阵;受 `no_factory_scrub.sh` 的 cpu/scf `build_*` 通配门约束。
- **工作区**:`internal::X2cWorkspace`(新,internal/x2c.h + src/cpu/scf/internal/x2c.cpp):纯线代退耦类,构造 `(S, T, V, W, double c = kLightSpeedAU)`(c 可注入,供 c→∞ 极限单测),构造即完成 Dirac 组装→广义对角化(复用 internal/eigen_solver.h)→X→R→h^X2C;成员 `hcore()` 返回 h^X2C。类私有、绝不进公共 API(Frame 纯净:方法私有工作区,随几何每次 `drive_scf_` 重建);正能解选择按本征值 > −c² 分支,病态(C_L 奇异)抛 `std::runtime_error`。
- **接线**:`IntegralPack` 装配路径加 W 开关(仅 x2c 时计算);`drive_scf_core_` 序幕在一电子装配后、06 环 ECP↔x2c 互斥门之后:`x2c=true` → 私有 `Hcore_` 整体替换为 h^X2C;Fock 循环、DIIS、密度、ERI 全部无感知。`forces()` 在 `x2c=true` 时抛 `std::runtime_error`(h^X2C 梯度未推导,无静默错误力——link 15 同款拦截)。
- **诚实面**:`mean_field_method.h` 的 `x2c` 成员 Doxygen 补消费语义 + PCE 警告(偶极/ESP/多极期望值未做 picture-change 变换);教程同述。本环不动 `run()`/SlaterDeterminant 签名(external-depends 只因触碰 03-unbag 落定的成员面)。

## Files to create or modify

- include/atomiverse/cpu/scf/internal/integrals.h — `pvp` 私有成员声明 + W 装配开关
- src/cpu/scf/integrals.cpp — x2c 时装配 W 的接线
- src/cpu/scf/one_electron.cpp — pVp OS 导数形式核实现
- include/atomiverse/cpu/scf/internal/x2c.h (new) — `X2cWorkspace` 声明 + kLightSpeedAU
- src/cpu/scf/internal/x2c.cpp (new) — Dirac 组装 / X / R / h^X2C 实现
- include/atomiverse/cpu/scf/mean_field_method.h — `x2c` 成员 Doxygen(消费语义 + PCE 警告)
- src/cpu/scf/mean_field_method.cpp — 序幕 Hcore 替换 + forces() 拦截
- CMakeLists.txt — 注册 src/cpu/scf/internal/x2c.cpp
- tests/cpu/scf/test_x2c.cpp (new) — W 钉值 / 极限 / 恒等式 / 门禁单元测试
- tests/CMakeLists.txt — 注册 `test_x2c`
- regressions/scf/x2c_argon/reg_x2c_argon.cpp (new) — Ar oracle 回归
- regressions/CMakeLists.txt — `atv_add_regression(scf x2c_argon)`(`ATV_BUILD_HF` 门内)
- docs/tutorial/ch20-hf-properties-environment.md — "标量相对论:x2c" 一节(由本链早前 link 创建;若尚未落地则新建章骨架后加节)

## Tasks

- [x] Write failing tests tests/cpu/scf/test_x2c.cpp(H₂/HeH⁺ STO-3G 的 W 矩阵元钉外部 PySCF `mol.intor('int1e_pnucp')` 值 ≤1e-7(Boys floor),W 对称;X2cWorkspace 注入 c=1e8 时 h^X2C 与 T+V 逐元素 ≤1e-6;H 原子/def2-SVP 精确退耦恒等式:h^X2C 最低本征值 == Dirac 矩阵最低正能本征值 ≤1e-10;互斥门双向:含 I 的 Frame `ecp="def2-ECP"; x2c=true` 抛含元素名 runtime_error,全电子 H₂ + x2c 正常收敛不抛;`x2c=true` 时 forces() 抛;x2c=false 时 Hcore/能量与现状逐字节不变)并注册 tests/CMakeLists.txt
- [x] Implement pvp kernel in src/cpu/scf/one_electron.cpp + 声明于 include/atomiverse/cpu/scf/internal/integrals.h(核吸引 OS 原语的 l±1 导数装配;IntegralPack 私有成员形态,过 no_factory_scrub 通配门)
- [x] Implement internal::X2cWorkspace in include/atomiverse/cpu/scf/internal/x2c.h + src/cpu/scf/internal/x2c.cpp(收缩基一步退耦:Dirac 块矩阵、广义本征、X、S̃、R、h^X2C;kLightSpeedAU;病态抛错)并注册 CMakeLists.txt
- [x] Wire x2c consumption in src/cpu/scf/integrals.cpp + src/cpu/scf/mean_field_method.cpp(x2c 时装配 W;drive_scf_core_ 序幕互斥门后 Hcore_ ← h^X2C;forces()+x2c 抛 runtime_error)
- [x] Add Doxygen(@brief/@param 带单位/@return)到所有新符号,并在 include/atomiverse/cpu/scf/mean_field_method.h 的 `x2c` 成员写明 PCE 警告(属性期望值未 picture-change 变换)
- [x] Add regression regressions/scf/x2c_argon/reg_x2c_argon.cpp(公开 API `cpu::scf::RHF`,Ar/def2-TZVP:x2c 能量 vs 钉死的 PySCF `x2c1e()`(xuncontract=False)≤1e-5 Ha;E_x2c−E_nonrel<0 且与钉死 Δ 相差 ≤5%;≤60 s)并注册 regressions/CMakeLists.txt
- [x] Extend docs/tutorial/ch20-hf-properties-environment.md:"标量相对论:x2c" 一节(重元素为何需要相对论——s/p 轨道收缩、d/f 膨胀叙事;x2c-1e 一页讲清;ECP vs x2c 决策指引与互斥;picture-change 诚实警告;Ar 上手示例;章末 PySCF 对照表行——标量进表)
- [x] Run full check + test suite

## Testing strategy

- **快乐路径(单元,秒级)**:W 矩阵元钉值(PySCF 版本+命令写死在测试源码注释,ctest 无 Python 依赖)、c→∞ 非相对论极限、一电子体系精确退耦机器恒等式;绝不在 tests/ 跑重原子完整 SCF。
- **边界**:`x2c=false` 全路径与现状逐字节等价;C_L 奇异构造抛错;互斥门双向(含 ECP 元素抛、全电子不抛);`x2c=true` + forces() 抛异常。
- **领域验证(regressions/,≤60 s)**:`reg_x2c_argon` 以公开 API 跑 Ar 非相对论与 x2c 两次 RHF,对钉死的 PySCF oracle 断言能量 ≤1e-5 Ha、相对论降低符号与幅值(≤5%);非零退出即失败;只挂 `regressions` target,不进 ctest。
- **门禁**:`no_factory_scrub.sh` 全程 PASS(新核为成员,无 build_* 自由函数)。

## Out of scope

- 属性算符的 picture-change 变换(PCE 修正的偶极/ESP/多极)——只文档警告;自旋-轨道 x2c(SOC)、DKH 系列、四分量。
- x2c 解析梯度 / 力(forces() 拦截抛错;后续环);x2c + ECP 组合(永久互斥,非"暂缓")。
- 退耦前基组 uncontract 精化(PySCF `xuncontract=True` 路线)与重元素(Z>18)全电子基组扩表;CUDA SCF 的 x2c 适配。
- 二电子相对论修正(ERI 保持非相对论——x2c-1e 定义即此)。
