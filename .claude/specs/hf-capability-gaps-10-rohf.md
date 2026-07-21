---
title: "ROHF 具体方法(Roothaan 有效 Fock,PySCF/Guest–Saunders 正则化约定)"
slug: hf-capability-gaps-10-rohf
status: code-complete
chain: hf-capability-gaps
link: 10/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# ROHF 具体方法(Roothaan 有效 Fock,PySCF/Guest–Saunders 正则化约定)

## Summary
新增开壳层限制性 Hartree–Fock(ROHF)作为一个新的具体方法类 `cpu::scf::ROHF`,沿 `uhf.h` 的形态:CRTP 继承 `MeanFieldMethod<ROHF>`(即 `CPUDriver<scf::Phase>` 的 SCF 循环节点),参数为成员(`n_alpha`/`n_beta`),`run(frame)` 以 Frame 为唯一 I/O。ROHF 内部复用现有 Pople–Nesbet 双密度(α/β)Fock 机制,在 diagonalize 接缝处插入 Roothaan 有效单 Fock 耦合——"哪些组件填充哪些阶段",不引入新的循环形态。产出:自旋纯(⟨S²⟩ 恰为 S(S+1))的单 MO 集(闭/开/虚三空间)开壳层解,总能量对 O₂(三重态)与 CH₃(双重态)以外部钉定的 PySCF `scf.ROHF` 为准绳到 1e-8 Ha;并扩写教程 ch03 的方法主线叙事。与并行链 scf-slater-determinant 对齐:方法持有 SlaterDeterminant、`build_*` 自由函数禁止、在 `methods.h` 注册。

## Domain basis
- Roothaan 单 Fock ROHF:F_eff 按闭(c)/开(o)/虚(v)三空间分块耦合;对角块系数即"耦合系数"。**用户裁定的绑定约定**:正则化采用 PySCF canonical ROHF(Guest–Saunders 族,`get_roothaan_fock`):对角块 cc=oo=vv=F_c≡(F_α+F_β)/2,非对角 co=F_β、ov=F_α、cv=F_c;AO 基下经密度投影子(P_c=P_β·S 等)装配。此约定必须写入本节与 `rohf.h` 的类文档。
- **PITFALL(必须在代码文档与教程中如实说明)**:Roothaan / Guest–Saunders / McWeeny–Diercksen 等耦合系数给出**相同的总能量**但**不同的轨道能**——总能量是可移植观测量,轨道能是约定绑定量;验收只在 PySCF 约定下比较轨道能,绝不以非 PySCF 约定的轨道能设门槛。
- 能量泛函与 UHF 相同(故对耦合系数不变):E = ½Tr[P_α(H_core+F_α)] + ½Tr[P_β(H_core+F_β)] + V_nn [Hartree];F_σ = H_core + J(P_α+P_β) − K(P_σ)。
- 自旋纯性:单 MO 集下 β 占据是 α 占据的子集,由 `spin_analysis.h` 公式得 ⟨S²⟩−S(S+1) 恰为 0(数值上 ≤1e-10)——与 UHF 自旋污染形成对照。
- 文献:Roothaan, Rev. Mod. Phys. 32, 179 (1960), DOI 10.1103/RevModPhys.32.179;Tsuchimochi & Scuseria, J. Chem. Phys. 133, 141102 (2010), DOI 10.1063/1.3503173(constrained-UHF 视角,作为现代辨析引用);Szabo & Ostlund §3.8。Oracle:外部运行 PySCF 2.13.1 `scf.ROHF` 钉定数值(依 pyscf-as-reference-oracle 规则:禁依赖、不禁知识)。
- 单位:内部 Hartree;写回 `frame.meta[kEnergy]` 为 eV(`core/units.h`)。

## Design
- `class ROHF : public MeanFieldMethod<ROHF>`(`rohf.h`/`rohf.cpp`),形态照抄 `UHF`:成员 `n_alpha`/`n_beta`、双构造(默认 + 显式计数)、`name()=="rohf"`、`resolve_occupation` 直取成员;构造默认镜像 UHF(Hcore 初猜、0.7 阻尼、DIIS 6 向量/第 5 步启动、tol 1e-8)。要求 `n_alpha ≥ n_beta`,违反即抛(无回退)。
- `SpinType` 增加 `RestrictedOpen`(现居 `scf_result.h`;02-rehome 落地后随迁 `scf_enums.h`);所有按 SpinType 分派的既有路径(gradient / hamiltonian_builder 等)对该值显式抛出而非静默走错分支。
- 新 stage 组件 `RohfCoupleStage`(header-only,`stages/rohf_couple_stage.h`,同 `DiagonalizeStage` 的命名引用接线风格):绑定于 `scf::Phase::diagonalize` 之前(`make_binding` 排序),读 F_α/F_β/S 与当前密度,按 Guest–Saunders 投影装配 F_eff 写入 α-Fock 通道;DIIS 误差改用单通道 F_eff·S·P_tot 对易子。
- `mean_field_method.h/.cpp` 的 `ensure_driver_built_` 接缝:RestrictedOpen 模式 = 非限制 Fock 构建 + 耦合 stage + 限制式对角化(单 MO 集,C_β=C_α)+ 分裂占据密度(P_α 取前 n_α 列、P_β 取前 n_β 列);显式实例化 `MeanFieldMethod<ROHF>`。`forces()`/`mp2()`/`fci()` 对 ROHF 抛出(见 Out of scope)。SlaterDeterminant 成员形态跟随 scf-slater-determinant-03-unbag 落地后的基类,不另起炉灶。
- `methods.h` 注册 `rohf.h`;禁止任何 `build_*`/`make_*` 自由函数;公有符号全部带 Doxygen(`@brief`/`@param` 含单位/`@return`),类文档记录正则化约定与耦合系数陷阱。
- 定标几何(oracle 钉定必须用同一几何,写入测试文件头):O₂ 三重态 R=1.2075 Å(n_α,n_β)=(9,7);CH₃ 双重态平面 D₃ₕ R(C–H)=1.079 Å(5,4);基组 STO-3G。

### 实现审计注记(2026-07-21)
- **PySCF 版本**:spec 写 2.13.1,oracle venv(`~/work/.x86_64`)实际为 **2.14.0**;钉定值以 2.14.0 生成并逐字记录(测试文件头 + REFERENCE.md 均注明)。
- **SpinType 现居 `scf_enums.h`**(02-rehome 已落地,`scf_result.h` 已删除);spec 文件清单里的 `scf_result.h` 改为 `scf_enums.h`。
- **RohfCoupleStage 绑定位置**:`scf::Phase::mix` @ order −1(在 MixStage 之前,因而也在 diagonalize 之前)——DIIS 必须看到 F_eff 才能用单通道 F_eff·S·P_tot 对易子;绑定在 diagonalize 相位内做不到这一点。
- **能量泛函读真实自旋 Fock**:couple stage 把 F_α/F_β 先保存到 `F_alpha_uncoupled_`/`F_beta_uncoupled_` 再覆写 F_eff;DensityStage 的 HF 能量分支接线到保存副本(F_eff 与 F_σ 在收敛点也不相等,直接读会永久偏差)。
- **额外显式拒绝**(spec 之外、同一"不静默走错分支"原则):ROHF + `soscf` 抛出(CIAH Hessian 无耦合项)、ROHF + `level_shift` 抛出(restricted 分支的 0.5·P_tot 不是开壳占据投影子)、ROHF 跳过 UHF 的 HOMO-LUMO 对称破缺初猜扰动。
- **UHF 对照的初猜**:本引擎 UHF 在 O₂ 上用 Hcore/MINAO 冷初猜收敛到更高的局部解(−147.3786 Ha,污染 1.27e-2);回归件与单测用收敛 ROHF 密度热启动 UHF(constrained-UHF 图像),落到 PySCF 同一全局解 −147.633946785450 Ha、污染 3.410858e-3。
- **文档顺带修正**:`docs/roadmap/hartree-fock.md` 与 `docs/roadmap/index.md` 中"ROHF does not exist"的过时陈述随本链节更新。

## Files to create or modify
- include/atomiverse/cpu/scf/rohf.h (new)
- src/cpu/scf/rohf.cpp (new)
- include/atomiverse/cpu/scf/stages/rohf_couple_stage.h (new)
- include/atomiverse/cpu/scf/scf_enums.h(SpinType 现居于此;spec 原文的 scf_result.h 已被 02-rehome 删除)
- include/atomiverse/cpu/scf/mean_field_method.h
- src/cpu/scf/mean_field_method.cpp
- include/atomiverse/cpu/scf/methods.h
- CMakeLists.txt(根;src 列表加 rohf.cpp)
- tests/cpu/scf/test_rohf_solver.cpp (new)
- tests/CMakeLists.txt
- regressions/scf/o2_rohf_sto3g/reg_o2_rohf_sto3g.cpp (new)
- regressions/scf/o2_rohf_sto3g/REFERENCE.md (new)
- regressions/CMakeLists.txt
- docs/tutorial/ch03-open-shell.md

## Tasks
- [x] Write failing tests for ROHF(tests/cpu/scf/test_rohf_solver.cpp,tests/CMakeLists.txt 注册 test_cpu_scf_rohf_solver):外部运行 PySCF 2.14.0(见审计注记)钉定 O₂/CH₃ 的 E_ROHF 与(Guest–Saunders 约定)轨道能,数值与几何写入文件头注释;含闭壳退化(H₂:ROHF(1,1) ≡ RHF 到 1e-8 Ha)、自旋纯性、n_α<n_β 抛出用例
- [x] Implement RohfCoupleStage in include/atomiverse/cpu/scf/stages/rohf_couple_stage.h(PySCF get_roothaan_fock 的 Guest–Saunders 投影装配 + F_eff DIIS 误差)
- [x] Implement SpinType::RestrictedOpen seam in scf_enums.h + mean_field_method.h/.cpp(接线耦合 stage、限制式对角化、分裂占据密度、`MeanFieldMethod<ROHF>` 显式实例化、gradient/mp2/fci 显式抛出)
- [x] Implement ROHF in include/atomiverse/cpu/scf/rohf.h + src/cpu/scf/rohf.cpp,并在 methods.h 与根 CMakeLists.txt 注册
- [x] Add docstring per Doxygen with units:rohf.h 类文档记录 PySCF/Guest–Saunders 正则化约定与"同能量、不同轨道能"陷阱
- [x] Verify against literature via regression regressions/scf/o2_rohf_sto3g(REFERENCE.md 引 Roothaan 1960 DOI + 钉定 PySCF 值与命令;≤60 s;regressions/CMakeLists.txt 注册)
- [x] Extend docs/tutorial/ch03-open-shell.md ROHF 一节(UHF 自旋污染回顾 → ROHF 自旋纯替代 → 耦合系数歧义如实讲清 → 何时选谁;O₂/CH₃ 上手;章末 PySCF 表行 E_total、⟨S²⟩,表格标量、无柱状图)
- [x] Run full check + test suite(login 节点 CPU 配方,-DATV_BUILD_HF=ON;含 no_factory_scrub 与 stage-typing scrub)

## Testing strategy
- 快乐路径(单元,秒级):O₂(9,7)/CH₃(5,4) STO-3G 收敛,`converged` 为真,总能量 vs 钉定 PySCF 到 1e-8 Ha;轨道能仅在 PySCF 约定下比较(1e-6 Ha)。
- 边界:n_α<n_β 抛 `std::invalid_argument`;ROHF(1,1) 于 H₂ 退化到 RHF 能量(1e-8 Ha);`forces()`/`mp2()` 对 ROHF 抛出;移动后重接线(沿用既有 wired_self_ 机制)不悬垂。
- 领域验证(science.required):自旋纯性 |⟨S²⟩−S(S+1)| ≤ 1e-10(O₂ S=1→2.0,CH₃ S=½→0.75),同几何 UHF 的 O₂ 污染 >1e-3 形成对照;回归件 o2_rohf_sto3g 出库消费公共头、≤60 s。
- 单元测试全部为秒级小体系;外部引擎只做离线钉数,构建/测试期零第三方读取。

## Out of scope
- ROHF 解析梯度/力(需含耦合项的 Lagrangian,`forces()` 显式抛出;后续链节)、ROHF-MP2 / ROKS(开壳 DFT)、CUDA 路径、非 PySCF 约定(Roothaan 原始 / McWeeny–Diercksen)的轨道能验收、稳定性分析联动(链节 08 已交付,仅在教程中引用)。
