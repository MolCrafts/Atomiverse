---
title: "HF capability gaps 04 — 布居/ESP 电荷与电离能"
slug: hf-capability-gaps-04-charges
status: code-complete
chain: hf-capability-gaps
link: 04/21
external-depends: scf-slater-determinant-03-unbag
created: 2026-07-20
---

# HF 能力补全 04 — 布居/ESP 电荷与电离能

## Summary
为 Gaussian-basis 平均场方法补上电荷与电离能分析层:新增 Löwdin 布居观察者与 ESP 拟合电荷(MK 嵌套 vdW 壳层与 CHELPG 立方格点两种采样),扩展 OrbitalInfoObserver 报告 Koopmans 电离能 IP = −ε_HOMO,并以现有方法成员(`charge`、`UHF::n_alpha/n_beta`)完成 ΔSCF IP/EA 工作流(非 Aufbau/MOM 占据控制的标准入口是 `SlaterDeterminant::with_occupation`,scf-slater-determinant-01-type 已定;本环的基态阳/阴离子 ΔSCF 只需既有成员)。所有观察者消费 `MeanFieldOutputs`(自 scf-slater-determinant-03-unbag 起其 `result` 为 `shared_ptr<const SlaterDeterminant>`),不新增任何 Config 包,不触 `IntegralPack`/`SCFResult`。教程侧扩写 ch02(Koopmans/ΔSCF 上手)并新增 ch20"布居分析与 ESP 电荷"一节。

## Domain basis
- Löwdin 布居:q_A = Z_A − Σ_{μ∈A}(S^{1/2} P S^{1/2})_{μμ}。Löwdin, J. Chem. Phys. 18, 365 (1950), DOI 10.1063/1.1747632。S^{1/2} 由对称本征分解构造。
- MK ESP 电荷:静电势采样于 1.4/1.6/1.8/2.0 × vdW 半径嵌套壳层;Besler, Merz, Kollman, J. Comput. Chem. 11, 431 (1990), DOI 10.1002/jcc.540110404。
- CHELPG:0.3 Å 立方格点,vdW 面外 0–2.8 Å 带内取点;Breneman, Wiberg, J. Comput. Chem. 11, 361 (1990), DOI 10.1002/jcc.540110311。
- 拟合为带 Lagrange 乘子的约束最小二乘(Σq = q_tot);V(r) = Σ_A Z_A/|r−R_A| − Σ_{μν} P_μν ⟨μ|1/|r−r'||ν⟩,电子项复用 `native::nuclear_block`(单位点电荷置于格点)。已知陷阱:埋藏原子病态(用 SVD/最小范数解并检验条件数)、对称 H 原子电荷必须相等。
- Koopmans 定理 IP ≈ −ε_HOMO:Koopmans, Physica 1, 104 (1934), DOI 10.1016/S0031-8914(34)90011-2;Szabo & Ostlund §3.3。Koopmans EA 是基组赝象,教程必须明示改用 ΔSCF。
- 神谕:N₂ HF −ε_HOMO ≈ 17.3 eV(实验 IP 15.6 eV,Szabo-Ostlund §3.3 讨论);H₂O Löwdin 对 PySCF 到 1e-6 e;CHELPG/MK 对 PySCF 锚定值。单位:电荷 e、能量报告 eV(内部 Hartree)、长度 Å。

## Design
- `cpu/observer/lowdin.h`:无状态自由函数 `lowdin_charge(const scf::MeanFieldOutputs&)` / `lowdin_spin(...)`,返回按原子序的 `std::vector<double>` [e]。密度按自旋道取自 `MeanFieldOutputs` 的 `SlaterDeterminant`(P = P^α + P^β);AO→原子映射与 S 由 `shells` 内部重建(观察者自建视图;公开签名不出现 `IntegralPack`/`SCFResult`,不扩债)。
- `cpu/observer/esp_charges.h`:两个具体类 `MKCharges` / `ChelpgCharges`(构造函数携带格点参数成员——壳层因子/格距/带宽,默认即文献值;非工厂、非 Config 包),成员函数 `fit(const scf::MeanFieldOutputs&, const Frame&) -> std::vector<double>` [e];共享的格点静电势内核与约束 LSQ 为 `.cpp` 内部实现(同时为 link 15 点电荷嵌入铺路)。Bondi vdW 半径为文件内静态表,不支持的元素抛异常(无回退)。
- `OrbitalInfoObserver` 扩展:新增 `ip_koopmans_eV()`(= −HOMO)与 TSV 列 `ip_koopmans`;不新增类。03-unbag 之后其 HOMO/LUMO 取值处为 `determinant().homo()/lumo()`,本环扩展跟随该形状。
- ΔSCF:零新 API——`MeanFieldMethod::charge` 与 `UHF{n_alpha, n_beta}` 已是成员;本环节以回归与教程钉住"两次收敛 SCF 之差"的用法。非 Aufbau 占据(激发态/核穴 ΔSCF)出局,留待 `with_occupation` 消费环。
- 依赖:link 02(AO 维度)、link 03(格点势内核相邻;ch20 文件若已由 03 创建则为扩写);external-depends scf-slater-determinant-03-unbag(`MeanFieldOutputs::result` 的 SlaterDeterminant 形状)。

### 实现审计注记(2026-07-20,实现时落定)
- **签名偏差**:`lowdin_charge` / `lowdin_spin` 取
  `(const scf::MeanFieldOutputs&, const cpu::Frame&)` —— Frame 参数为 AO→原子
  映射与 Z_A 所必需(shell 中心按 Bohr、容差 1e-8 与 Frame 原子位置匹配;
  未匹配即抛)。规格原文只写了 `lowdin_charge(const scf::MeanFieldOutputs&)`。
- **Löwdin AO 归一化约定**:Löwdin 布居对逐 AO 缩放不不变,而笛卡尔 d 归一
  化各码不同(PySCF cart 裸 libcint d 自重叠 2.513/0.838,本引擎 3/1)。两侧
  统一为:先单位归一化每个 AO(S̄=DSD, P̄=D⁻¹PD⁻¹, D=diag(1/√S_ii))再对称
  正交化 —— 数值与工程约定无关;钉定值即该约定下的值。
- **对称 H 等价的容差**:确定性格点(Fibonacci 螺旋相位、立方格点原点)不
  严格镜像对称,MK/CHELPG 拟合电荷的两 H 差 ~1e-3 e(神谕在同一格点上给出
  同样的不对称;C++ 对钉定值逐点 ≤1e-9 e)。回归断言两 H 差 ≤2e-3 e,
  Löwdin 两 H 差 ≤1e-6。见 acceptance ac-004 的 note。
- **N₂ Koopmans 文献带**:17.3 eV 是 HF 极限值;6-31G* 下 −ε_HOMO =
  16.6467 eV。主断言钉在同基组 PySCF 值 ≤1e-3 eV,文献带作宽松 sanity
  (15.5–17.5 eV)。见 acceptance ac-005 的 note。

## Files to create or modify
- include/atomiverse/cpu/observer/lowdin.h (new)
- src/cpu/observer/lowdin.cpp (new)
- include/atomiverse/cpu/observer/esp_charges.h (new)
- src/cpu/observer/esp_charges.cpp (new)
- include/atomiverse/cpu/scf/orbital_info_observer.h
- src/cpu/scf/orbital_info_observer.cpp
- tests/cpu/observer/test_lowdin.cpp (new)
- tests/cpu/observer/test_esp_charges.cpp (new)
- tests/cpu/scf/test_orbital_info_observer.cpp
- tests/CMakeLists.txt
- CMakeLists.txt
- regressions/scf/h2o_charges_pyscf/reg_h2o_charges_pyscf.cpp (new)
- regressions/scf/h2o_charges_pyscf/REFERENCE.md (new)
- regressions/scf/n2_ionization/reg_n2_ionization.cpp (new)
- regressions/scf/n2_ionization/REFERENCE.md (new)
- regressions/CMakeLists.txt
- docs/tutorial/ch02-rhf.md
- docs/tutorial/ch20-hf-properties-environment.md (new)

## Tasks
- [x] Write failing tests for Löwdin populations (tests/cpu/observer/test_lowdin.cpp):单 s 函数解析例 q = Z − P₀₀、双原子对称合成密度下两原子电荷相等、Σq 守恒;纯手工 shells+SlaterDeterminant,不跑 SCF
- [x] Implement lowdin_charge / lowdin_spin in include/atomiverse/cpu/observer/lowdin.h + src/cpu/observer/lowdin.cpp,消费 MeanFieldOutputs(SlaterDeterminant 自旋道密度),Doxygen 含单位 [e]
- [x] Write failing tests for ESP fitting (tests/cpu/observer/test_esp_charges.cpp):对 H₂O 几何摆放已知点电荷 (+0.4, −0.8, +0.4)、以解析库仑势喂给 MK 与 CHELPG 两套格点的约束 LSQ,须精确回收电荷、Σq 约束严格成立、对称 H 相等到 1e-8、埋藏原子病态例走最小范数解不发散;不跑 SCF
- [x] Implement MKCharges / ChelpgCharges in include/atomiverse/cpu/observer/esp_charges.h + src/cpu/observer/esp_charges.cpp(格点生成、nuclear_block 复用的势内核、Lagrange 约束 LSQ、Bondi 静态表)
- [x] Write failing test extension for ip_koopmans in tests/cpu/scf/test_orbital_info_observer.cpp(手工构造 SlaterDeterminant,校验 ip_koopmans_eV() == −homo_eV() 与 TSV 新列)
- [x] Implement ip_koopmans_eV() + TSV 列 in include/atomiverse/cpu/scf/orbital_info_observer.h + src/cpu/scf/orbital_info_observer.cpp(HOMO/LUMO 经 determinant() 访问器)
- [x] Add regressions:regressions/scf/h2o_charges_pyscf(H₂O RHF:Löwdin 对 PySCF 锚定值 ≤1e-6 e,CHELPG 与 MK 各对 PySCF ≤0.01 e,两 H 相等)与 regressions/scf/n2_ionization(N₂:−ε_HOMO 对 PySCF 锚定 ≤1e-3 eV 且落在 ≈17.3 eV 文献带,ΔSCF IP 仅经 charge/n_alpha/n_beta 成员完成)+ 各自 REFERENCE.md 引文;登记 regressions/CMakeLists.txt 与 tests/CMakeLists.txt
- [x] Extend docs/tutorial/ch02-rhf.md(Jensen 3.4 式 Koopmans/ΔSCF 动手小节,含"Koopmans EA 为基组赝象、用 ΔSCF"警示)并写 docs/tutorial/ch20-hf-properties-environment.md 之"布居分析与 ESP 电荷"节(章末 PySCF 对照表——标量进表格、禁柱状图;ESP 格点/电荷可视化经 molplot/molvis)
- [x] Run full check + test suite

## Testing strategy
- 快路径单元:Löwdin/ESP 全部走手工构造的 shells、SlaterDeterminant 与解析点电荷势——tests/ 内绝无 SCF 求解,整套秒级。
- 边界:Σq 约束的精确性(乘子行)、埋藏原子病态矩阵(SVD 截断后残差有限)、不支持元素的 vdW 查表抛异常、对称等价原子电荷相等。
- 领域验证(regressions/,≤60 s,引文见 REFERENCE.md):H₂O 电荷三法对 PySCF 锚定值;N₂ Koopmans 与 ΔSCF 电离能对 PySCF 锚定值及 Szabo-Ostlund 文献带。

## Out of scope
- 点电荷 QM/MM 嵌入(link 15,仅内核铺路)、RESP 双阶段拟合与拟合约束组、NPA/NBO、偶极/多极本身(link 03)、CUDA 侧观察者、AM1-BCC 通路改动。
- 非 Aufbau/MOM ΔSCF(`SlaterDeterminant::with_occupation` 的消费)——激发态/核穴 ΔSCF 留待后续。
