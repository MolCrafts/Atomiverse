# scf/properties

One case; every section below is run by `atv_regressions --case scf/properties` and keeps its own oracle record.


## Reference — H2O RHF/6-31G* dipole moment

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | 6-31G* (cartesian-6d, the set's defining convention) |
| 收敛 | `conv_tol = 1e-12`, RHF; E = −76.01050498825705 Ha |
| PySCF 版本 | 2.14.0 |
| 命令 | `mf = scf.RHF(mol); mf.kernel(); mf.dip_moment(unit='AU')` |
| **μ (a.u.)** | **(0, 0, −0.8757858707840785)** |
| μ (Debye) | (0, 0, −2.2260255980254606) |
| 对照容差 | 1e-6 a.u. 逐分量 |

物理背景:HF 系统性高估水的偶极(≈2.23 D vs 实验 1.85 D,Szabo & Ostlund
properties 一节)— 本回归是同理论级 engine-vs-engine 对照,不是对实验。
积分内核:移位中心闭式(Obara & Saika, *J. Chem. Phys.* **84**, 3963 (1986),
DOI 10.1063/1.450106);原点约定 = 核电荷中心,随数值一并报告。


## Reference — H2O RHF/6-31G* Löwdin + MK/CHELPG ESP 电荷

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md(外部运行、钉定);
本用例的钉定脚本:`tools/pin_charges_oracle.py`(只在外部 venv 里跑,
绝不进 CMake/ctest)。

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | 6-31G* (cartesian-6d) |
| 收敛 | `conv_tol = 1e-12`, RHF; E = −76.010504988257 Ha |
| PySCF 版本 | 2.14.0 |
| Löwdin 命令 | `S=mf.get_ovlp(); P=mf.make_rdm1();` 先单位归一化 AO(S̄=DSD, P̄=D⁻¹PD⁻¹, D=diag(1/√S_ii)),再 `S12=U√s Uᵀ; q=Z−diag(S12·P̄·S12)` 按原子聚合 |
| **Löwdin q (e)** | **O −0.7284015337; H +0.3642007669; H +0.3642007669** |
| ESP 神谕 | PySCF `mol.intor('int1e_grids')` 密度 ESP,拟合在 Atomiverse 文档化格点上(见下)|
| **MK q (e)** | **O −0.8070921421; H₁ +0.4040081384; H₂ +0.4030840037**(364 点)|
| **CHELPG q (e)** | **O −0.8050061239; H₁ +0.4020088594; H₂ +0.4029972644**(8485 点)|
| 对照容差 | Löwdin ≤1e-6 e;MK/CHELPG ≤1e-3 e;Σq ≤1e-10;两 H 差 ≤2e-3 e |

### 格点约定(两侧逐字相同;esp_charges.h 里的正式定义)

- **MK**:每原子、每壳层因子 s ∈ {1.4, 1.6, 1.8, 2.0},半径 r = s·r_Bondi,
  黄金角 Fibonacci 球面 N = ceil(1.0·4πr²) 点;点落入任一其它原子的同因子
  壳层内(|p−R_B| < s·r_Bondi(B))则弃。
- **CHELPG**:0.3 Å 立方格点,包围盒每轴外扩 2.8 Å;保留"处处在 vdW 面外
  且距至少一个原子 vdW 面 ≤2.8 Å"的点。
- Bondi 半径 (1964):H 1.20、C 1.70、N 1.55、O 1.52、F 1.47 等(静态表,
  不支持元素抛异常)。
- 格点两侧都以 Å 定义(单位漂移为零);ESP 与拟合在原子单位进行。

### Löwdin 的 AO 归一化约定

Löwdin 布居对逐 AO 缩放**不**不变(Mulliken/ESP 则不变),而笛卡尔 d 函数
的工程归一化各码不同:PySCF cart 的裸 libcint d AO 自重叠为 2.513/0.838,
Atomiverse 引擎为 3/1。两侧都先把每个 AO 单位归一(S_ii = 1)再做对称正交
化,数值即与工程约定无关 —— 上表钉的就是这个约定下的值(两侧同一步骤,
lowdin.cpp 与 pin 脚本中都有文档)。

### 已知偏差(有意为之,已文档化)

确定性格点不精确镜像对称(Fibonacci 螺旋相位、格点原点相对分子对称面偏
移),故对称 H 的拟合电荷只在格点离散化水平(~1e-3 e)相等 —— PySCF 神谕
在同一格点上给出同样的不对称,两侧逐点对照仍到 ≤1e-3 e。

PySCF 无内置 MK/CHELPG:神谕 = PySCF 密度 ESP + 本仓库文档化格点与约束
LSQ 的 numpy 复刻(命令与算法见 tools/pin_charges_oracle.py)。

物理背景:HF/6-31G* 类的 CHELPG 水氧电荷 ≈ −0.74…−0.8 e(Breneman &
Wiberg 1990 表 III 一带);MK 类似(Besler, Merz, Kollman 1990)。Löwdin
比 Mulliken 对基组更稳,但两者都是"轨道空间划分",与 ESP 拟合电荷
(再现分子外静电势的有效点电荷)是不同的可观测目标。

引文:
- Löwdin, *J. Chem. Phys.* **18**, 365 (1950), DOI 10.1063/1.1747632。
- Besler, Merz, Kollman, *J. Comput. Chem.* **11**, 431 (1990),
  DOI 10.1002/jcc.540110404(MK 嵌套 vdW 壳层)。
- Breneman, Wiberg, *J. Comput. Chem.* **11**, 361 (1990),
  DOI 10.1002/jcc.540110311(CHELPG 立方格点)。
- Bondi, *J. Phys. Chem.* **68**, 441 (1964)(vdW 半径表)。


## Reference: `wannier-centers`

### Method

Foster–Boys localization (Foster & Boys, *Rev. Mod. Phys.* **32**, 300
(1960)) of occupied closed-shell RHF/STO-3G MOs via Jacobi pair rotations.
Public API: `cpu::scf::WannierCenters::compute(MeanFieldOutputs, Frame)`.

### Pins (no CP2K binary)

| System | Check | Source |
| ------ | ----- | ------ |
| H₂ (0.74 Å) | 1 center at bond midpoint \(z=0.37\) Å (±1e-4) | unit algebra / symmetry of \(\langle z\rangle\) for \(\sigma_g\) |
| H₂O | 5 centers; \(\Omega_\mathrm{tot}\) decreases vs canonical | Boys functional definition |
| H₂O | classic Boys geometry: O-core, 2 OH-bond, 2 rabbit-ear LP | textbook MLWF of water (e.g. Marzari et al. reviews; PySCF/CP2K qualitative) |
| H₂O | \(\sum_i x_i=\sum_i y_i=0\) (C₂ᵥ) | symmetry |

CP2K `LOCALIZE` numerical centers may be collected offline when a binary is
available; this regression is **`perf_exempt`** and does not require it.

### Geometry

H₂O same as `h2o_rhf_dipole` (Å):

```
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
```


## Reference — H₂O RHF/STO-3G static polarizability (CPHF)

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
No live PySCF at ctest time.

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G (Cartesian; pure s/p — identical to spherical for this set) |
| Method | RHF, `conv_tol = 1e-14` |
| E₀ | −74.96302313846286 Ha |
| PySCF version | 2.14.0 |
| Observable | Static dipole polarizability α_ab = ∂μ_a/∂F_b [a.u.] |
| Origin | Center of nuclear charge (origin = 0 for this geometry) |
| Oracle path | Analytic CPHF on PySCF MOs/ERIs: ¹(A+B)·U^{F_b} = −r_b^{vo}, α_ab = −4 Σ_ai U^b_ai ⟨a\|r_a\|i⟩; cross-checked by finite-field dμ/dF and −∂²E/∂F² (h = 2×10⁻⁴ a.u., h1 = F·⟨r⟩ matching Atomiverse `ExternalField`) |
| **α_xx** | **0.04190863** a.u. |
| **α_yy** | **5.20973255** a.u. |
| **α_zz** | **2.12715820** a.u. |
| Off-diagonal | ~0 (C₂v molecular plane yz) |
| Gate | 1×10⁻⁵ a.u. per diagonal component |

### Pin commands (offline; never in ctest)

```bash
bash tools/setup_pyscf_env.sh
## Analytic CPHF (PySCF MOs + ao2mo ERI) and finite-field dμ/dF with the
## repository water geometry; see agent pin log 2026-07-23.
```

### Theory notes

- Closed-shell CPHF electric-field response: S^{(F)} = 0; skeleton Fock
  derivative is the length-gauge dipole matrix (electron charge −1 a.u.
  ⇒ h += F·r). Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure
  Theory*, response chapter; Pople et al. for the nuclear CPHF twin used
  by `Hessian<RHF>`.
- STO-3G lacks diffuse / polarization functions — α_xx (out of plane) is
  unphysically small (~0.04 a.u.). The regression is engine-vs-engine at
  identical theory level, not a comparison to experiment (~9–10 a.u.
  isotropic for water).
