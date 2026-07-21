# Reference — H2O RHF/6-31G* dipole moment

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
