# Reference — I₂ RHF/def2-SVP+def2-ECP nuclear gradient

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
Energy sibling: regressions/scf/ecp_heavy_atom (same I₂ geometry/basis/ECP).

| 字段 | 值 |
|---|---|
| 几何 (Å) | I(0, 0, 0); I(0, 0, 2.666) |
| 基组 / ECP | def2-SVP (spherical, 5d/7f) / def2-ECP (I ECP28MWB, 50 价电子) |
| 收敛 | `conv_tol = 1e-12`, RHF |
| PySCF 版本 | 2.14.0 |
| 命令 | `gto.M(atom="I 0 0 0; I 0 0 2.666", basis="def2-svp", ecp="def2-svp", cart=False)`; `mf=scf.RHF(mol); mf.conv_tol=1e-12; mf.kernel()`; `g=mf.nuc_grad_method().kernel()` |
| **E_RHF** | **−593.3161818794325 Ha** (matches the energy sibling) |
| 对照容差 | 元素级 1e-5 Ha/Bohr;平移不变性 1e-8 Ha/Bohr |

## Pinned analytic gradient dE/dR [Ha/Bohr]

`mf.nuc_grad_method().kernel()` (PySCF has **no** `kick_start`; the entry point
is `.kernel()`). Homonuclear diatomic along z → equal-and-opposite, x/y ~1e-16.

| 原子 | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| I (0) | 0 | 0 | +5.752712116941e-03 |
| I (1) | 0 | 0 | −5.752712116951e-03 |

Σ over atoms ≈ 1.1e-14 Ha/Bohr(平移不变)。`max|∂E/∂R| = 5.752712e-03`。

## 为什么 1e-5 阈值 + 半数值 dU/dR 说明

Atomiverse 的 ECP 梯度是**半数值**的:除 ECP 单电子块 dU/dR 外全部解析;
dU/dR 用中心差分(h=1e-4 Bohr)对 ECP 矩阵求导——U 是每几何构建一次的廉价
单电子矩阵(非 O(N⁴)),且引擎的 U 与 PySCF `ECPscalar` 一致到 ~3e-13,故中心
差分误差(截断 O(h²) + 舍入)约 1e-8,远低于 1e-5 阈值。ECP 之外的解析项
(dS/dT/dV_ne/dERI,均已用 Z_val)与 PySCF 解析梯度一致到 ~1e-9。完整解析
dU/dR 递推是未来升级(见 spec `ponytail:` ceiling)。

文献:Kahn, Baybutt, Truhlar, *J. Chem. Phys.* **65**, 3826 (1976),
DOI 10.1063/1.432900(半定域 ECP 形式);McMurchie, Davidson,
*J. Comput. Phys.* **44**, 289 (1981), DOI 10.1016/0021-9991(81)90053-X
(type-1/2 积分);Weigend & Ahlrichs, *PCCP* **7**, 3297 (2005)
(def2-ECP 谱系);PySCF `pyscf/grad/rhf.py`(数值 oracle)。
