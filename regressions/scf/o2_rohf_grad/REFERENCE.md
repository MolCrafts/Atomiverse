# Reference — O₂ triplet ROHF/STO-3G nuclear gradient

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
Energy sibling: regressions/scf/o2_rohf_sto3g (same O₂ geometry/basis).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0); O(0, 0, 1.2075) |
| 基组 | STO-3G (Cartesian) |
| 自旋 | triplet, ROHF(9,7) — ⟨S²⟩ = 2 exactly |
| 收敛 | `conv_tol = 1e-12`, ROHF |
| PySCF 版本 | 2.14.0 |
| 命令 | `gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2, cart=True)`; `mf=scf.ROHF(mol); mf.conv_tol=1e-12; mf.kernel()`; `g=mf.nuc_grad_method().kernel()` |
| **E_ROHF** | **−147.6321669906824 Ha** (matches the energy sibling) |
| 对照容差 | 元素级 1e-5 Ha/Bohr;平移不变性 1e-8 Ha/Bohr |

## Pinned analytic gradient dE/dR [Ha/Bohr]

`mf.nuc_grad_method().kernel()`. Homonuclear diatomic along z → equal-and-opposite.

| 原子 | ∂E/∂z |
|---|---|
| O (0) | +2.607023240746e-02 |
| O (1) | −2.607023240746e-02 |

x/y components ~1e-16. Engine reproduces this to **5.8e-11 Ha/Bohr**.

## 为什么这是真正的 ROHF 判别测试

ROHF 梯度复用 unrestricted 的 J−K 导数收缩(共享空间轨道的 α/β 密度),唯一
ROHF 特有项是能量加权密度 W。**朴素的 W = Σ εᵢ CᵢCᵢᵀ 对 ROHF 错 ~7×**
(本几何:0.188 vs 0.0261 Ha/Bohr)——因为 effective-Fock 本征值是约定相关的,
不是物理 Lagrangian。正确构造(PySCF `grad/rohf.py` `make_rdm1e`):

    W = O_a·F_a·O_a + O_b·F_b·O_b

F_a/F_b 为收敛密度下的**自旋 Fock 矩阵**(非 effective Fock),O_a/O_b 为
occupied-α / doubly-occupied MO 投影算符。本测试对 W 的错误约定极度敏感
(错则 ≥ 1e-1),1e-5 阈值干净判别。

文献:PySCF `pyscf/grad/rohf.py`(`make_rdm1e`,`grad_elec = uhf_grad.grad_elec`);
Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960)(ROHF effective Fock);
Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* §10.
