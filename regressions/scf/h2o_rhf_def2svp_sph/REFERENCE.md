# Reference — H2O RHF/def2-SVP, spherical harmonics (5d)

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | def2-SVP, **spherical** (`cart=False`; the set's defining convention) |
| 收敛 | `conv_tol = 1e-12`, RHF |
| PySCF 版本 | 2.14.0 |
| 命令 | `gto.M(atom=..., unit='Angstrom', basis='def2-svp', cart=False)`; `scf.RHF(mol).kernel()` |
| **E_RHF** | **−75.96098398708 Ha** |
| 对照容差 | 1e-8 Ha |

对照锚点(同几何、同版本):cartesian(6d)运行 `cart=True` 给
E = −75.9622197549911 Ha,nao 25 → 24;两者之差 ~1.2 mHa 即 6d 的 s-污染
分量的变分贡献 — 教程 ch04「笛卡尔与球谐高斯基」一节引用本数据。

文献:Weigend & Ahlrichs, *PCCP* **7**, 3297 (2005), DOI 10.1039/B508541A
(def2-SVP 定义,球谐约定);Schlegel & Frisch, *Int. J. Quantum Chem.* **54**,
83 (1995), DOI 10.1002/qua.560540202(c2s 闭式系数)。
