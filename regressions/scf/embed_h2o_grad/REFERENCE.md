# Reference — H₂O RHF/STO-3G point-charge-embedding (QM/MM) forces

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
Energy sibling: regressions/scf/h2o_embedding_field (same geometry/charges).

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | STO-3G (Cartesian) |
| MM 点电荷 (Å, e) | (0,0,3.0, +0.5); (0,0,−3.0, −0.5) |
| 收敛 | `conv_tol = 1e-12`, RHF |
| PySCF 版本 | 2.14.0 |
| 命令 | `mf = qmmm.mm_charge(scf.RHF(mol), mm_coords, mm_charges, unit='Angstrom')`; `mf.conv_tol=1e-12; mf.kernel()`; `g = mf.nuc_grad_method().kernel()` |
| **E** | **−74.98418931920533 Ha** (matches the energy sibling, MM–MM excluded by PySCF) |
| 对照容差 | 元素级 1e-5 Ha/Bohr |

## Pinned analytic gradient on QM atoms dE/dR [Ha/Bohr]

| QM 原子 | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O | 0 | 0 | −6.3053441384e-02 |
| H | 0 | −1.7882928801e-02 | +3.1195488026e-02 |
| H | 0 | +1.7882928801e-02 | +3.1195488026e-02 |

**No translational invariance:** the FIXED MM charges exert a net force on the
QM system, so Σ_QM ∂E/∂R ≠ 0 (PySCF Σ_QM,z ≈ 6.6e-4 Ha/Bohr; the balancing
force is on the MM sites, which qmmm's QM gradient does not report). Engine
reproduces the QM gradient to **5.4e-11 Ha/Bohr**.

## 物理与实现

嵌入 1e 算符 V_embed = Σ_C −q_C⟨μ|1/|r−R_C||ν⟩ 在 SCF 前折进 Hcore(密度无关,
同 ECP U),经典核–MM 库仑 Σ_A Σ_C Z_A q_C/|R_A−R_C| 进 v_nuc。梯度加两个显式导数
(密度响应已在收敛的 P/W 里):
- 电子–MM:Tr[P·dV_embed/dR_A](对 `PointChargeEmbedding::core_matrix` 中心差分,
  半数值,同 ECP 模式,h=1e-4 Bohr);
- 核–MM:−Σ_C Z_A q_C (R_A−R_C)/|R_A−R_C|³(解析)。
MM–MM 对 QM 核导数为 0。外场力见 `field_h2o_grad`；C-PCM 自由能力见
`pcm-grad-h2o`（真空 HF 项 + 固定密度下 ΔG_el 的半数值导数）。

文献:Warshel & Levitt, *J. Mol. Biol.* **103**, 227 (1976),
DOI 10.1016/0022-2836(76)90311-9(QM/MM 静电嵌入);PySCF `pyscf/qmmm`
(数值 oracle)。
