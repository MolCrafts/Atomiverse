# scf/embedding

One case; every section below is run by `atv_regressions --case scf/embedding` and keeps its own oracle record.


## Reference — H₂O RHF/STO-3G point-charge-embedding (QM/MM) forces

Oracle protocol: docs/tutorial/appendix-pyscf-oracle.md (external run, pinned).
Energy sibling: regressions/scf/embedding (same geometry/charges).

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

### Pinned analytic gradient on QM atoms dE/dR [Ha/Bohr]

| QM 原子 | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| O | 0 | 0 | −6.3053441384e-02 |
| H | 0 | −1.7882928801e-02 | +3.1195488026e-02 |
| H | 0 | +1.7882928801e-02 | +3.1195488026e-02 |

**No translational invariance:** the FIXED MM charges exert a net force on the
QM system, so Σ_QM ∂E/∂R ≠ 0 (PySCF Σ_QM,z ≈ 6.6e-4 Ha/Bohr; the balancing
force is on the MM sites, which qmmm's QM gradient does not report). Engine
reproduces the QM gradient to **5.4e-11 Ha/Bohr**.

### 物理与实现

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


## embed_h2o_mm_forces

### System

- H₂O RHF/STO-3G (Cartesian), same geometry as `embed_h2o_grad`
- Two external charges: `q = ±0.5 e` at `(0, 0, ±3)` Å
- `embed_self_energy = false` — PySCF `qmmm.mm_charge` energy does **not**
  include classical MM–MM, so the FD pin is the QM↔MM seam only

### Oracle

- **Code:** PySCF 2.14.0 `qmmm.mm_charge` + central FD of `mf.e_tot` w.r.t. MM
  coordinates (Δ = 1e-4 Bohr), offline capture only.
- **Recipe:** same charge layout as `examples/qmmm/30-force_on_mm_particles.py`
  (force-on-MM-particles pattern); numbers below are the other code’s output.

### Pin — ∂E/∂R_MM [Ha/Bohr]

| site | x | y | z |
| ---- | - | - | - |
| +0.5 @ z=+3 | 0 | 0 | `+4.053776692103e-03` |
| −0.5 @ z=−3 | 0 | 0 | `−3.391311764744e-03` |

Tolerance: element-wise ≤ `1e-5` Ha/Bohr.

### Also checked

- With `embed_self_energy = false`, `|Σ_QM F + Σ_MM F| ≤ 1e-9` eV/Å (Newton’s
  third law on the QM↔MM seam).


## Reference — H₂O RHF/STO-3G external-field forces

Oracle protocol: analytic-vs-finite-difference of the **field-SCF energy**.
The field ENERGY is externally validated elsewhere (the −dE/dF ≡ μ identity vs
PySCF in `regressions/scf/embedding`), so the force — being the
nuclear derivative of that validated energy — is correctly gated by comparing
the analytic force against a central finite difference of the same energy.
PySCF has no turnkey external-field gradient, so no external force pin exists.

| 字段 | 值 |
|---|---|
| 几何 (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| 基组 | STO-3G (Cartesian) |
| 外场 F (a.u.) | (0, 0, 0.01) |
| 收敛 | `conv_tol = 1e-12`, RHF |
| oracle | central FD of `rhf.energy()` (field-SCF), h = 1e-3 Bohr |
| 对照容差 | 元素级 1e-5 Ha/Bohr(FD 截断下限;引擎更准) |

实测 analytic-vs-FD 最大偏差 **1.5e-7 Ha/Bohr**。沿场方向 z 有非零力
(O_z≈−0.062, H_z≈+0.031 Ha/Bohr;外场极化/推动分子),x/y ~1e-11。

### 物理与实现

场 1e 算符 V_field = Σ_a F_a⟨μ|(r−R₀)_a|ν⟩ 折进 Hcore,经典核场能
−Σ_A Z_A F·(R_A−R₀) 进 v_nuc(R₀ = 核电荷中心)。dE_field/dR_A 加两项显式导数
(密度响应已在 P/W):对 `ExternalField::core_matrix` 与 `nuclear_energy` 中心差分
(半数值,同 embed;位移原子 A 时其 shells 与 R₀ 一起重建)。中性分子场能
与原点无关,故力良定义。C-PCM(自洽面电荷,密度相关)力仍是后续链节。

文献:Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* §10
(分子性质 / 场响应)。


## H₂O RHF/STO-3G — external field + point-charge embedding

### Geometry (Å)

Same H₂O as `h2o_rhf_dipole` / multipole oracles:

| Atom | x | y | z |
|------|---|---|---|
| O | 0.0000 | 0.0000 | 0.1173 |
| H | 0.0000 | 0.7572 | −0.4692 |
| H | 0.0000 | −0.7572 | −0.4692 |

### Theory

- RHF / STO-3G, `conv_tol = 1e-12`
- Finite field: F_z = ±1×10⁻⁴ a.u., origin = center of nuclear charge
  (for this geometry R₀ = 0)
- Point charges (Å, e): q₁ = +0.5 at (0, 0, 3.0), q₂ = −0.5 at (0, 0, −3.0)

### PySCF 2.14.0 pins (external oracle, not a runtime dependency)

```text
E0 (vacuum)              = -74.96302313846292  Ha
mu_z                     = -0.6787872654524012 a.u.
E_mm (qmmm.mm_charge)    = -74.98418931920534  Ha
  (QM energy + QM–MM nuclear + electron–MM; NO MM–MM self energy)
MM–MM classical          = q1*q2 / |R1-R2|
                         = (0.5)*(-0.5) / (6.0 * a0⁻¹)
                         = -0.022049050455259  Ha
                         (a0⁻¹ = 1.8897261245650618 Bohr/Å)
```

Atomiverse includes MM–MM in `v_nuc_`, so:

```text
E_atv (embed) = E_mm_pyscf + MM_MM
              = -75.0062383696606 Ha
```

Finite-field central difference:

```text
(E(+F) − E(−F)) / (2F)  ≃  −μ_z   within 1e-5 a.u.
```

Script: one-off external `python3` + PySCF (not run by ctest).
