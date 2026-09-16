# Reference — H₂O RHF/STO-3G external-field forces

Oracle protocol: analytic-vs-finite-difference of the **field-SCF energy**.
The field ENERGY is externally validated elsewhere (the −dE/dF ≡ μ identity vs
PySCF in `regressions/scf/h2o_embedding_field`), so the force — being the
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

## 物理与实现

场 1e 算符 V_field = Σ_a F_a⟨μ|(r−R₀)_a|ν⟩ 折进 Hcore,经典核场能
−Σ_A Z_A F·(R_A−R₀) 进 v_nuc(R₀ = 核电荷中心)。dE_field/dR_A 加两项显式导数
(密度响应已在 P/W):对 `ExternalField::core_matrix` 与 `nuclear_energy` 中心差分
(半数值,同 embed;位移原子 A 时其 shells 与 R₀ 一起重建)。中性分子场能
与原点无关,故力良定义。C-PCM(自洽面电荷,密度相关)力仍是后续链节。

文献:Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* §10
(分子性质 / 场响应)。
