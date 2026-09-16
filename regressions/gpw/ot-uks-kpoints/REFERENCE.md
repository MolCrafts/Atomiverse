# Reference — multi-k UKS GPW OT

## Method

Collinear **UKS GPW** with **Orbital Transformation (OT)** on the **complex
multi-k** path (`use_ot = true` + `n_alpha`/`n_beta` + `k_mesh` / explicit
`k_points`). Dual spin manifolds with open-shell occupation 1; joint PR+ CG
and Armijo on the rebuilt total energy (same fixed point as multi-k UKS DIIS).

Product API: `cpu::gpw::GpwDft` members `use_ot`, `n_alpha`/`n_beta`,
`k_mesh` / `k_points`+`k_point_weights`. No public `*Config`, no factories.

OT: VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003); Kühne et al.,
*J. Chem. Phys.* **152**, 194103 (2020) §VII C.

## System (regression)

| Item | Value |
| ---- | ----- |
| Molecule | high-spin H₂ (compact DZV-GTH-PADE) |
| Cell | **8 Bohr** cubic, periodic |
| Bond | ±0.7 Bohr from cell centre along *x* |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) |
| XC | PADE LSD |
| Cutoff | **100 Ry**, single grid |
| k | A2: κ = (¼,0,0), w=1; A3: MP 2×1×1 TR |

## Offline CP2K oracle

- **Input:** `tools/cp2k_oracle/inputs/ot-uks-kpoints/h2_uks_ot_kpoints.inp`
  (DIAG multi-k UKS documentation twin).
- **Engine:** CP2K **2026.1** (`…/work/cp2k/install/bin/cp2k.psmp`).
- **OT+KPOINTS:** CP2K aborts at `qs_scf_initialization`
  (`OT not possible with kpoint calculations`). There is **no** CP2K OT+k
  energy twin on this release — ATV multi-k UKS OT is a capability beyond the
  offline oracle’s OT surface; absolute energy uses the frozen first-green
  band + OT↔DIIS exactness.
- **DIAG twin:** compact high-spin H₂ multi-k UKS DIAG did not converge
  reliably (outer-SCF energy oscillation); not used as a digit pin.
- **Pin strategy:** internal exactness identities are decisive (same pattern
  as `reg_ot-kpoints`).

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| -------- | ----: |
| multi-k UKS OT E (κ=¼, 100 Ry) | **−0.7061343842 Ha** (11 OT iters) |
| multi-k UKS DIIS same | −0.7061343842 Ha (6 DIIS iters) |
| \|OT − DIIS\| | 4.8×10⁻¹² Ha |
| real Γ UKS OT | −0.6687052609 Ha |
| k-path Γ UKS OT | −0.6687052608 Ha |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | real Γ UKS OT ≡ k-path Γ UKS OT | 1e-6 Ha |
| A2 | multi-k UKS OT ≡ multi-k UKS DIIS (κ=¼) | 1e-4 Ha |
| A3 | MP 2×1×1 OT ≡ DIIS | 1e-4 Ha |
| A4 | absolute vs frozen first-green | 1e-3 Ha |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | CP2K 2026.1 rejects OT+KPOINTS; no fair OT wall twin. Identity suite only. |
| r = T_atv/T_cp2k | n/a |
| `within_10x` | n/a (exempt) |

**ctest never launches CP2K.**
