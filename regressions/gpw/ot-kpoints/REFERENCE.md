# Reference — multi-k GPW OT (Monkhorst–Pack / explicit k)

## Method

Closed-shell **GPW** Kohn–Sham DFT with **Orbital Transformation (OT)** SCF
on the **complex multi-k** path (`use_ot = true` + `k_mesh` / explicit
`k_points`). Same OT manifold as Γ (VandeVondele & Hutter 2003; Kühne 2020
§VII C), Hermitian-generalized for Bloch \(H_k,S_k,C_k\).

Product API: concrete `cpu::gpw::GpwDft` members `use_ot`, `k_mesh`,
`k_points` / `k_point_weights`. No public `*Config`, no factories.

## System (regression)

| Item | Value |
| ---- | ----- |
| Molecule | H₂ (compact DZV-GTH-PADE) |
| Cell | **8 Bohr** cubic, periodic |
| Bond | ±0.7 Bohr from cell centre along *x* |
| XC | PADE |
| Cutoff | **100 Ry**, single grid |
| k | A2: explicit κ = (¼,0,0), w=1; A3: MP 2×1×1 TR |
| n_occ | 1 |

## Offline CP2K oracle

- **Input:** `tools/cp2k_oracle/inputs/ot-kpoints/h2_ot_kpoints.inp`
- **Status (2026-07-26):** no CP2K module/binary on the worker pin host
  (`module avail cp2k` empty). Absolute CP2K digits **not invented**.
- **Pin strategy:** internal exactness identities (OT vs DIIS multi-k;
  explicit-Γ OT vs real Γ OT) are decisive. Replace A5 frozen band with a
  cited CP2K OT+KPOINTS energy when offline CP2K is available.

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| -------- | ----: |
| multi-k OT E (κ=¼, 100 Ry) | **−1.1080204833 Ha** (13 OT iters) |
| multi-k DIIS same | −1.1080204833 Ha (5 DIIS iters) |
| \|OT − DIIS\| | 6.2×10⁻¹¹ Ha |
| real Γ OT | −1.1123413990 Ha |
| k-path Γ OT | −1.1123413989 Ha |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | real Γ OT ≡ k-path Γ OT | 1e-6 Ha |
| A2 | multi-k OT ≡ multi-k DIIS (κ=¼) | 1e-4 Ha |
| A3 | MP 2×1×1 OT ≡ DIIS | 1e-4 Ha |
| A4 | multi-k UKS OT builds (product in `reg_ot-uks-kpoints`) | — |
| A5 | absolute vs frozen first-green | 1e-3 Ha |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on pin host for fair wall twin. |
| r = T_atv/T_cp2k | n/a |
| `within_10x` | n/a (exempt) |

**ctest never launches CP2K.**
