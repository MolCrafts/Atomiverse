# Multi-k UKS GPW Hirshfeld-scaled MBD@rsSCS — O₂ / PBE / DZVP-MOLOPT-SR-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-hirshfeld-uks-kpoints` |
| Regression | `regressions/gpw/gpw-hirshfeld-uks-kpoints/` |
| Public API | `cpu::gpw::GpwDft` / `GPWEvaluator`: `n_alpha`/`n_beta`, `k_mesh`, `mbd`, `hirshfeld`; `GpwResult::e_disp` [Ha], `volume_ratios` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion — multi-k UKS Hirshfeld |
| Queue | CP2K capability loop (`docs/developer/cp2k-method-queue.md`) |

## Summary

**Hirshfeld volume ratios** from the final **multi-k collinear UKS** GPW PBE
total density scale free-atom **MBD@rsSCS** (Tkatchenko–Scheffler family).
Product path: `fold_dispersion_k_uks` — k-summed
\(\rho=\sum_k w_k(\rho_k^\alpha+\rho_k^\beta)\) → in-tree Hirshfeld kernel →
scaled MBD. Pin system: **triplet O₂** on a 10 Å periodic cube with
Monkhorst–Pack **2×2×1** (TR → 2 k).

Public path: construct `cpu::gpw::GpwDft` with lookup constructors
`GaussianBasis` / `GthPotential`, set members (`n_alpha`, `n_beta`, `mbd`,
`hirshfeld`, `k_mesh`, …), `build` → `run`. **No** public `*Config` bag and
**no** `make_*` / `build_*` factories. Product gate:
`reg_gpw-hirshfeld-uks-kpoints`. Wall is **perf_exempt** (E_disp pin; stock
CP2K has no TS/MBD twin).

Related: free-atom MBD (`gpw-mbd-ice`), Γ Hirshfeld RKS
(`gpw-mbd-hirshfeld`), multi-k RKS Hirshfeld (`gpw-hirshfeld-kpoints`),
Γ-UKS Hirshfeld (`gpw-hirshfeld-uks`).

## Method

**GPW host.** Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997).

**Multi-k UKS.** Dual complex Bloch densities \(P_k^\alpha\), \(P_k^\beta\);
Hirshfeld uses the **total** density only (not spin-resolved volumes).

**Hirshfeld + MBD@rsSCS.** Same free-atom exponential model and libMBD tables
as sibling slugs; scaling \(\alpha\propto v\), \(C_6\propto v^2\),
\(R\propto v^{1/3}\). Isolated MBD (`mbd_k_mesh={0,0,0}`).

## Hard pin

**E_disp = −1.0427466367435301×10⁻⁴ Ha** (CUTOFF 120 Ry, k=2×2×1 TR).
See `REFERENCE.md` for free-atom dual and geometry.

## Citations

- Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009).
- Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014).
- Monkhorst & Pack, *Phys. Rev. B* **13**, 5188 (1976).
