# gpw-vdw-uks-kpoint-multigrid

## Goal

Dedicated dual-gate pin for the **triple product** already wired on the
multi-k collinear UKS GPW runner:

| Axis | Prior product |
| ---- | ------------- |
| multi-k UKS + nonlocal vdW-DF | `reg_gpw-vdw-uks-kpoints` |
| Γ UKS + nonlocal + REL_CUTOFF multigrid | `reg_gpw-vdw-uks-multigrid` |
| multi-k UKS + REL_CUTOFF multigrid (no nonlocal) | `reg_gpw-uks-multigrid-kpoints` |

Claim: with `n_alpha`/`n_beta` + `vdw_df` (also `vdw_df2` / `rvv10`) +
`k_mesh` (or explicit `k_points`) + `ngrids>1` + `rel_cutoff>0`,
self-consistent \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) of
the k-summed **total** density is added to **both** spin potentials on the
**fine** grid **before** per-level restriction / phase-summed
\(V_k^\sigma\) integration, so the dual KS matrices stay the exact gradient
of the collocated \(E_H+E_{xc}+E_c^{\mathrm{nl}}\). Post-SCF
\(E_c^{\mathrm{nl}}\) lands on `GpwResult::e_disp` via
`fold_dispersion_k_uks` (multigrid product density through
`collocate_rho_spin_k`).

Write-first: no new algorithm. Public surface is already legal after
`gpw-vdw-uks-kpoints`. This slug freezes a CP2K-matched offline pin and
internal single-grid ↔ multigrid consistency on that triple path.

## Product surface

- `cpu::gpw::GpwDft` / `GPWEvaluator`:
  - `n_alpha` / `n_beta` (collinear UKS)
  - `vdw_df = true` (DRSLL; also `vdw_df2` / `rvv10`)
  - `k_mesh = {2,1,1}` (or equivalent), `k_time_reversal = false` for 1:1
    CP2K MP mesh
  - `ngrids = 4`, `rel_cutoff = 40` [Ry], `cutoff_ry = 200`
  - `xc = XcKind::PBE`
- Lookup ctors: `GaussianBasis` / `GthPotential` (no factories, no Config bags)
- Frame public energy remains eV; regression gates use internal Ha

## Oracle

| Role | Source |
| ---- | ------ |
| \(E_{\mathrm{tot}}\), \(E_c^{\mathrm{nl}}\) (Dispersion energy) | offline CP2K **2026.1** `cp2k.psmp` git:5e54ba2 — LSD MULT 1 water dimer, PBE + NON_LOCAL DRSLL + MP 2×1×1 + NGRIDS 4 / REL_CUTOFF 40 (`tools/cp2k_oracle/inputs/gpw-vdw-uks-kpoint-multigrid/`) |
| Fair wall | same single twin job, 1 MPI × fixed OMP; report \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) on the multigrid SCF only |
| Internal consistency | multi-k UKS single-grid ↔ multigrid (\(\lvert\Delta E\rvert\) envelope matching `gpw-vdw-multigrid`) |

Fair twin class: GPW ↔ GPW GTH/MOLOPT, same spin/k/XC (collinear UKS / LSD,
PBE, DRSLL, MP mesh, multigrid cutoffs). No cross-representation window.

## Acceptance

See `acceptance.md`.
