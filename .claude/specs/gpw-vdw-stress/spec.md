# Spec — gpw-vdw-stress

## Summary

Analytic fixed-\(P\) **cell stress** with nonlocal vdW-DF / DF2 / rVV10
(\(E_{\mathrm{c}}^{\mathrm{nl}}\)) folded under orthorhombic strain — dedicated
pin for the SC nonlocal correlation stress contribution.

## Domain basis

- Nonlocal vdW energy / SC potential: `reg_gpw-vdw-scf-potential`, `gpw-vdw-*` ✓
- Analytic stress product: `reg_gpw-analytic-stress` / multi-k stress ✓
- Multi-k vdW forces: `reg_gpw-vdw-kpoint-forces` ✓
- Residual: dedicated stress pin with NON_LOCAL under strain (Γ or multi-k as
  product path allows) missing or unpinned.

## Design

1. Fixed-\(P\) Lagrangian stress includes \(\partial E_{\mathrm{c}}^{\mathrm{nl}}/\partial h\)
   (or dual-gate strain-FD of total \(E\) with frozen density policy matching
   existing stress product).
2. Twin: CP2K 2026.1 `STRESS_TENSOR` + `NON_LOCAL` (DRSLL|LMKLL|RVV10) on
   GTH-GPW water-dimer / ice-class system (matched GPW).
3. Dual-gate analytic vs multi-SCF FD + offline CP2K NUMERICAL or ENERGY+stress
   pins; fair wall on single SCF+stress twin.

## Files (expected)

| Path | Role |
| --- | --- |
| `gpw_stress.cpp` / nonlocal fold | product stress |
| `regressions/gpw/gpw-vdw-stress/` | reg + REFERENCE |
| `tools/cp2k_oracle/inputs/gpw-vdw-stress/` | CP2K twin |

## Tasks

1. Confirm product fold or implement missing \(\sigma^{\mathrm{nl}}\).
2. Offline CP2K twin + REFERENCE.
3. Regression dual-gate + fair wall ≤ 10.

## Out of scope

- Stock VV10 (blocked); density-dependent XDM; shear off-diagonal if product is diag-only.
