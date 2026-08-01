# Spec — gpw-d4-kpoint-stress

## Goal

Dedicated offline pin that the **multi-k** GPW analytic fixed-\(P_k\) cell
stress Lagrangian folds **DFT-D4** (Becke–Johnson two-body, PBE parameters;
ATM off) into \(\sigma\). The k-path fold is already productized in
`energy_fixed_P_k` (`gpw_stress.cpp`) after `gpw-d4-stress` +
`gpw-kpoint-stress`; this slug closes the multi-k pin gap.

## Acceptance

1. **E_disp** from multi-k SCF equals the offline **dftd4 3.7.0** unstrained
   pin (tol 1e-8 Ha) — same value as Γ `gpw-d4-water` (geometry-only; k-mesh
   independent).
2. **CP2K multi-k \(\Delta\sigma\):** product
   \(\sigma(\mathrm{d4})-\sigma(\mathrm{plain})\) matches the offline CP2K
   2026.1 NUMERICAL stress-difference pin (tol 2e-9 Ha/Bohr³) from a
   **libdftd4-linked** binary.
3. **Internal dual gate:** analytic fixed-\(P_k\) vs multi-SCF strain-FD on
   the same multi-k `d4=true` method (tol 2e-5 Ha/Bohr³).
4. Multi-k payload present (`n_kpts ≥ 1`, `C_occ_k` non-empty); orthorhombic
   off-diagonals ~0; Frame `kStress` unit conversion.
5. Product wall ≤ 60 s. **Fair wall:** \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\)
   on the **same** CP2K D4 multi-k NUMERICAL-stress twin (1 MPI × fixed OMP).
   Not `perf_exempt`.

## Twin

- Geometry: periodic water dimer 12 Å cube (same as `gpw-d4-water` /
  `gpw-d4-stress`).
- Method: GPW PBE + DFT-D4 BJ, ATM off, electronic mesh Monkhorst–Pack **1×1×2**.
- Hard \(\Delta\sigma\) oracle: **CP2K 2026.1 + libdftd4** twin pair
  (`water_dimer_d4_kpoint_stress.inp` / `water_dimer_plain_kpoint_stress.inp`).
- Hard E_disp oracle: **dftd4 3.7.0** (ATV backend; CP2K printed E_disp is
  recorded in REFERENCE for provenance).
- Fair wall: CP2K D4 multi-k NUMERICAL-stress job only.

## Non-goals

D4 ATM (\(s_9\)), multi-k UKS, D3/D3(BJ)/TS/XDM/MBD k-stress (sibling
slugs), productizing a new fold (already present).
