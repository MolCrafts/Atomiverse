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
2. **Strained-cell E_disp** (±h per axis, fixed fractional coordinates)
   matches fresh offline dftd4 pins; derived \(\sigma^{\mathrm{D4}}\) FD
   terms feed the wiring gate.
3. **Internal dual gate:** analytic fixed-\(P_k\) vs multi-SCF strain-FD on
   the same multi-k `d4=true` method (tol 2e-5 Ha/Bohr³).
4. **Wiring:** \(\sigma(\mathrm{d4})-\sigma(\mathrm{plain})\) equals the
   dftd4 strain-FD dispersion term per axis (tol 2e-9 Ha/Bohr³).
5. Multi-k payload present (`n_kpts ≥ 1`, `C_occ_k` non-empty); orthorhombic
   off-diagonals ~0; Frame `kStress` unit conversion.
6. Product wall ≤ 60 s. **Perf:** host CP2K 2026.1 source build
   (`cp2kflags` has no `dftd4`) cannot run `TYPE DFTD4` — same
   **perf_exempt** policy as `gpw-d4-stress` / `gpw-d4-water`. Hard gates
   are dftd4 pins + dual gate.

## Twin

- Geometry: periodic water dimer 12 Å cube (same as `gpw-d4-water` /
  `gpw-d4-stress`).
- Method: GPW PBE + DFT-D4 BJ, ATM off (`--mbdscale 0`), electronic mesh
  Monkhorst–Pack **1×1×2**.
- Hard E_disp oracle: **dftd4 3.7.0** strained-cell POSCAR pins
  (`tools/cp2k_oracle/inputs/gpw-d4-kpoint-stress/`).
- Optional CP2K `.inp` retained for hosts with libdftd4 (not runnable on
  this host's source binary).

## Non-goals

D4 ATM (\(s_9\)), multi-k UKS, D3/D3(BJ)/TS/XDM/MBD k-stress (sibling
slugs), productizing a new fold (already present).
