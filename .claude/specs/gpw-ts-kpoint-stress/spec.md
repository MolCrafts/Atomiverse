---
slug: gpw-ts-kpoint-stress
status: approved
created: 2026-08-01
revised: 2026-08-01
domain: gpw
queue: cp2k-method-loop
depends_on: [gpw-ts-stress]
---

# Spec: `gpw-ts-kpoint-stress` — multi-k analytic cell stress with free-atom TS

**Slug:** `gpw-ts-kpoint-stress`  
**Domain:** GPW (CPU)  
**Queue:** CP2K method loop  
**Depends on:** `gpw-ts-stress` (Γ free-atom TS stress fold ✓)

## Summary

Close the residual that Γ fixed-\(P\) analytic cell stress already rebuilds
geometry-only free-atom Tkatchenko–Scheffler dispersion
(\(E_{\mathrm{disp}}^{TS}\), \(V/V_{\mathrm{free}}=1\)) at every strained cell
while the **multi-k** product path needs a dedicated dual-gate pin. The
matched algorithm path is the k-summed Hellmann–Feynman fixed-\(\{P_k,W_k\}\)
Lagrangian in `energy_fixed_P_k` → `gpw_stress_analytic_impl` with
`cfg.ts = true` and a non-Γ `k_mesh`.

## Capability (already on product surface)

- `cpu::gpw::GPWEvaluator` members: `ts = true`, `k_mesh = {nx,ny,nz}`,
  `compute_stress = true`, `stress_use_fd = false`.
- Multi-k stress Lagrangian folds free-atom TS via the same `ts_pbe` call as
  the energy and Γ stress paths (pure RS lattice sum, rcut 50 Å, \(s_R=0.94\),
  \(d=20\)), rebuilt at every strained cell.
- Public Frame `kStress` in eV/Å³; internal compare via
  `last_stress_ha_bohr3()` [Ha/Bohr³].

## Oracle policy

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no** stock free-atom `TYPE TS`. Hard pins
are offline **libMBD** Fortran `get_ts_energy` pure-RS branch (same family as
`gpw-ts-ice` / `gpw-ts-stress`). Dual gate:

1. Strained-cell \(E_{\mathrm{disp}}\) vs libMBD pins +
   \(\sigma(\mathrm{ts})-\sigma(\mathrm{plain})\) vs libMBD strain-FD terms.
2. Analytic fixed-\(P_k\) \(\sigma\) vs multi-SCF strain-FD on the same
   multi-k method (`gpw_stress_fd_impl`).

No live oracle in ctest. **`oracle: literature` / libMBD** — not a fake
`oracle: cp2k`. Perf wall vs CP2K is **N/A** (`perf_exempt`; no matched twin).

## Acceptance system

Water dimer (same geometry as `gpw-ts-ice` / `gpw-ts-stress`), cell 12³ Å,
DZVP-MOLOPT-SR-GTH + GTH-PBE, PBE, MP **1×1×2**, cutoff 120 Ry, strain
\(h=2\times10^{-3}\).

## Deliverables

| Artifact | Path |
| -------- | ---- |
| Regression + REFERENCE | `regressions/gpw/gpw-ts-kpoint-stress/` |
| Offline pin driver | `tools/cp2k_oracle/inputs/gpw-ts-kpoint-stress/pin_ts_strain.f90` |
| Pin notes | `tools/cp2k_oracle/inputs/gpw-ts-kpoint-stress/water_dimer_ts_kpoint_stress.note` |
| CMake registration | `atv_add_regression(gpw gpw-ts-kpoint-stress …)` |
