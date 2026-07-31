---
slug: gpw-ts-stress
status: approved
created: 2026-07-31
revised: 2026-07-31
domain: gpw / dispersion / cell stress
queue: cp2k-method-loop
deferred: false
depends_on: []
unblocks: [gpw-ts-stress]
grilled: false
sibling: gpw-xdm-stress
---

# Spec: `gpw-ts-stress` — dedicated pins for the TS dispersion cell-stress fold

**Slug:** `gpw-ts-stress`
**Domain:** GPW Γ-point analytic fixed-P cell stress × Tkatchenko–Scheffler
free-atom dispersion (CPU)
**Queue row:** `gpw-ts-stress` (wave-15 thin pin-only; hold released after the
wave-1 stress workers drained `gpw_stress.cpp`)

## Summary

The Γ fixed-\(P\) stress Lagrangian (`gpw_stress.cpp`, both the Γ and the
k-summed `energy_fixed_P*` paths) already **rebuilds the geometry-only TS
energy at every strained cell** (`cfg.ts` branch beside D3/D4/XDM/MBD), so
`GpwDft::ts = true` + `compute_stress = true` folds \(E_{\mathrm{disp}}^{TS}\)
into \(\sigma\). What is missing is the **capability closure**: no dedicated
strained-cell oracle pins and no regression gating that the fold is correct —
`gpw-ts-ice` pins only the unstrained energy, and the shipped stress
regressions cover D3(BJ)/D4/MBD, never TS.

This slug is **pin-only**: no product code change is expected. Deliverables
are (1) offline libMBD strained-cell TS pins, (2) the
`regressions/gpw/gpw-ts-stress/` dual-gated case, (3) REFERENCE.md with full
provenance.

## Oracle

**libMBD** (git `88069c6`, Fortran `get_ts_energy`) — the same oracle family
as the shipped `gpw-ts-ice` / `gpw-ts-forces` pins. CP2K 2026.1 stock
`PAIR_POTENTIAL TYPE` has no `TS` (only DFTD2/D3/D3BJ/D4), so CP2K cannot
produce a free-atom-TS stress twin.

Pin driver: `tools/cp2k_oracle/inputs/gpw-ts-stress/pin_ts_strain.f90` — a
thin Fortran main that links libMBD and calls **libMBD's own**
`get_ts_energy` with `ewald_on = .false.` and
`dipole_cutoff = 50 Å` (ATV Å→Bohr), i.e. libMBD's pure real-space lattice-sum
path with ATV's cutoff convention. The driver sets inputs and prints libMBD's
numbers; it reimplements nothing. Free-atom \(\alpha_0, C_6, R_{\mathrm{vdW}}\)
come from libMBD's `ts_vdw_params`; damping Fermi \(s_R = 0.94\), \(d = 20\)
(PBE). Never launched from ctest.

This driver also upgrades the provenance of the **existing** periodic TS pin
(−6.0996307733…×10⁻⁴ Ha, previously recorded as "dual pure-RS / ATV
`ts_pbe`"): the unstrained libMBD pure-RS number must reproduce it, and the
`gpw-ts-ice` / `gpw-ts-forces` REFERENCE provenance rows are updated to cite
the genuine libMBD run.

## Geometry

The `gpw-ts-forces` / `gpw-ts-ice` input, extended by strain: H-bonded water
dimer (O H H O H H, coordinates as in those REFERENCEs), periodic 12 Å cube;
strained cells scale one axis by \(1 \pm h\) (h = 2e-3) at fixed fractional
coordinates. `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6`/`-q1`, PBE, Γ.

## Gates (regression `reg_gpw-ts-stress`)

- **A** — unstrained `E_disp` from the product SCF (`ts = true`) matches the
  libMBD pure-RS pin (tol 1e-8 Ha; the same number as the `gpw-ts-ice` pin).
- **B** — `E_disp` at each of the 6 strained cells (±h per axis, standalone
  `ts_pbe` at the strained geometry — the exact call the stress Lagrangian
  makes) matches fresh libMBD strained-cell pins to **1e-12 Ha class**
  (regression tol 1e-8 Ha gate, expected ~1e-15).
- **C** — analytic fixed-P stress (`stress_use_fd = false`) vs the multi-SCF
  strain-FD oracle `gpw_stress_fd_impl` (both `ts = true`), diagonal ≤ 2e-5
  Ha/Bohr³ (SCF-noise tier shared by every sibling stress case).
- **D** — TS wiring: \(\sigma(\mathrm{ts}) - \sigma(\mathrm{plain})\) per axis
  equals the libMBD strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\)
  ≤ 2e-9 Ha/Bohr³ (the ≤1e-8 Ha/Bohr³-class gate of
  `gpw-mbd-kpoint-stress`, gate D).
- **E** — orthorhombic off-diagonals exactly 0; `kStress` [eV/Å³] conversion;
  product wall ≤ 60 s.

## Out of scope

Hirshfeld-scaled TS stress (volume-ratio gradients under strain), XDM stress
(`gpw-xdm-stress` sibling), shear/off-diagonal strain, CUDA.

## Perf

**perf_exempt** — CP2K has no stock free-atom-TS twin to time against;
geometry-only hard-gate policy identical to `gpw-ts-ice` / `gpw-ts-forces` /
`gpw-mbd-stress`.
