# gpw-d3-atm-kpoint-forces

## Goal

Dedicated multi-k GPW pin of **analytic ionic forces** with Grimme DFT-D3
**three-body ATM (C9)** folded in, matching an offline CP2K 2026.1
`ENERGY_FORCE` twin with Monkhorst–Pack k-mesh + `CALCULATE_C9_TERM`.

## Status of the algorithm path

Multi-k force assembly already calls `dftd3_pbe(..., calculate_c9 =
cfg.d3_atm)` on the k path (`gpw_forces.cpp`), identical to Γ. This slug is
**capability certification by pin**, not a new force kernel:

- Deps green: `gpw-d3-atm-forces` (Γ ATM F), `gpw-d3-atm-kpoint-stress`
  (multi-k ATM stress fold).
- Gap: no dedicated multi-k ATM **force** regression / offline pin.

## Twin design

- System: periodic water dimer **6 Å** cell (same as Γ ATM force / multi-k
  ATM stress) so lattice-image triple classes engage.
- Electronic: MP **1×1×2**, DIAGONALIZATION (OT is Γ-only under `&KPOINTS`),
  PBE, DZVP-MOLOPT-SR-GTH, GTH-PBE, CUTOFF 300.
- Dispersion: DFT-D3 zero, `R_CUTOFF 10.0`, pair of runs C9 T/F.
- Pin: \(F(\mathrm{C9}) - F(\mathrm{2b})\) isolates pure ATM gradient
  (geometry-only; electronic force cancels). Fair wall on the C9
  `ENERGY_FORCE` job only (1 MPI × 8 OMP).

## Acceptance

See `acceptance.md`.
