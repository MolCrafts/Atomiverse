# qeq-fluctuating-charge

oracle: lammps
oracle_version: LAMMPS 30 Mar 2026 development (patch_30Mar2026-1074-g7f680de296), QEQ package

## What is pinned

QEq / fluctuating-charge equilibration (`cpu::md::QEq`, the LAMMPS
`fix qeq/point` twin) on a formaldehyde-like H₂CO 4-atom organic toy:
per-atom equilibrated charges, the QEq objective, and the `coul/cut`
Coulomb energy of the equilibrated system. Units: eV, Å, e (`units metal`).

Oracle run offline 2026-07-27 (`lmp -in in.qeq_point`; inputs preserved at
`tools/metalwalls_oracle/inputs/lammps/qeq-fluctuating-charge/`, procedure in
`tools/metalwalls_oracle/pin_notes_qeq-fluctuating-charge.md`). ctest never
launches LAMMPS.

## System

`boundary f f f`, 40 Å cube; C (20, 20, 20), O (20, 20, 21.205),
H (20.943, 20, 19.412), H (19.057, 20, 19.412).
QEq params (χ [eV], η [eV/e²]): C 5.343/10.126, O 8.741/13.364,
H 4.528/13.8904. `fix qeq/point 1 10.0 1.0e-10 500 param.qeq`,
`pair_style coul/cut 10.0`, `run 0`.

## Pinned values (hard-coded in the regression, tol 1e-6)

| quantity | LAMMPS pin |
|---|---|
| q(C) | +0.046279970820 e |
| q(O) | −0.230795571743 e |
| q(H) ×2 | +0.092257800461 e |
| Σq | 0 (two-vector constraint, tol 1e-12) |
| E_coul (`coul/cut`, qqr2e = 14.399645) | −0.254699311955 eV |
| E_qeq = Σχq + ½qᵀ(ηδ + 1/r)q (derived from pinned q; bare-1/r kernel) | −0.467311783767 eV |

Kernel identity verified against LAMMPS source `src/QEQ/fix_qeq_point.cpp`:
off-diagonal bare `1/r` (no qqrd2e), diagonal η, RHS −χ, constraint
`q = s − (Σs/Σt)t`.

## Fair perf twin

`in.qeq_perf`: 1000 repeated `fix qeq/point` solves (dt = 0), LAMMPS
`Loop time of 0.000610294 s` (1 proc). The regression times 1000
`QEq::launch` equilibrations (dense rebuild + Cholesky each launch) on the
same geometry and prints `T_atv`, `T_lmp`, and the ratio to stdout — reported,
not asserted (micro-workload; both sides are sub-ms).

Measured 2026-07-27 (x86_64 epyc login node, 1 thread):
`T_atv = 0.000275 s`, `T_lmp = 0.000610 s` ⇒ **ratio 0.45** (ATV within 10×;
dense factor+two substitutions beats warm-started CG at N = 4).
