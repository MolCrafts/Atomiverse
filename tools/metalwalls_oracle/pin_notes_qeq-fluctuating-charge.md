# Pin notes — qeq-fluctuating-charge (LAMMPS `fix qeq/point`)

oracle: lammps
oracle_version: LAMMPS 30 Mar 2026 development (patch_30Mar2026-1074-g7f680de296), QEQ package
binary: `lmp` on PATH (x86_64 login node), run offline 2026-07-27 — **never at ctest time**

## System

Formaldehyde-like H₂CO 4-atom cluster (small organic toy), `boundary f f f`,
40 Å cube, `units metal` (eV / Å / e):

| id | type | element | x | y | z |
|---|---|---|---|---|---|
| 1 | 1 | C | 20.0 | 20.0 | 20.0 |
| 2 | 2 | O | 20.0 | 20.0 | 21.205 |
| 3 | 3 | H | 20.943 | 20.0 | 19.412 |
| 4 | 3 | H | 19.057 | 20.0 | 19.412 |

QEq parameters (`param.qeq`, RG91-flavored; χ [eV], η [eV/e²]; gamma/zeta/qcore
unused by the point style):

| type | χ | η |
|---|---|---|
| 1 (C) | 5.343 | 10.126 |
| 2 (O) | 8.741 | 13.364 |
| 3 (H) | 4.528 | 13.8904 |

Command: `fix q all qeq/point 1 10.0 1.0e-10 500 param.qeq` with
`pair_style coul/cut 10.0`; `run 0` (setup equilibrates charges before output).

## Algorithm identity (verified against LAMMPS source `src/QEQ/fix_qeq_point.cpp`)

- Off-diagonal coupling `H_ij = 1/r_ij` for `r ≤ cutoff` — **no qqrd2e
  prefactor** (`compute_H`: `H.val = 0.5/r` on a full neighbor list, applied
  symmetrically in `sparse_matvec`).
- Diagonal `η_i`; RHS `b_s = −χ_i`, `b_t = −1`; charges
  `q = s − (Σs/Σt)·t` ⇒ Σq = 0 exactly (`calculate_Q`).
- The `coul/cut` pair energy afterwards *does* use qqr2e = 14.399645 eV·Å/e².

An independent dense solve of `(η_i δ_ij + 1/r_ij) q = −χ + λ` reproduced every
printed digit of the LAMMPS dump before pinning.

## Pins (hard-coded in `regressions/md/qeq-fluctuating-charge/`)

- q(C, id 1)  = **0.046279970820 e**
- q(O, id 2)  = **−0.230795571743 e**
- q(H, id 3)  = **0.092257800461 e**
- q(H, id 4)  = **0.092257800461 e**
- Σq = 0 (exact, two-vector constraint)
- E_coul (`pair coul/cut 10.0` with equilibrated q) = **−0.254699311955 eV**
- Derived QEq objective E = Σχq + ½qᵀ(η δ + 1/r)q (k≡1 kernel convention)
  = **−0.467311783767 eV** (self-consistency value for `QEq::qeq_energy()`,
  computed from the pinned charges; not a LAMMPS thermo output)

## Perf twin (`in.qeq_perf`)

1000 repeated equilibrations (dt = 0, `fix nve`, qeq every step):
`Loop time of 0.000610294 s on 1 procs for 1000 steps with 4 atoms`
⇒ T_lmp ≈ 0.61 µs/solve (CG, warm-started by fix qeq history).
ATV twin: 1000 `QEq::launch` calls on the same geometry (dense rebuild+solve
each launch), timed inside the regression binary and reported to stdout.

## Files

- `inputs/lammps/qeq-fluctuating-charge/in.qeq_point` — pin run
- `inputs/lammps/qeq-fluctuating-charge/in.qeq_perf` — perf twin
- `inputs/lammps/qeq-fluctuating-charge/param.qeq` — QEq parameters
