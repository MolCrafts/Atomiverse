# Reference — external uniform E-field on MD charges

oracle: lammps
oracle_version: 30 Mar 2026 - Development (patch_30Mar2026-1074-g7f680de296); binary `/home/jicli594/.local/x86_64/bin/lmp`

## Physical system

Three point charges in a cubic orthorhombic box, **no pair potential**
(LAMMPS `pair_style zero`). Uniform external field applied with
`fix efield` (constant path). Metal units throughout.

| Quantity | Value |
| --- | --- |
| Box | \(L = 20\) Å cubic, PBC |
| \(\mathbf{E}\) | \((0.1,\,-0.05,\,0.2)\) V/Å |
| Atom 1 | \(q=+1\), \(\mathbf{r}=(5,10,10)\) Å |
| Atom 2 | \(q=-1\), \(\mathbf{r}=(7,10,10)\) Å |
| Atom 3 | \(q=+0.5\), \(\mathbf{r}=(10,12,8)\) Å |
| Image flags | all zero (wrapped ≡ unwrapped) |

## Offline pin (LAMMPS `run 0`)

Produced with `tools/metalwalls_oracle/inputs/lammps/in.external_efield`
(`lmp -in in.external_efield`). Dump + fix scalar:

| Atom | \(q\) | \(F_x\) | \(F_y\) | \(F_z\) [eV/Å] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +1 | 0.1 | −0.05 | 0.2 |
| 2 | −1 | −0.1 | 0.05 | −0.2 |
| 3 | +0.5 | 0.05 | −0.025 | 0.1 |

- Field energy (LAMMPS fix scalar `f_1`): \(U = -0.8\) eV  
  Formula: \(U = -\sum_i q_i \mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}\)  
  (source: `fix_efield.cpp` constant path, `fsum[0] -= F·unwrap`).
- Pair PE = 0 (`pair_style zero`).
- Force formula: \(\mathbf{F}_i = q_i \mathbf{E}\) with metal `qe2f = 1`.

## What Atomiverse asserts

`reg_external-efield-md` builds the same Frame, runs
`cpu::Evaluate` + `cpu::md::FixEfield{0.1, -0.05, 0.2}`, and checks:

1. Per-atom forces match the table within \(10^{-12}\) eV/Å.
2. `frame.meta.get(kEnergy)` matches \(U = -0.8\) eV within \(10^{-12}\) eV.

Dual gate: **force and energy** (not a self-golden of ATV output).

## Fair wall (perf gate)

Comparable pure-efield workload (pair zero, NVE, efield only on charges):

| Engine | Workload | Wall [s] (OMP=1) |
| --- | --- | ---: |
| LAMMPS (`lmp`) | 1000 atoms, 5000 steps, `pair zero` + `fix efield 0.05 0 0` + `fix nve`, dt=1 fs | 0.21 |
| Atomiverse | 1000 atoms, 5000 steps, inert `PairLJCut` (rc=2 Å ≪ spacing) + `FixEfield{0.05,0,0}` + VV, dt=1 fs | 0.126 |

`ratio_atv_over_oracle = T_atv / T_lammps = 0.126 / 0.21 ≈ 0.60` (≤ 10).  
within_10x: true. Host: same login node, single-thread, 2026-07-23.

## Provenance

- LAMMPS docs: <https://docs.lammps.org/fix_efield.html>
- Source: `src/fix_efield.cpp` (constant efield, charge interactions).
- Map: `.claude/notes/metalwalls-capability-map.md` slice **E3**.
