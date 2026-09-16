# Reference — oscillating (AC) external E-field on MD charges

oracle: lammps
oracle_version: 30 Mar 2026 - Development (patch_30Mar2026-1074-g7f680de296); binary `/home/jicli594/.local/x86_64/bin/lmp`

## Physical system

Three point charges in a cubic orthorhombic box, **no pair potential**
(LAMMPS `pair_style zero`). Time-dependent uniform field via LAMMPS
`fix efield` equal-style variable (AC cosine). Metal units in the oracle
input; Atomiverse public surface uses **fs** for time (see conversion).

| Quantity | LAMMPS (metal) | Atomiverse |
| --- | --- | --- |
| Box | \(L = 20\) Å cubic, PBC | same |
| \(\mathbf{E}_0\) | \((0,\,0,\,0.2)\) V/Å | same |
| Period \(T\) | \(1.0\) ps | \(1000\) fs |
| Timestep \(\Delta t\) | \(0.001\) ps | \(1\) fs |
| Phase \(\varphi\) | \(0\) | \(0\) |
| Atom 1 | \(q=+1\), \(\mathbf{r}=(5,10,10)\) Å | same |
| Atom 2 | \(q=-1\), \(\mathbf{r}=(7,10,10)\) Å | same |
| Atom 3 | \(q=+0.5\), \(\mathbf{r}=(10,12,8)\) Å | same |
| Image flags | all zero | same |

Field:

\[
E_z(t) = 0.2\,\cos(2\pi\, t / T),\qquad
\mathbf{F}_i = q_i\,\mathbf{E}(t),\qquad
U = -\sum_i q_i E_z(t)\, z_i
\]

(with \(t = n\Delta t\); \(T/\Delta t = 1000\) steps in both unit systems).

## Offline pin (LAMMPS `run 0` at selected steps)

Produced with
`tools/metalwalls_oracle/inputs/lammps/oscillating-efield-md/in.oscillating_efield`
(`lmp -in in.oscillating_efield`). Force from dump; energy from fix scalar
with `potential v_phi` (`phi atom "-v_Ez*z"`) so variable-path \(U\) matches
the constant-path formula at frozen \(E(t)\).

| step \(n\) | \(t\) (ps) | \(E_z\) [V/Å] | \(U\) [eV] | \(F_{z,1}\) | \(F_{z,2}\) | \(F_{z,3}\) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.2 | −0.8 | 0.2 | −0.2 | 0.1 |
| 125 | 0.125 | 0.14142135623731000 | −0.5656854249492400 | \(q_1 E_z\) | \(q_2 E_z\) | \(q_3 E_z\) |
| 250 | 0.25 | ~0 (≤ 2e−17) | ~0 | ~0 | ~0 | ~0 |
| 375 | 0.375 | −0.14142135623731000 | +0.5656854249492400 | … | … | … |
| 500 | 0.5 | −0.2 | +0.8 | −0.2 | 0.2 | −0.1 |
| 750 | 0.75 | ~0 | ~0 | ~0 | ~0 | ~0 |
| 1000 | 1.0 | 0.2 | −0.8 | 0.2 | −0.2 | 0.1 |

Regression asserts steps **0, 125, 500** (max / mid-quarter / min) with dual
force+energy gates. Analytic \(U = -4\,E_z\) for this geometry
(\(-\sum q_i z_i = -4\)).

## What Atomiverse asserts

`reg_oscillating-efield-md` builds the same Frame, constructs

```cpp
cpu::md::FixOscillatingEfield{0.0, 0.0, 0.2, /*period=*/1000.0, /*dt=*/1.0};
```

and at each pin step \(n\) launches with `ctx.step = n`, checking:

1. Per-atom forces match \(q_i E(t)\) within \(10^{-12}\) eV/Å (or \(10^{-10}\) at
   near-zero crossings).
2. `frame.meta.get(kEnergy)` matches pin \(U(t)\) within \(10^{-12}\) eV
   (looser \(10^{-10}\) near zeros).

Dual gate: **force and energy** at multiple times (not a self-golden).

## Fair wall (perf gate)

Comparable pure oscillating-efield workload (pair zero / inert, NVE, AC field
on charges):

| Engine | Workload | Wall [s] (OMP=1) |
| --- | --- | ---: |
| LAMMPS (`lmp`) | 1000 atoms, 5000 steps, `pair zero` + variable `fix efield 0 0 v_Ez` + `fix nve`, dt=0.001 ps | **0.0585** (Loop time) |
| Atomiverse | 1000 atoms, 5000 steps, inert `PairLJCut` (rc=2 Å ≪ spacing) + `FixOscillatingEfield{0,0,0.05,1000,1}` + VV, dt=1 fs | **0.124** |

`ratio_atv_over_oracle = T_atv / T_lammps = 0.124 / 0.0585 ≈ 2.12` (≤ 10).  
within_10x: true. Host: same login node, single-thread, 2026-07-24.

## Provenance

- LAMMPS docs: <https://docs.lammps.org/fix_efield.html> (equal-style variable)
- Source: `src/fix_efield.cpp` (variable path forces; potential keyword for \(U\))
- Map: `.claude/notes/metalwalls-capability-map.md` slice **E3** (oscillating residual)
- Constant foothold: `regressions/md/external-efield-md/`
