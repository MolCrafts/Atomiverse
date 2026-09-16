# Reference — constant-force steered ion pull

oracle: lammps
oracle_version: 30 Mar 2026 - Development (patch_30Mar2026-1074-g7f680de296); binary `/home/jicli594/.local/x86_64/bin/lmp`

## Physical system

Two atoms in a cubic orthorhombic box, **no pair potential**
(LAMMPS `pair_style zero`). Constant external force applied with
`fix addforce` on the type-1 ion only. Metal units throughout.

| Quantity | Value |
| --- | --- |
| Box | \(L = 20\) Å cubic, PBC |
| \(\mathbf{F}^{\mathrm{add}}\) | \((0,\,0,\,0.05)\) eV/Å on type 1 |
| Atom 1 (ion, type 1) | \(q=+1\), \(m=39.948\) amu, \(\mathbf{r}=(10,10,5)\) Å |
| Atom 2 (bg, type 2) | \(q=0\), \(m=39.948\) amu, \(\mathbf{r}=(10,10,15)\) Å |
| Image flags | all zero at start |
| Integrator (Gate B) | NVE, \(dt=1\) fs (= LAMMPS metal `timestep 0.001`), 100 steps, \(v_0=0\) |

## Offline pin (LAMMPS)

Produced with
`tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/in.addforce_pin`
(`lmp -in in.addforce_pin`).

### Gate A — `run 0` forces + bias energy

| Atom | type | \(F_x\) | \(F_y\) | \(F_z\) [eV/Å] |
| ---: | ---: | ---: | ---: | ---: |
| 1 (ion) | 1 | 0 | 0 | 0.05 |
| 2 (bg) | 2 | 0 | 0 | 0 |

- Bias energy (LAMMPS fix scalar `f_1`): \(U = -0.25\) eV  
  Formula: \(U = -\sum_{i\in\mathrm{group}} \mathbf{F}^{\mathrm{add}}\cdot\mathbf{r}_i^{\mathrm{u}}\)  
  (source: `fix_addforce.cpp` constant path, `foriginal[0] -= F·unwrap`).
- Pair PE = 0 (`pair_style zero`).

### Gate B — short NVE (100 steps, \(dt=1\) fs)

| Quantity | Value |
| --- | --- |
| \(z_{\mathrm{ion}}\) final | \(5.060381832778068\) Å |
| \(\Delta z\) | \(0.0603818327780683\) Å |
| Work proxy \(W=F_z\Delta z\) | \(0.00301909163890342\) eV |
| Bias energy final `f_1` | \(-0.2530190916389034\) eV |
| \(z_{\mathrm{bg}}\) | \(15\) Å (unmoved) |
| \(F_z\) ion (still) | \(0.05\) eV/Å |

Tolerances used by the regression: force/energy Gate A \(10^{-12}\);
Gate B \(z\) \(5\times10^{-6}\) Å, work/U \(5\times10^{-7}\) eV
(accounts for `force_to_accel` rounding vs LAMMPS metal `ftm2v`).

Gate B seeds \(\mathbf{F}\) on the Frame before `MD::run` so the first
VelocityVerlet half-step matches LAMMPS `setup`/`run 0` force evaluation
(ATV step order is integrate-then-force).

## What Atomiverse asserts

`reg_constant-force-pull-ion` builds the same Frame and checks:

1. **Gate A:** `cpu::Evaluate` + `cpu::md::FixAddForce{0,0,0.05,type=1}` —
   per-atom forces and total bias energy vs table.
2. **Gate B:** `cpu::md::MD` + `VelocityVerlet{1.0}` + same `FixAddForce` —
   final \(z\) of the ion, work proxy \(W=F_z\Delta z\), bias energy, and
   background immobility vs pin.

Dual gate: **force+energy** and **short-MD displacement / work** (not a
self-golden of ATV output).

## Fair wall (perf gate)

Comparable pure-addforce workload (pair zero / inert, NVE, addforce on one type):

| Engine | Workload | Wall [s] (OMP=1) |
| --- | --- | ---: |
| LAMMPS (`lmp`) | 1000 atoms (type-1 group), 5000 steps, `pair zero` + `fix addforce 0 0 0.05` + `fix nve`, dt=1 fs | **0.455** (Loop time) |
| Atomiverse | 1000 atoms, 5000 steps, `FixAddForce{0,0,0.05,type=1}` + VV, dt=1 fs | **0.055** (chrono around `MD::run`) |

`ratio_atv_over_oracle = T_atv / T_lammps = 0.055 / 0.455 ≈ 0.12` (≤ 10).  
within_10x: true. Host: same login node, single-thread, 2026-07-24.

Notes: ATV MD step order is integrate-then-force; fair trajectory match seeds
\(F\) on the Frame before `run` (LAMMPS `setup` role). The bench seeds the same
way and does not include that one-time seed in the timed region beyond the first
step's natural recompute.

## Provenance

- LAMMPS docs: <https://docs.lammps.org/fix_addforce.html>
- Source: `src/fix_addforce.cpp` (constant force path).
- Queue: shared classical MD ○ / EDL transport twin (`constant-force-pull-ion`).
- Not `fix smd` (spring / constant-velocity CV) — deferred.
