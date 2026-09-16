# Reference — freeze electrode / wall atoms (LAMMPS `fix setforce 0`)

oracle: lammps
oracle_version: develop / patch_30Mar2026-1074-g7f680de296 (30 Mar 2026 Development; binary `lmp` at pin time)

## Physical setup

Immobile electrode (wall) atoms under classical LJ NVE with a mobile fluid.
This is the MetalWalls E1 wall-immobility foothold and the LAMMPS twin of
`fix setforce 0.0 0.0 0.0` on a wall group plus `velocity wall set 0 0 0`.

| Quantity | Value |
|---|---|
| Box | cubic \(L = 10\) Å, PBC |
| Wall (type 2) | 4 atoms at \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å; mass 196.9665 amu |
| Fluid (type 1) | 4 atoms at \(z=3.5\) above each wall site; mass 39.948 amu |
| Pair | LJ 12-6, \(\varepsilon = 0.010341\) eV, \(\sigma = 3.4\) Å, \(r_c = 8.0\) Å (all type pairs) |
| Ensemble | NVE velocity-Verlet, \(dt = 1\) fs |
| Steps (correctness pin) | 1000 |
| Wall initial \(v\) | \(0\) |
| Fluid initial \(T\) | Maxwell–Boltzmann 200 K (seed 99; RNG not bit-matched to LAMMPS) |

Offline input: `tools/metalwalls_oracle/inputs/lammps/freeze-electrode-atoms/in.lammps`.

## What is asserted (not a self-golden)

From LAMMPS `fix setforce 0.0 0.0 0.0` on group `wall` (`fix_setforce.cpp`):
after forces are computed, wall forces are zeroed so the NVE half-kicks do not
change wall velocities. With \(v_{\mathrm{wall}}(0)=0\), wall positions remain
exactly at their create positions for the entire trajectory.

**Offline LAMMPS pin (1000 steps, metal units, `timestep 0.001` ps = 1 fs):**

| Observable | Pin value |
|---|---|
| Wall positions after 1000 steps | identical to create: \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å |
| Wall velocities | all components exactly `0` |
| Wall \(z\) span | \(z \in [0, 0]\) throughout (thermo `c_min_dz`/`c_max_dz`) |

Dump excerpt (`wall_final.lammpstrj` after 1000 steps):

```
1 2 2 2 0 0 0 0
2 2 2 6 0 0 0 0
3 2 6 2 0 0 0 0
4 2 6 6 0 0 0 0
```

Atomiverse regression asserts the same **immobility invariants** (not fluid
coordinates — fluid RNG and unit-velocity conventions differ):

1. \(\max_i |x_i(t) - x_i(0)| \le 10^{-10}\) Å on the wall type.
2. \(\max |v|\) components on wall ≤ \(10^{-12}\) Å/fs.
3. At least one fluid atom displaces by more than \(10^{-3}\) Å (freeze is selective).

## Units note

LAMMPS `units metal` time is **ps**; Atomiverse public time is **fs**.
`timestep 0.001` (ps) ⇔ `dt = 1.0` fs. Velocities in metal dumps are Å/ps;
the freeze pin is zero in either convention.

## Perf gate (fair wall)

Same geometry, **20000** NVE steps, single OpenMP thread (`OMP_NUM_THREADS=1`),
no dump/thermo thrash:

| Engine | Wall time | Notes |
|---|---|---|
| LAMMPS `lmp` (1 rank, OMP=1) | Loop time **0.0186 s** | `in.bench` |
| Atomiverse `FixFreeze` + VV + LJ | **0.0740 s** (instrumented in reg binary) | same N, steps, dt |

\[
\frac{T_{\mathrm{atv}}}{T_{\mathrm{oracle}}} = \frac{0.0740}{0.0186} \approx \mathbf{3.98}
\]

Gate: ≤ 10.0 → **PASS** (`within_10x=true`). Tiny 8-atom system; ratio is
not a production throughput claim — only a fair twin ceiling for this slug.


## API

```cpp
cpu::md::FixFreeze freeze(/*type_id=*/2);  // freeze electrode type
sim.add(std::move(freeze));
// with VelocityVerlet + PairLJCut on cpu::md::MD
```
