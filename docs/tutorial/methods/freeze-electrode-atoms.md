# Freeze electrode / wall atoms — LJ NVE (`FixFreeze`)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `freeze-electrode-atoms` |
| Regression | `regressions/md/freeze-electrode-atoms/` |
| Public API | `cpu::md::FixFreeze{type_id}` on `cpu::md::MD` (with `VelocityVerlet` + pair potential) |
| Roadmap / queue | MetalWalls map **E1** wall-immobility foothold; [metalwalls-method-queue.md](../../developer/metalwalls-method-queue.md) → **done** (wave 1, 2026-07-23) |
| Dual gate | Correctness (immobility invariants); speed fair wall vs LAMMPS \(r\approx 3.98\le 10\) |

## Summary

Immobile electrode (wall) atoms under classical LJ molecular dynamics: selected
atom **type** rows keep \(v=\mathbf{0}\) and fixed positions while the fluid
evolves under NVE velocity-Verlet. Public component is
`cpu::md::FixFreeze` — a `Constraint` / `ConstraintProvider` bound into the MD
preset’s stages. This is the MetalWalls **E1** wall-immobility foothold and the
LAMMPS twin of `fix setforce 0.0 0.0 0.0` on a wall group (plus
`velocity wall set 0 0 0`). **Oracle is LAMMPS** (not MetalWalls): the algorithm
exists in LAMMPS, so offline pins and the fair wall use `lmp`. MetalWalls
`mobile = false` electrode practice is the same physics end-state but is not
the binding pin for this slug.

## Method

**Physical model.** Classical pairwise LJ fluid above a lattice of wall atoms
that must not integrate as free particles. Freezing is **selective by atom
type**: fluid atoms move; wall atoms stay at create positions.

**LAMMPS correspondence** (`fix_setforce.cpp`, audit ALIGNED for type groups):

\[
\mathbf{F}_i \leftarrow \mathbf{0}
\quad\text{for } i\in\text{wall group, after force accumulate (post\_force).}
\]

With \(\mathbf{v}_i(0)=\mathbf{0}\) and \(\mathbf{F}_i=\mathbf{0}\), velocity-Verlet
leaves wall positions unchanged for the whole trajectory.

**Atomiverse `FixFreeze` (superset of setforce-0 immobility).**

| Stage | Binding | Action |
| ----- | ------- | ------ |
| `post_force` | order \(+1000\) | zero \(\mathbf{F}\) on frozen type (after physical + Langevin at \(+500\)) |
| `post_integrate` | order \(0\) | restore build-time \(\mathbf{x}\) by tag; zero \(\mathbf{v}\) |
| `final_integrate` | order \(+200\) | re-zero \(\mathbf{v}\) after the final half-kick |

At `build()`, wall rows are resolved, positions are snapshotted keyed by stable
global **tag**, and wall \(v\)/\(F\) are zeroed (so `InitVelocity` cannot leave
wall momentum). `ConstraintProvider::n_constraints` reports **3 DOF per frozen
atom** so thermostats that consume Ndof remain consistent. Selection is a single
integer **type id** (textbook electrode slabs share one type); full LAMMPS
`group` region/id predicates are out of scope for this foothold.

**System (regression / offline pin).**

| Quantity | Value |
| -------- | ----- |
| Box | cubic \(L = 10\) Å, PBC |
| Wall (type 2) | 4 atoms at \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å; mass 196.9665 amu |
| Fluid (type 1) | 4 atoms at \(z=3.5\) above each wall site; mass 39.948 amu |
| Pair | LJ 12-6, \(\varepsilon = 0.010341\) eV, \(\sigma = 3.4\) Å, \(r_c = 8.0\) Å (all type pairs) |
| Neighbor | skin 0.3 Å |
| Ensemble | NVE velocity-Verlet, \(\Delta t = 1\) fs |
| Steps (correctness) | 1000 |
| Wall initial \(v\) | \(\mathbf{0}\) |
| Fluid initial \(T\) | Maxwell–Boltzmann 200 K (seed 99; RNG **not** bit-matched to LAMMPS) |

**Units.** Atomiverse public surface: eV, Å, fs, amu, eV/Å, K. LAMMPS
`units metal` time is **ps**; pin inputs use `timestep 0.001` (ps) \(\Leftrightarrow\)
`dt = 1.0` fs. Freeze velocities are zero in either convention.

**Excluded (non-goals of this slug)**

- Constant-potential charge solve, Gaussian electrode Coulomb, slab Ewald,
  external E-field (sibling E1–E3 slugs).
- Region/id group predicates; multi-type freeze in one component (compose
  multiple `FixFreeze`).
- Bit-identical fluid trajectories vs LAMMPS (RNG / velocity-init conventions).
- Live LAMMPS or MetalWalls inside `ctest`.
- Public `*Config` bags / `make_*` factories; new Driver category.

## Implementation

Public surface only: `cpu::Frame`, MD preset categorical setters, and
`cpu::md::FixFreeze(int type_id)`. Headers:
`include/atomiverse/cpu/dynamics/fix_freeze.h`,
`md.h`, `velocity_verlet.h`,
`include/atomiverse/cpu/potentials/pair_lj_cut.h`,
`include/atomiverse/cpu/neighbor/neighbor_list.h`. Unit tests (not e2e):
`tests/cpu/dynamics/test_fix_freeze.cpp`.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

// Geometry matches tools/metalwalls_oracle/inputs/lammps/freeze-electrode-atoms/
const int wall_type = 2;
const int fluid_type = 1;
const double wall_xyz[4][3] = {
    {2.0, 2.0, 0.0}, {2.0, 6.0, 0.0}, {6.0, 2.0, 0.0}, {6.0, 6.0, 0.0},
};

cpu::Frame frame(8);
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_x)[i] = wall_xyz[i][0];
    frame.span(kAtom_y)[i] = wall_xyz[i][1];
    frame.span(kAtom_z)[i] = wall_xyz[i][2];
    frame.span(kAtom_mass)[i] = 196.9665;
    frame.span(kAtom_type)[i] = wall_type;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_vx)[i] = frame.span(kAtom_vy)[i] = frame.span(kAtom_vz)[i] = 0.0;
}
for (int i = 0; i < 4; ++i) {
    const int j = 4 + i;
    frame.span(kAtom_x)[j] = wall_xyz[i][0];
    frame.span(kAtom_y)[j] = wall_xyz[i][1];
    frame.span(kAtom_z)[j] = 3.5;
    frame.span(kAtom_mass)[j] = 39.948;
    frame.span(kAtom_type)[j] = fluid_type;
    frame.span(kAtom_tag)[j] = j + 1;
}
frame.box = cpu::Box::cubic(10.0);

cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{/*eps [eV]=*/0.010341, /*sigma [Å]=*/3.4,
                                 /*cutoff [Å]=*/8.0});
sim.set_neighbor(cpu::md::NeighborList<>{8.0, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.set_observer(cpu::md::InitVelocity{/*T [K]=*/200.0, /*seed=*/99u});
sim.set_observer(cpu::md::FixFreeze{wall_type});  // electrode type
sim.build();
sim.run(frame, 1000);
```

```bash
# MD-only configure (binding for this loop): HF/PW/GPW/CUDA OFF
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_freeze-electrode-atoms -j"$(nproc)"
./build/regressions/reg_freeze-electrode-atoms
```

Offline LAMMPS twin (never launched by ctest):
`tools/metalwalls_oracle/inputs/lammps/freeze-electrode-atoms/in.lammps`
(correctness, 1000 steps) and `in.bench` (fair wall, 20000 steps).

## Validation (vs LAMMPS)

**Oracle: LAMMPS** develop / `patch_30Mar2026-1074-g7f680de296`
(30 Mar 2026 Development; binary `lmp` at pin time). Source:
`regressions/md/freeze-electrode-atoms/REFERENCE.md` (offline pin; never live
in ctest). Dump excerpt after 1000 steps (`wall_final.lammpstrj`): wall atoms
remain at create coordinates with zero velocities.

From LAMMPS `fix setforce 0.0 0.0 0.0` on group `wall`: after forces are
computed, wall forces are zeroed so NVE half-kicks do not change wall
velocities. With \(v_{\mathrm{wall}}(0)=0\), wall positions remain exactly at
create positions for the entire trajectory.

| Observable | Atomiverse / pin | LAMMPS | Tol / gate | Notes |
| ---------- | ---------------: | -----: | ---------- | ----- |
| Wall positions after 1000 steps | create \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å | identical (dump) | \(\max\|x_i(t)-x_i(0)\| \le 10^{-10}\) Å | by stable tag (sort-safe) |
| Wall velocities | all components 0 | all components 0 | \(\max\|v\| \le 10^{-12}\) Å/fs | metal dump v is Å/ps; pin is zero either way |
| Wall \(z\) span | \(z\in[0,0]\) | thermo `c_min_dz`/`c_max_dz` | exact | throughout |
| Fluid displacement | \(\ge 1\) atom moves | fluid free under LJ | \(\max\) fluid disp \(> 10^{-3}\) Å | freeze is **selective**, not global |

Fluid coordinates are **not** golden-matched: fluid RNG and unit-velocity
conventions differ between engines. The regression asserts immobility
invariants plus selective fluid motion only.

## Performance

Source: `REFERENCE.md` fair wall — same geometry, **20000** NVE steps, single
OpenMP thread (`OMP_NUM_THREADS=1`), no dump/thermo thrash. LAMMPS input:
`tools/metalwalls_oracle/inputs/lammps/freeze-electrode-atoms/in.bench`.
Atomiverse time instrumented in the regression binary (same \(N\), steps, \(dt\)).

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ------------------------------------------: | ---- | ------ |
| `FixFreeze` + VV + LJ vs `fix setforce 0` + NVE | **0.0186** (1 rank, OMP=1) | **0.0740** | **3.98** | ✓ PASS | Tiny \(N=8\); not a production throughput claim — fair twin ceiling only |

Gate: \(r \le 10.0\) → **PASS** (`within_10x=true`).

## Limitations

- **Foothold scale.** Eight atoms pin algorithm correctness and a fair ceiling,
  not large-slab electrode throughput.
- **Type-only selection.** One `type_id` per `FixFreeze`; region/id groups and
  multi-type freeze require composition or follow-on API.
- **Superset vs pure setforce.** Position restore + multi-stage \(v\) zeroing
  is stricter than LAMMPS post_force alone; good for electrode safety, not a
  claim of bit-identical intermediate force arrays mid-step.
- **No const-pot / charging physics here.** Pair this component with
  `ConstPotBO`, charge observers, and slab Ewald for full E1 stacks
  (`const-pot-md-nvt-short`, etc.).
- **Single-rank regression.** MPI migration re-resolves rows each launch in
  product code; this e2e is single-rank.

## References

- Offline pin: [`regressions/md/freeze-electrode-atoms/REFERENCE.md`](../../../regressions/md/freeze-electrode-atoms/REFERENCE.md)
- Regression consumer: `regressions/md/freeze-electrode-atoms/reg_freeze-electrode-atoms.cpp`
- LAMMPS offline inputs: `tools/metalwalls_oracle/inputs/lammps/freeze-electrode-atoms/`
- Component: `include/atomiverse/cpu/dynamics/fix_freeze.h`, `src/cpu/dynamics/fix_freeze.cpp`
- Queue / handoff: [metalwalls-method-queue.md](../../developer/metalwalls-method-queue.md),
  [metalwalls-method-handoff.md](../../developer/metalwalls-method-handoff.md)
- Design: [metalwalls-method-loop.md](../../developer/metalwalls-method-loop.md)
- LAMMPS: `fix setforce` documentation; source `src/fix_setforce.cpp`
  (develop pin version in REFERENCE)
- MetalWalls electrode `mobile = false` practice (physics end-state; **not** the
  binding oracle for this slug)
