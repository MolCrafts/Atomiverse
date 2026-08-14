# Steele 10-4-3 continuum wall — single Ar (LAMMPS `wall/lj1043`)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `steele-wall-argon` |
| Regression | `regressions/forcefield/steele-wall-argon/` |
| Public API | `cpu::md::FixWallLJ1043` (`Bias` Component; default `md::Phase::post_force`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **E8 done** (wave 2, 2026-07-24); force-fields note Steele wall |
| Oracle | **LAMMPS** `fix wall/lj1043` (metal units) |

## Summary

**Steele 10-4-3** is the integrated Lennard-Jones continuum wall for a planar
solid surface (Steele, *Surf. Sci.* **36**, 317 (1973)). Atomiverse ships it as
`cpu::md::FixWallLJ1043`, a **Bias** Component on `cpu::md::MD` (or on
`cpu::Evaluate` for a single-point), mirroring LAMMPS `fix wall/lj1043` — not
`wall/lj93` (that fix is the **9-3** integrated form). Energy is shifted so
\(U(r_c)=0\); force is along the wall normal only. The product gate
`reg_steele-wall-argon` is a **one-atom Ar** single-point above a \(z\)-lo wall
at \(z=0\), pinned offline to LAMMPS metal-unit PE and \(F_z\). Fair wall vs
LAMMPS is **`perf_exempt`** (sub-millisecond 1-atom pin; startup noise).

## Method

**Physics.** For perpendicular distance \(r\) from the wall plane to a fluid-side
atom, with well depth \(\varepsilon\) and length \(\sigma\), the shifted Steele
potential (LAMMPS `FixWallLJ1043`, metal units) is

\[
U(r)=2\pi\varepsilon\left[
  \frac{2}{5}\left(\frac{\sigma}{r}\right)^{10}
  -\left(\frac{\sigma}{r}\right)^{4}
  -\frac{\sqrt{2}\,\sigma^{3}}{3\bigl(r+(0.61/\sqrt{2})\sigma\bigr)^{3}}
\right]-U(r_c),\qquad 0 < r < r_c.
\]

For \(r\ge r_c\), \(U=0\) and \(F=0\). Force follows LAMMPS `wall_particle`:
only the normal component is nonzero; transverse forces are identically zero
for a planar wall. Sign convention: lo wall (`side = -1`) places fluid at
coordinates greater than `wall_pos`; the regression \(F_z < 0\) at \(r=4\) Å
means attraction toward the wall for Ar at the equilibrium side of the well.

**Oracle naming note.** Queue / map text sometimes wrote `wall/lj93` for
“Steele 10-4-3.” In LAMMPS, **`wall/lj93` is 9-3** and **`wall/lj1043` is
10-4-3**. This pin and public type use **`wall/lj1043`** only.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Atoms | 1 Ar (mass 39.948 amu) |
| Position | \((10, 10, 4)\) Å |
| Box | \(20\times 20\times 30\) Å orthorhombic (`Box::orthorhombic`) |
| Wall | axis \(z\) (`axis=2`), lo (`side=-1`), `wall_pos=0` Å |
| \(\varepsilon\) | 0.010338 eV |
| \(\sigma\) | 3.40 Å |
| \(r_c\) | 12.0 Å |
| Pair | none (wall only; LAMMPS twin uses `pair_style zero`) |
| Binding | `cpu::Evaluate` @ `evaluate::Phase::evaluate` (single-point) |

**Key exclusions (non-goals of this slug)**

- Variable / moving wall position, multi-wall in one object, lattice units.
- `fld yes` / field-only modes.
- Multi-atom bulk slab fair-bench twin (deferred; pin is textbook 1-atom).
- Live LAMMPS inside ctest; MetalWalls oracle (feature exists in LAMMPS).
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `FixWallLJ1043` ctor args / public members;
no Config bag). Minimal single-point matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_wall_lj1043.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame f(1);
f.span(kAtom_x)[0] = 10.0;
f.span(kAtom_y)[0] = 10.0;
f.span(kAtom_z)[0] = 4.0;   // r = 4 Å above zlo wall at 0
f.span(kAtom_mass)[0] = 39.948;
f.span(kAtom_type)[0] = 0;
f.span(kAtom_tag)[0] = 1;
f.box = cpu::Box::orthorhombic(20.0, 20.0, 30.0);

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::md::FixWallLJ1043{/*axis=*/2, /*side=*/-1, /*wall_pos=*/0.0,
                                /*epsilon [eV]=*/0.010338,
                                /*sigma [Å]=*/3.40,
                                /*cutoff [Å]=*/12.0});
eval.build();
eval.run(f);

const double E = f.meta.get(kEnergy);   // [eV]
const double fz = f.span(kAtom_fz)[0];  // [eV/Å]
```

MD binding (component defaults to `post_force` order +100 after pair/bonded/
kspace):

```cpp
cpu::md::MD sim;
// ... set_neighbor, set_evaluator, set_integrator as needed ...
sim.add(cpu::md::FixWallLJ1043{2, -1, 0.0, 0.010338, 3.40, 12.0});
// or: sim.add_ref(wall);  // borrow
sim.build();
sim.run(frame, nsteps);
```

Architecture rules:

- **Driver stays `cpu::md::MD` / `cpu::Evaluate`.** Walls are Components
  (`Bias`), never a new Driver category.
- Parameters are **constructor arguments and public members** — no
  `FixWallLJ1043Config`.
- Frame is the sole I/O currency; energy accumulates on `kEnergy`, force on
  the normal component of `kAtom_f*`.

Build and run the regression (MD-only configure; CPU login pattern from
`AGENTS.md` / metalwalls loop):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_steele-wall-argon -j"$(nproc)"
  ./build/regressions/reg_steele-wall-argon
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/fix_wall_lj1043.h` | Public `FixWallLJ1043` |
| `src/cpu/dynamics/fix_wall_lj1043.cpp` | Precompute + `launch` (LAMMPS coeffs) |
| `regressions/forcefield/steele-wall-argon/reg_steele-wall-argon.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/in.steele_wall_argon` | Offline twin input |

## Validation (vs LAMMPS)

Source:
[`regressions/forcefield/steele-wall-argon/REFERENCE.md`](../../../regressions/forcefield/steele-wall-argon/REFERENCE.md)
(offline pin; **never** live LAMMPS in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).
Twin input:
[`tools/metalwalls_oracle/inputs/lammps/in.steele_wall_argon`](../../../tools/metalwalls_oracle/inputs/lammps/in.steele_wall_argon).

**Oracle:** LAMMPS **30 Mar 2026 - Development** (`develop` /
`patch_30Mar2026-1074-g7f680de296`), metal units, `fix wall/lj1043 zlo 0.0
0.010338 3.40 12.0`, `fix_modify energy yes`, `pair_style zero`, `run 0`.

**Why LAMMPS (not MetalWalls):** the algorithm exists as LAMMPS
`fix wall/lj1043`. Binding oracle priority for this loop is **LAMMPS first**
when the feature overlaps. MetalWalls is reserved for MW-only physics.

### Accuracy

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E\) [eV] | **−0.035247965699900359** | same | \(\lvert\Delta E\rvert\le 10^{-10}\) | hard-coded gate |
| \(F_z\) [eV/Å] | **−0.025162258813859549** | same | \(\lvert\Delta F_z\rvert\le 10^{-10}\) | normal force |
| \(F_x, F_y\) [eV/Å] | **0** | 0 | \(\lvert F_{x,y}\rvert\le 10^{-12}\) | planar wall |
| Analytic vs LAMMPS dump | — | forces.dump %.16e | ≥ 1e-15 | independent closed-form check (REFERENCE) |

Cross-check heights (same \(\varepsilon,\sigma,r_c\); **not** regression gates):

| \(r\) [Å] | \(E\) [eV] | \(F_z\) [eV/Å] |
| --------: | ---------: | -------------: |
| 3.0 | −2.892345800523042×10⁻² | +1.508268784423809×10⁻¹ |
| 4.0 | −3.524796569990036×10⁻² | −2.516225881385955×10⁻² |
| 5.0 | −1.687854854072667×10⁻² | −1.207703871038502×10⁻² |
| ≥ \(r_c\) | 0 | 0 |

Geometry, parameters, and tolerances: **Method** table and REFERENCE (match
`reg_steele-wall-argon`).

### Offline pin repro (never ctest)

```bash
# Requires a local LAMMPS binary with fix wall/lj1043 (standard package)
lmp -in tools/metalwalls_oracle/inputs/lammps/in.steele_wall_argon
# PE / Fz pins recorded in REFERENCE.md (2026-07-24)
```

## Performance

Source: REFERENCE.md; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| 1-atom wall single-point (`run 0`) | — | — | **perf_exempt** | n/a | Wall dominated by process startup, not kernel; fair multi-atom slab twin deferred |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | single-atom wall single-point pin; wall dominated by process startup, not kernel. Fair multi-atom slab twin deferred. |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a multi-atom continuum-wall twin is
defined offline, record version + walls and replace `perf_exempt` in REFERENCE.

## Limitations

- **Foothold scope.** Fixed planar wall; no variable wall motion, multi-wall
  object, or lattice-unit wall coordinates.
- **Behind-wall atoms.** \(r\le 0\) raises `std::runtime_error` naming the axis,
  the wall position and the offending coordinate. The 10-4-3 form is defined
  only on \(0<r<r_c\) — it diverges as \(r\to0^+\) and turns attractive past its
  pole at \(r=-(0.61/\sqrt2)\sigma\) — so a behind-wall atom has no wall energy
  to report. Reduce the timestep or move the wall.
- **No fair speed ratio** until a multi-atom slab twin is pinned offline.
- **E8 force-field slice only.** Related continuum / pair styles (Born,
  Fumi–Tosi, Buckingham) are sibling slugs with their own pages when written.
- **Oracle is LAMMPS `wall/lj1043` only** — do not compare to `wall/lj93`
  (different functional form).

## References

- Offline pin: [`regressions/forcefield/steele-wall-argon/REFERENCE.md`](../../../regressions/forcefield/steele-wall-argon/REFERENCE.md)
- Product gate: [`regressions/forcefield/steele-wall-argon/reg_steele-wall-argon.cpp`](../../../regressions/forcefield/steele-wall-argon/reg_steele-wall-argon.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/fix_wall_lj1043.h`](../../../include/atomiverse/cpu/dynamics/fix_wall_lj1043.h)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Roadmap: [`docs/roadmap/force-fields.md`](../../roadmap/force-fields.md) (Steele 10-4-3 / `FixWallLJ1043`)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), twin [`tools/metalwalls_oracle/inputs/lammps/in.steele_wall_argon`](../../../tools/metalwalls_oracle/inputs/lammps/in.steele_wall_argon)
- Literature: Steele, *Surf. Sci.* **36**, 317 (1973); LAMMPS `fix wall` / `wall/lj1043` documentation
