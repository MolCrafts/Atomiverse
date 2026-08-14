# Constant-force ion pull — LAMMPS `fix addforce`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `constant-force-pull-ion` |
| Regression | `regressions/md/constant-force-pull-ion/` |
| Public API | `cpu::md::FixAddForce` (`Bias` Component; default `md::Phase::post_force`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **shared classical MD / EDL transport twin** (wave 3, 2026-07-24); `docs/roadmap/molecular-dynamics.md` → FixAddForce |
| Oracle | **LAMMPS** `fix addforce` (metal units) |

## Summary

**Constant-force pull** applies a fixed external force \(\mathbf{F}^{\mathrm{add}}\)
to a selected atom type each step — the classical MD foothold for steered
transport of an ion (or tagged group) under a constant load, without a
pair potential on the pin. Atomiverse ships it as `cpu::md::FixAddForce`, a
**Bias** Component on `cpu::md::MD` (or on `cpu::Evaluate` for a single-point),
mirroring LAMMPS `fix addforce` on the **constant** force path. The product
gate `reg_constant-force-pull-ion` is a **two-atom** textbook case (pulled ion
+ inert background, `pair_style zero` twin): Gate A checks force and bias
energy at `run 0`; Gate B checks short NVE displacement and work vs an offline
LAMMPS pin. Fair wall vs LAMMPS reports **ratio \(r \approx 0.12\)** (within
10×) on a 1000-atom / 5000-step pure-addforce twin.

## Method

**Physics.** For every local atom \(i\) with `type_id` matching the selected
type, each launch accumulates

\[
\mathbf{F}_i \mathrel{+}= \mathbf{F}^{\mathrm{add}},\qquad
U \mathrel{+}= -\mathbf{F}^{\mathrm{add}}\cdot\mathbf{r}_i^{\mathrm{u}},
\]

with \(\mathbf{F}^{\mathrm{add}}\) in eV/Å, \(\mathbf{F}\) in eV/Å, and \(U\)
in eV. Unwrapped coordinates use orthorhombic image flags
(\(\mathbf{r}^{\mathrm{u}} = \mathbf{r} + \mathrm{image}\cdot\mathbf{L}\)).
Atoms of other types receive zero added force. There is **no** pair potential
on this pin (LAMMPS twin: `pair_style zero`); the only PE is the bias scalar
above (LAMMPS fix scalar `f_1`).

**Trajectory (Gate B).** From rest under constant \(F_z\) with VelocityVerlet
NVE, the ion accelerates along \(z\). Work proxy and bias energy at the final
geometry:

\[
W = F_z\,\Delta z,\qquad
U = -\mathbf{F}^{\mathrm{add}}\cdot\mathbf{r}^{\mathrm{u}}.
\]

ATV MD stage order is integrate-then-force; to match LAMMPS `setup`/`run 0`
force evaluation before the first half-kick, Gate B **seeds** \(\mathbf{F}\)
on the Frame once before `MD::run` (documented in REFERENCE; not a separate
public API).

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Atoms | 2 (type 1 ion, type 2 inert background) |
| Box | \(L=20\) Å cubic, PBC |
| \(\mathbf{F}^{\mathrm{add}}\) | \((0,\,0,\,0.05)\) eV/Å on type 1 only |
| Ion (type 1) | \(q=+1\), \(m=39.948\) amu, \(\mathbf{r}=(10,10,5)\) Å, \(\mathbf{v}=\mathbf{0}\) |
| Background (type 2) | \(q=0\), \(m=39.948\) amu, \(\mathbf{r}=(10,10,15)\) Å, \(\mathbf{v}=\mathbf{0}\) |
| Pair | none (`pair_style zero` twin) |
| Gate A | `cpu::Evaluate` + `FixAddForce` single-point |
| Gate B | `cpu::md::MD` + `VelocityVerlet{1.0}` fs, 100 NVE steps |

**Key exclusions (non-goals of this slug)**

- Variable (`v_`) forces, region keyword, `energy v_…`, RESPA level.
- LAMMPS `fix smd` (spring / constant-velocity CV) — deferred.
- Virial tally for the added force; triclinic unwrap (foothold is orthorhombic only).
- Live LAMMPS inside ctest; MetalWalls oracle (feature exists in LAMMPS).
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `FixAddForce` ctor args / public members;
no Config bag). Minimal single-point matching Gate A of the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/fix_addforce.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame f(2);
f.span(kAtom_x)[0] = 10.0; f.span(kAtom_y)[0] = 10.0; f.span(kAtom_z)[0] = 5.0;
f.span(kAtom_x)[1] = 10.0; f.span(kAtom_y)[1] = 10.0; f.span(kAtom_z)[1] = 15.0;
f.span(kAtom_type)[0] = 1;
f.span(kAtom_type)[1] = 2;
f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 39.948;
f.span(kAtom_charge)[0] = 1.0;
f.span(kAtom_charge)[1] = 0.0;
f.span(kAtom_tag)[0] = 1;
f.span(kAtom_tag)[1] = 2;
f.box = cpu::Box::cubic(20.0);

const double Fz = 0.05;  // [eV/Å]
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::md::FixAddForce{/*Fx=*/0.0, /*Fy=*/0.0, /*Fz=*/Fz, /*type=*/1});
eval.build();
eval.run(f);

const double U = f.meta.get(kEnergy);   // [eV]  bias only → −0.25
const double fz = f.span(kAtom_fz)[0];  // [eV/Å] → 0.05 on type 1
```

Short NVE MD (Gate B shape — component defaults to `post_force` order +100).
Frame is the only I/O token; `FixAddForce` is bound on the driver:

```cpp
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.add(cpu::md::FixAddForce{0.0, 0.0, Fz, /*type=*/1});
// or: sim.add_ref(fix);  // borrow
sim.build();
sim.run(frame, /*n_steps=*/100);
```

Architecture rules:

- **Driver stays `cpu::md::MD` / `cpu::Evaluate`.** External force is a Component
  (`Bias`), never a new Driver category and never folded into `KSpace*`.
- Parameters are **constructor arguments and public members** (`Fx`, `Fy`,
  `Fz`, `type_id`) — no `FixAddForceConfig`.
- Frame is the sole I/O currency; bias energy accumulates on `kEnergy`, force
  on `kAtom_f*`.

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
  cmake --build build --target reg_constant-force-pull-ion -j"$(nproc)"
  ./build/regressions/reg_constant-force-pull-ion
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/fix_addforce.h` | Public `FixAddForce` |
| `src/cpu/dynamics/fix_addforce.cpp` | `launch` (constant path + bias energy) |
| `regressions/md/constant-force-pull-ion/reg_constant-force-pull-ion.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/in.addforce_pin` | Offline twin input |

## Validation (vs LAMMPS)

Source:
[`regressions/md/constant-force-pull-ion/REFERENCE.md`](../../../regressions/md/constant-force-pull-ion/REFERENCE.md)
(offline pin; **never** live LAMMPS in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).
Twin input:
[`tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/in.addforce_pin`](../../../tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/in.addforce_pin).

**Oracle:** LAMMPS **30 Mar 2026 - Development** (`develop` /
`patch_30Mar2026-1074-g7f680de296`), metal units, `fix addforce 0 0 0.05` on
type-1 group, `pair_style zero`, Gate A `run 0` + Gate B NVE 100 steps at
`timestep 0.001` ps (= 1 fs).

**Why LAMMPS (not MetalWalls):** the algorithm exists as LAMMPS
`fix addforce`. Binding oracle priority for this loop is **LAMMPS first**
when the feature overlaps. MetalWalls is reserved for MW-only physics.

### Accuracy

Dual gate: force+energy (Gate A) and short-MD displacement / work (Gate B) —
not a self-golden of ATV output.

**Gate A — single-point (`run 0`)**

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(U\) bias [eV] | **−0.25** | −0.25 (`f_1`) | \(10^{-12}\) | \(U=-F_z z_0\), \(z_0=5\) Å |
| \(F_x,F_y,F_z\) ion [eV/Å] | **(0, 0, 0.05)** | same | \(10^{-12}\) | type 1 only |
| \(F_x,F_y,F_z\) bg [eV/Å] | **(0, 0, 0)** | same | \(10^{-12}\) | type 2 inert |
| Pair PE [eV] | **0** | 0 | — | `pair_style zero` |

**Gate B — NVE 100 steps, \(dt=1\) fs, \(v_0=0\)**

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(z_{\mathrm{ion}}\) final [Å] | **5.060381832778068** | same | \(5\times10^{-6}\) | `force_to_accel` vs metal `ftm2v` |
| \(\Delta z\) [Å] | **0.0603818327780683** | same | (from \(z\)) | |
| Work proxy \(W=F_z\Delta z\) [eV] | **0.00301909163890342** | same | \(5\times10^{-7}\) | |
| Bias \(U\) final [eV] | **−0.2530190916389034** | `f_1` same | \(5\times10^{-7}\) | recomputed last `post_force` |
| \(z_{\mathrm{bg}}\) [Å] | **15** | 15 | \(10^{-12}\) | unmoved |
| \(F_z\) ion still [eV/Å] | **0.05** | 0.05 | \(10^{-12}\) | constant force path |

Geometry, parameters, and tolerances: **Method** table and REFERENCE (match
`reg_constant-force-pull-ion`).

### Offline pin repro (never ctest)

```bash
# Requires a local LAMMPS binary with fix addforce (standard package)
lmp -in tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/in.addforce_pin
# U / F / z / work pins recorded in REFERENCE.md (2026-07-24)
```

## Performance

Source: REFERENCE.md; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md);
queue
[`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md).

Comparable pure-addforce workload (pair zero / inert, NVE, addforce on one
type; OMP=1; same login node, 2026-07-24):

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| 1000 atoms, 5000 steps, `fix addforce` / `FixAddForce` + NVE, \(dt=1\) fs | **0.455** (Loop time) | **0.055** (chrono around `MD::run`) | **0.12** | ✓ | Integrate-then-force vs LAMMPS setup; seed \(F\) outside timed region beyond natural first-step recompute |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **false** |
| `ratio_atv_over_oracle` | **0.12** (\(T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\)) |
| `within_10x` | **true** |

Do **not** invent wall-clock ratios. Numbers above are the offline fair-bench
recorded in REFERENCE / handoff / queue for this slug.

## Limitations

- **Constant path only.** No variable force expressions, regions, or RESPA
  levels; no `energy v_…` override of the default \(-\mathbf{F}\cdot\mathbf{r}^{\mathrm{u}}\) bias.
- **Not SMD.** Constant-velocity / spring `fix smd` is deferred (roadmap ○).
- **Orthorhombic unwrap.** Triclinic image unwrap throws in this foothold.
- **No virial for the added force** in this implementation.
- **MD stage ordering.** First-step trajectory parity with LAMMPS requires
  seeding forces before `MD::run` (or accepting a half-step lag); production
  multi-potential runs recompute every `post_force` after pairs/kspace.
- **Oracle is LAMMPS `fix addforce` only** — not MetalWalls, not a literature
  free-particle formula as the primary pin (though the NVE trajectory is
  elementary under constant \(F\)).

## References

- Offline pin: [`regressions/md/constant-force-pull-ion/REFERENCE.md`](../../../regressions/md/constant-force-pull-ion/REFERENCE.md)
- Product gate: [`regressions/md/constant-force-pull-ion/reg_constant-force-pull-ion.cpp`](../../../regressions/md/constant-force-pull-ion/reg_constant-force-pull-ion.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/fix_addforce.h`](../../../include/atomiverse/cpu/dynamics/fix_addforce.h)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Roadmap: [`docs/roadmap/molecular-dynamics.md`](../../roadmap/molecular-dynamics.md) (`FixAddForce` / constant-force pull)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), twin [`tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/in.addforce_pin`](../../../tools/metalwalls_oracle/inputs/lammps/constant-force-pull-ion/in.addforce_pin)
- LAMMPS: <https://docs.lammps.org/fix_addforce.html>; source `src/fix_addforce.cpp` (constant force path)
