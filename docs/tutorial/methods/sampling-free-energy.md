# Free-energy methods

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Free-energy and steered-sampling workflows composed from ordinary Components on
the classical drivers. The page documents the constant-force pull: a fixed
external force on a selected atom type, shipped as `cpu::md::FixAddForce`, a
`Bias` Component (default `md::Phase::post_force`) bound on `cpu::md::MD` via
`set_evaluator`, or added at `evaluate::Phase::evaluate` on `cpu::Evaluate` for
a single point. It mirrors the constant-force path of LAMMPS `fix addforce`
(metal units). Coverage is CPU only: force and bias energy at a single point,
and a short NVE trajectory with its work proxy.

The case also registers the section `fep_soft_dimer` (`REQUIRES PATH`, see
`case.cmake`), which has no section on this page.

Build: the section requires the `MD` module (CPU login shell with
`buildenv-gcc/2026.03-mpich`, as in the top-level build instructions).

Case: `ctest -R '^sampling/free-energy$'` — sources and oracle record in [`regressions/sampling/free-energy/`](../../../regressions/sampling/free-energy/). One section of the case per `##` heading below: `atv_regressions --case sampling/free-energy --section <name>`.

## constant-force-pull-ion — Constant-force ion pull (LAMMPS `fix addforce`) {#constant-force-pull-ion}

### Summary

Constant-force pull applies a fixed external force \(\mathbf{F}^{\mathrm{add}}\)
to a selected atom type each step — the classical MD foothold for steered
transport of an ion (or tagged group) under a constant load. The gate is a
two-atom textbook case (pulled ion + inert background, `pair_style zero` twin):
Gate A checks force and bias energy at `run 0`; Gate B checks short NVE
displacement and work against an offline LAMMPS pin. Fair wall versus LAMMPS
gives ratio \(r \approx 0.12\) (within 10×) on a 1000-atom / 5000-step
pure-addforce twin.

### Method

For every local atom \(i\) whose `type_id` matches the selected type, each
launch accumulates

\[
\mathbf{F}_i \mathrel{+}= \mathbf{F}^{\mathrm{add}},\qquad
U \mathrel{+}= -\mathbf{F}^{\mathrm{add}}\cdot\mathbf{r}_i^{\mathrm{u}},
\]

with \(\mathbf{F}^{\mathrm{add}}\) and \(\mathbf{F}\) in eV/Å and \(U\) in eV.
Unwrapped coordinates use orthorhombic image flags
(\(\mathbf{r}^{\mathrm{u}} = \mathbf{r} + \mathrm{image}\cdot\mathbf{L}\)).
Atoms of other types receive zero added force. There is no pair potential on
this pin (LAMMPS twin: `pair_style zero`); the only PE is the bias scalar above
(LAMMPS fix scalar `f_1`).

**Trajectory (Gate B).** From rest under constant \(F_z\) with VelocityVerlet
NVE, the ion accelerates along \(z\). Work proxy and bias energy at the final
geometry:

\[
W = F_z\,\Delta z,\qquad
U = -\mathbf{F}^{\mathrm{add}}\cdot\mathbf{r}^{\mathrm{u}}.
\]

To match the LAMMPS `setup`/`run 0` force evaluation before the first
half-kick, Gate B seeds \(\mathbf{F}\) on the Frame with a single-point
`cpu::Evaluate` + `FixAddForce` before `MD::run` (not a separate public API).

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

**Out of scope:** variable (`v_`) forces, the region keyword, `energy v_…`,
RESPA levels; LAMMPS `fix smd` (spring / constant-velocity CV); a virial tally
for the added force; triclinic unwrap (orthorhombic only).

### Usage

Parameters are constructor arguments / public members (`Fx`, `Fy`, `Fz`,
`type_id`); bias energy accumulates on `kEnergy`, force on `kAtom_f*`. The
external force is a Component, never a Driver category and never folded into
`KSpace*`. Single point (Gate A):

```cpp
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

Short NVE MD (Gate B; the component defaults to `post_force`, order +100):

```cpp
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.set_evaluator(cpu::md::FixAddForce{0.0, 0.0, Fz, /*type=*/1});
sim.build();
sim.run(frame, /*n_steps=*/100);
```

Public header `fix_addforce.h`; implementation `src/cpu/dynamics/fix_addforce.cpp`
(`launch`: constant path + bias energy).

### Validation (oracle: LAMMPS)

Oracle: LAMMPS **30 Mar 2026 - Development** (`develop` /
`patch_30Mar2026-1074-g7f680de296`), metal units, `fix addforce 0 0 0.05` on
the type-1 group, `pair_style zero`, Gate A `run 0` + Gate B NVE 100 steps at
`timestep 0.001` ps (= 1 fs). Offline pins only — never live LAMMPS in ctest.
LAMMPS is the binding oracle because the algorithm exists as `fix addforce`
(MetalWalls is reserved for MetalWalls-only physics).

**Gate A — single point (`run 0`)**

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

Offline repro (never ctest): `lmp -in <LAMMPS oracle input for this case>`
with a LAMMPS binary carrying `fix addforce` (standard package); pins are
recorded in `REFERENCE.md`.

### Performance

Comparable pure-addforce workload (pair zero / inert, NVE, addforce on one
type; OMP=1; same login node):

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| 1000 atoms, 5000 steps, `fix addforce` / `FixAddForce` + NVE, \(dt=1\) fs | **0.455** (Loop time) | **0.055** (chrono around `MD::run`) | **0.12** | ✓ | Integrate-then-force vs LAMMPS setup; seed \(F\) outside timed region beyond natural first-step recompute |

`perf_exempt` false; ratio \(T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} = 0.12\);
`within_10x` true.

### Limitations

- **Constant path only.** No variable force expressions, regions, or RESPA
  levels; no `energy v_…` override of the default \(-\mathbf{F}\cdot\mathbf{r}^{\mathrm{u}}\) bias.
- **Not SMD.** Constant-velocity / spring `fix smd` is not implemented (roadmap ○).
- **Orthorhombic unwrap.** Triclinic image unwrap throws.
- **No virial for the added force.**
- **MD stage ordering.** First-step trajectory parity with LAMMPS requires
  seeding forces before `MD::run` (or accepting a half-step lag); production
  multi-potential runs recompute every `post_force` after pairs/kspace.
- **Oracle is LAMMPS `fix addforce` only** — not MetalWalls, not a literature
  free-particle formula as the primary pin (though the NVE trajectory is
  elementary under constant \(F\)).

### References

- Roadmap: [`docs/roadmap/molecular-dynamics.md`](../../roadmap/molecular-dynamics.md) (`FixAddForce` / constant-force pull)
- LAMMPS: <https://docs.lammps.org/fix_addforce.html>; source `src/fix_addforce.cpp` (constant force path)
