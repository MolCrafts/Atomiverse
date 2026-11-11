# External electric and displacement fields

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Uniform electric fields on classical point charges, and the finite-field
routes of electrochemical MD built on them. Every variant is a `Bias`
Component on `cpu::md::MD` (default `md::Phase::post_force`, order +100,
matching LAMMPS `POST_FORCE`) or on `cpu::Evaluate` for a single-point check —
never a new Driver category, never a charge solve inside `KSpace*`. Parameters
are constructor arguments and public members; there is no `*Config` bag and no
factory. Forces accumulate on `kAtom_f*`, energies on `kEnergy`.

- `cpu::md::FixEfield{Ex, Ey, Ez}` — constant field (LAMMPS `fix efield`).
- `cpu::md::FixOscillatingEfield(Ex0, Ey0, Ez0, period, dt, phase)` — AC cosine
  field (LAMMPS `fix efield` with an equal-style variable).
- `cpu::md::ConstPotBO` (members `Ex`/`Ey`/`Ez` on the BO right-hand side) +
  `FixEfield` + `cpu::md::ClassicalPolarization` — finite field with a single
  constant-potential electrode (Dufils).
- `cpu::md::FixDfield(Dx, Dy, Dz)` with `polarization_feedback = true` — the
  Zhang–Sprik constant-\(D\) ensemble.

All sections run on CPU only. The case also registers the section
`displacement-dfield-md` (the vacuum `FixDfield` foothold, \(\mathbf{E}\equiv
\mathbf{D}\); `REQUIRES KSPACE MD`), which has no section on this page.

Case: `ctest -R '^md/electric-fields$'` — sources and oracle record in [`regressions/md/electric-fields/`](../../../regressions/md/electric-fields/). One section of the case per `##` heading below: `atv_regressions --case md/electric-fields --section <name>`.

## Common setup

- **Build:** the sections below require the `MD` module
  (`constant-d-polarization-feedback` also `KSPACE`); CPU login shell with
  `buildenv-gcc/2026.03-mpich`, as in the top-level build instructions.
- **Oracle policy.** When the feature exists in LAMMPS, LAMMPS is the binding
  oracle for both the offline pin and the fair wall (`external-efield-md`,
  `oscillating-efield-md`). MetalWalls-only physics (single-wall finite field,
  constant-\(D\)) pins against closed-form literature expressions of the
  functional MetalWalls implements, because no MetalWalls binary is installed on
  the agent host. Offline pins only — never live LAMMPS or MetalWalls in ctest.
- **LAMMPS version** (both LAMMPS sections): 30 Mar 2026 - Development
  (`patch_30Mar2026-1074-g7f680de296`), binary recorded in `REFERENCE.md` as
  `/home/jicli594/.local/x86_64/bin/lmp`, metal units.
- **Field physics.** In metal units the force conversion factor is
  \(q_{\mathrm{e2f}} = 1\), so for a field \(\mathbf{E}\) [V/Å] on charges
  \(q_i\) [e]

\[
\mathbf{F}_i = q_i\,\mathbf{E},
\qquad
U = -\sum_i q_i\,\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}
\quad\text{[eV/Å, eV]},
\]

  with orthorhombic unwrapping (LAMMPS `domain->unmap`, Frame image flags)
  \(\mathbf{r}_i^{\mathrm{u}} = \mathbf{r}_i + \mathrm{image}_i \odot
  \mathbf{L}\). The energy identity is the LAMMPS `fix_efield.cpp` constant path
  (`fsum[0] -= F·unwrap`). Triclinic boxes throw at launch.
- **Three-charge pin geometry** (`external-efield-md`, `oscillating-efield-md`):
  cubic \(L = 20\) Å, PBC; atom 1 \(q=+1\) at \((5,10,10)\) Å, atom 2 \(q=-1\) at
  \((7,10,10)\) Å, atom 3 \(q=+0.5\) at \((10,12,8)\) Å; image flags all zero;
  no pair (LAMMPS twin `pair_style zero`). Here \(\sum_i q_i z_i = 4\) Å·e.
- **Fair-wall twins** (LAMMPS sections): 1000 atoms, 5000 steps, OMP=1, same
  login node; LAMMPS `pair zero` + `fix nve`; ATV inert `PairLJCut` (rc = 2 Å ≪
  spacing) + VelocityVerlet, dt = 1 fs. The walls isolate field bookkeeping + MD
  step overhead, not a neighbor/kspace-heavy EDL bench.
- **Shared limitations.** Charges are inputs: the field Components do not
  induce or solve electrode charges. No `region` keyword, no dipole torque path.
  The MetalWalls capability map is `.claude/notes/metalwalls-capability-map.md`
  (E3; "Displacement D-field", P2).

### References

- LAMMPS `fix efield`: <https://docs.lammps.org/fix_efield.html>; source `src/fix_efield.cpp`
- A. Marin-Laflèche et al., *J. Open Source Softw.* **5**, 2373 (2020) (MetalWalls)

## external-efield-md — Constant external field (LAMMPS `fix efield`) {#external-efield-md}

### Summary

Uniform, time-constant external field on classical point charges through
`cpu::md::FixEfield` (public members `Ex`/`Ey`/`Ez` [V/Å]). This is the
constant-field foothold of the electrochemical E3 slice; finite-field electrode
work builds on it. The pin is the LAMMPS `fix efield` `run 0` on the
three-charge geometry with \(\mathbf{E} = (0.1,\,-0.05,\,0.2)\) V/Å (dump forces
+ fix scalar `f_1`). Expected forces are \(\mathbf{F}_i = q_i\mathbf{E}\)
exactly; field energy \(U = -0.8\) eV. Fair wall \(r =
T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} = 0.60\) (≤10×).

**Out of scope:** variable (`v_`) fields, the minimization-only variable energy
keyword, time-varying or gradient fields (see
[`oscillating-efield-md`](#oscillating-efield-md)), and the single-electrode
finite field ([`finite-field-single-electrode`](#finite-field-single-electrode)).

### Usage

```cpp
cpu::Frame frame(3);
// set kAtom_x/y/z, kAtom_charge, kAtom_mass, kAtom_type, kAtom_tag, zero forces
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::md::FixEfield{/*Ex=*/0.1, /*Ey=*/-0.05, /*Ez=*/0.2});
eval.build();
eval.run(frame);

const double U = frame.meta.get(kEnergy);  // [eV]
// forces on frame.span(kAtom_fx/fy/fz) [eV/Å]
```

On classical MD the component self-binds at `post_force` (order +100), so
pair/kspace forces accumulate first. Implementation:
`src/cpu/dynamics/fix_efield.cpp` (OMP force + unwrapped energy; accumulates into
`kEnergy`); unit tests in `tests/cpu/dynamics/test_fix_efield.cpp` (including
nonzero image unwrap).

### Validation (oracle: LAMMPS)

| Atom | \(q\) [e] | \(F_x\) | \(F_y\) | \(F_z\) [eV/Å] | Tol |
| ---: | --------: | ------: | ------: | -------------: | --- |
| 1 | +1 | 0.1 | −0.05 | 0.2 | \(10^{-12}\) |
| 2 | −1 | −0.1 | 0.05 | −0.2 | \(10^{-12}\) |
| 3 | +0.5 | 0.05 | −0.025 | 0.1 | \(10^{-12}\) |

| Quantity | Atomiverse / pin | LAMMPS | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| Field energy \(U\) [eV] | −0.8 | −0.8 (`f_1`) | \(10^{-12}\) | \(U=-\sum q_i\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}\) |
| Pair PE [eV] | 0 | 0 | — | twin uses `pair_style zero` |

Force and energy must both match. With zero image flags, wrapped ≡ unwrapped
for the energy pin.

### Performance

| Twin | Workload | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\) | ≤10× |
| ---- | -------- | --------------------------: | -----------------------: | -----------------------------------------: | ---- |
| Pure efield MD | \(N=1000\), 5000 steps, dt = 1 fs; LAMMPS `fix efield 0.05 0 0`; ATV `FixEfield{0.05,0,0}` | 0.21 | 0.126 | **0.60** | ✓ |

`within_10x` true at \(r \approx 0.60\) (ATV faster on this pure-bias wall).

### Limitations

- **Constant field only.** No `v_` variable field and no spatial region mask.
- **Single-rank path.** MPI DD ownership uses `owned_count`, but multi-rank
  efield energy reduction is not part of the pin.

## oscillating-efield-md — Oscillating (AC) external field {#oscillating-efield-md}

### Summary

Time-dependent uniform field on classical point charges: an AC cosine envelope
\(\mathbf{E}(t)=\mathbf{E}_0\cos(2\pi t/T+\varphi)\) with forces
\(\mathbf{F}_i=q_i\mathbf{E}(t)\) and constant-path potential energy at frozen
\(E(t)\), via `cpu::md::FixOscillatingEfield`. Offline pin and fair wall use
LAMMPS variable `fix efield` (equal-style `v_Ez`). Atomiverse public time is
**fs** (metal-unit twin: \(T=1\) ps, \(\Delta t=0.001\) ps \(\Leftrightarrow\)
\(T=1000\) fs, \(\Delta t=1\) fs). Fair ratio
\(r=T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\approx\mathbf{2.12}\) (≤ 10×).

### Method

\[
\mathbf{E}(t) = \mathbf{E}_0\cos\!\bigl(2\pi\, t / T + \varphi\bigr),
\qquad t = n_{\mathrm{step}}\,\Delta t,
\]

\[
\mathbf{F}_i \mathrel{+}= q_i\,\mathbf{E}(t),\qquad
U \mathrel{+}= -q_i\,\mathbf{E}(t)\cdot\mathbf{r}_i^{\mathrm{u}}
\quad\text{[eV, eV/Å]}.
\]

Instantaneous energy freezes \(\mathbf{E}\) over the timestep (LAMMPS
variable-path convention; the oracle pins energy with atom-style `potential`
\(\phi=-\mathbf{E}\cdot\mathbf{r}\)). Pin settings on the three-charge geometry:
\(\mathbf{E}_0 = (0,\,0,\,0.2)\) V/Å, period \(T = 1000\) fs (LAMMPS metal
\(1.0\) ps), \(\Delta t = 1\) fs (LAMMPS metal \(0.001\) ps), \(\varphi = 0\).
With \(\sum_i q_i z_i = 4\) Å·e,

\[
U(t) = -4\,E_z(t)\ \mathrm{eV},\qquad
F_{z,i}(t) = q_i E_z(t).
\]

Offline pin table (LAMMPS `run 0` at selected steps; full table in the pin
notes):

| step \(n\) | \(t\) (metal ps / ATV fs) | \(E_z\) [V/Å] | \(U\) [eV] | \(F_{z,1}\) | \(F_{z,2}\) | \(F_{z,3}\) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | **0.2** | **−0.8** | 0.2 | −0.2 | 0.1 |
| 125 | 0.125 / 125 | **0.14142135623731000** | **−0.5656854249492400** | \(q_i E_z\) | … | … |
| 250 | 0.25 | ~0 (≤ 2e−17) | ~0 | ~0 | ~0 | ~0 |
| 375 | 0.375 | **−0.14142135623731000** | **+0.5656854249492400** | … | … | … |
| 500 | 0.5 | **−0.2** | **+0.8** | −0.2 | 0.2 | −0.1 |
| 750 | 0.75 | ~0 | ~0 | ~0 | ~0 | ~0 |
| 1000 | 1.0 | **0.2** | **−0.8** | 0.2 | −0.2 | 0.1 |

The section gates steps **0, 125, 500** (max / mid-quarter / min): per-atom
\(\mathbf{F}\) and `frame.meta.get(kEnergy)` against the pin. Each pin step is a
single-point `cpu::Evaluate` at step 0 with the step folded into the phase,
\(\varphi = 2\pi n\,\Delta t/T\), so \(E(0,\varphi) = E(n, 0)\).

**Out of scope:** atom-style spatial fields, an arbitrary equal-style expression
engine beyond the AC cosine envelope, coupling the AC field into the
`ConstPotBO` right-hand side (Dufils AC + constant potential).

### Usage

```cpp
cpu::Frame frame(3);
frame.span(kAtom_charge)[0] = 1.0;   // atom 1: q=+1 at (5,10,10)
frame.span(kAtom_x)[0] = 5.0;
frame.span(kAtom_y)[0] = 10.0;
frame.span(kAtom_z)[0] = 10.0;
// … fill atoms 2–3; tags/types/mass as needed
frame.box = cpu::Box::cubic(20.0);

// Amplitudes [V/Å], period and dt [fs], phase [rad] (default 0).
const double phi = 2.0 * std::acos(-1.0) * step * dt / period;
cpu::md::FixOscillatingEfield osc(/*Ex0=*/0.0, /*Ey0=*/0.0, /*Ez0=*/0.2,
                                  /*period [fs]=*/1000.0, /*dt [fs]=*/1.0, phi);
cpu::Evaluate eval;
eval.set_evaluator_ref(osc);
eval.build();
eval.run(frame);
// osc.last_Ez() → 0.2 [V/Å] at step 0; frame.meta.get(kEnergy) → −0.8 [eV]
// Fz[i] → q[i] * Ez
```

On `cpu::md::MD` the Component binds at `post_force` +100 and forms
\(t=n\cdot\Delta t\) from the step index; the caller must pass a `dt`
consistent with the integrator timestep. Public members: `Ex0`/`Ey0`/`Ez0`,
`period`, `dt`, `phase`. Files: `fix_oscillating_efield.h`,
`src/cpu/dynamics/fix_oscillating_efield.cpp`; unit tests in
`tests/cpu/dynamics/test_fix_oscillating_efield.cpp`.

### Validation (oracle: LAMMPS)

Oracle: LAMMPS variable `fix efield`, equal-style cosine.

| Quantity | Atomiverse gate | LAMMPS pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| \(E_z(0)\) [V/Å] | `last_Ez()` | **0.2** | abs ≤ **1e-12** | \(\cos 0 = 1\) |
| \(U(0)\) [eV] | `kEnergy` | **−0.8** | abs ≤ **1e-12** | \(U=-4 E_z\) |
| \(F_{z,1..3}(0)\) [eV/Å] | per-atom \(q_i E_z\) | **0.2, −0.2, 0.1** | abs ≤ **1e-12** | force gate |
| \(E_z(125)\) [V/Å] | `last_Ez()` | **0.14142135623731000** | abs ≤ **1e-12** | \(0.2\cos(\pi/4)\) |
| \(U(125)\) [eV] | `kEnergy` | **−0.5656854249492400** | abs ≤ **1e-12** | mid-quarter |
| \(F_{z,i}(125)\) | \(q_i E_z\) | \(q_i\times\) pin \(E_z\) | abs ≤ **1e-12** | force gate |
| \(E_z(500)\) [V/Å] | `last_Ez()` | **−0.2** | abs ≤ **1e-12** | half-period min |
| \(U(500)\) [eV] | `kEnergy` | **+0.8** | abs ≤ **1e-12** | sign flip |
| \(F_{z,1..3}(500)\) | \(q_i E_z\) | **−0.2, 0.2, −0.1** | abs ≤ **1e-12** | force gate |

Near-zero crossings in the full offline table carry looser notes
(\(10^{-10}\)) in `REFERENCE.md`; the section asserts the three high-signal
steps at \(10^{-12}\). Offline repro: `lmp -in <LAMMPS oracle input for this
case>` prints PIN lines (step, time, Ez, Ufield from fix scalar + potential
`v_phi`). Metal ↔ ATV time map: `period=1.0` ps / `dt=0.001` ps maps to
`period=1000` fs / `dt=1` fs with the same integer step index \(n\).

### Performance

| Twin | Workload | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| oscillating efield | LAMMPS variable `fix efield 0 0 v_Ez`, dt=0.001 ps; ATV `FixOscillatingEfield{0,0,0.05,1000,1}`, dt=1 fs | **0.0585** (Loop time) | **0.124** | **2.12** | ✓ | Single-thread; field amp 0.05 V/Å on fair twin (not the 0.2 pin) |

`perf_exempt` false; ratio \(0.124/0.0585 = 2.12\); `within_10x` true.

### Limitations

- **Uniform AC cosine only.** No atom-style \(E(\mathbf{r})\), no region mask,
  no arbitrary equal-style expression beyond
  \(\mathbf{E}_0\cos(2\pi t/T+\varphi)\).
- **Caller supplies `dt`.** The Component does not read the integrator's
  timestep — `period`/`dt` must stay consistent with the MD schedule for
  \(t=n\Delta t\) to match the pin.
- **No pair / electrolyte in the pin.** The gate is the pure field on frozen
  charges; production MD composes with real potentials.

## finite-field-single-electrode — Finite field with a single electrode (Dufils) {#finite-field-single-electrode}

### Summary

Finite-field constant-potential electrode in 3D PBC with a single wall: one
Gaussian electrode group at applied \(\Psi\), free total charge
(`electroneutrality = false`), and a uniform external field \(\mathbf{E}\)
folded into the BO right-hand side so the induced electrode charge responds to
\(\Phi_{\mathrm{ext}} = -\mathbf{E}\cdot\mathbf{r}\). Three Components compose on
`cpu::md::MD`:

1. `cpu::md::ConstPotBO` — solves \(A Q = \Psi - b + \mathbf{E}\cdot\mathbf{r}\)
   (default `pre_force`, −50) and writes \(Q\) to `kAtom_charge`.
2. `cpu::md::FixEfield` — free-ion forces \(\mathbf{F}=q\mathbf{E}\) (default
   `post_force`, +100).
3. `cpu::md::ClassicalPolarization` — observer
   \(\mathbf{P}=V^{-1}\sum q\mathbf{r}^{u}\) (default `end_of_step`, +50).

`ConstPotBO::{Ex,Ey,Ez}` (charge response) and `FixEfield` (forces) are
orthogonal; a full finite-field MD step needs both. The pin is a vacuum
single-site analytic Reed self-term. Fair wall is `perf_exempt`.

### Method

Siepmann–Sprik / Reed Gaussian electrode charges of width \(\eta\) [1/Å]. With
\(k_e =\) `units::coulomb` \(= 14.399645\) eV·Å/e², the self elastance is

\[
A_{ii} = k_e\,\eta\sqrt{\frac{2}{\pi}}.
\]

For a uniform external field the BO equation on electrode sites is

\[
(A Q)_i = \Psi_i - b_i + \mathbf{E}\cdot\mathbf{r}_i,
\]

where \(b_i\) is the electrolyte contribution to the potential at site \(i\)
(zero in the vacuum pin). Electroneutrality is off for a free total \(Q\) on a
single wall (MetalWalls / Dufils single-electrode geometry). Forces on free
charges come from `FixEfield` (Common setup), and the classical polarization
density (MetalWalls / Dufils observable) is

\[
\mathbf{P} = \frac{1}{V}\sum_i q_i\,\mathbf{r}_i^{u}
\quad [e/\mathrm{Å}^2].
\]

**Why not LAMMPS.** LAMMPS ELECTRODE `ffield on` implements the
**two-electrode** finite-field path (\(E_z=\Delta\Psi/L_z\), requires exactly two
groups); the single-wall Dufils geometry is MetalWalls-only.

| Item | Setting |
| ---- | ------- |
| Box | \(L=20\) Å cubic, 3D PBC |
| \(\eta\) | \(2.0\) Å⁻¹ |
| \(\Psi\) | \(0\) V |
| \(\mathbf{E}\) | \((0,\,0,\,0.1)\) V/Å |
| Electrode | type 1 at \(\mathbf{r}=(0,0,10)\) Å, mass 196.966553 amu |
| Neutrality | **off** |
| \(xy\) images | `0` (primary cell only) |
| Electrolyte | none (vacuum pin); the field-force check is a separate one-atom frame |

Analytic numbers:

\[
A_{ii} = 22.978508853084353\ \mathrm{V}/e,
\quad
Q_{\mathrm{pin}} = \frac{E_z z}{A_{ii}} = 0.04351892485250505\,e,
\]

\[
P_{z,\mathrm{pin}} = \frac{Q z}{V} = 5.439865606563132\times 10^{-5}\ e/\mathrm{Å}^2
\quad (V=L^3=8000\,\mathrm{Å}^3).
\]

Field force on a test ion \(q=+1\): \(F_z = q E_z = 0.1\) eV/Å.

**Out of scope:** the two-electrode LAMMPS ELECTRODE `ffield on` product; a full
electrolyte or MetalWalls liquid input as the pin; an oscillating field.

### Usage

```cpp
cpu::Frame frame(1);
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 10.0;   // [Å]
frame.span(kAtom_charge)[0] = 0.0;
frame.span(kAtom_type)[0] = 1;
frame.span(kAtom_mass)[0] = 196.966553;
frame.span(kAtom_tag)[0] = 1;
frame.box = cpu::Box::cubic(20.0);

cpu::md::ConstPotBO conp(/*eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.Ez = 0.1;                   // [V/Å] — BO RHS only
conp.electroneutrality = false;  // free total Q (single wall)
conp.rebuild_matrix_every_step = true;

cpu::md::ClassicalPolarization pol;
cpu::md::FixEfield field(0.0, 0.0, 0.1);

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/2.0, /*skin=*/0.3});
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0, /*sigma=*/1.0, /*cutoff=*/2.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.set_observer_ref(conp);    // pre_force −50
sim.set_observer_ref(field);   // post_force +100
sim.set_observer_ref(pol);     // end_of_step +50
sim.build();
sim.run(frame, /*nsteps=*/1);

const double Q = conp.total_charge(0);  // [e] → pin 0.04351892485250505
// pol.Pz() → pin 5.439865606563132e-5 [e/Å²]
```

The standalone charge/polarization check uses the same composition without
`FixEfield` and with `VelocityVerlet{0.0}`; the field-force check is a
one-atom `cpu::Evaluate` with `FixEfield{0.0, 0.0, 0.1}` at
`evaluate::Phase::evaluate` (\(q = +1\) → \(F_z = 0.1\) eV/Å). Headers:
`const_pot_bo.h`, `fix_efield.h`, `classical_polarization.h`; unit tests in
`tests/cpu/dynamics/test_const_pot_ffield.cpp`,
`tests/cpu/dynamics/test_fix_efield.cpp`,
`tests/cpu/observe/test_classical_polarization.cpp`.

### Validation (oracle: literature analytic)

Dufils et al., *Phys. Rev. Lett.* **123**, 195501 (2019) finite field + single
constant-potential electrode; Reed / Siepmann–Sprik Gaussian electrode kernels
(same family as MetalWalls and `ConstPotBO`). `oracle: metalwalls`,
`oracle_version: literature analytic`.

| Quantity | Atomiverse gate / pin | Analytic pin | Tol | Notes |
| -------- | --------------------: | -----------: | --- | ----- |
| \(A_{ii}\) [V/e] | \(k_e\eta\sqrt{2/\pi}\) | **22.978508853084353** | identity | Reed self-term |
| \(Q\) [e] | within **1e-10** of pin | **0.04351892485250505** | abs ≤ **1e-10** | \(Q=(E_z z)/A_{ii}\), \(\Psi=0\), \(b=0\) |
| \(P_z\) [e/Å²] | within **1e-12** of pin | **5.439865606563132×10⁻⁵** | abs ≤ **1e-12** | standalone `ClassicalPolarization` |
| \(P_x,P_y\) | ~0 | 0 | abs ≤ **1e-14** | vacuum pin orientation |
| \(F_z\) ion [eV/Å] | within **1e-12** of \(E_z\) | **0.1** | abs ≤ **1e-12** | `FixEfield`, \(q=+1\) |
| MD \(Q\) (1 VV step) | same as pin | **0.04351892485250505** | abs ≤ **1e-10** | `ConstPotBO` + `FixEfield` + pol on `MD` |
| MD \(P_z\) | same as pin | **5.439865606563132×10⁻⁵** | abs ≤ **1e-10** | large mass / \(v=0\); looser than standalone |

Offline repro:

```python
import math
coulomb = 14.399645
eta, Ez, z, L = 2.0, 0.1, 10.0, 20.0
Aii = coulomb * eta * math.sqrt(2 / math.pi)
Q = (Ez * z) / Aii
Pz = Q * z / (L ** 3)
print(Aii, Q, Pz)
# 22.978508853084353  0.04351892485250505  5.439865606563132e-05
```

### Performance

`perf_exempt`: no MetalWalls binary on the agent host, and LAMMPS `ffield`
requires two electrodes, so there is no fair twin wall (`ratio_atv_over_oracle`
n/a, `within_10x` exempt).

### Limitations

- **Vacuum single-site pin.** No electrolyte \(b\) term, no multi-site wall —
  only the Reed self-term identity for \(A Q = \Psi + \mathbf{E}\cdot\mathbf{r}\).
- **Primary-cell elastance.** `xy_image_range = 0` on this pin; planar image
  sums and full electrode Ewald are covered by the constant-potential pages.
- **Free total \(Q\).** Electroneutrality projection is intentionally off;
  parallel-plate `symm on` is
  [`const-pot-bo-parallel-plate`](md-constant-potential.md#const-pot-bo-parallel-plate).
- **Constant \(\mathbf{E}\) only.**

### References

- Dufils et al., *Phys. Rev. Lett.* **123**, 195501 (2019)
- Reed / Siepmann–Sprik Gaussian electrode charges

## constant-d-polarization-feedback — Constant-D polarization feedback (Zhang–Sprik) {#constant-d-polarization-feedback}

### Summary

With `polarization_feedback = true`, `cpu::md::FixDfield(Dx, Dy, Dz)` turns the
vacuum D-field foothold (\(\mathbf{E}\equiv\mathbf{D}\), no screening) into the
self-consistent constant-\(D_z\) ensemble of Zhang & Sprik: the Maxwell field
along z responds to the instantaneous itinerant polarization of the cell,
\(E_z = D_z - 4\pi k_e P_z\), and the electric-enthalpy term
\(U_D = V E_z^2/(8\pi k_e)\) enters the energy so the force
\(F_{i,z} = q_i E_z\) is exactly \(-\partial U_D/\partial z_i\). Free charges
screen the applied displacement field — the physical content of MetalWalls'
finite-field route for bulk electrolytes. Accessors: `last_energy()`,
`last_Pz()`, `last_Ez()`. The section pins a ±1 e pair in a cubic cell against
the closed-form constant-D expressions: static screening at \(D_z=0.5\) V/Å,
the Cox–Sprik plasma-like polarization oscillation at \(D_z=0\), and the
vacuum-foothold twin that must not move.

### Method

In metal units (eV, Å, e, fs; \(k_e =\) `units::coulomb` \(= 14.399645\)
eV·Å/e²), along the cell z axis:

\[
P_z = \frac{1}{V}\sum_i q_i\, z_i^{\mathrm u} \quad [\mathrm{e/\AA^2}]
\qquad\text{(itinerant polarization, unwrapped coordinates)}
\]

\[
E_z = D_z - 4\pi k_e P_z \quad [\mathrm{V/\AA}],
\qquad
U_D = \frac{V\,E_z^2}{8\pi k_e} \quad [\mathrm{eV}],
\qquad
F_{i,z} = -\frac{\partial U_D}{\partial z_i} = q_i E_z .
\]

This is the electric-enthalpy functional
\(F = E_{\mathrm{PBC}} + V(D-4\pi P)^2/8\pi\) that MetalWalls implements for its
finite-field route. Because the force is the exact gradient of \(U_D\),
constant-D MD conserves \(KE + U_D\) when the field does the only work. \(P_z\)
uses unwrapped coordinates (`kAtom_image`) so it tracks charge transport across
the boundary. Two consequences become gates:

- **Static screening.** At fixed geometry, a polarized cell reduces the
  Maxwell field below the applied \(D_z\) (the foothold twin keeps
  \(E_z = D_z\) exactly).
- **Plasma-like oscillation.** At \(D_z = 0\) the itinerant polarization of a
  ±q pair sits in a harmonic well \(U = 2\pi k_e q^2 d^2/V\) (Cox & Sprik), so
  the pair separation oscillates as \(d(t) = d_0\cos\omega t\) with
  \(\omega = \sqrt{4\pi k_e q^2/(\mu V)\cdot\texttt{force\_to\_accel}}\).

x and y keep the foothold coupling — MetalWalls applies \(D\) along the cell
normal only. Mainline LAMMPS has no constant-D ensemble, so the pins are the
closed-form expressions of the finite-field papers evaluated with the project's
Coulomb constant.

| Quantity | Value |
| -------- | ----- |
| Charges | +1 e at \(z_0 + d/2\), −1 e at \(z_0 - d/2\) (on the z axis) |
| Separation \(d\) | 2 Å |
| Box | cubic \(L = 10\) Å (\(V = 1000\) Å³) |
| Masses | 1 amu each (reduced mass \(\mu = 0.5\) amu) |
| Gate A field | \(D_z = 0.5\) V/Å, static (no MD) |
| Gate B field | \(D_z = 0\), VV dt = 0.1 fs, 532 steps (half period) |
| Gate C | feedback **off**, \(D = 0\) — vacuum-foothold twin |

Gate A pins (static, \(D_z = 0.5\) V/Å):

| Quantity | Value | Units |
| -------- | ----: | ----- |
| \(P_z\) | 2.0e−3 | e/Å² |
| \(E_z\) | 1.3809744842959204e−1 | V/Å |
| \(U_D\) | 5.2696244279044818e−2 | eV |
| \(F_z(+q)\) | 1.3809744842959204e−1 | eV/Å |
| foothold twin \(E_z\) | 0.5 (= \(D_z\), unscreened) | V/Å |

Gate B (dynamics, \(D_z = 0\)): \(\omega =\) **5.9091697154893236e−2** fs⁻¹
(period 106.329 fs), \(d(t) = d_0\cos\omega t\) with \(d_0 = 2\) Å, total
energy \(U_0 =\) **3.6190255157040796e−1** eV.

**Out of scope:** constant-D along x/y; Berry-phase / electronic polarization
(point charges only); coupling to induced dipoles or electrode charge solvers in
this pin (compose `FixDfield` with `ConstPotBO` separately).

### Usage

Static single evaluation (Gate A):

```cpp
cpu::Frame f(2);                       // ±1 e pair on z, cubic L = 10 Å
f.span(kAtom_z)[0] = 6.0;  f.span(kAtom_charge)[0] = +1.0;
f.span(kAtom_z)[1] = 4.0;  f.span(kAtom_charge)[1] = -1.0;
// ... x/y/mass/type/tag/image as in the section source ...
f.box = cpu::Box::cubic(10.0);

cpu::md::FixDfield d(/*Dx=*/0.0, /*Dy=*/0.0, /*Dz [V/Å]=*/0.5);
d.polarization_feedback = true;        // constant-D ensemble (default: foothold)

cpu::Evaluate eval;
eval.set_evaluator_ref(d);
eval.build();
eval.run(f);

double Pz = d.last_Pz();       // [e/Å²]  itinerant polarization
double Ez = d.last_Ez();       // [V/Å]   D_z − 4π k_e P_z
double U  = d.last_energy();   // [eV]    V E_z² / (8π k_e)
// forces q_i E_z added to frame.span(kAtom_fz)
```

Constant-D MD (Gate B). Forces are seeded on the Frame by a single-point
`Evaluate` with the same `FixDfield` before the first VV half-kick (the role of
LAMMPS `fix setup()`), then:

```cpp
cpu::md::MD sim;
sim.set_threads(1);            // 2 atoms: fork/join dwarfs the physics
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
cpu::md::FixDfield dfield(0.0, 0.0, 0.0);
dfield.polarization_feedback = true;
sim.set_observer_ref(dfield);
sim.build();
sim.run(frame, 266);           // probe at 26.6 fs
sim.run(frame, 266);           // half a polarization-oscillation period (532 steps)
```

Files: `fix_dfield.h` (`Dx/Dy/Dz`, `polarization_feedback`,
`last_Pz/last_Ez/last_energy`), `src/cpu/dynamics/fix_dfield.cpp` (\(P_z\) from
unwrapped coords; \(E_z = D_z - 4\pi k_e P_z\); \(U_D\) + exact-gradient
forces).

### Validation (oracle: literature)

| Quantity | ATV | Oracle pin (closed-form constant-D) | Tol | Notes |
| -------- | --- | ----------------------------------: | --- | ----- |
| A: \(P_z\) [e/Å²] | `last_Pz()` | **2.0e−3** | abs ≤ **1e-13** | \(q\,d/V\), static |
| A: \(E_z\) [V/Å] | `last_Ez()` | **1.3809744842959204e−1** | abs ≤ **1e-13** | \(D_z - 4\pi k_e P_z\), screened below \(D_z=0.5\) |
| A: \(U_D\) [eV] | `last_energy()` | **5.2696244279044818e−2** | abs ≤ **1e-13** | \(V E_z^2/(8\pi k_e)\) |
| A: \(F_z(+q)\) [eV/Å] | `span(kAtom_fz)[0]` | **1.3809744842959204e−1** | abs ≤ **1e-13** | \(q_i E_z\) = exact gradient |
| A: twin \(E_z\) [V/Å] | feedback off | **0.5** (= \(D_z\), unscreened) | exact | vacuum foothold |
| B: \(d(t{=}26.6\,\mathrm{fs})\) [Å] | VV, dt = 0.1 fs | \(d_0\cos(\omega\,26.6)\), \(\omega=\) **5.9091697154893236e−2** fs⁻¹ | abs ≤ **5e-5** | VV error ≤ 4.6e−6 Å |
| B: \(d(t{=}53.2\,\mathrm{fs})\) [Å] | VV, dt = 0.1 fs | \(d_0\cos(\omega\,53.2)\approx-d_0\) | abs ≤ **5e-5** | half period (106.329 fs / 2) |
| B: \(KE + U_D\) [eV] | at step 532 | **3.6190255157040796e−1** | abs ≤ **1e-6** | energy conservation, \(U_0\) |
| B: \(E_z + 4\pi k_e P_z\) [V/Å] | every probe | **0** (self-consistency at \(D_z=0\)) | abs ≤ **1e-13** | field tracks motion |
| C: \(z_+, z_-\) [Å] | after 532 steps | **6.0, 4.0** (unmoved) | exact | feedback off, \(D=0\) → zero force |
| C: \(E_z\) [V/Å] | after 532 steps | **0** (= \(D_z\)) | exact | foothold twin |

Runtime is ~1064 MD steps of a 2-atom system, well under 60 s. Offline repro:
`python <MetalWalls oracle pin for this case>`.

### Performance

`perf_exempt`: no MetalWalls binary on the agent host and no LAMMPS constant-D
ensemble for a fair twin; the O(N) field evaluation on 2 atoms is sub-second
(`ratio_atv_over_oracle` n/a, `within_10x` exempt).

### Limitations

- **z axis only** for the self-consistent ensemble; x,y keep the vacuum
  foothold coupling.
- **Point charges / itinerant polarization only** — no Berry-phase or
  electronic polarization, no induced-dipole contribution to \(P_z\) in this
  pin (PIM dipoles: [`md-polarizable.md`](md-polarizable.md)).
- **Uniform-field coupling** — the constant-D term sits on top of whatever
  Coulomb pipeline the simulation composes; this pin runs it stand-alone (no
  PME) so the closed-form well is exact.
- **No constant-E ↔ constant-D Legendre-transform toggle** as a separate API —
  constant-E at \(\varepsilon=1\) is the foothold path
  (`polarization_feedback = false`).

### References

- C. Zhang, M. Sprik, *Phys. Rev. B* **93**, 144201 (2016), DOI [10.1103/PhysRevB.93.144201](https://doi.org/10.1103/PhysRevB.93.144201) (constant-D electric enthalpy \(F = E_{\mathrm{PBC}} + V(D-4\pi P)^2/8\pi\))
- S. J. Cox, M. Sprik, *J. Chem. Phys.* **151**, 064506 (2019), DOI [10.1063/1.5099207](https://doi.org/10.1063/1.5099207) (finite-field bulk electrolyte formalism)
- M. Stengel, N. A. Spaldin, D. Vanderbilt, *Nat. Phys.* **5**, 304 (2009), DOI [10.1038/nphys1185](https://doi.org/10.1038/nphys1185) (constant-D functional origin)
