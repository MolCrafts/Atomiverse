# Chapter 9. Gaussian and Plane Waves

Data: in-tree GTH / MOLOPT tables — no runtime parsers.
API: `GthPotential(Element, name)`, `GaussianBasis(Element, name)`, `GPWEvaluator`.

## Live result: isolated H₂O / PBE / DZVP-MOLOPT-GTH

CP2K getting-started geometry (Martyna–Tuckerman isolated electrostatics):

| | Atomiverse | CP2K reference |
|---|---:|---:|
| Converged | yes (9 iters) | — |
| \(E_\mathrm{total}\) (Ha) | **−17.21275807** | −17.21233714 |
| \(\lvert\Delta E\rvert\) (Ha) | **0.000421** | — |

Geometry is the CP2K getting-started water (coordinates in the example source).
What matters is the converged energy against the literature reference, not a
ball-and-stick view of three atoms.

```cpp
const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
// … H kind …
cpu::gpw::GPWEvaluator eval(
    /*kinds*/ {O, H}, /*cutoff_ry*/ 400.0, /*max_iter*/ 200, /*tol*/ 1e-7,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true);
cpu::single_point(frame, eval);
```

Analytic forces must match finite differences; a known pitfall is
`local_pp_on_grid` (member on `GpwDft` / `GPWEvaluator`) — energy and gradient both read it
(see `cpu/gpw/internal/local_pp.h`).

The manual has now presented the two great basis families of electronic
structure. The molecular chapters ([Chapter 2](ch02-rhf.md) through
[Chapter 6](ch06-hybrid-dft.md)) used **atom-centered Gaussians** — compact,
few functions per atom, but the two-electron Coulomb term costs formally
$\mathcal{O}(N^4)$ integrals. [Chapter 8](ch08-plane-wave.md) used **plane waves** —
a uniform basis that makes the Hartree potential an $\mathcal{O}(N \log N)$ FFT,
but needs hundreds of functions per atom.

**GPW — Gaussian and Plane Waves** — is the method that takes the best of each.
It is the electronic-structure engine behind CP2K's *Quickstep*, and Atomiverse
reproduces its numbers for the same input.

> **The GPW idea.** Keep the *wavefunction* in a small Gaussian basis, but
> represent the *density* on a uniform real-space grid. The Kohn-Sham matrix is
> built in the Gaussian basis (like [Chapter 5](ch05-kohn-sham-dft.md)); the Hartree and
> exchange-correlation potentials are evaluated on the grid by FFT (like
> [Chapter 8](ch08-plane-wave.md)). A dual representation, bridged by mapping the Gaussian
> density onto the grid and back.

You will learn:

1. **The concept** — the dual Gaussian/grid representation and why it scales
   better than pure Gaussians for the electrostatics.
2. **The representation** — a `gpw_solver` inner engine behind a
   `GPWEvaluator` component, the third member of the evaluator family.
3. **The C++ API** — `Element`, `GPWEvaluator` (constructor parameters), and `single_point`,
   walked through for an isolated water molecule.
4. **The expected result** — CP2K Quickstep parity, to the sub-µHartree.
5. **The validation** — against CP2K, and the honest experimental scope.
6. **The recurring pattern** — a third evaluator that drops into the same
   Evaluate / MD / Minimize drivers as the Gaussian and plane-wave ones.

GPW needs FFTW3 and is off by default; build with
GPW enabled in the build.

## The Concept: Two Representations, One Density

The Kohn-Sham energy has terms that live most naturally in *different* bases.

The **kinetic energy** and the **short-range pseudopotential** are cheap and
accurate in a small Gaussian basis — a handful of contracted functions per atom,
exactly as in the molecular DFT of [Chapter 5](ch05-kohn-sham-dft.md). But the **Hartree
term** — the classical electron-electron Coulomb repulsion — is the expensive
one for Gaussians, because it couples every basis-function pair to every other.

Plane waves invert this. The Hartree potential is trivial in reciprocal space:
solve Poisson's equation by dividing the density's Fourier coefficients by
$G^2$, an FFT away. GPW therefore does the following each SCF iteration:

1. Build the density matrix in the **Gaussian** basis.
2. **Collocate** it onto a uniform real-space **grid** — map the compact
   Gaussian density $\rho(\mathbf r)$ onto grid points.
3. FFT to reciprocal space, solve Poisson for the Hartree potential, evaluate
   the XC potential on the grid — all $\mathcal{O}(N \log N)$.
4. **Integrate** the resulting potential back against the Gaussian basis
   functions to form the potential matrix contribution.
5. Add kinetic + pseudopotential (Gaussian side), diagonalize, rebuild the
   density — the familiar SCF loop.

Core electrons are removed by **GTH (Goedecker-Teter-Hutter) pseudopotentials**,
the same analytic norm-conserving form used in the plane-wave chapter, which
keeps the required grid cutoff modest. The "plane-wave cutoff" here is the
resolution of that real-space grid, set in Rydberg.

The result: Gaussian compactness for the wavefunction, plane-wave-grid scaling
for the electrostatics. This is why Quickstep can run condensed-phase DFT and
AIMD on thousands of atoms.

## The Representation: An Inner Solver Behind an Evaluator

GPW follows the exact architecture of the other electronic-structure methods in
this manual. The SCF machinery is an **inner solver** — `cpu::gpw::GpwDft`,
driven by `cpu::gpw::GPWEvaluator` — and it is *not* a `Driver` subclass, just
as [Gaussian SCF is an inner solver](ch02-rhf.md) behind `SCF` and
[plane-wave DFT](ch08-plane-wave.md) is one behind `PWEvaluator`.

`GPWEvaluator` reports `role() == Evaluator` and `capabilities() == Energy`
(plus `Force` when analytic forces are enabled). Its `build()` records
the per-element basis and pseudopotential templates; each `launch()` assembles
the atoms at the frame's current geometry (converting Å → Bohr), runs the GPW
SCF, and writes the total energy to `frame.meta.get(kEnergy)` in eV — the same
contract every evaluator honours. Because it is an ordinary Energy evaluator, it
plugs into the `Evaluate`, `MD`, and `Minimize` drivers with no special support.

## The C++ API: Elements, Config, `single_point`

Practice:
an isolated water molecule, the electronic structure of CP2K's getting-started
`h2o.inp`. Keep it open beside this text.

**Step 1 — build the Frame.** GPW reads `frame.span(kAtom_type)` as the atomic number
$Z$ and needs a cell (`cpu::Box`); an isolated molecule sits in a box large
enough to avoid its images:

```cpp
cpu::Frame f(3);
const int    z[3]  = {8, 1, 1};                    // O, H, H
const double x[3]  = {5.0000, 5.7586, 4.2414};     // Å
const double y[3]  = {5.0000, 5.0000, 5.0000};
const double zc[3] = {5.0000, 5.5043, 5.5043};
for (int i = 0; i < 3; ++i) {
    f.span(kAtom_type)[i] = z[i];
    f.span(kAtom_x)[i] = x[i];
    f.span(kAtom_y)[i] = y[i];
    f.span(kAtom_z)[i] = zc[i];
}
f.box = cpu::Box::cubic(10.0);                   // 10 Å box
```

**Step 2 — declare a `Kind` per element.** A `cpu::gpw::Kind` is CP2K's
`&KIND` concept: one chemical element (`molrs::Element`, the molrs
periodic-table enum) paired with its contracted Gaussian basis and GTH
pseudopotential, both from the in-tree data tables:

```cpp
const cpu::gpw::Kind O{molrs::Element::O,
                       cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
                       cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{molrs::Element::H,
                       cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
                       cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};
```

The `q6`/`q1` suffixes are the valence electron counts the pseudopotential keeps
(6 for oxygen, 1 for hydrogen) — the core is replaced by the GTH form.

**Step 3 — construct and run.** `GPWEvaluator` takes kinds and SCF/grid knobs
in the constructor (no public Config bag):

```cpp
cpu::gpw::GPWEvaluator eval(
    /*kinds*/ {O, H},
    /*cutoff_ry*/ 400.0,                   // real-space grid resolution [Ry]
    /*max_iter*/ 200,
    /*tol*/ 1e-7,
    /*mix*/ 0.3,
    /*xc*/ cpu::gpw::XcKind::PBE,          // PADE (LDA) or PBE (GGA)
    /*nonperiodic*/ true);                 // isolated: Martyna-Tuckerman electrostatics

cpu::single_point(frame, eval);            // kEnergy now set (eV)

const double e_ha = eval.last_energy_hartree();          // total energy (Hartree)
const bool   ok   = eval.last_result().converged;
const int    n    = eval.last_result().iters;
```

Set `nonperiodic = false` for a genuinely periodic system (a crystal or
slab) — then the grid electrostatics are periodic, as in the small wurtzite-ZnO
cell of `/* benchmarks removed; re-add later */ gpw/validation/test_cpu_gpw_zno.cpp`. Optional knobs: `mix` (density
mixing), `max_iter`, `tol`, `d3` (Grimme DFT-D3 dispersion from the
in-tree tables), and `compute_forces` (analytic forces — see
the scope note below).

## The Expected Result: CP2K Quickstep Parity

Because GPW is *the same method* as Quickstep — same basis, same
pseudopotentials, same grid electrostatics — the target is not "close to CP2K"
but "CP2K's number." For the water input above (PBE, DZVP-MOLOPT-GTH,
Martyna-Tuckerman) the CP2K reference total energy is

$$ E_\text{total} = -17.21233714\ \text{Hartree}, $$

and the run reports `|dE|` against it. The `GpwResult` also exposes
the energy decomposition — `e_kinetic`, `e_loc_sr` (short-range local PP),
`e_nonlocal`, `e_es` (total electrostatic), `e_xc` — so you can check parity
term by term, which is how you localise a discrepancy if the total drifts.

This is a **numerical-tolerance** result against a reference *engine* (see the
[benchmark validation policy](https://github.com/MolCrafts/Atomiverse/blob/dev/benchmarks/README.md)): the two codes
implement the same equations, so agreement should be to the convergence
threshold, not merely to chemical accuracy.

## The Validation: Against CP2K

CP2K Quickstep is the reference oracle for GPW, exactly as PySCF is for the
Gaussian SCF chapters and ABINIT/CP2K for the plane-wave chapter. The validation
strategy is CP2K parity on:

- **Isolated molecules** (Martyna-Tuckerman) — the water single point above.
- **Periodic systems** — the small wurtzite-ZnO cell of
  `/* benchmarks removed; re-add later */ gpw/validation/test_cpu_gpw_zno.cpp` (DZVP-MOLOPT-SR-GTH,
  GTH-PBE), pinned against a constexpr anchor energy.

The full anchor suite lives in `/* benchmarks removed; re-add later */ gpw/validation/`
(H₂ / Ar / H₂O / ZnO / NaCl, plus forces, geoopt, AIMD and OT-vs-diag);
until then the two examples are the executable checks, each printing its
deviation from the CP2K reference.

## The Recurring Pattern: A Third Evaluator, Same Drivers

Stand back and count the electronic-structure evaluators this manual now has:

| evaluator | basis | electrostatics | chapter |
|---|---|---|---|
| `SCF` | Gaussian | analytic ERIs | [2](ch02-rhf.md)–[6](ch06-hybrid-dft.md) |
| `PWEvaluator` | plane wave | reciprocal-space FFT | [8](ch08-plane-wave.md) |
| `GPWEvaluator` | Gaussian + grid | grid FFT (Poisson) | 9 (this) |

All three are `role() == Evaluator`, `capabilities() == Energy`, and all three
write `frame.meta.get(kEnergy)`. So all three are interchangeable at the *driver*
level: `cpu::single_point(frame, eval)` for an energy, `on(md::Phase::force)
.add(eval)` for AIMD, `on(minimizer::Phase::force).add(eval)` for geometry
optimization. The physics inside each is completely different; the seam the
driver sees — an Energy-capable component at the force phase — is identical.
This is "swap the evaluator, keep the driver" ([Chapter 15](ch15-cross-method.md))
paying off for a *third* electronic-structure family.

## Scope and Status

GPW is **experimental**. Be precise about what works today:

- **Closed-shell only** — a restricted (spin-paired) formulation. No open-shell
  GPW.
- **Γ-point only** — no k-point sampling; correct for large periodic cells where
  a single k-point suffices, not for small unit cells needing a k-mesh.
- **XC:** PADE (LDA) and PBE (GGA). No hybrid functionals on the grid.
- **Forces are analytic**: `cfg.compute_forces = true` runs one SCF plus one
  gradient pass per evaluation, validated against the central finite-difference
  oracle in `/* benchmarks removed; re-add later */ gpw/validation/test_cpu_gpw_forces.cpp`.
- All basis/pseudopotential/D3 data is compiled in as static `.h` tables
  (`cpu/gpw/basis_data/`); there are no file parsers — an entry outside the
  supported set throws.

For the periodic plane-wave method GPW builds on, see
[Chapter 8](ch08-plane-wave.md); for the molecular Gaussian DFT it shares a basis with,
see [Chapter 5](ch05-kohn-sham-dft.md).

## Validation Reference

The runnable GPW example validates against a pinned reference energy printed at
the end of the run:

- Isolated H₂O (PBE, DZVP-MOLOPT-GTH, Martyna-Tuckerman) vs the pinned anchor
  `-17.21233714` Ha.

The full anchor suite (H₂ / Ar / H₂O / ZnO / NaCl single points, forces vs the
FD oracle, geoopt, AIMD, OT-vs-diag) lives in `/* benchmarks removed; re-add later */ gpw/validation/` —
every case is a minimal textbook system and runs in minutes.
