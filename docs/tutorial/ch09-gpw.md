# Gaussian and Plane Waves

Status tables: [DFT roadmap](../roadmap/dft.md).

Localized orbitals and regular grids solve different representation problems
well. GPW uses Gaussian orbitals for chemical structure and a grid-compatible
density representation for efficient electrostatic and functional operations.
Understanding this division helps us diagnose errors without changing
unrelated settings.

**Learning goals.** Distinguish orbital and density resolutions, understand
collocation, and test a water calculation under controlled boundaries.

**Before you begin.** [Basis Sets](ch04-basis-sets.md),
[Density Functional Theory](ch05-kohn-sham-dft.md), and
[Plane Waves](ch08-plane-wave.md).

## Two representations of one state

For real AO functions the density is

$$\rho(\mathbf r)=\sum_{\mu\nu}P_{\mu\nu}
\chi_\mu(\mathbf r)\chi_\nu(\mathbf r).$$

Placing these basis products on a grid is often called collocation. Their
spatial widths differ from the parent functions, so density resolution needs
its own convergence test. Grid potentials are integrated back against basis
products to produce matrix contributions. GPW is not a plane-wave orbital
calculation with a Gaussian correction added afterward.

Orbital basis, density grid, and SCF tolerance control distinct errors.
Periodic systems also require cell and k-point checks. More grid points
cannot provide absent orbital flexibility; a large basis on a coarse grid
can instead make forces noisy. Compatible pseudopotentials and valence
bases remain part of the Hamiltonian definition.

Multigrid techniques distribute products across resolutions. Orbital
transformation methods optimize orbitals under overlap constraints rather
than relying on the same diagonalization at every step. These choices
change the numerical procedure, not the meaning of an electron density.

## Practice plan

Begin with water at a fixed geometry and compatible oxygen and hydrogen
data. Supply the hydrogen kind omitted in a short fragment. Verify what
the selected nonperiodic treatment actually changes, and record the cell.

At fixed orbital basis refine the density cutoff. Translate the whole
molecule by a fraction of a grid spacing: residual energy changes reveal
grid-related translation error. Then enlarge the basis and repeat. Before
dynamics, check displaced forces; before cell optimization, check stress
against controlled strains. Match all these conventions in an external comparison.

## Orbital and density representations

The Kohn-Sham energy has terms that live most naturally in *different* bases.

The **kinetic energy** and the **short-range pseudopotential** are cheap and
accurate in a small Gaussian basis — a handful of contracted functions per atom,
exactly as in the molecular DFT of [Density Functional Theory](ch05-kohn-sham-dft.md). But the **Hartree
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

## Electronic solve and evaluator

GPW follows the exact architecture of the other electronic-structure methods in
this manual. The concrete method is `cpu::gpw::GpwDft` — a
`cpu::Driver<gpw::Phase>` loop node (KS build → mix → diagonalize → density →
check), Frame in / Frame out, the same shape as [`cpu::scf::RHF`](ch02-rhf.md).
It owns per-element kinds, cutoff and XC as constructor identity, with SCF/grid
controls as public members;
`dft.run(frame)` writes the total energy to `frame.meta.get(kEnergy)` in eV and
`dft.forces(frame)` additionally writes `kAtom_fx/fy/fz` in eV/Å.

For Born–Oppenheimer outer-loop use (Evaluate, MD, Minimize), mount `GpwDft`
directly on the parent Driver. It runs on the parent's Runtime, accumulates
energy/forces, and assembles stress when the parent plan consumes it.
`cpu::gpw::GPWEvaluator` remains for electronic propagators and callers
awaiting migration; examples using that wrapper show the transitional path.

## A GPW water calculation

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
`&KIND` concept: one chemical element (`molrs::Element`, the molpy
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

**Step 3 — construct and run.** `GpwDft` takes the kinds, the density cutoff
and the functional in its constructor; SCF/grid knobs are public members (no
public Config bag):

```cpp
cpu::gpw::GpwDft dft{{O, H},
                     400.0,                   // real-space grid resolution [Ry]
                     cpu::gpw::XcKind::PBE};  // PADE (LDA) or PBE (GGA)
dft.max_iter    = 200;
dft.tol         = 1e-7;
dft.mix         = 0.3;
dft.nonperiodic = true;                    // isolated: Martyna-Tuckerman electrostatics

dft.build(f);                              // fail-fast validation
const cpu::gpw::GpwResult& r = dft.run(f); // kEnergy now set (eV)

const double e_ha = r.e_total;             // total energy (Hartree)
const bool   ok   = r.converged;
const int    n    = r.iters;
```

Set `nonperiodic = false` for a genuinely periodic system (a crystal or
slab) — then the grid electrostatics are periodic, as in the small wurtzite-ZnO
cell of `/* benchmarks removed; re-add later */ gpw/validation/test_cpu_gpw_zno.cpp`. Optional knobs: `mix` (density
mixing), `max_iter`, `tol`, `d3` (Grimme DFT-D3 dispersion from the
in-tree tables); `dft.forces(f)` runs the SCF plus analytic forces — see
the scope note below.

## Using GPW in workflows

The three electronic-structure families share the outer Driver interface:

| concrete method | basis | electrostatics | chapter |
|---|---|---|---|
| `RHF` / `UHF` / `RKS` / `UKS` | Gaussian | analytic ERIs | [2](ch02-rhf.md)–[6](ch06-hybrid-dft.md) |
| `PW` | plane wave | reciprocal-space FFT | [8](ch08-plane-wave.md) |
| `GpwDft` | Gaussian + grid | grid FFT (Poisson) | 9 (this) |

Each concrete method owns its electronic loop and writes Frame energy through
`run(frame)`. Mount it on the parent's force phase for Born–Oppenheimer MD or
geometry optimization; the parent requests forces/stress and the nested method
accumulates those quantities. `GPWEvaluator` is retained for electronic
propagators and unmigrated callers. No public `role()` / `capabilities()`
inspection is required.
This is "swap the evaluator, keep the driver" ([Comparing Models](ch15-cross-method.md))
paying off for a *third* electronic-structure family.

## Scope and Status

GPW is **experimental**. Be precise about what works today:

- **Closed-shell only** — a restricted (spin-paired) formulation. No open-shell
  GPW.
- **Γ-point only** — no k-point sampling; correct for large periodic cells where
  a single k-point suffices, not for small unit cells needing a k-mesh.
- **XC:** PADE (LDA) and PBE (GGA). No hybrid functionals on the grid.
- **Forces are analytic**: `dft.forces(frame)` (or `GPWEvaluator` with
  `compute_forces = true`) runs one SCF plus one gradient pass per evaluation, validated against the central finite-difference
  oracle in `/* benchmarks removed; re-add later */ gpw/validation/test_cpu_gpw_forces.cpp`.
- All basis/pseudopotential/D3 data is compiled in as static `.h` tables
  (`cpu/gpw/basis_data/`); there are no file parsers — an entry outside the
  supported set throws.

For the periodic plane-wave method GPW builds on, see
[Plane Waves](ch08-plane-wave.md); for the molecular Gaussian DFT it shares a basis with,
see [Density Functional Theory](ch05-kohn-sham-dft.md).

The following numerical material is retained from the source snapshot.
It was not rerun for this editorial revision; match its input and provenance
before using it as a validation target.

## Reference results: isolated H₂O / PBE / DZVP-MOLOPT-GTH

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
cpu::gpw::GpwDft dft{{O, H}, 400.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = true;
dft.build(frame);
dft.run(frame);
```

Analytic forces must match finite differences; a known pitfall is
`local_pp_on_grid` (member on `GpwDft`) — energy and gradient both read it
(see `cpu/gpw/internal/local_pp.h`).

The manual has now presented the two great basis families of electronic
structure. The molecular chapters ([Hartree–Fock Theory](ch02-rhf.md) through
[Hybrid Functionals](ch06-hybrid-dft.md)) used **atom-centered Gaussians** — compact,
few functions per atom, but the two-electron Coulomb term costs formally
$\mathcal{O}(N^4)$ integrals. [Plane Waves](ch08-plane-wave.md) used **plane waves** —
a uniform basis that makes the Hartree potential an $\mathcal{O}(N \log N)$ FFT,
but needs hundreds of functions per atom.

**GPW — Gaussian and Plane Waves** — is the method that takes the best of each.
It is the electronic-structure engine behind CP2K's *Quickstep*, and Atomiverse
reproduces its numbers for the same input.

> **The GPW idea.** Keep the *wavefunction* in a small Gaussian basis, but
> represent the *density* on a uniform real-space grid. The Kohn-Sham matrix is
> built in the Gaussian basis (like [Density Functional Theory](ch05-kohn-sham-dft.md)); the Hartree and
> exchange-correlation potentials are evaluated on the grid by FFT (like
> [Plane Waves](ch08-plane-wave.md)). A dual representation, bridged by mapping the Gaussian
> density onto the grid and back.

You will learn:

1. **The concept** — the dual Gaussian/grid representation and why it scales
   better than pure Gaussians for the electrostatics.
2. **The representation** — the `GpwDft` Driver node (Frame in / Frame out),
   mounted directly into outer drivers for Born–Oppenheimer work, with
   `GPWEvaluator` retained for electronic propagators and unmigrated callers.
3. **The C++ API** — `Element`, `Kind`, `GpwDft` (kinds + public members), and `run(frame)`,
   walked through for an isolated water molecule.
4. **The expected result** — comparison with CP2K Quickstep at the recorded
   basis/grid settings and tolerances.
5. **The validation** — against CP2K, and the honest experimental scope.
6. **The recurring pattern** — an electronic method that drops into the same
   Evaluate / MD / Minimize drivers as the Gaussian and plane-wave ones.

GPW needs FFTW3 and is off by default; build with
GPW enabled in the build.

## Reference comparison

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
[benchmark validation policy](https://github.com/MolCrafts/Atomiverse/blob/master/benchmarks/README.md)): the two codes
implement the same equations, so agreement should be to the convergence
threshold, not merely to chemical accuracy.

## Matched GPW validation

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

## Reference provenance

The runnable GPW example validates against a pinned reference energy printed at
the end of the run:

- Isolated H₂O (PBE, DZVP-MOLOPT-GTH, Martyna-Tuckerman) vs the pinned anchor
  `-17.21233714` Ha.

The full anchor suite (H₂ / Ar / H₂O / ZnO / NaCl single points, forces vs the
FD oracle, geoopt, AIMD, OT-vs-diag) lives in `/* benchmarks removed; re-add later */ gpw/validation/` —
every case is a minimal textbook system and runs in minutes.

## Questions for reflection

Why are basis size and density cutoff independent convergence variables? What can a rigid-translation test expose that a single energy comparison cannot?
