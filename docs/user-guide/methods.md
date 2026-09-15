# Methods

Which **preset** and **evaluator** to use for each job. For the underlying
vocabulary (Driver / Component / Frame / Criterion), start with
[Core concepts](concepts.md). Scientific theory and literature numbers live
in the [Tutorial](../tutorial/).

Every calculation is a **driver** (what kind of job) plus an **evaluator**
component (how energy and forces are computed). The two axes are independent:
any evaluator can be bound to any driver, which is how the same
`cpu::md::MD` loop runs both a Lennard-Jones fluid and ab-initio MD.

## Drivers — what kind of job

Each driver below is a **preset**: an ordinary loop whose constructor
pre-attaches the policy and infrastructure that kind of job wants. There is
no category hierarchy behind them — one loop-node class (`cpu::Driver<P>`)
serves them all, and *when the loop stops* is a policy object you can replace.

| Driver | Job type | Stops when |
|---|---|---|
| `cpu::Evaluate` | single point | after exactly one pass |
| `cpu::md::MD` | molecular dynamics | the step budget is exhausted |
| `cpu::minimizer::Minimize` | geometry optimization | the `ForceConverged` criterion says so |
| `cpu::vib::Hessian` | vibrational analysis | all 6N displacements are done |
| `cpu::mc::MC` | Monte Carlo sampling | the trial-move budget is exhausted |
| `cpu::PathDriver` | transition path (CI-NEB) | every replica of the band has converged |

`cuda::Evaluate`, `cuda::md::MD`, and `cuda::minimizer::Minimize` mirror the
first three on the GPU with the same API.

Two policy slots let you change *when* a run ends without writing a driver:

- `driver.add_criterion(c)` — **plural**. A `cpu::Criterion` is a Component the
  driver polls once per
  iteration; it returns `Continue`, `Converged`, or `Stop`; several compose,
  most-severe-wins. In-tree: `ForceConverged`, `AllReplicasConverged`,
  `StopOnNaN`. Writing your own (LAMMPS's `fix halt`: "stop above 2000 K") is
  about ten lines, from outside the library. Ask the driver for the outcome:
  `driver.converged()`, `driver.stop_reason()`.
- `driver.set_outcome(p)` — **singular**. A `cpu::OutcomePolicy` decides
  whether the iteration that just ran is kept (`Commit`) or undone
  (`Rollback`). `MC` attaches `cpu::Metropolis`; a GCMC or REMD acceptance
  rule goes here too.

## Energy / force sources — how the energy is computed

| Method | Component (kind) | Build flag |
|---|---|---|
| Lennard-Jones + bonded styles | `PairLJCut` / `BondHarmonic` / … (`Potential`) | `ATV_BUILD_POTENTIALS` |
| Reciprocal-space electrostatics | `KSpaceEwald` / `KSpacePME` / `KSpacePPPM` (`Potential`) | `ATV_BUILD_KSPACE` |
| Gaussian SCF — Hartree-Fock and Kohn-Sham | `cpu::scf::RHF` / `UHF` / `ROHF` / `GHF` / `RKS` / `UKS` / `ROKS` (`HF`) | `ATV_BUILD_HF` |
| Plane-wave DFT | `cpu::pw::PWEvaluator` (`PlaneWave`) | `ATV_BUILD_PW` |
| Gaussian-and-plane-waves DFT | `cpu::gpw::GPWEvaluator` (`GPW`) | `ATV_BUILD_GPW` |
| NDDO semiempirics | `cpu::semiempirical::AM1Evaluator` / `RM1Evaluator` (`Semiempirical`) | `ATV_BUILD_SEMIEMPIRICAL` |
| DFTB | `cpu::semiempirical::DFTB0` / `DFTB2` / `DFTB3` (`Semiempirical`) | `ATV_BUILD_SEMIEMPIRICAL` |
| Extended tight binding | `cpu::semiempirical::GFN0` / `GFN1` / `GFN2` / `IPEA` (`Semiempirical`) | `ATV_BUILD_SEMIEMPIRICAL` |

Most methods are a **concrete type** carrying its parameters as constructor
arguments and public members, with `run(frame)` as the entry point —
`cpu::scf::RHF`/`UHF`/`RKS`/`UKS` (`cpu/scf/methods.h`), `cpu::pw::PW`
(`cpu/planewave/pw_solver.h`), `cpu::semiempirical::AM1`/`RM1`
(`cpu/semiempirical/am1_solver.h`). `PW` and `AM1` want one `build(reference)`
before the first `run()`; the SCF methods build themselves.

`cpu::gpw::GpwDft` (`cpu/gpw/gpw_solver.h`) is the exception: its `run()` takes
a `std::vector<GpwAtom>` — positions in **Bohr**, with the contracted basis and
the GTH pseudopotential already resolved per atom — not a `Frame`. The
Frame-shaped entry point for GPW is therefore the `cpu::gpw::GPWEvaluator`
component (`cpu/gpw/gpw_evaluator.h`), which owns the solver and does that
translation; use it, not `GpwDft`, from user code.

Bind the matching evaluator component when a Driver needs energy/forces on a
Plan slot.

## Classical molecular dynamics

Pair and bonded styles, neighbor lists, NVE/NVT/NPT, and reciprocal-space
electrostatics (Ewald / PME under `ATV_BUILD_KSPACE`). Components are attached
with LAMMPS-shaped categorical setters.

```cpp
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
using namespace ATV;

cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.8});     // eps (eV), sigma (Å), rcut (Å)
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});     // cutoff (Å), skin (Å)
sim.set_integrator(cpu::md::VelocityVerlet{0.01});       // dt (fs)
sim.set_thermostat(cpu::md::NoseHoover{300.0, 100.0, 0.01});  // T (K), tau_T (fs), dt (fs)
sim.set_thermo(100);
sim.build();
sim.run(frame, 1000);
```

The thermostat takes the timestep too — it integrates its own chain variable on
the same clock as the integrator, so pass the same `dt`.

See [Chapter 10](../tutorial/ch10-classical-md.md) and
[Chapter 11](../tutorial/ch11-ensembles.md).

## Gaussian-basis SCF

RHF, UHF, and Kohn-Sham DFT (LSDA, PBE, B3LYP, PBE0) for molecules, with
analytic gradients. Basis sets: `STO3G`, `Pople631G`, `Pople631Gs`, `Def2SVP`,
`Def2TZVP`.

There is **no** public freestanding integral bag / `SCFConfig` / `SCFSolver`.
Construct a **concrete method** with its parameters as constructor arguments and
pass a `Frame`:

```cpp
#include "atomiverse/cpu/scf/methods.h"
using namespace ATV;

cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10,
                  /*max_iter=*/100};
rhf.run(frame);                  // returns void — results land on the object
                                 // and on the Frame
double e_ha = rhf.energy();                  // total energy [Hartree]
double e_ev = frame.meta.get(ATV::kEnergy);  // the same energy [eV]
bool ok = rhf.converged();

// forces:
// cpu::scf::GradientResult g = rhf.forces(frame);  // eV/Å, also written to
//                                                  // the fx/fy/fz columns
```

`run()` writes eV to `frame.meta[kEnergy]`; the Hartree internals stay on the
method (`energy()`, `converged()`, `iterations()`, `determinant()`).

Open-shell and KS:

```cpp
#include "atomiverse/cpu/scf/methods.h"
using namespace ATV;

cpu::scf::UHF uhf(/*n_alpha=*/5, /*n_beta=*/4);
uhf.charge = 1;
uhf.run(frame);

cpu::scf::RKS rks(cpu::scf::XCType::PBE);
rks.run(frame);

cpu::scf::UKS uks(1, 1, cpu::scf::XCType::LSDA);
uks.run(frame);
```

Chapters [2](../tutorial/ch02-rhf.md)–[7](../tutorial/ch07-correlation.md)
cover this arc, from a bare fixed-point loop to hybrid functionals.

## Ab-initio MD

The same `MD` driver, with the SCF evaluator bound at the force phase. This is
the whole point of the architecture: nothing about the loop changes.

```cpp
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/scf/methods.h"
using namespace ATV;

cpu::md::MD aimd;
aimd.set_integrator(cpu::md::VelocityVerlet{0.5});
aimd.on(md::Phase::force).add(cpu::scf::RHF{});  // concrete method as force component
aimd.build();
aimd.run(frame, 100);
```

## Plane-wave and GPW DFT

Plane-wave Kohn-Sham DFT (`cpu::pw`) targets periodic crystals: norm-conserving
pseudopotentials (analytic GTH in-tree, or a UPF file), a plane-wave cutoff, a
k-mesh, per-k Davidson diagonalization, and Pulay mixing. GPW (`cpu::gpw`) is
the CP2K-style hybrid — Gaussian orbitals with a plane-wave density — carrying
in-tree GTH pseudopotentials, MOLOPT basis sets, and DFT-D3 dispersion, so it
needs no external data files.

A plane-wave run has **no default pseudopotential**: set exactly one of
`gth_name` (analytic Goedecker-Teter-Hutter, shipped in-tree — H, O, Na, Cl, Ar,
Si, Ni, Zn) or `upf_path` (a norm-conserving UPF file). Setting neither, or
both, throws.

```cpp
#include <utility>

#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
using namespace ATV;

cpu::pw::PW pw;
pw.gth_name = "GTH-PBE-q4";               // in-tree analytic GTH (Si, 4 valence e⁻)
pw.ecut_ry = 30.0;                        // wavefunction cutoff [Ry]
pw.k_mesh = 4;                            // Monkhorst-Pack mesh per dimension
cpu::pw::PWEvaluator pwe{std::move(pw)};  // PW is move-only
cpu::single_point(frame, pwe);            // frame.meta.get(ATV::kEnergy) (eV)
```

See [Chapter 8](../tutorial/ch08-plane-wave.md) and
[Chapter 9](../tutorial/ch09-gpw.md).

## Geometry optimization

L-BFGS or conjugate gradient (Polak-Ribière+), hot-swappable on the same driver,
over *any* force provider — classical or quantum.

```cpp
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/scf/methods.h"
using namespace ATV;

cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});  // memory, alpha0, max_ls
// Convergence is the criterion's, not the optimizer's. Omit it and Minimize
// attaches cpu::ForceConverged{} with its defaults.
opt.add_criterion(cpu::ForceConverged{/*ftol=*/0.05});
opt.on(minimizer::Phase::force).add(cpu::scf::RHF{});
opt.build();
opt.run(frame, 100);
```

See [Chapter 12](../tutorial/ch12-geometry-optimization.md).

## Monte Carlo

NVT Metropolis sampling. The MC driver reuses the MD force pipeline for trial
energies, so a system you can run with MD you can also sample with MC — you only
add a **Mover**.

```cpp
#include "atomiverse/cpu/montecarlo/mc.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
using namespace ATV;

cpu::mc::MC mc{/*T (K)=*/300.0, /*seed=*/12345u};
mc.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, true});
mc.set_mover(cpu::mc::TranslateMove{/*delta (Å)=*/0.2});
mc.build();
mc.run(frame, 100000);                    // 100k trial moves
double e = mc.current_energy();            // (eV); mc.acceptance_ratio() in [0,1]
```

## Semiempirical and tight binding

Roughly SCF accuracy at a small fraction of the cost — the right choice when a
full Gaussian SCF is too expensive per step. Three families, all CPU-only:

- **NDDO** — `AM1` and `RM1`, with analytic nuclear forces and AM1-BCC charge
  assignment. Pinned against OpenMOPAC.
- **DFTB** — `DFTB0` (non-SCC), `DFTB2` (SCC) and `DFTB3` (third order) on the
  in-tree mio-1-1 and 3ob-3-1 Slater-Koster tables for H/C/N/O. DFTB2 carries
  analytic forces and analytic cell stress for closed and open shell, at Gamma
  and on a k-mesh, plus D3 dispersion, DFTB+U, and linear-response TD-DFTB.
  Pinned against DFTB+.
- **Extended tight binding** — `GFN0`, `GFN1`, `GFN2` and `IPEA`, each with
  analytic forces and an FD Hessian. Pinned against xtb.

Each is a concrete type with its parameters as members; bind the matching
evaluator (or the method itself, which is a `Semiempirical` component) into any
Driver that needs energy and forces. See
[Chapter 17](../tutorial/ch17-semiempirical-am1.md).

## Vibrations

A finite-difference Hessian plus thermochemistry, driven by
`cpu::vib::Hessian` over any force provider. See
[Chapter 14](../tutorial/ch14-vibrations.md).

## Observables

Derived quantities are **Observer** components that any driver can carry: radial
distribution functions, virial stress, Mulliken populations, and electron
densities. Writing them out is a separate **Recorder** component (XYZ, cube,
thermo). See [Chapter 13](../tutorial/ch13-observables.md).
