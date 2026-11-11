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
| Plane-wave DFT | `cpu::pw::PW` (nested `Driver<pw::Phase>`) | `ATV_BUILD_PW` |
| Gaussian-and-plane-waves DFT | `cpu::gpw::GpwDft` (nested `Driver<gpw::Phase>`) | `ATV_BUILD_GPW` |
| NDDO semiempirics | `cpu::semiempirical::AM1` / `RM1`; CUDA counterparts provide CHNO molecular E/q/analytic F (nested `Driver<nddo::Phase>`) | `ATV_BUILD_SEMIEMPIRICAL`; CUDA additionally `ATV_BUILD_CUDA` |
| DFTB | `cpu::semiempirical::DFTB0` / `DFTB3` (`Semiempirical`); `DFTB2` is also a nested `Driver<scc::Phase>` | `ATV_BUILD_SEMIEMPIRICAL` |
| Extended tight binding | `cpu::semiempirical::GFN0` / `GFN1` / `GFN2` / `IPEA` (`Semiempirical`) | `ATV_BUILD_SEMIEMPIRICAL` |

Most methods are a **concrete type** carrying its parameters as constructor
arguments and public members, with `run(frame)` as the entry point —
`cpu::scf::RHF`/`UHF`/`RKS`/`UKS` (`cpu/scf/methods.h`), `cpu::pw::PW`
(`cpu/planewave/pw_solver.h`), `cpu::gpw::GpwDft` (`cpu/gpw/gpw_solver.h`),
`cpu::semiempirical::AM1`/`RM1` (`cpu/semiempirical/am1_solver.h`). `PW` and
`GpwDft` want one `build(reference)` before the first `run()`; Gaussian SCF
and AM1/RM1 build themselves on first use. AM1/RM1 also permit explicit
`build(reference)` to prepare their workspace.

`GpwDft` takes one `Kind` (element + contracted Gaussian basis + GTH
pseudopotential) per element and reads positions [Å], Z (`kAtom_atomic_number`) and the
orthorhombic box from the Frame; `run(frame)` writes the energy [eV] and
`forces(frame)` also writes `kAtom_f*` [eV/Å]. For Born–Oppenheimer work,
mount the method directly on the outer Driver:

```cpp
cpu::gpw::GpwDft dft{{O, H}, 400.0, cpu::gpw::XcKind::PBE};          // cpu::gpw::Kind per element
cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(dft);
driver.run(frame);
```

The parent phase requests energy/forces; stress is assembled when the plan
consumes it. The borrowed method must outlive the Driver.
`cpu::gpw::GPWEvaluator` remains for electronic propagators (CPMD,
second-gen CP, XL-BOMD) and callers awaiting migration.

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

cpu::scf::UHF uhf{cpu::scf::BasisSet::STO3G, /*n_alpha=*/5, /*n_beta=*/4, /*charge=*/1};
uhf.run(frame);

cpu::scf::RKS rks{cpu::scf::BasisSet::STO3G, cpu::scf::XCType::PBE};
rks.run(frame);

cpu::scf::UKS uks{cpu::scf::BasisSet::STO3G, /*n_alpha=*/1, /*n_beta=*/1, cpu::scf::XCType::LSDA};
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
aimd.on(md::Phase::force).add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G});  // concrete method as force component
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

A plane-wave run takes a `shared_ptr<const electronic::NormConservingPotential>`.
Use `GthPseudopotential(element, name)`, `UpfPseudopotential(path)`, or derive
the public physical interface in an independent project. The built-ins and
custom providers use the same CPU NC operators; every Frame atom must match
the provider's explicit element. See the [independent extension](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/extensions/nc/README.md).

```cpp
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
using namespace ATV;

cpu::pw::PW pw{std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::Si, "GTH-PBE-q4"),  // in-tree analytic GTH (Si, 4 valence e⁻)
               30.0,                                       // wavefunction cutoff [Ry]
               4};                                         // Monkhorst-Pack mesh per dimension
cpu::single_point(frame, pw);             // frame.meta.get(ATV::kEnergy) (eV)
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
opt.on(minimizer::Phase::force).add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G});
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

These parameterized electronic models have distinct accuracy and coverage
limits. All three families have CPU implementations; native CUDA currently
adds a restricted AM1/RM1 energy/charge/analytic-force path:

- **NDDO** — `AM1` and `RM1`, with analytic nuclear forces from three-direction
  Cartesian derivatives of the pair integrals, contracted with the converged
  density. CPU convergence requires both density and energy tolerances. Model
  parameters are unchanged; executed OpenMOPAC 23.2.5 pins cover energy, heat
  of formation, charges and full forces for distorted H₂O, CO and NH₃.
- **DFTB** — `DFTB0` (non-SCC), `DFTB2` (SCC) and `DFTB3` (third order) on the
  shipped Slater-Koster tables for H/C/N/O. DFTB2 supports mio-1-1 and
  molecular 3ob-3-1. Its implemented variants carry
  analytic forces and analytic cell stress for closed and open shell, at Gamma
  and on a k-mesh. Additional capabilities include two-body D3 dispersion,
  energy-only DFTB+U (no +U forces/stress), and TD-DFTB excitation energies.
  DFTB0/2 have external DFTB+ pins within the documented variants; multi-k
  external evidence remains open. DFTB3 uses a restricted mixed parameter
  setup with internal snapshots and FD derivative checks, not a completed
  external DFTB3 Hamiltonian match.
- **Extended tight binding** — `GFN0`, `GFN1`, `GFN2` and `IPEA`, each with
  analytic forces and an FD Hessian. Existing xtb cases pass; GFN2 still needs
  tight external full-force-vector validation before its CUDA migration.

Mount CPU AM1/RM1 directly in the parent phase; their SCF uses the parent's
Runtime. Standalone `run(Frame&)` overwrites E/q and preserves forces;
`run(const Frame&)` updates only the method result; `forces(Frame&)` overwrites
E/q/F. Mounted evaluation adds energy and requested forces and writes charges.
Both SCF residual criteria must pass, and failed/trial/initial-force solves do
not replace the accepted electronic guess. CPU methods remain molecular and
closed shell, with existing charged and s/p-element support; RM1's shipped
parameters cover H/C/N/O/P/S. Neither supplies periodic NDDO stress or UHF.

`cuda::semiempirical::AM1` and `RM1` expose the same standalone E/q/F semantics
for neutral closed-shell CHNO isolated molecules in FP64, including heat of
formation and analytic forces. They explicitly reject stress, nonzero charge,
open shells, non-CHNO elements and periodic cells. CUDA Evaluate mounts these
methods directly; standalone/mounted results, additive contributions and
failure/retry are verified. Both methods pass CUDA MD initial-force preparation,
2+2 versus four-step continuation and CPU/CUDA four-step agreement at `0.01 fs`.
Three steepest-descent steps lower energy, without establishing optimization
convergence. Line-search trial/rejection, host edits between chunks and longer
NVE/time-step convergence remain unverified.
Python exposes CPU AM1; RM1 and CUDA NDDO Python parity remain open.

See [Chapter 17](../tutorial/ch17-semiempirical-am1.md) for usage and the
[method-by-method roadmap](../roadmap/semiempirical.md) for executed evidence
and variant limits. AM1/RM1 support is not completion of the full SQM method
set or input compatibility.

### AM1-BCC charges (GAFF / GAFF2)

`AM1BCCCharges` computes molecular AM1-BCC or ABCG2 partial charges.
The correction family is independent of the GAFF generation.

1. It runs AM1 at the molecular graph's geometry to get the Mulliken base charges.
2. It averages those charges over topologically equivalent atoms.
3. It adds the AM1-BCC (`"bcc"`) or ABCG2 (`"abcg2"`) bond charge increments.

The fixed-geometry benchmark compares the result with AmberTools
`antechamber -c bcc` / `-c abcg2`. The graph must carry the molecule's bonds
and coordinates; the net charge is the `AM1` constructor argument `charge`.

```cpp
#include "atomiverse/cpu/semiempirical/am1_bcc_charges.h"
using namespace ATV::cpu::semiempirical;

AM1 am1{/*charge=*/-1};  // net molecular charge (e)
auto charges = AM1BCCCharges{am1, "bcc"}.assign(molecule); // owned charges [e]
```

The geometry is used as given. By default antechamber first optimises the
structure with sqm. To get the same, run a `Minimize` over a mounted `AM1`
before calling `assign`. The charge model accepts a chemical molecular graph
and returns charges in atom order without modifying that graph. Python uses
`AM1BCCCharges(method=AM1(charge=-1), correction="bcc").assign(molecule)`,
where `molecule` is a molpy `Atomistic`.

## Vibrations

A finite-difference Hessian plus thermochemistry, driven by
`cpu::vib::Hessian` over any force provider. See
[Chapter 14](../tutorial/ch14-vibrations.md).

## Observables

Derived quantities are **Observer** components that any driver can carry: radial
distribution functions, virial stress, Mulliken populations, and electron
densities. Writing them out is a separate **Recorder** component (XYZ, cube,
thermo). See [Chapter 13](../tutorial/ch13-observables.md).
