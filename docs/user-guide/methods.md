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
no category hierarchy behind them — one loop-node class (`cpu::CPUDriver<P>`)
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
| Lennard-Jones + bonded styles | `PairLJCut` / `BondHarmonic` / … (`Potential`) | *(always on)* |
| Gaussian SCF — RHF / UHF / Kohn-Sham DFT | `cpu::scf::SCF` (`HF`) | `ATV_BUILD_HF` |
| Plane-wave DFT | `cpu::pw::PWEvaluator` (`PlaneWave`) | `ATV_BUILD_PW` |
| Gaussian-and-plane-waves DFT | `cpu::gpw::GPWEvaluator` (`GPW`) | `ATV_BUILD_GPW` |
| NDDO / AM1 semiempirics | `cpu::semiempirical::AM1Evaluator` (`Semiempirical`) | `ATV_BUILD_HF` |

Each method's iteration loop lives in an inner **solver**
(`cpu::scf::SCFSolver`, `cpu::pw::PWSolver`, `cpu::gpw::GPWSolver`,
`cpu::semiempirical::NDDOSolver`) that you can also drive directly when you
want the raw result struct rather than a frame.

## Classical molecular dynamics

Pair and bonded styles, neighbor lists, NVE/NVT/NPT, and reciprocal-space
electrostatics (Ewald / PME under `ATV_BUILD_KSPACE`). Components are attached
with LAMMPS-shaped categorical setters.

```cpp
cpu::md::MD sim;
sim.add_potential(cpu::PairLJCut{0.0104, 3.4, 6.8});          // eps (eV), sigma (Å), rcut (Å)
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{0.01});       // dt (fs)
sim.set_thermostat(cpu::md::NoseHoover{300.0, 100.0});   // T (K), tau (fs)
sim.set_thermo(100);
sim.build();
sim.run(frame, 1000);
```

See [Chapter 10](../tutorial/ch10-classical-md.md) and
[Chapter 11](../tutorial/ch11-ensembles.md).

## Gaussian-basis SCF

RHF, UHF, and Kohn-Sham DFT (LSDA, PBE, B3LYP, PBE0) for molecules, with
analytic gradients. Basis sets: `STO3G`, `Pople631G`, `Pople631Gs`, `Def2SVP`,
`Def2TZVP`.

There is **no** public `IntegralPack` / `SCFConfig` / `SCFSolver`. Construct a
**concrete method**, set members, pass a `Frame`:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-10;
rhf.max_iter = 100;
const auto& res = rhf.run(frame);   // res.energy [Hartree]
// forces:
// auto g = rhf.forces(frame);      // eV/Å on frame force columns
```

Open-shell and KS:

```cpp
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
cpu::md::MD aimd;
aimd.set_integrator(cpu::md::VelocityVerlet{0.5});
aimd.on(md::Phase::force).add(cpu::scf::RHF{});  // concrete method as force component
aimd.build();
aimd.run(frame, 100);
```

## Plane-wave and GPW DFT

Plane-wave Kohn-Sham DFT (`cpu::pw`) targets periodic crystals: UPF
pseudopotentials, a plane-wave cutoff, a k-mesh, per-k Davidson diagonalization,
and Pulay mixing. GPW (`cpu::gpw`) is the CP2K-style hybrid — Gaussian orbitals
with a plane-wave density — carrying in-tree GTH pseudopotentials, MOLOPT basis
sets, and DFT-D3 dispersion, so it needs no external data files.

```cpp
cpu::pw::PWEvaluator pwe{cpu::pw::PWConfig{.ecut_ry = 30.0, .k_mesh = 4}};
cpu::single_point(frame, pwe);            // frame.meta.get(kEnergy) (eV)
```

See [Chapter 8](../tutorial/ch08-plane-wave.md) and
[Chapter 9](../tutorial/ch09-gpw.md).

## Geometry optimization

L-BFGS or conjugate gradient (Polak-Ribière+), hot-swappable on the same driver,
over *any* force provider — classical or quantum.

```cpp
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
cpu::mc::MC mc{/*T (K)=*/300.0, /*seed=*/12345u};
mc.add_potential(cpu::PairLJCut{eps, sigma, cutoff});
mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, true});
mc.set_mover(cpu::mc::TranslateMove{/*delta (Å)=*/0.2});
mc.build();
mc.run(frame, 100000);                    // 100k trial moves
double e = mc.current_energy();            // (eV); mc.acceptance_ratio() in [0,1]
```

## Semiempirical

NDDO with the AM1 parameterization, and AM1-BCC charge assignment. Roughly SCF
accuracy at force-field cost, useful when a full Gaussian SCF is too expensive
per step.

## Vibrations

A finite-difference Hessian plus thermochemistry, driven by
`cpu::vib::Hessian` over any force provider. See
[Chapter 14](../tutorial/ch14-vibrations.md).

## Observables

Derived quantities are **Observer** components that any driver can carry: radial
distribution functions, virial stress, Mulliken populations, and electron
densities. Writing them out is a separate **Recorder** component (XYZ, cube,
thermo). See [Chapter 13](../tutorial/ch13-observables.md).
