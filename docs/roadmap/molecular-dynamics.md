# Molecular Dynamics

Legend and caveats: [Roadmap index](index.md).

The Dynamics driver (`cpu::md::MD`, `cuda::md::MD`) is the most complete part of
the engine: NVE/NVT/NPT all run on both backends, with MPI domain decomposition,
halo exchange, and migration on CPU, and multi-GPU support on CUDA. Constraints
(RATTLE) run on CPU under MPI. The gaps are specific rather than structural — no NPH,
only uniaxial deformation, and no rigid bodies.

## 1. Molecular-Dynamics Integrators

| Method                            | CPU | CUDA |
| --------------------------------- | :-: | :--: |
| Velocity Verlet                   |  ✓  |   ✓  |
| Forest–Ruth integrator            |  ○  |   ○  |
| Reversible RESPA                  |  ○  |   ○  |
| Langevin dynamics                 |  ✓  |   ✓  |
| BAOAB Langevin integrator         |  ○  |   ○  |
| GJF Langevin integrator           |  ○  |   ○  |
| Langevin-middle integrator        |  ○  |   ○  |
| Brownian dynamics                 |  ○  |   ○  |
| Generalized Langevin equation     |  ○  |   ○  |
| Rigid-body quaternion integration |  ○  |   ○  |

Langevin dynamics is the Schneider–Stoll additive-force form (`FixLangevin`),
not one of the named splitting schemes. `NVELimit` (LAMMPS `nve/limit`, a
per-step displacement cap) is also available on both backends for relaxing
overlapping start configurations.

## 2. Thermostats

| Method                                   | CPU | CUDA |
| ---------------------------------------- | :-: | :--: |
| Langevin thermostat                      |  ✓  |   ✓  |
| Nosé–Hoover thermostat                   |  ✓  |   ✓  |
| Nosé–Hoover chain thermostat             |  ✓  |   ✓  |
| Bussi stochastic velocity rescaling      |  ✓  |   ○  |
| Gaussian isokinetic thermostat           |  ○  |   ○  |
| Massive Nosé–Hoover chains               |  ○  |   ○  |
| Dissipative particle dynamics thermostat |  ○  |   ○  |
| Dual thermostat for Drude particles      |  ○  |   ○  |
| PILE thermostat                          |  ○  |   ○  |
| PIGLET thermostat                        |  ○  |   ○  |

Two caveats behind those ✓ marks:

- The standalone `NoseHoover` component is a **single-chain** thermostat (one
  scalar ξ) — a documented divergence from LAMMPS, which chains by default.
- The real **M-chain NHC** (default M = 3, with `drag`) exists only *inside* the
  MTK NPT integrator, not as a standalone NVT thermostat. Choosing NVT with
  chains today means running the NPT integrator, or accepting the single chain.

Bussi/CSVR (`FixCSVR`) is CPU-only.

## 3. Barostats

| Method                              | CPU | CUDA |
| ----------------------------------- | :-: | :--: |
| Berendsen barostat                  |  ✓  |   ✓  |
| Parrinello–Rahman barostat          |  ○  |   ○  |
| Martyna–Tobias–Klein barostat       |  ✓  |   ✓  |
| Langevin piston                     |  ○  |   ○  |
| Monte Carlo isotropic barostat      |  ○  |   ○  |
| Monte Carlo anisotropic barostat    |  ○  |   ○  |
| Isotropic pressure coupling         |  ✓  |   ✓  |
| Semi-isotropic pressure coupling    |  ✓  |   ✓  |
| Fully anisotropic pressure coupling |  ○  |   ○  |
| Constant-surface-tension coupling   |  ○  |   ○  |
| Constant-stress dynamics            |  ○  |   ○  |

The MTK integrator couples the **cell diagonal** only: isotropic, or per-axis
via `couple_axis[3]` (an axis left off is held rigid). There is no off-diagonal
or shear coupling, so the cell cannot tilt — which is exactly why
Parrinello–Rahman, constant surface tension, and constant-stress dynamics are
all ○. Berendsen is isotropic only.

## 4. Molecular-Dynamics Ensembles

| Method                              | CPU | CUDA |
| ----------------------------------- | :-: | :--: |
| Microcanonical dynamics, NVE        |  ✓  |   ✓  |
| Canonical dynamics, NVT             |  ✓  |   ✓  |
| Isothermal–isobaric dynamics, NPT   |  ✓  |   ✓  |
| Isoenthalpic–isobaric dynamics, NPH |  ○  |   ○  |
| Constant surface-tension dynamics   |  ○  |   ○  |
| Constant stress dynamics            |  ○  |   ○  |
| Simulated annealing                 |  ○  |   ○  |
| Temperature-ramp dynamics           |  ✓  |   ✓  |
| Pressure-ramp dynamics              |  ○  |   ○  |

**NPH is unreachable by construction**: the MTK integrator fuses the thermostat
chain with the barostat, and there is no thermostat-free barostat integrator.

Temperature ramps are supported by both the Langevin thermostat and the MTK NPT
integrator (`T_target` → `T_stop`). Simulated annealing is marked ○ because it
is a *protocol* over those ramps, not a component — you can express it today by
driving the ramp yourself, but the engine ships no annealing schedule.

## 5. Constraint and Rigid-Body Methods

| Method                       | CPU | CUDA |
| ---------------------------- | :-: | :--: |
| RATTLE                       |  ✓  |   ○  |
| P-LINCS                      |  ○  |   ○  |
| SETTLE                       |  ○  |   ○  |
| CCMA                         |  ○  |   ○  |
| Distance constraints         |  ✓  |   ○  |
| Angle constraints            |  ✓  |   ○  |
| Rigid-water dynamics         |  ○  |   ○  |
| General rigid-body dynamics  |  ○  |   ○  |
| Virtual-site dynamics        |  ○  |   ○  |
| Hydrogen-mass repartitioning |  ○  |   ○  |

`cpu::md::FixRattle` (Andersen 1983) holds distance and 1–3 (angle) constraints, runs
under MPI, and throws rather than return a molecule that is merely *nearly* rigid. It
holds TIP3P geometry to 5 × 10⁻¹¹ relative over 5 ps at `dt = 2 fs`
(`regressions/md/rigid_water`).

**SETTLE** is listed and unwritten. It is a closed-form solution for exactly one cluster
shape — a three-site rigid water — so it is an *optimization* of the general solver rather
than a replacement for it: worth a few percent of step time on water, worth nothing to any
molecule that is not water. It gets written when a profile asks for it.

Before RATTLE, every molecule was fully flexible: water models had to be flexible
variants and the timestep was bounded by the fastest X–H stretch (~9 fs period for O–H,
so ~0.5 fs steps). Constrained water runs at `dt = 2 fs`. What is still missing for
biomolecular work is **rigid bodies** and **virtual sites**.

## 6. Nonequilibrium Molecular Dynamics

| Method                              | CPU | CUDA |
| ----------------------------------- | :-: | :--: |
| Constant electric-field MD          |  ○  |   ○  |
| Oscillating electric-field MD       |  ○  |   ○  |
| Constant-force pulling              |  ○  |   ○  |
| Constant-velocity pulling           |  ○  |   ○  |
| Steered molecular dynamics          |  ○  |   ○  |
| SLLOD shear dynamics                |  ○  |   ○  |
| Lees–Edwards shear dynamics         |  ○  |   ○  |
| Uniaxial-deformation MD             |  ✓  |   ✓  |
| Biaxial-deformation MD              |  ○  |   ○  |
| Triaxial-deformation MD             |  ○  |   ○  |
| Oscillatory-shear MD                |  ○  |   ○  |
| Reverse nonequilibrium MD           |  ○  |   ○  |
| Müller–Plathe heat-flux method      |  ○  |   ○  |
| Homogeneous nonequilibrium MD       |  ○  |   ○  |
| Temperature-gradient MD             |  ○  |   ○  |
| Pressure-driven flow                |  ○  |   ○  |
| Poiseuille-flow MD                  |  ○  |   ○  |
| Shock-wave MD                       |  ○  |   ○  |

`Deform` drives one axis at a time, with affine coordinate remap, in either
engineering strain (`erate`, `L = L₀(1 + Rt)`) or true strain (`trate`,
`L = L₀·exp(Rt)`). The CUDA port supports `erate` only. This is enough for a
uniaxial tension/compression run (see `/* benchmarks removed; re-add later */ pe/`, polyethylene under
tension) and nothing more — no shear, no flow, no heat flux.

## 7. Ab Initio Molecular Dynamics

| Method                                  | CPU | CUDA |
| --------------------------------------- | :-: | :--: |
| Hartree–Fock Born–Oppenheimer MD        |  ✓  |   ○  |
| Kohn–Sham Born–Oppenheimer MD           |  ✓  |   ○  |
| Semiempirical Born–Oppenheimer MD       |  ✓  |   ○  |
| DFTB molecular dynamics                 |  ○  |   ○  |
| xTB molecular dynamics                  |  ○  |   ○  |
| Car–Parrinello molecular dynamics       |  ○  |   ○  |
| Extended-Lagrangian Born–Oppenheimer MD |  ○  |   ○  |
| Second-generation Car–Parrinello MD     |  ○  |   ○  |

AIMD works by binding a force-providing Evaluator to `md::Phase::force`. Which
evaluators qualify is decided by one thing — whether they declare
`Capability::Force`:

- `cpu::scf::SCF` with `with_gradient(true)` → **HF and KS BOMD** (both
  restricted and unrestricted), with an ASPC density predictor for warm starts.
- `cpu::gpw::GPWEvaluator` with `compute_forces` → **periodic KS BOMD** (Γ-point,
  closed-shell).
- `cpu::pw::PWEvaluator` is **energy-only** — plane waves cannot drive MD.
- `cpu::semiempirical::AM1Evaluator` with `with_gradient(true)` (default) writes
  nuclear forces (eV/Å) (validated vs re-SCF FD). Semiempirical BOMD is
  pinned by `regressions/semiempirical/am1_water_bomd` (short NVE-style smoke).
- `cuda::scf::SCFEvaluator` has no gradient kernel — **there is no GPU AIMD**.

ASPC is a density *predictor* (warm start), not an extended-Lagrangian
propagator, so the XL-BOMD row stays ○.

## 8. Nuclear Quantum Dynamics

| Method                                 | CPU | CUDA |
| -------------------------------------- | :-: | :--: |
| Path-integral molecular dynamics       |  ○  |   ○  |
| Ring-polymer molecular dynamics        |  ○  |   ○  |
| Centroid molecular dynamics            |  ○  |   ○  |
| Thermostatted ring-polymer MD          |  ○  |   ○  |
| Normal-mode PIMD                       |  ○  |   ○  |
| Staging-coordinate PIMD                |  ○  |   ○  |
| Ring-polymer contraction               |  ○  |   ○  |
| Quantum thermal bath                   |  ○  |   ○  |
| Path-integral free-energy perturbation |  ○  |   ○  |

## 9. Nonadiabatic Dynamics

| Method                                | CPU | CUDA |
| ------------------------------------- | :-: | :--: |
| Ehrenfest dynamics                    |  ○  |   ○  |
| Fewest-switches surface hopping       |  ○  |   ○  |
| Landau–Zener surface hopping          |  ○  |   ○  |
| Decoherence-corrected surface hopping |  ○  |   ○  |
| Spin-mixed surface hopping            |  ○  |   ○  |
| Mapping-variable dynamics             |  ○  |   ○  |
| Ab initio multiple spawning           |  ○  |   ○  |
| Diabatic-state dynamics               |  ○  |   ○  |

Nonadiabatic dynamics requires excited states, and there are none — see
[Excited States and Many-Body](excited-states.md).

## 10. Vibrational Analysis

Not part of the reference taxonomy, but it is what the Response driver does, and
it is what stands in for every "analytic Hessian" row in this roadmap.

| Method                                    | CPU | CUDA |
| ----------------------------------------- | :-: | :--: |
| Finite-difference Hessian (central, 6N)   |  ✓  |   ○  |
| Mass-weighted normal modes / frequencies  |  ✓  |   ○  |
| Harmonic thermochemistry (RRHO ideal gas) |  ✓  |   ○  |
| Analytic Hessian (any method)             |  ○  |   ○  |

`cpu::vib::Hessian` displaces each Cartesian degree of freedom and differences
the analytic forces, so it works with any force-providing evaluator — classical
or SCF. It is molecular: there is no supercell/q-point phonon machinery.
