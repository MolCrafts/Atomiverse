# Multiscale Methods and Properties

Legend and caveats: [Roadmap index](index.md).

Everything that couples two levels of theory, or extracts a chemical property
from a converged wavefunction. Almost all of it is unimplemented; the exceptions
are the two charge schemes Atomiverse actually needs today — Mulliken (as an
Observer over the SCF density) and AM1-BCC (built on the
[AM1](semiempirical.md) solver plus molrs bond-charge-correction typing).

There is **no QM/MM coupling of any kind**, no implicit solvent, no embedding,
and no relativistic treatment. The only "pseudopotentials" in the engine are the
plane-wave norm-conserving UPF and the GPW GTH sets — those are frozen-core
approximations, not scalar-relativistic ECPs in the quantum-chemistry sense.

## 1. Atomic-Charge Methods

| Method                                 | CPU | CUDA |
| -------------------------------------- | :-: | :--: |
| Mulliken charges                       |  ✓  |   ✓  |
| Löwdin charges                         |  ○  |   ○  |
| Natural population analysis            |  ○  |   ○  |
| Hirshfeld charges                      |  ○  |   ○  |
| Iterative Hirshfeld charges            |  ○  |   ○  |
| CM5 charges                            |  ○  |   ○  |
| Bader charges                          |  ○  |   ○  |
| DDEC charges                           |  ○  |   ○  |
| Merz–Kollman ESP charges               |  ○  |   ○  |
| CHELPG charges                         |  ○  |   ○  |
| RESP charges                           |  ○  |   ○  |
| RESP2 charges                          |  ○  |   ○  |
| AM1-BCC charges                        |  ✓  |   ○  |
| Gasteiger charges                      |  ○  |   ○  |
| Electronegativity-equalization charges |  ○  |   ○  |
| QEq charges                            |  ○  |   ○  |
| EQeq charges                           |  ○  |   ○  |

Löwdin is ○ deliberately: the Löwdin **orthogonalization** (S^(−1/2)) is used
inside the SCF, but no Löwdin *population analysis* is exposed. Charges are
component Capabilities (`Capability::Charge`), never Drivers — there is no
"charge driver".

## 2. Implicit-Solvent Methods

| Method                             | CPU | CUDA |
| ---------------------------------- | :-: | :--: |
| Polarizable continuum model, PCM   |  ○  |   ○  |
| Integral-equation-formalism PCM    |  ○  |   ○  |
| Conductor-like PCM                 |  ○  |   ○  |
| COSMO                              |  ○  |   ○  |
| SMD                                |  ○  |   ○  |
| SM12                               |  ○  |   ○  |
| Generalized Born                   |  ○  |   ○  |
| GBSA                               |  ○  |   ○  |
| ALPB                               |  ○  |   ○  |
| Poisson–Boltzmann                  |  ○  |   ○  |
| PBSA                               |  ○  |   ○  |
| Nonequilibrium PCM                 |  ○  |   ○  |
| State-specific excited-state PCM   |  ○  |   ○  |
| Implicit-solvent analytic gradient |  ○  |   ○  |
| Implicit-solvent analytic Hessian  |  ○  |   ○  |

## 3. QM/MM Methods

| Method                           | CPU | CUDA |
| -------------------------------- | :-: | :--: |
| Mechanical embedding QM/MM       |  ○  |   ○  |
| Electrostatic embedding QM/MM    |  ○  |   ○  |
| Polarizable embedding QM/MM      |  ○  |   ○  |
| Additive QM/MM                   |  ○  |   ○  |
| Subtractive QM/MM                |  ○  |   ○  |
| ONIOM two-layer method           |  ○  |   ○  |
| ONIOM multilayer method          |  ○  |   ○  |
| Link-atom QM/MM                  |  ○  |   ○  |
| Localized-orbital boundary QM/MM |  ○  |   ○  |
| Pseudobond QM/MM                 |  ○  |   ○  |
| Periodic QM/MM                   |  ○  |   ○  |
| Long-range PME QM/MM             |  ○  |   ○  |
| Adaptive QM/MM                   |  ○  |   ○  |
| Multi-region QM/MM               |  ○  |   ○  |
| QM/MM geometry optimization      |  ○  |   ○  |
| QM/MM molecular dynamics         |  ○  |   ○  |
| Excited-state QM/MM              |  ○  |   ○  |

The Component model makes QM/MM *expressible* in principle — a Plan can bind a
QM evaluator and an MM pair style to the same force phase — but nothing performs
the region partitioning, boundary treatment, or embedding-charge coupling that
makes it QM/MM rather than a naive superposition. Treat every row here as
not-started.

## 4. Quantum Embedding Methods

| Method                            | CPU | CUDA |
| --------------------------------- | :-: | :--: |
| Fixed point-charge embedding      |  ○  |   ○  |
| Distributed-multipole embedding   |  ○  |   ○  |
| Polarizable embedding             |  ○  |   ○  |
| Effective fragment potential      |  ○  |   ○  |
| Frozen-density embedding          |  ○  |   ○  |
| Projection-based embedding        |  ○  |   ○  |
| Density-matrix embedding theory   |  ○  |   ○  |
| Subsystem DFT                     |  ○  |   ○  |
| Fragment molecular orbital method |  ○  |   ○  |

## 5. Relativistic Methods

| Method                                        | CPU | CUDA |
| --------------------------------------------- | :-: | :--: |
| Scalar-relativistic effective core potentials |  ○  |   ○  |
| Spin-orbit effective core potentials          |  ○  |   ○  |
| Douglas–Kroll–Hess second order               |  ○  |   ○  |
| Scalar ZORA                                   |  ○  |   ○  |
| Spin-orbit ZORA                               |  ○  |   ○  |
| Scalar X2C                                    |  ○  |   ○  |
| Spin-orbit X2C                                |  ○  |   ○  |
| Exact two-component molecular mean field      |  ○  |   ○  |
| Four-component Dirac–Hartree–Fock             |  ○  |   ○  |
| Four-component Dirac–Kohn–Sham                |  ○  |   ○  |
| Breit interaction                             |  ○  |   ○  |
| Spin-orbit mean-field approximation           |  ○  |   ○  |
| Relativistic analytic gradient                |  ○  |   ○  |

No ECP support means heavy elements are out of reach in the Gaussian path — an
all-electron Cartesian basis is the only option there. Periodic calculations
sidestep this through the UPF/GTH pseudopotentials.

## 6. Intermolecular Interaction Methods

### 6.1 Symmetry-adapted perturbation theory

| Method                 | CPU | CUDA |
| ---------------------- | :-: | :--: |
| SAPT0                  |  ○  |   ○  |
| Scaled SAPT0           |  ○  |   ○  |
| SAPT2                  |  ○  |   ○  |
| SAPT2+                 |  ○  |   ○  |
| SAPT2+(3)              |  ○  |   ○  |
| SAPT2+3                |  ○  |   ○  |
| SAPT(DFT)              |  ○  |   ○  |
| XSAPT                  |  ○  |   ○  |
| XSAPT+MBD              |  ○  |   ○  |
| SAPT analytic gradient |  ○  |   ○  |

### 6.2 Energy-decomposition methods

| Method                                   | CPU | CUDA |
| ---------------------------------------- | :-: | :--: |
| Morokuma energy decomposition            |  ○  |   ○  |
| ALMO energy-decomposition analysis       |  ○  |   ○  |
| Natural energy-decomposition analysis    |  ○  |   ○  |
| Fragment molecular orbital decomposition |  ○  |   ○  |
| Many-body expansion                      |  ○  |   ○  |
| Counterpoise correction                  |  ○  |   ○  |

## 7. Composite and Multiscale Methods

| Method                                       | CPU | CUDA |
| -------------------------------------------- | :-: | :--: |
| Delta-learning QM correction                 |  ○  |   ○  |
| ML/MM hybrid potential                       |  ○  |   ○  |
| ML/QM delta correction                       |  ○  |   ○  |
| Fragment-based quantum calculation           |  ○  |   ○  |
| Many-body expansion potential                |  ○  |   ○  |
| Adaptive-resolution simulation               |  ○  |   ○  |
| Concurrent atomistic–continuum coupling      |  ○  |   ○  |
| Quantum embedding molecular dynamics         |  ○  |   ○  |
| Multiple-time-scale mixed-potential dynamics |  ○  |   ○  |

Blocked upstream on two counts: there is no ML potential (the torch backend is a
differentiable *classical* force field, not a network — see
[Force Fields](force-fields.md#6-machine-learning-potentials)), and no QM/MM.
