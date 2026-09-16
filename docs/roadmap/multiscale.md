# Multiscale Methods and Properties

Legend and caveats: [Roadmap index](index.md).

Everything that couples two levels of theory, or extracts a chemical property
from a converged wavefunction. Charge schemes still anchor the product surface
(Mulliken observer; AM1-BCC via [AM1](semiempirical.md) + molrs typing). Beyond
that, several *environment* hooks now exist on the SCF mean-field methods:

- **C-PCM / COSMO / IEF-PCM** electrostatic free energy in the Fock build
  (`reg_h2o_rhf_631gs_cpcm`, `reg_ief-pcm-h2o`) and C-PCM free-energy forces
  (`reg_pcm-grad-h2o`) — non-electrostatic terms absent.
- **Fixed point-charge embedding** + **uniform external electric field**, with
  analytic forces (`reg_embed_h2o_grad`, `reg_field_h2o_grad`).
- **ECP** (semi-numerical nuclear forces) and **x2c-1e** (energy only; forces
  throw).

**Textbook QM/MM** is a connection layer over any embedding-capable `QmEngine`
(RHF/RKS/… for electrostatic/polarizable; any `QmEngine` for mechanical).
Compose under one outer Driver like AIMD: `md.set_evaluator(MM…)` +
`md.set_evaluator(Qmmm*.set_engine(QM…))`. The QM region is the per-atom
**`kAtomQmZ` mask** (0 = MM, >0 = QM with Z). **Link cuts** are derived from
mask × existing bond topology (no bonds ⇒ zero links; no `set_links`). MM–MM
LJ/Coulomb belong to ordinary pair styles; couplings own QM–MM cross terms and
(electrostatic) third-law site forces. NVE MD conservation
(`reg_qmmm_md_conservation`) ships on CPU. Still no ONIOM / subtractive product
path, no SMD/GB, and no picture-change / DKH / SOC stack.

## 1. Atomic-Charge Methods

| Method                                 | CPU | CUDA | Kokkos |
| -------------------------------------- | :-: | :--: | :---: |
| Mulliken charges                       |  ✓  |   ✓  |   ○  |
| Löwdin charges                         |  ✓  |   ○  |   ○  |
| Natural population analysis            |  ○  |   ○  |   ○  |
| Hirshfeld charges                      |  ○  |   ○  |   ○  |
| Iterative Hirshfeld charges            |  ○  |   ○  |   ○  |
| CM5 charges                            |  ○  |   ○  |   ○  |
| Bader charges                          |  ○  |   ○  |   ○  |
| DDEC charges                           |  ○  |   ○  |   ○  |
| Merz–Kollman ESP charges               |  ✓  |   ○  |   ○  |
| CHELPG charges                         |  ✓  |   ○  |   ○  |
| RESP charges                           |  ○  |   ○  |   ○  |
| RESP2 charges                          |  ○  |   ○  |   ○  |
| AM1-BCC charges                        |  ✓  |   ○  |   ○  |
| Gasteiger charges                      |  ○  |   ○  |   ○  |
| Electronegativity-equalization charges |  ○  |   ○  |   ○  |
| QEq charges                            |  ○  |   ○  |   ○  |
| EQeq charges                           |  ○  |   ○  |   ○  |

Löwdin population analysis (`lowdin_charge` / `lowdin_spin`,
`cpu/observe/lowdin.h`) and grid-fitted ESP charges — Merz–Kollman shells and
CHELPG cube grid (`MKCharges` / `ChelpgCharges`, `cpu/observe/esp_charges.h`)
— are gated by `regressions/scf/h2o_charges_pyscf` (PySCF 2.14.0; Löwdin
≤ 1e-6 e, MK/CHELPG ≤ 1e-3 e). RESP is the missing next rung on the ESP
ladder (restraints on top of the existing MK grid). Charges are Observer /
method outputs written to Frame, never Drivers — there is no "charge
driver".

## 2. Implicit-Solvent Methods

| Method                             | CPU | CUDA | Kokkos |
| ---------------------------------- | :-: | :--: | :---: |
| Polarizable continuum model, PCM   |  ✓  |   ○  |   ○  |
| Integral-equation-formalism PCM    |  ✓  |   ○  |   ○  |
| Conductor-like PCM                 |  ✓  |   ○  |   ○  |
| COSMO                              |  ✓  |   ○  |   ○  |
| SMD                                |  ○  |   ○  |   ○  |
| SM12                               |  ○  |   ○  |   ○  |
| Generalized Born                   |  ○  |   ○  |   ○  |
| GBSA                               |  ○  |   ○  |   ○  |
| ALPB                               |  ○  |   ○  |   ○  |
| Poisson–Boltzmann                  |  ○  |   ○  |   ○  |
| PBSA                               |  ○  |   ○  |   ○  |
| Nonequilibrium PCM                 |  ○  |   ○  |   ○  |
| State-specific excited-state PCM   |  ○  |   ○  |   ○  |
| Implicit-solvent analytic gradient |  ✓  |   ○  |   ○  |
| Implicit-solvent analytic Hessian  |  ○  |   ○  |   ○  |

C-PCM / COSMO / IEF-PCM share one cavity implementation (`internal::Pcm`;
`SolventModel::{CPCM,COSMO,IEFPCM}`). Conductor models use \(S q=-f V\)
(x=0 vs x=0.5); IEF-PCM builds the \(K q=R V\) system with the double-layer
operator \(D\) (Cancès–Mennucci–Tomasi; PySCF `method='IEF-PCM'`). Energy
regressions: `reg_h2o_rhf_631gs_cpcm`, `reg_ief-pcm-h2o`. Free-energy forces:
`reg_pcm-grad-h2o` (vacuum HF terms on the solvent density + fixed-P FD of
ΔG_el). Surface-switching / SWIG smoothing remains out of scope (hard
bury-delete cavity).

## 3. QM/MM Methods

| Method                           | CPU | CUDA | Kokkos |
| -------------------------------- | :-: | :--: | :---: |
| Mechanical embedding QM/MM       |  ✓  |   ○  |   ○  |
| Electrostatic embedding QM/MM    |  ✓  |   ○  |   ○  |
| Polarizable embedding QM/MM      |  ✓  |   ○  |   ○  |
| Additive QM/MM                   |  ✓  |   ○  |   ○  |
| Subtractive QM/MM                |  ✓  |   ○  |   ○  |
| ONIOM two-layer method           |  ✓  |   ○  |   ○  |
| ONIOM multilayer method          |  ○  |   ○  |   ○  |
| Link-atom QM/MM                  |  ✓  |   ○  |   ○  |
| Localized-orbital boundary QM/MM |  ○  |   ○  |   ○  |
| Pseudobond QM/MM                 |  ○  |   ○  |   ○  |
| Periodic QM/MM                   |  ○  |   ○  |   ○  |
| Long-range PME QM/MM             |  ○  |   ○  |   ○  |
| Adaptive QM/MM                   |  ○  |   ○  |   ○  |
| Multi-region QM/MM               |  ○  |   ○  |   ○  |
| QM/MM geometry optimization      |  ✓  |   ○  |   ○  |
| QM/MM molecular dynamics         |  ✓  |   ○  |   ○  |
| Excited-state QM/MM              |  ○  |   ○  |   ○  |

Fixed point-charge embedding on the SCF method (external MM charges in the
one-electron Hamiltonian, with analytic forces) remains a lower-level hook.
The product QM/MM components are:

- `QmmmMechanical` — any `QmEngine` + classical QM–MM LJ/Coulomb (`reg_qmmm_h2_ar`,
  `reg_qmmm_opt_h2_ar` Minimize composition)
- `QmmmElectrostatic` — embedding engine + mask region + QM–MM LJ + third-law
  site forces (`reg_qmmm_es_h2o_pyscf`, `reg_qmmm_link_ch3oh`,
  `reg_qmmm_md_conservation`)
- `QmmmPolarizable` — embedding engine + density-derived permanent field +
  isotropic induced MM dipoles (`reg_qmmm_pol_h2o`)
- `QmmmOniom` — two-layer subtractive ONIOM
  \(E_{\mathrm{high}}(\mathrm{model})+E_{\mathrm{low}}(\mathrm{real})
  -E_{\mathrm{low}}(\mathrm{model})\) (`reg_qmmm_oniom_h2_ar`); low = classical
  nonbonded; high = vacuum `QmEngine`

No automatic region partitioner, no 3-layer ONIOM, no PME long-range QM/MM.
MC posture: `reset_history_each_launch` cold-starts the engine each trial.
Geometry optimization is the same composition as MD: `Minimize` +
`set_evaluator(Qmmm*)` (not a `QmmmOpt` driver).

## 4. Quantum Embedding Methods

| Method                            | CPU | CUDA | Kokkos |
| --------------------------------- | :-: | :--: | :---: |
| Fixed point-charge embedding      |  ✓  |   ○  |   ○  |
| Distributed-multipole embedding   |  ○  |   ○  |   ○  |
| Polarizable embedding             |  ○  |   ○  |   ○  |
| Effective fragment potential      |  ○  |   ○  |   ○  |
| Frozen-density embedding          |  ○  |   ○  |   ○  |
| Projection-based embedding        |  ○  |   ○  |   ○  |
| Density-matrix embedding theory   |  ○  |   ○  |   ○  |
| Subsystem DFT                     |  ○  |   ○  |   ○  |
| Fragment molecular orbital method |  ○  |   ○  |   ○  |

## 5. Relativistic Methods

| Method                                        | CPU | CUDA | Kokkos |
| --------------------------------------------- | :-: | :--: | :---: |
| Scalar-relativistic effective core potentials |  ✓  |   ○  |   ○  |
| Spin-orbit effective core potentials          |  ○  |   ○  |   ○  |
| Douglas–Kroll–Hess second order               |  ○  |   ○  |   ○  |
| Scalar ZORA                                   |  ○  |   ○  |   ○  |
| Spin-orbit ZORA                               |  ○  |   ○  |   ○  |
| Scalar X2C                                    |  ✓  |   ○  |   ○  |
| Spin-orbit X2C                                |  ○  |   ○  |   ○  |
| Exact two-component molecular mean field      |  ○  |   ○  |   ○  |
| Four-component Dirac–Hartree–Fock             |  ○  |   ○  |   ○  |
| Four-component Dirac–Kohn–Sham                |  ○  |   ○  |   ○  |
| Breit interaction                             |  ○  |   ○  |   ○  |
| Spin-orbit mean-field approximation           |  ○  |   ○  |   ○  |
| Relativistic analytic gradient                |  ○  |   ○  |   ○  |

Gaussian ECPs are available with semi-numerical forces (`reg_ecp_i2_grad`,
`reg_ecp_heavy_atom`). x2c-1e replaces the one-electron Hamiltonian (energy
path validated on Ar); analytic forces with `x2c=true` throw. Stacking x2c
on an ECP atom is refused. Periodic calculations still use UPF/GTH
pseudopotentials, separate from the Gaussian ECP tables.

## 6. Intermolecular Interaction Methods

### 6.1 Symmetry-adapted perturbation theory

| Method                 | CPU | CUDA | Kokkos |
| ---------------------- | :-: | :--: | :---: |
| SAPT0                  |  ○  |   ○  |   ○  |
| Scaled SAPT0           |  ○  |   ○  |   ○  |
| SAPT2                  |  ○  |   ○  |   ○  |
| SAPT2+                 |  ○  |   ○  |   ○  |
| SAPT2+(3)              |  ○  |   ○  |   ○  |
| SAPT2+3                |  ○  |   ○  |   ○  |
| SAPT(DFT)              |  ○  |   ○  |   ○  |
| XSAPT                  |  ○  |   ○  |   ○  |
| XSAPT+MBD              |  ○  |   ○  |   ○  |
| SAPT analytic gradient |  ○  |   ○  |   ○  |

### 6.2 Energy-decomposition methods

| Method                                   | CPU | CUDA | Kokkos |
| ---------------------------------------- | :-: | :--: | :---: |
| Morokuma energy decomposition            |  ○  |   ○  |   ○  |
| ALMO energy-decomposition analysis       |  ○  |   ○  |   ○  |
| Natural energy-decomposition analysis    |  ○  |   ○  |   ○  |
| Fragment molecular orbital decomposition |  ○  |   ○  |   ○  |
| Many-body expansion                      |  ○  |   ○  |   ○  |
| Counterpoise correction                  |  ✓  |   ○  |   ○  |

## 7. Composite and Multiscale Methods

| Method                                       | CPU | CUDA | Kokkos |
| -------------------------------------------- | :-: | :--: | :---: |
| Delta-learning QM correction                 |  ○  |   ○  |   ○  |
| ML/MM hybrid potential                       |  ○  |   ○  |   ○  |
| ML/QM delta correction                       |  ○  |   ○  |   ○  |
| Fragment-based quantum calculation           |  ○  |   ○  |   ○  |
| Many-body expansion potential                |  ○  |   ○  |   ○  |
| Adaptive-resolution simulation               |  ○  |   ○  |   ○  |
| Concurrent atomistic–continuum coupling      |  ○  |   ○  |   ○  |
| Quantum embedding molecular dynamics         |  ○  |   ○  |   ○  |
| Multiple-time-scale mixed-potential dynamics |  ○  |   ○  |   ○  |

Blocked upstream where product ML potentials are still ○ (see
[Force Fields](force-fields.md#6-machine-learning-potentials)). Textbook
**additive** mechanical / electrostatic / polarizable QM/MM + **two-layer
ONIOM/subtractive** + link atoms + MD + geometry optimization ship on CPU
(esmm-qmmm-01…07). Open product gaps: 3-layer ONIOM, periodic / PME QM/MM,
adaptive multi-region, excited-state, electronic-embedding ONIOM.
