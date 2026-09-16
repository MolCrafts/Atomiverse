# Hartree–Fock

Legend and caveats: [Roadmap index](index.md).

Hartree–Fock shares one solver with [Kohn–Sham DFT](dft.md): the SCF config is
the 2×2 product of {restricted, unrestricted} × {HF, KS}, so RHF, UHF, RKS and
UKS are the same code path with a different Fock build. Correlated methods built
on the HF reference live in [Post-Hartree–Fock](post-hf.md).

## 1. Molecular Hartree–Fock

| Method                                     | CPU | CUDA | Kokkos |
| ------------------------------------------ | :-: | :--: | :---: |
| Restricted Hartree–Fock, RHF               |  ✓  |   ✓  |   ○  |
| Unrestricted Hartree–Fock, UHF             |  ✓  |   ○  |   ○  |
| Restricted open-shell Hartree–Fock, ROHF   |  ✓  |   ○  |   ○  |
| Generalized Hartree–Fock, GHF              |  ✓  |   ○  |   ○  |
| Complex generalized Hartree–Fock           |  ○  |   ○  |   ○  |
| Spinor Hartree–Fock                        |  ○  |   ○  |   ○  |
| Finite-temperature Hartree–Fock            |  ○  |   ○  |   ○  |
| Constrained Hartree–Fock                   |  ○  |   ○  |   ○  |
| Broken-symmetry Hartree–Fock               |  ○  |   ○  |   ○  |
| Delta-SCF Hartree–Fock                     |  ✓  |   ○  |   ○  |
| Maximum-overlap excited-state Hartree–Fock |  ✓  |   ○  |   ○  |
| RHF analytic gradient                      |  ✓  |   ✓  |   ○  |
| UHF analytic gradient                      |  ✓  |   ○  |   ○  |
| ROHF analytic gradient                     |  ✓  |   ○  |   ○  |
| GHF analytic gradient                      |  ✓  |   ○  |   ○  |
| RHF analytic Hessian                       |  ✓  |   ○  |   ○  |
| UHF analytic Hessian                       |  ✓  |   ○  |   ○  |
| ROHF analytic Hessian                      |  ✓  |   ○  |   ○  |
| Hartree–Fock stability analysis            |  ✓  |   ○  |   ○  |

ROHF is the Roothaan single-effective-Fock method in the PySCF canonical
(Guest–Saunders) convention (`cpu::scf::ROHF`): energy + **analytic nuclear
forces** (spin-Fock energy-weighted density; O₂ vs PySCF) + **ROMP2**
(`ROHF::mp2`; `reg_romp2-o2-sto3g`) + nuclear Hessian via FD of forces
(`ROHF::hessian`; `reg_rohf-hessian-o2`). SOSCF and level shift still throw.
UHF remains the default open-shell workhorse; ROHF is the spin-pure
(⟨S²⟩ = S(S+1) exactly) alternative. GHF is energy + **analytic nuclear
gradient** (real two-component spinors; spinor energy-weighted density +
four-block exchange; `reg_h3_ghf_sto3g`, `reg_ghf-grad-h3-sto3g`);
mp2 / fci / soscf / level_shift still throw.

**Static polarizability** on RHF ships via CPHF (`RHF::polarizability`,
`reg_static-polariz-h2o` vs PySCF CPHF/FD). **RHF / UHF analytic nuclear
Hessians** ship via CPHF (`RHF::hessian` / `UHF::hessian`;
`reg_h2o_rhf_freq`, `reg_uhf-hessian-o2`). **ROHF nuclear Hessian** ships as
central FD of analytic ROHF forces (`ROHF::hessian`; `reg_rohf-hessian-o2`) —
PySCF has no `hessian.ROHF`; full three-space ROHF CPHF is a follow-up. KS
analytic Hessians and the generic FD vib path (`cpu::vib::Hessian`) remain
open rows — see [Molecular Dynamics](molecular-dynamics.md). Seeger–Pople
stability analysis over the orbital Hessian is implemented (internal channel
+ triplet RHF→UHF).

## 2. Periodic Hartree–Fock

| Method                | CPU | CUDA | Kokkos |
| --------------------- | :-: | :--: | :---: |
| Γ-point periodic RHF  |  ✓  |   ○  |   ○  |
| k-point periodic RHF  |  ○  |   ○  |   ○  |
| Γ-point periodic UHF  |  ○  |   ○  |   ○  |
| k-point periodic UHF  |  ○  |   ○  |   ○  |

Γ-point closed-shell periodic RHF is a truncated-lattice foothold
(`RHF.periodic = true`): J, K, S, T, V and E_nn share
`m_d = floor(rcut/L_d)`. A cell larger than the Gaussian pair reach has
only T = 0 and recovers molecular RHF (`regressions/scf/periodic-rhf-gamma`).
Bare 1/r image sums are not cutoff-stable (need Ewald); the unit gate uses
short-range erfc. k-point / UHF / Ewald are follow-ons. PW/GPW product
paths remain semi-local DFT — see [Periodic DFT](dft.md#10-periodic-kohnsham-methods).

## 3. SCF infrastructure

| Capability | CPU | CUDA | Kokkos |
| ---------- | :-: | :--: | :---: |
| DIIS | ✓ | ✓ (device CDIIS) |   ○  |
| Linear damping | ✓ | ○ |   ○  |
| Virtual-orbital level shift (Saunders–Hillier) | ✓ | ✓ |   ○  |
| SOSCF (second-order SCF) | ✓ | ○ |   ○  |
| Canonical orthogonalization + linear-dependence filtering | ✓ | ○ |   ○  |
| Dual energy + DIIS-error convergence test | ✓ | ○ |   ○  |
| Hcore initial guess | ✓ | ✓ |   ○  |
| MINAO initial guess | ✓ | ○ |   ○  |
| ASPC density predictor (AIMD warm start) | ✓ | ○ |   ○  |
| ⟨S²⟩ spin-contamination analysis | ✓ | ○ |   ○  |
| Spherical (pure) AO transform | ✓ | ○ |   ○  |
| RI-J / RI-JK density fitting | ✓ | ○ |   ○  |
| CFMM-J (continuous FMM for Coulomb) | ✓ | ○ |   ○  |
| Native Rys ERI engine (l ≤ 3, s/p/d/f) | ✓ | ○ |   ○  |
| Effective core potentials (ECP) | ✓ | ○ |   ○  |
| x2c-1e scalar relativity (energy) | ✓ | ○ |   ○  |
| Device ERI kernels | — | ✓ (s/p, l ≤ 1) |   ○  |
| Diagonalization | ✓ (LAPACK) | ✓ (cuSOLVER) |   ○  |

Basis sets: STO-3G, 6-31G, 6-31G\*, def2-SVP, def2-TZVP. Cartesian is the
default engine space; spherical (pure) is available for sets defined that way
(def2-SVP regression). Density fitting is RI-J / RI-JK over def2-universal
JKFIT (SCF only — not RI-MP2). CFMM-J is an alternate Coulomb algorithm
(`Coulomb::CFMM`). ECP forces are semi-numerical; x2c forces throw.

The MINAO guess table covers H, C, N, O, F only; other elements fall back to
Hcore.

**The CUDA column is a restricted-HF engine, not a general SCF.** The SCF loop
is device-resident: Fock and density builds, cuSOLVER diagonalization, device
CDIIS (Pulay) Fock mixing (history on device; only the tiny (m+1) Pulay solve
runs on host), and the Saunders–Hillier level shift on the device Fock path.
Device-native integrals **and analytic nuclear forces** cover s/p shells
(l ≤ 1, STO-3G class — mixed Z up to O). Energy on other bases still
host-fills S/Hcore/ERI and uploads (SCF loop on device). Requesting
forces outside that envelope **throws** — use `cpu::scf::RHF`. GPU BOMD
smoke: `reg_h2_rhf_sto3g_aimd` (GH200 **1332540**). Still missing on
CUDA: UHF and any Gaussian-basis Kohn–Sham path.
