# Hartree–Fock

Legend and caveats: [Roadmap index](index.md).

Hartree–Fock shares one solver with [Kohn–Sham DFT](dft.md): the SCF config is
the 2×2 product of {restricted, unrestricted} × {HF, KS}, so RHF, UHF, RKS and
UKS are the same code path with a different Fock build. Correlated methods built
on the HF reference live in [Post-Hartree–Fock](post-hf.md).

## 1. Molecular Hartree–Fock

| Method                                     | CPU | CUDA |
| ------------------------------------------ | :-: | :--: |
| Restricted Hartree–Fock, RHF               |  ✓  |   ✓  |
| Unrestricted Hartree–Fock, UHF             |  ✓  |   ○  |
| Restricted open-shell Hartree–Fock, ROHF   |  ✓  |   ○  |
| Generalized Hartree–Fock, GHF              |  ○  |   ○  |
| Complex generalized Hartree–Fock           |  ○  |   ○  |
| Spinor Hartree–Fock                        |  ○  |   ○  |
| Finite-temperature Hartree–Fock            |  ○  |   ○  |
| Constrained Hartree–Fock                   |  ○  |   ○  |
| Broken-symmetry Hartree–Fock               |  ○  |   ○  |
| Delta-SCF Hartree–Fock                     |  ○  |   ○  |
| Maximum-overlap excited-state Hartree–Fock |  ○  |   ○  |
| RHF analytic gradient                      |  ✓  |   ○  |
| UHF analytic gradient                      |  ✓  |   ○  |
| ROHF analytic gradient                     |  ○  |   ○  |
| GHF analytic gradient                      |  ○  |   ○  |
| RHF analytic Hessian                       |  ○  |   ○  |
| UHF analytic Hessian                       |  ○  |   ○  |
| ROHF analytic Hessian                      |  ○  |   ○  |
| Hartree–Fock stability analysis            |  ○  |   ○  |

ROHF is the Roothaan single-effective-Fock method in the PySCF canonical
(Guest–Saunders) convention (`cpu::scf::ROHF`): energy only — the analytic
gradient (coupling-term Lagrangian), ROHF-MP2, SOSCF and level shift throw.
UHF remains the default open-shell workhorse; ROHF adds the spin-pure
(⟨S²⟩ = S(S+1) exactly) alternative.

Second derivatives are finite-difference everywhere (see the vibrational driver
in [Molecular Dynamics](molecular-dynamics.md)); no CPHF is implemented, so
every analytic-Hessian row in this roadmap is ○.

## 2. Periodic Hartree–Fock

| Method                | CPU | CUDA |
| --------------------- | :-: | :--: |
| Γ-point periodic RHF  |  ○  |   ○  |
| k-point periodic RHF  |  ○  |   ○  |
| Γ-point periodic UHF  |  ○  |   ○  |
| k-point periodic UHF  |  ○  |   ○  |

Exact exchange under periodic boundary conditions is not implemented in any
engine. The periodic solvers (plane-wave, GPW) are pure/semi-local DFT only —
see [Periodic DFT](dft.md#10-periodic-kohnsham-methods). The Γ-point lattice sum
for one-electron integrals (overlap, kinetic, pseudopotential) *does* exist in
GPW and is validated to machine precision, so the infrastructure a periodic HF
would need is partly in place; the four-index periodic exchange is not.

## 3. SCF infrastructure

| Capability | CPU | CUDA |
| ---------- | :-: | :--: |
| DIIS | ✓ | ○ |
| Linear damping | ✓ | ○ |
| Virtual-orbital level shift (Saunders–Hillier) | ✓ | ✓ |
| Canonical orthogonalization + linear-dependence filtering | ✓ | ○ |
| Dual energy + DIIS-error convergence test | ✓ | ○ |
| Hcore initial guess | ✓ | ✓ |
| MINAO initial guess | ✓ | ○ |
| ASPC density predictor (AIMD warm start) | ✓ | ○ |
| ⟨S²⟩ spin-contamination analysis | ✓ | ○ |
| Native Rys ERI engine (l ≤ 3, s/p/d/f) | ✓ | ○ |
| Device ERI kernels | — | ○ (s-type only) |
| Diagonalization | ✓ (LAPACK) | ✓ (cuSOLVER) |

Basis sets: STO-3G, 6-31G, 6-31G\*, def2-SVP, def2-TZVP — **Cartesian only**, no
spherical-harmonic transform. Density fitting / RI is not implemented.

The MINAO guess table covers H, C, N, O, F only; other elements silently fall
back to Hcore.

**The CUDA column is a restricted-HF energy engine, not a general SCF.** Fock
build, density build, and diagonalization run on device, but there is no DIIS,
no UHF, no Kohn–Sham path, and **no gradient kernel** — hence no GPU AIMD. Its
device-native integrals are s-type only; higher shells require a host-built
integral pack.
