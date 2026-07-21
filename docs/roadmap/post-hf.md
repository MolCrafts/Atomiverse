# Post-Hartree–Fock Correlation

Legend and caveats: [Roadmap index](index.md).

Correlated methods built on a converged [Hartree–Fock](hartree-fock.md)
reference. Two exist, both as post-SCF members on the concrete method
(`RHF::mp2(frame)` / `RHF::fci(frame)`) consuming the converged
`SlaterDeterminant` plus the method's privately materialized ERI tensor (they
are not Drivers or Components, and register in no Plan; both throw
`std::invalid_argument` on a non-converged reference):

- **RMP2** — closed-shell canonical MP2, the production correlation method.
- **FCI** — a determinant-basis exactness oracle for tiny systems, not a
  production method.

Coupled cluster and multireference theory are entirely absent.

## 1. Møller–Plesset Perturbation Theory

| Method                             | CPU | CUDA |
| ---------------------------------- | :-: | :--: |
| Restricted MP2, RMP2               |  ✓  |   ○  |
| Unrestricted MP2, UMP2             |  ○  |   ○  |
| Restricted open-shell MP2, ROMP2   |  ○  |   ○  |
| Frozen-core MP2                    |  ○  |   ○  |
| Density-fitted MP2                 |  ○  |   ○  |
| RI-MP2                             |  ○  |   ○  |
| Laplace-transformed MP2            |  ○  |   ○  |
| Spin-component-scaled MP2, SCS-MP2 |  ○  |   ○  |
| Scaled-opposite-spin MP2, SOS-MP2  |  ○  |   ○  |
| SCS-MI-MP2                         |  ○  |   ○  |
| Orbital-optimized MP2, OOMP2       |  ○  |   ○  |
| κ-OOMP2                            |  ○  |   ○  |
| σ²-OOMP2                           |  ○  |   ○  |
| Local MP2                          |  ○  |   ○  |
| Pair-natural-orbital MP2           |  ○  |   ○  |
| DLPNO-MP2                          |  ○  |   ○  |
| MP2-F12                            |  ○  |   ○  |
| MP2 analytic gradient              |  ○  |   ○  |
| MP2 analytic Hessian               |  ○  |   ○  |
| MP2.5                              |  ○  |   ○  |
| MP3                                |  ○  |   ○  |
| SCS-MP3                            |  ○  |   ○  |
| Orbital-optimized MP3              |  ○  |   ○  |
| MP4(SDTQ)                          |  ○  |   ○  |
| Full MP4                           |  ○  |   ○  |

RMP2 runs four sequential O(N⁵) quarter transforms to the `(ia|jb)` ovov block,
then the canonical energy expression. Energy only: no frozen core, no density
fitting/RI, no gradients. It requires the full in-memory ERI tensor (which
`mp2()` materializes in the method's private workspace), and that is what
bounds its system size.

## 2. Configuration Interaction

| Method                                 | CPU | CUDA |
| -------------------------------------- | :-: | :--: |
| Configuration interaction singles, CIS |  ○  |   ○  |
| Full configuration interaction, FCI    |  ✓  |   ○  |
| Direct CI                              |  ○  |   ○  |
| Determinant-driven CI                  |  ✓  |   ○  |
| Integral-driven CI                     |  ○  |   ○  |
| Selected CI                            |  ○  |   ○  |
| CIPSI                                  |  ○  |   ○  |
| Heat-bath CI                           |  ○  |   ○  |
| Semistochastic heat-bath CI            |  ○  |   ○  |
| Adaptive sampling CI                   |  ○  |   ○  |
| CIS analytic gradient                  |  ○  |   ○  |

FCI is determinant-driven (Slater–Condon assembly, Jordan–Wigner phases) with a
**dense** CI matrix and `eigh` diagonalization — so it is explicitly *not* a
direct CI. It throws past a hard cap (`n_mo > 14` or `N_det > 2000`). Its purpose
is to pin the exact answer for H₂/STO-3G-scale systems in validation, not to
correlate real ones.

## 3. Coupled Cluster

| Method                       | CPU | CUDA |
| ---------------------------- | :-: | :--: |
| Coupled-cluster doubles, CCD |  ○  |   ○  |
| Brueckner doubles, BD        |  ○  |   ○  |
| CC2                          |  ○  |   ○  |
| CCSD                         |  ○  |   ○  |
| CCSD(T)                      |  ○  |   ○  |
| CCSDT-1a                     |  ○  |   ○  |
| CCSDT-1b                     |  ○  |   ○  |
| CCSDT-2                      |  ○  |   ○  |
| CCSDT-3                      |  ○  |   ○  |
| CCSDT                        |  ○  |   ○  |
| CCSDT(Q)                     |  ○  |   ○  |
| CCSDTQ                       |  ○  |   ○  |
| CEPA/0                       |  ○  |   ○  |
| ACPF                         |  ○  |   ○  |
| AQCC                         |  ○  |   ○  |
| Orbital-optimized CCD        |  ○  |   ○  |
| Density-fitted CCSD          |  ○  |   ○  |
| Cholesky-decomposed CCSD     |  ○  |   ○  |
| CCSD-F12a                    |  ○  |   ○  |
| CCSD-F12b                    |  ○  |   ○  |
| CCSD(T)-F12                  |  ○  |   ○  |
| CCSD analytic gradient       |  ○  |   ○  |
| CCSD analytic Hessian        |  ○  |   ○  |
| CCSD(T) analytic gradient    |  ○  |   ○  |

### Local-correlation coupled cluster

| Method                         | CPU | CUDA |
| ------------------------------ | :-: | :--: |
| Local CCSD                     |  ○  |   ○  |
| Pair-natural-orbital CCSD      |  ○  |   ○  |
| DLPNO-CCSD                     |  ○  |   ○  |
| DLPNO-CCSD(T0)                 |  ○  |   ○  |
| DLPNO-CCSD(T)                  |  ○  |   ○  |
| LNO-CCSD                       |  ○  |   ○  |
| LNO-CCSD(T)                    |  ○  |   ○  |
| Frozen-natural-orbital CCSD    |  ○  |   ○  |
| Frozen-natural-orbital CCSD(T) |  ○  |   ○  |

## 4. Multireference Methods

### 4.1 Active-space methods

| Method                               | CPU | CUDA |
| ------------------------------------ | :-: | :--: |
| Complete-active-space CI, CASCI      |  ○  |   ○  |
| Complete-active-space SCF, CASSCF    |  ○  |   ○  |
| State-specific CASSCF                |  ○  |   ○  |
| State-averaged CASSCF                |  ○  |   ○  |
| Restricted-active-space CI, RASCI    |  ○  |   ○  |
| Restricted-active-space SCF, RASSCF  |  ○  |   ○  |
| Generalized-active-space CI, GASCI   |  ○  |   ○  |
| Generalized-active-space SCF, GASSCF |  ○  |   ○  |
| DMRG-CI                              |  ○  |   ○  |
| DMRG-SCF                             |  ○  |   ○  |
| Selected-CI active-space solver      |  ○  |   ○  |
| FCIQMC active-space solver           |  ○  |   ○  |
| CASCI analytic gradient              |  ○  |   ○  |
| CASSCF analytic gradient             |  ○  |   ○  |
| CASSCF analytic Hessian              |  ○  |   ○  |

The FCI kernel is the natural seed for a CASCI active-space solver, but no
active-space selection or orbital optimization exists on top of it.

### 4.2 Multireference correlation

| Method                                            | CPU | CUDA |
| ------------------------------------------------- | :-: | :--: |
| Multireference CISD, MRCISD                       |  ○  |   ○  |
| Internally contracted MRCI                        |  ○  |   ○  |
| Uncontracted MRCI                                 |  ○  |   ○  |
| MRCI+Q                                            |  ○  |   ○  |
| CASPT2                                            |  ○  |   ○  |
| Multistate CASPT2                                 |  ○  |   ○  |
| Extended multistate CASPT2                        |  ○  |   ○  |
| Strongly contracted NEVPT2                        |  ○  |   ○  |
| Partially contracted NEVPT2                       |  ○  |   ○  |
| Quasidegenerate NEVPT2                            |  ○  |   ○  |
| MRMP2                                             |  ○  |   ○  |
| Multiconfiguration pair-density functional theory |  ○  |   ○  |
| Multireference analytic gradient                  |  ○  |   ○  |
