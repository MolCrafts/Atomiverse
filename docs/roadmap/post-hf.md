# Post-Hartree–Fock Correlation

Legend and caveats: [Roadmap index](index.md).

Correlated methods built on a converged [Hartree–Fock](hartree-fock.md)
reference. MP2 / MP3 / MP4 / CC2 / CCSD / CCSD(T) / CASCI / FCI ship as
post-SCF members on the concrete method (`RHF::mp2` / `UHF::mp2` /
`ROHF::mp2` / `RHF::mp3` / `RHF::mp4` / `RHF::cc2` / `RHF::ccsd` /
`RHF::ccsd_t` / `RHF::casci` / `RHF::fci`) consuming the converged
`SlaterDeterminant` plus the method's privately materialized ERI tensor
(they are not Drivers or Components, and register in no Plan; they throw
`std::invalid_argument` on a non-converged reference):

- **RMP2** — closed-shell canonical MP2, the production correlation method.
- **UMP2** — unrestricted spin-orbital MP2 on a UHF reference (`UHF::mp2`).
- **ROMP2** — UMP2 on a spin-pure ROHF reference (`ROHF::mp2`).
- **RMP3** — closed-shell third-order MP (`RHF::mp3`; `reg_mp3-h2-sto3g`).
- **RMP4(SDTQ)** — closed-shell fourth-order MP (`RHF::mp4`;
  `reg_mp4-h2-sto3g`).
- **CC2** — approximate CC doubles foothold (`RHF::cc2`; `reg_cc2-h2-sto3g`).
- **CCSD** — closed-shell coupled-cluster singles and doubles (`RHF::ccsd`;
  `reg_ccsd-h2-sto3g`).
- **CCSD(T)** — non-iterative triples on closed-shell CCSD (`RHF::ccsd_t`;
  `reg_ccsd-t-h2o-sto3g`).
- **CASCI** — complete-active-space CI on the FCI kernel (`RHF::casci`;
  `reg_casci-h2-sto3g`).
- **CASSCF** — orbital-optimized CAS on the same FCI active-space kernel
  (`RHF::casscf`; `reg_casscf-h2-sto3g`).
- **FCI** — a determinant-basis exactness oracle for tiny systems, not a
  production method.

**SC-NEVPT2** ships (`RHF::nevpt2`); PC/QD-NEVPT2, CASPT2, and full
density-based CC analytic gradients remain open. CCSD nuclear forces ship
as FD of the CCSD energy (`RHF::ccsd_forces`).

## 1. Møller–Plesset Perturbation Theory

| Method                             | CPU | CUDA | Kokkos |
| ---------------------------------- | :-: | :--: | :---: |
| Restricted MP2, RMP2               |  ✓  |   ○  |   ○  |
| Unrestricted MP2, UMP2             |  ✓  |   ○  |   ○  |
| Restricted open-shell MP2, ROMP2   |  ✓  |   ○  |   ○  |
| Frozen-core MP2                    |  ✓  |   ○  |   ○  |
| Density-fitted MP2                 |  ✓  |   ○  |   ○  |
| RI-MP2                             |  ✓  |   ○  |   ○  |
| Laplace-transformed MP2            |  ○  |   ○  |   ○  |
| Spin-component-scaled MP2, SCS-MP2 |  ✓  |   ○  |   ○  |
| Scaled-opposite-spin MP2, SOS-MP2  |  ✓  |   ○  |   ○  |
| SCS-MI-MP2                         |  ○  |   ○  |   ○  |
| Orbital-optimized MP2, OOMP2       |  ○  |   ○  |   ○  |
| κ-OOMP2                            |  ○  |   ○  |   ○  |
| σ²-OOMP2                           |  ○  |   ○  |   ○  |
| Local MP2                          |  ○  |   ○  |   ○  |
| Pair-natural-orbital MP2           |  ○  |   ○  |   ○  |
| DLPNO-MP2                          |  ○  |   ○  |   ○  |
| MP2-F12                            |  ○  |   ○  |   ○  |
| MP2 analytic gradient              |  ✓  |   ○  |   ○  |
| MP2 analytic Hessian               |  ○  |   ○  |   ○  |
| MP2.5                              |  ○  |   ○  |   ○  |
| MP3                                |  ✓  |   ○  |   ○  |
| SCS-MP3                            |  ○  |   ○  |   ○  |
| Orbital-optimized MP3              |  ○  |   ○  |   ○  |
| MP4(SDTQ)                          |  ✓  |   ○  |   ○  |
| Full MP4                           |  ○  |   ○  |   ○  |

RMP2 runs four sequential O(N⁵) quarter transforms to the `(ia|jb)` ovov block,
then the canonical energy expression. **UMP2** (`UHF::mp2`) uses independent
α/β MOs with same-spin (¼ antisymmetrized) and opposite-spin channels
(`reg_ump2-o2-sto3g` vs PySCF `mp.UMP2`). **RI-MP2** is the closed-shell entry
with `rhf.ri = RI::{J,JK}` and an aux basis (`MP2Result::ri = true`,
`reg_h2o_ri_mp2`) — Coulomb-metric density fitting for `(ia|jb)`. **Frozen-core**
is the member `rhf.frozen = N` (lowest N occupied spatial MOs excluded from the
doubles sum; PySCF `mp.MP2(mf, frozen=N)` convention) on closed-shell RMP2 and
RI-MP2 (`reg_frozen-core-mp2-h2o`, `MP2Result::frozen`). **SCS-MP2** is the same
RMP2 path with Grimme scales on the opposite-/same-spin channels via members
`mp2_os_scale` / `mp2_ss_scale` (defaults 1.0; SCS: 1.2 / ⅓;
`reg_scs-mp2-h2o-sto3g`; SOS: 1.3 / 0, Jung et al. JCP 121, 9793 (2004),
`reg_sos-mp2-h2o-sto3g`). **RMP2 analytic nuclear gradient** ships
(`RHF::mp2_forces`; Lagrangian 1-RDM + non-separable 2-RDM + Z-vector;
`reg_mp2-grad-h2o-sto3g` vs PySCF `mp.MP2.nuc_grad_method`). **UMP2 analytic
nuclear gradient** ships on UHF (`UHF::mp2_forces`; spin-resolved Lagrangian
1-RDM + non-separable 2-RDM + unrestricted Z-vector;
`reg_ump2-grad-o2-sto3g` vs PySCF `mp.UMP2.nuc_grad_method`). **ROMP2**
(`ROHF::mp2`) is UMP2 on the spin-pure ROHF reference with semi-canonical
spin-Fock denominators (PySCF `mp.UMP2(ROHF)` convention;
`reg_romp2-o2-sto3g`). **RMP3** (`RHF::mp3`) adds the third-order doubles
diagrams on the same closed-shell MO ERIs (`reg_mp3-h2-sto3g`; offline pin on
PySCF RHF orbitals — PySCF 2.x has no `mp.MP3`). **RMP4(SDTQ)** (`RHF::mp4`)
adds fourth-order RSPT through singles/doubles/triples/quadruples on the same
closed-shell path (`reg_mp4-h2-sto3g`; offline det-space RSPT-4 twin — PySCF
2.x has no `mp.MP4`). No ROMP2 gradients, no frozen-core UMP2, no RI-UMP2.
Canonical mode still materializes the full in-memory ERI tensor in the
method's private workspace.

## 2. Configuration Interaction

| Method                                 | CPU | CUDA | Kokkos |
| -------------------------------------- | :-: | :--: | :---: |
| Configuration interaction singles, CIS |  ○  |   ○  |   ○  |
| Full configuration interaction, FCI    |  ✓  |   ○  |   ○  |
| Direct CI                              |  ○  |   ○  |   ○  |
| Determinant-driven CI                  |  ✓  |   ○  |   ○  |
| Integral-driven CI                     |  ○  |   ○  |   ○  |
| Selected CI                            |  ○  |   ○  |   ○  |
| CIPSI                                  |  ○  |   ○  |   ○  |
| Heat-bath CI                           |  ○  |   ○  |   ○  |
| Semistochastic heat-bath CI            |  ○  |   ○  |   ○  |
| Adaptive sampling CI                   |  ○  |   ○  |   ○  |
| CIS analytic gradient                  |  ○  |   ○  |   ○  |

FCI is determinant-driven (Slater–Condon assembly, Jordan–Wigner phases) with a
**dense** CI matrix and `eigh` diagonalization — so it is explicitly *not* a
direct CI. It throws past a hard cap (`n_mo > 14` or `N_det > 2000`). Its purpose
is to pin the exact answer for H₂/STO-3G-scale systems in validation, not to
correlate real ones.

## 3. Coupled Cluster

| Method                       | CPU | CUDA | Kokkos |
| ---------------------------- | :-: | :--: | :---: |
| Coupled-cluster doubles, CCD |  ○  |   ○  |   ○  |
| Brueckner doubles, BD        |  ○  |   ○  |   ○  |
| CC2                          |  ✓  |   ○  |   ○  |
| CCSD                         |  ✓  |   ○  |   ○  |
| CCSD(T)                      |  ✓  |   ○  |   ○  |
| CCSDT-1a                     |  ○  |   ○  |   ○  |
| CCSDT-1b                     |  ○  |   ○  |   ○  |
| CCSDT-2                      |  ○  |   ○  |   ○  |
| CCSDT-3                      |  ○  |   ○  |   ○  |
| CCSDT                        |  ○  |   ○  |   ○  |
| CCSDT(Q)                     |  ○  |   ○  |   ○  |
| CCSDTQ                       |  ○  |   ○  |   ○  |
| CEPA/0                       |  ○  |   ○  |   ○  |
| ACPF                         |  ○  |   ○  |   ○  |
| AQCC                         |  ○  |   ○  |   ○  |
| Orbital-optimized CCD        |  ○  |   ○  |   ○  |
| Density-fitted CCSD          |  ○  |   ○  |   ○  |
| Cholesky-decomposed CCSD     |  ○  |   ○  |   ○  |
| CCSD-F12a                    |  ○  |   ○  |   ○  |
| CCSD-F12b                    |  ○  |   ○  |   ○  |
| CCSD(T)-F12                  |  ○  |   ○  |   ○  |
| CCSD analytic gradient       |  ✓  |   ○  |   ○  |
| CCSD analytic Hessian        |  ○  |   ○  |   ○  |
| CCSD(T) analytic gradient    |  ○  |   ○  |   ○  |

**CC2** (`RHF::cc2`) is the approximate CC doubles model (full singles residual;
doubles residual linear in \(T_2\)) on a closed-shell RHF reference
(`reg_cc2-h2-sto3g` vs PySCF `cc.RCCSD` with `cc2=True`).
**CCSD** (`RHF::ccsd`) is closed-shell spin-orbital Gauss–Stanton on a
converged RHF reference (`reg_ccsd-h2-sto3g` vs PySCF `cc.CCSD`).
**CCSD(T)** (`RHF::ccsd_t`) adds the non-iterative (T) triples correction on
the same path (`reg_ccsd-t-h2o-sto3g` vs PySCF `cc.CCSD(...).ccsd_t()`).
**CCSD nuclear gradient** (`RHF::ccsd_forces`; `reg_ccsd-grad-h2o-sto3g`) is a
product foothold: central FD of the analytic CCSD total energy, gated against
PySCF analytic `cc.CCSD.nuc_grad_method` pins. A density-based Lagrangian /
Z-vector analytic path (matching `pyscf.grad.ccsd`) is a follow-up. No
open-shell CC, no density-fitted CC, no CCSD Hessian / CCSD(T) forces yet.

### Local-correlation coupled cluster

| Method                         | CPU | CUDA | Kokkos |
| ------------------------------ | :-: | :--: | :---: |
| Local CCSD                     |  ○  |   ○  |   ○  |
| Pair-natural-orbital CCSD      |  ○  |   ○  |   ○  |
| DLPNO-CCSD                     |  ○  |   ○  |   ○  |
| DLPNO-CCSD(T0)                 |  ○  |   ○  |   ○  |
| DLPNO-CCSD(T)                  |  ○  |   ○  |   ○  |
| LNO-CCSD                       |  ○  |   ○  |   ○  |
| LNO-CCSD(T)                    |  ○  |   ○  |   ○  |
| Frozen-natural-orbital CCSD    |  ○  |   ○  |   ○  |
| Frozen-natural-orbital CCSD(T) |  ○  |   ○  |   ○  |

## 4. Multireference Methods

### 4.1 Active-space methods

| Method                               | CPU | CUDA | Kokkos |
| ------------------------------------ | :-: | :--: | :---: |
| Complete-active-space CI, CASCI      |  ✓  |   ○  |   ○  |
| Complete-active-space SCF, CASSCF    |  ✓  |   ○  |   ○  |
| State-specific CASSCF                |  ○  |   ○  |   ○  |
| State-averaged CASSCF                |  ○  |   ○  |   ○  |
| Restricted-active-space CI, RASCI    |  ○  |   ○  |   ○  |
| Restricted-active-space SCF, RASSCF  |  ○  |   ○  |   ○  |
| Generalized-active-space CI, GASCI   |  ○  |   ○  |   ○  |
| Generalized-active-space SCF, GASSCF |  ○  |   ○  |   ○  |
| DMRG-CI                              |  ○  |   ○  |   ○  |
| DMRG-SCF                             |  ○  |   ○  |   ○  |
| Selected-CI active-space solver      |  ○  |   ○  |   ○  |
| FCIQMC active-space solver           |  ○  |   ○  |   ○  |
| CASCI analytic gradient              |  ○  |   ○  |   ○  |
| CASSCF analytic gradient             |  ○  |   ○  |   ○  |
| CASSCF analytic Hessian              |  ○  |   ○  |   ○  |

**CASCI** (`RHF::casci(frame, ncas, nelecas)`) freezes the inactive core and
runs the dense FCI kernel in the active space (`reg_casci-h2-sto3g` vs PySCF
`mcscf.CASCI`). **CASSCF** (`RHF::casscf(frame, ncas, nelecas)`) adds
two-step orbital optimization on the same active-space FCI kernel
(`reg_casscf-h2-sto3g` vs PySCF `mcscf.CASSCF`). **SC-NEVPT2**
(`RHF::nevpt2(frame, ncas, nelecas)`; `reg_nevpt2-h2-sto3g` vs PySCF
`mrpt.NEVPT`) is the in-tree multireference PT foothold on a closed-shell
CASSCF reference. No RAS/GAS, no state-averaged CASSCF product path, no
CASSCF gradient, no PC-NEVPT2 / QD-NEVPT2 / CASPT2.

### 4.2 Multireference correlation

| Method                                            | CPU | CUDA | Kokkos |
| ------------------------------------------------- | :-: | :--: | :---: |
| Multireference CISD, MRCISD                       |  ○  |   ○  |   ○  |
| Internally contracted MRCI                        |  ○  |   ○  |   ○  |
| Uncontracted MRCI                                 |  ○  |   ○  |   ○  |
| MRCI+Q                                            |  ○  |   ○  |   ○  |
| CASPT2                                            |  ○  |   ○  |   ○  |
| Multistate CASPT2                                 |  ○  |   ○  |   ○  |
| Extended multistate CASPT2                        |  ○  |   ○  |   ○  |
| Strongly contracted NEVPT2                        |  ✓  |   ○  |   ○  |
| Partially contracted NEVPT2                       |  ○  |   ○  |   ○  |
| Quasidegenerate NEVPT2                            |  ○  |   ○  |   ○  |
| MRMP2                                             |  ○  |   ○  |   ○  |
| Multiconfiguration pair-density functional theory |  ○  |   ○  |   ○  |
| Multireference analytic gradient                  |  ○  |   ○  |   ○  |
