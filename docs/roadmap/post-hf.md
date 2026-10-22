# Post-Hartree–Fock Correlation

Tutorial: [Electron correlation](../tutorial/ch07-correlation.md).

Correlation on a converged HF reference. Each method is its own type
(`MP2{mf}`, `MP3`, `MP4`, `CC2`, `CCSD` with `ccsd_t`, `CASCI`, `CASSCF`,
`FCI`, `NEVPT2`), not a member of `MeanFieldMethod`. Not Drivers — they throw
on a non-converged reference. Legend: [Roadmap index](index.md).

## 1. Møller–Plesset

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RMP2 | ✓ | ○ | Canonical O(N⁵) |
| UMP2 | ✓ | ○ | `MP2` on `UHF` |
| ROMP2 | ✓ | ○ | UMP2 on ROHF reference |
| Frozen-core MP2 | ✓ | ○ | `MP2::frozen = N`; RMP2 pinned (RI-MP2 combination reachable) |
| RI-MP2 | ◐ | ○ | `ri = RI::{J,JK}` + aux; RHF only. No external pin; self-pinned case removed |
| SCS-MP2 / SOS-MP2 | ✓ | ○ | `MP2::os_scale` / `ss_scale` |
| RMP2 / UMP2 analytic gradient | ✓ | ○ | Lagrangian + Z-vector |
| ROMP2 gradient | ○ | ○ | |
| MP2 Hessian | ○ | ○ | |
| RMP3 | ✓ | ○ | Closed-shell |
| RMP4(SDTQ) | ✓ | ○ | Closed-shell |
| Local / DLPNO / F12 MP2 | ○ | ○ | |

## 2. Configuration interaction

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| FCI (determinant-driven) | ✓ | ○ | Dense matrix; cap `n_mo > 14` or `N_det > 2000` — oracle tool, not production |
| Selected / heat-bath CI | ○ | ○ | |

CIS excitation energies live under
[Excited States](excited-states.md) (`TDA<RHF>::run`), not here.

## 3. Coupled cluster

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| CC2 | ✓ | ○ | Closed-shell |
| CCSD | ✓ | ○ | Closed-shell Gauss–Stanton |
| CCSD(T) | ✓ | ○ | Non-iterative triples |
| CCSD nuclear gradient | ✓ | ○ | FD of CCSD energy (not Lagrangian/Z-vector yet) |
| CCSD(T) gradient / CCSD Hessian | ○ | ○ | |
| Open-shell / DF / F12 / local CC | ○ | ○ | |

## 4. Multireference

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| CASCI | ✓ | ○ | Dense FCI in active space |
| CASSCF | ✓ | ○ | Two-step orbital opt |
| SC-NEVPT2 | ✓ | ○ | On closed-shell CASSCF |
| State-averaged CAS / RAS / GAS / DMRG | ○ | ○ | |
| CASPT2 / PC-NEVPT2 / QD-NEVPT2 | ○ | ○ | |
| CASSCF analytic gradient | ○ | ○ | |
