# Semiempirical and Tight-Binding Methods

Tutorial: [Semiempirical models](../tutorial/ch17-semiempirical-am1.md).

CPU-only domain. Oracles: **OpenMOPAC** (NDDO), **DFTB+** 24.1 (DFTB), **xtb
6.7.1** (GFN/IPEA). Legend: [Roadmap index](index.md).

## 1. NDDO

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| AM1 | ✓ | ○ | `cpu::semiempirical::AM1`; OpenMOPAC ΔHf pins |
| RM1 | ✓ | ○ | Same NDDO path; MOPAC `calpar` free-atom repair |
| PM3 / PM6 / PM7 | ○ | ○ | Tables away |
| OM3 | ○ | ○ | **Blocked** — no OpenMOPAC primary surface |
| Nuclear forces (product) | ✓ | ○ | Fixed-density FD of SCF energy |
| Closed-form integral-derivative forces | ○ | ○ | |
| FD Hessian via `vib::Hessian` | ✓ | ○ | |
| Heat of formation (MOPAC-style) | ✓ | ○ | |
| NDDO UHF / COSMO / Γ-PBC | ○ | ○ | Molecular AM1 under wrap unwraps an isolated fragment |

## 2. DFTB

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| DFTB0 | ✓ | ○ | Analytic nuclear forces (HF + Pulay + repulsive); no SCC γ |
| DFTB2 (SCC) | ✓ | ○ | mio-1-1 + 3ob-3-1 (H/C/N/O); `lattice` γ is Ewald |
| DFTB3 | ✓ | ○ | Energy + closed-form F + Γ-PBC F; no k / stress / spin / 3ob / D3 |
| DFTB3 γ^h (DampXH) | ○ | ○ | ζ = 4 X–H damping; MioSketch DFTB3 runs undamped |
| DFTB3 Ewald γ | ○ | ○ | Periodic γ cut at the SK sphere; DFTB2 γ is Ewald |
| Collinear spin-DFTB2 | ✓ | ○ | MioSketch HOAO (`ShellResolvedSpin = No`); ThreeOb open shell throws |
| Shell-resolved collinear spin | ○ | ○ | DFTB+ `ShellResolvedSpin = Yes` throws |
| DFTB+U | ✓ | ○ | Equal-filling energy; analytic F and per-shell \(U_\ell\) still ○ |
| D3 dispersion (mol + Γ-PBC) | ✓ | ○ | DFTB2 two-body BJ, orthorhombic PBC. ATM, triclinic cells, and D4 are ○ |
| DFTB-D3 three-body (ATM) | ○ | ○ | `calculate_c9=false`; DFTB+ / s-dftd3 ATM |
| DFTB-D3 triclinic cell | ○ | ○ | `d3_periodic` throws unless the box is orthorhombic |
| SK beyond H/C/N/O | ○ | ○ | mio-1-1 ships P and S; in-tree tables are H/C/N/O and throw |
| On-site correction | ○ | ○ | DFTB+ `OnSiteCorrection` |
| Γ-point / multi-k periodic DFTB2 | ✓ | ○ | Analytic F/σ on MioSketch; ThreeOb is molecular only |
| DFTB2 analytic gradient / stress | ✓ | ○ | Pair virial; all product paths |
| TD-DFTB | ✓ | ○ | Closed-shell Γ MioSketch. 3ob and collinear-reference Casida still ○. See [Excited States](excited-states.md) |
| LC-DFTB / Fermi filling / GBSA | ○ | ○ | |

## 3. Extended tight binding (xTB)

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| GFN0 / GFN1 / GFN2 | ✓ | ○ | H/C/N/O; analytic G; FD Hessian; integer occ |
| GFN2 AES (quadrupole) + self-consistent D4 | ✓ | ○ | Cumulative multipoles of every atom; AES and D4 in the SCC Fock; gradient is the exact derivative of E |
| IPEA-xTB | ✓ | ○ | Energy + G + FD Hessian + ΔSCC VIP/VEA; CHNO only |
| Periodic / spin / ALPB / GBSA / CPCM-X / Fermi / sTDA-xTB | ○ | ○ | `--gbsa` is not ALPB. Extra-Z tables too |
| GFN-FF | ○ | ○ | Classical FF scope-out |

### Gaps

- Elements beyond H/C/N/O in shipped SK / xTB tables (mio-1-1 P/S and
  xtb's all-element tables; AM1 NDDO rows are broader).
- DFTB3 is not a DFTB+ Hamiltonian until γ^h and, under PBC, Ewald γ land.
- D3 is two-body and orthorhombic; the ATM three-body term and a general
  cell are still open. On-site correction (`OnSiteCorrection`) is open.
- TD-DFTB Casida is closed-shell MioSketch at Γ; 3ob and a collinear
  reference are open.
- Analytic (closed-form) Hessians anywhere in this chapter — all FD via
  `cpu::vib::Hessian`. xtb `--hess` is numerical too.
- No CUDA path.
