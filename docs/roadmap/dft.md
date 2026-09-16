# Density-Functional Theory

Legend and caveats: [Roadmap index](index.md).

The molecular Kohn–Sham path shares the Gaussian SCF solver with
[Hartree–Fock](hartree-fock.md): the same 2×2 config product of
{restricted, unrestricted} × {HF, KS}, the same DIIS/level-shift machinery, the
same analytic-gradient assembly.

The molecular functional set includes pure LDA/GGA selectors (`LSDA`, `PBE`,
`PBEsol`, `BLYP`), global hybrids (`B3LYP`, `PBE0`), and double hybrid
`B2PLYP` — evaluated on Becke-partitioned atomic grids with Lebedev angular
quadrature. All functionals are hand-coded and gated against libxc to 1e-12 in
tests; libxc is never a runtime dependency.

**Periodic DFT is DFT**, so it lives here too: sections 10–11 cover the
plane-wave and GPW engines, their functionals, forces and stress, phonons, and
what is still missing there. The periodic-only gaps are **hybrids**,
**meta-GGA**, **GAPW/PAW**, and stock **VV10** — the last is blocked rather than
scheduled, because CP2K 2026.1 has no `TYPE VV10` to pin it against and this
project does not ship a periodic functional without an external twin.

## 1. Kohn–Sham Methods

| Method                                    | CPU | CUDA | Kokkos |
| ----------------------------------------- | :-: | :--: | :---: |
| Restricted Kohn–Sham DFT, RKS             |  ✓  |   ○  |   ○  |
| Unrestricted Kohn–Sham DFT, UKS           |  ✓  |   ○  |   ○  |
| Restricted open-shell Kohn–Sham DFT, ROKS |  ✓  |   ○  |   ○  |
| Generalized Kohn–Sham DFT                 |  ○  |   ○  |   ○  |
| Noncollinear spin DFT                     |  ○  |   ○  |   ○  |
| Spinor Kohn–Sham DFT                      |  ○  |   ○  |   ○  |
| Finite-temperature DFT                    |  ○  |   ○  |   ○  |
| Constrained DFT                           |  ○  |   ○  |   ○  |
| Delta-SCF DFT                             |  ○  |   ○  |   ○  |
| Maximum-overlap excited-state DFT         |  ○  |   ○  |   ○  |
| Ensemble DFT                              |  ○  |   ○  |   ○  |
| DFT+U                                     |  ○  |   ○  |   ○  |
| RKS analytic gradient                     |  ✓  |   ○  |   ○  |
| UKS analytic gradient                     |  ✓  |   ○  |   ○  |
| ROKS analytic gradient                    |  ✓  |   ○  |   ○  |
| RKS analytic Hessian                      |  ✓  |   ○  |   ○  |
| UKS analytic Hessian                      |  ✓  |   ○  |   ○  |

**ROKS** (`cpu::scf::ROKS`) is energy + **analytic nuclear gradient** on the
Roothaan effective-Fock path (Guest–Saunders / PySCF canonical;
`reg_roks-o2-sto3g` energy, `reg_roks-grad-o2-sto3g` forces — O₂ triplet
LSDA/STO-3G). Forces reuse the UKS electronic gradient on split-occupation
α/β densities with the ROHF-style energy-weighted density from **KS spin
Focks** (not effective-Fock eigenvalues). KS gradients (RKS/UKS/ROKS) include
the semi-local XC contribution (LDA, GGA, and the hybrid exact-exchange
admixture) and the Becke grid-response terms. **RKS nuclear Hessian**
(`RKS::hessian`; `reg_rks-hessian-h2o-sto3g`) and **UKS nuclear Hessian**
(`UKS::hessian`; `reg_uks-hessian-o2-sto3g`) are product footholds via FD of
analytic KS forces (same style as `ROHF::hessian`), gated against PySCF FD
twins; full CPKS / UCPKS closed-form `hessian.RKS` / `hessian.UKS` is a
follow-up. There is no Kohn–Sham path on CUDA at all — the device SCF solver
is restricted HF only.

## 2. LDA Functionals

| Functional           | CPU | CUDA | Kokkos |
| -------------------- | :-: | :--: | :---: |
| Slater exchange      |  ✓  |   ✓  |   ○  |
| VWN correlation      |  ✓  |   ✓  |   ○  |
| SVWN                 |  ✓  |   ○  |   ○  |
| Perdew–Wang 1992 LDA |  ✓  |   ○  |   ○  |
| Spin-polarized LDA   |  ✓  |   ✓  |   ○  |

`LSDA` = Slater (Dirac) exchange + VWN5 correlation. VWN-RPA is also
implemented, as the LDA correlation component inside B3LYP. PW92 is present as
the LDA base of PBE. The CUDA column here is **plane-wave only**: Slater + VWN5
(`XCType::LSDA`, `cuda/planewave/internal/xc_lda.cu` — deliberately NOT the
PW92-correlation σ→0 limit of PBE), closed-shell **and collinear spin** LSDA
(fixed-occupation UKS; `cuda-pw-collinear-spin`) — see
[Periodic Kohn–Sham Methods](#10-periodic-kohnsham-methods) below. There is no
XC on the device in the Gaussian path.

## 3. GGA Functionals

| Functional | CPU | CUDA | Kokkos |
| ---------- | :-: | :--: | :---: |
| PBE        |  ✓  |   ✓  |   ○  |
| PBEsol     |  ✓  |   ○  |   ○  |
| revPBE     |  ✓  |   ○  |   ○  |
| RPBE       |  ✓  |   ○  |   ○  |
| BLYP       |  ✓  |   ○  |   ○  |
| BP86       |  ○  |   ○  |   ○  |
| B97        |  ○  |   ○  |   ○  |
| OLYP       |  ○  |   ○  |   ○  |
| HCTH       |  ○  |   ○  |   ○  |
| HCTH/407   |  ○  |   ○  |   ○  |
| SOGGA      |  ○  |   ○  |   ○  |

The PBE CUDA ✓ is **plane-wave** closed-shell **and collinear fixed-occupation
spin-GGA energy + forces + stress** (`cuda-pw-spin-pbe` / `reg_cuda-pw-spin-pbe`;
GH200 **1335945** \|ΔE\|=1.14e-13 Ha, max\|ΔF\|=2.0e-12 eV/Å,
max‖Δσ‖=4.9e-12 eV/Å³). Spin×smear and PBE-family variants remain open:
the `cuda::pw::PW` density side runs
device PBE (`cuda/planewave/internal/xc_pbe.cu` — PBE-X + PW92-based PBE-C;
closed-shell ζ=0 + spin `xc_pbe_spin_grid`; spectral ∇ρ / divergence in
`DeviceDensity`) against the CPU stage helpers as oracle. GPW has a device
closed-shell PBE **KS stage** (`KsChainDevice` / `cuda-gpw-ks-gga`) composed
into hybrid public `cuda::gpw::GpwDft` (product PBE H₂ dual-gate **landed**
`cuda-gpw-pbe-product-gate` / GH200 **1127166**; water×PBE + path×PBE F/σ
still open). No CUDA GGA on the molecular Gaussian path; the PBE-family
variants (PBEsol/revPBE/RPBE) stay CPU-only.
`XCType::BLYP` is pure B88+LYP (unit weights, no HF admixture); regression
`reg_blyp-water-sto3g` pins H2O/STO-3G vs PySCF xc=blyp. **BLYP on GPW**
ships as `XcKind::BLYP` (`reg_gpw-blyp-water`; dual-gate vs PBE). **PBEsol**
(Perdew et al. 2008; solid-state μ/β) shares `evaluate_xc` and ships on the
molecular Gaussian path and **GPW** RKS (`XcKind::PBEsol`;
`reg_gpw-pbesol-si`) **and Γ-UKS** (`reg_gpw-uks-pbesol`; offline CP2K O₂
twin / `xc_grid_pbesol_spin`).
**revPBE** (Zhang–Yang κ=1.245; `XCType::RevPBE` / `XcKind::RevPBE`) ships on
the molecular path and **GPW** RKS (`reg_gpw-revpbe-si`) **and Γ-UKS**
(`reg_gpw-uks-revpbe`; dual-gate vs PBE UKS / offline CP2K O₂).
**RPBE** (Hammer–Hansen–Nørskov; `XCType::RPBE` / `XcKind::RPBE`) ships on
the molecular path and **GPW** RKS (`reg_gpw-rpbe-water`; dual-gate vs PBE)
**and Γ-UKS** (`reg_gpw-uks-rpbe`; offline CP2K O₂ twin).

## 4. Meta-GGA Functionals

| Functional | CPU | CUDA | Kokkos |
| ---------- | :-: | :--: | :---: |
| TPSS       |  ○  |   ○  |   ○  |
| revTPSS    |  ○  |   ○  |   ○  |
| r2SCAN     |  ○  |   ○  |   ○  |
| M06-L      |  ○  |   ○  |   ○  |
| MN12-L     |  ○  |   ○  |   ○  |
| MN15-L     |  ○  |   ○  |   ○  |
| B97M-V     |  ○  |   ○  |   ○  |
| TASK       |  ○  |   ○  |   ○  |

No kinetic-energy-density (τ) channel exists in the grid engine, so the
meta-GGA rung is structurally absent, not merely unparameterized.

## 5. Global Hybrid Functionals

| Functional | CPU | CUDA | Kokkos |
| ---------- | :-: | :--: | :---: |
| B3LYP      |  ✓  |   ○  |   ○  |
| B3PW91     |  ○  |   ○  |   ○  |
| PBE0       |  ✓  |   ○  |   ○  |
| BHLYP      |  ○  |   ○  |   ○  |
| BHandHLYP  |  ○  |   ○  |   ○  |
| X3LYP      |  ○  |   ○  |   ○  |
| TPSSh      |  ○  |   ○  |   ○  |
| M06        |  ○  |   ○  |   ○  |
| M06-2X     |  ○  |   ○  |   ○  |
| M06-HF     |  ○  |   ○  |   ○  |
| BMK        |  ○  |   ○  |   ○  |
| SCAN0      |  ○  |   ○  |   ○  |
| r2SCAN0    |  ○  |   ○  |   ○  |
| MN15       |  ○  |   ○  |   ○  |

B3LYP carries a 0.20 exact-exchange fraction, PBE0 0.25. The hybrid path builds
a real exchange matrix from the ERIs, so it is available anywhere the Gaussian
SCF runs — but not in GPW or plane waves.

## 6. Range-Separated and Screened Hybrid Functionals

| Functional                                 | CPU | CUDA | Kokkos |
| ------------------------------------------ | :-: | :--: | :---: |
| CAM-B3LYP                                  |  ✓  |   ○  |   ○  |
| LC-ωPBE                                    |  ○  |   ○  |   ○  |
| LC-BLYP                                    |  ○  |   ○  |   ○  |
| ωB97                                       |  ○  |   ○  |   ○  |
| ωB97X                                      |  ○  |   ○  |   ○  |
| ωB97X-D                                    |  ○  |   ○  |   ○  |
| ωB97X-D3                                   |  ○  |   ○  |   ○  |
| ωB97M-V                                    |  ○  |   ○  |   ○  |
| HSE03                                      |  ○  |   ○  |   ○  |
| HSE06                                      |  ○  |   ○  |   ○  |
| Optimally tuned range-separated hybrid DFT |  ○  |   ○  |   ○  |

CAM-B3LYP is a closed-shell RKS energy foothold (`XCType::CAMB3LYP`):
`rys_eri_erf_block` supplies \(K[\mathrm{erf}(\mu r)/r]\) and the Fock builder
mixes \(\alpha K[1/r]+\beta K[\mathrm{erf}]\) with Yanai
\((\alpha,\beta,\mu)=(0.19,0.46,0.33)\). Semi-local XC reuses the B3LYP
kernels (not libxc ITYH). Pin: `regressions/scf/cam-b3lyp-h2-sto3g`.
Other RSH functionals still need the same engine with new catalog rows.

## 7. Double-Hybrid Functionals

| Functional                              | CPU | CUDA | Kokkos |
| --------------------------------------- | :-: | :--: | :---: |
| B2PLYP                                  |  ✓  |   ○  |   ○  |
| B2GP-PLYP                               |  ○  |   ○  |   ○  |
| PWPB95                                  |  ○  |   ○  |   ○  |
| DSD-BLYP                                |  ○  |   ○  |   ○  |
| DSD-PBEP86                              |  ○  |   ○  |   ○  |
| revDSD-PBEP86-D4                        |  ○  |   ○  |   ○  |
| PBE0-DH                                 |  ○  |   ○  |   ○  |
| PBE-QIDH                                |  ○  |   ○  |   ○  |
| ωB97X-2                                 |  ○  |   ○  |   ○  |
| Spin-component-scaled double-hybrid DFT |  ○  |   ○  |   ○  |

**B2PLYP** (`XCType::B2PLYP` + `RKS::mp2` with `hybrid_pt2_fraction` a_c=0.27)
composes hybrid KS (a_x=0.53 HF + 0.47 B88 + 0.73 LYP) with scaled RMP2 on the
KS orbitals (`reg_b2plyp-h2o` vs offline PySCF-composed pin). Other double
hybrids and SCS-DH remain ○.

## 8. Dispersion and Nonlocal Correlation

| Method                           | CPU | CUDA | Kokkos |
| -------------------------------- | :-: | :--: | :---: |
| DFT-D3 zero damping              |  ✓  |   ○  |   ○  |
| DFT-D3 Becke–Johnson damping     |  ✓  |   ○  |   ○  |
| DFT-D4                           |  ✓  |   ○  |   ○  |
| Tkatchenko–Scheffler dispersion  |  ✓  |   ○  |   ○  |
| Many-body dispersion, MBD        |  ✓  |   ○  |   ○  |
| Exchange-hole dipole moment, XDM |  ✓  |   ○  |   ○  |
| VV10                             |  ○  |   ○  |   ○  |
| rVV10                            |  ✓  |   ○  |   ○  |
| vdW-DF                           |  ✓  |   ○  |   ○  |
| vdW-DF2                          |  ✓  |   ○  |   ○  |
| optB88-vdW                       |  ✓  |   ○  |   ○  |

DFT-D3 zero damping (PBE: s6 = 1.0, s8 = 0.722, sr6 = 1.217) is wired into
**GPW**, from in-tree reference tables. DFT-D3 **Becke–Johnson** damping
(PBE: s6 = 1, s8 = 0.7875, a1 = 0.4289, a2 = 4.4407 Bohr) is on the
**molecular Gaussian** path via `cpu::scf::D3BJ` / `MeanFieldMethod::d3bj`
(PySCF `disp='d3bj'` twin; regression `d3bj-molecular-pbe`) **and on GPW**
via `GpwDft::d3bj` / `GPWEvaluator::d3bj` (CP2K CN + BJ pin;
`reg_gpw-pbe-d3bj-ice`). **DFT-D4**
(Caldeweyher et al.; two-body BJ + EEQ/CN-dependent C6) is on the molecular
path via `cpu::scf::D4` / `MeanFieldMethod::d4` (PySCF `disp='d4'` twin;
regression `dft-d4-h2o-pbe`; three-body ATM not included) **and on GPW** via
`GpwDft::d4` / `GPWEvaluator::d4` (dftd4 3.7.0 offline pin;
`reg_gpw-d4-water`; three-body ATM off). **Tkatchenko–Scheffler (TS)** free-atom
pairwise dispersion ships on GPW (`GpwDft::ts` / `GPWEvaluator::ts`; libMBD
tables; `reg_gpw-ts-ice`). **MBD@rsSCS** free-atom (volume-ratio = 1) many-body
dispersion ships on GPW (`GpwDft::mbd` / `GPWEvaluator::mbd`; libMBD Fortran
pin; `reg_gpw-mbd-ice`). **Hirshfeld density volumes** (density-scaled
TS/MBD via `GpwDft::hirshfeld`) ship on GPW at **Γ RKS**
(`reg_gpw-mbd-hirshfeld`), **multi-k RKS** (`reg_gpw-hirshfeld-kpoints`),
**Γ-UKS** (`reg_gpw-hirshfeld-uks`), and **multi-k UKS**
(`reg_gpw-hirshfeld-uks-kpoints`).
**vdW-DF** (DRSLL nonlocal correlation; `GpwDft::vdw_df`) ships on GPW
(`reg_gpw-vdw-df-ice`). **vdW-DF2 (LMKLL)** (`GpwDft::vdw_df2`;
`reg_gpw-vdw-df2-ice`) and **rVV10** (`GpwDft::rvv10`; `reg_gpw-rvv10-ice`)
ship on GPW. **Nonlocal SCF potential + forces** (Román–Pérez–Soler
back-propagation) productize for vdW-DF/DF2/rVV10
(`reg_gpw-vdw-scf-potential`), including on the **REL_CUTOFF multigrid**
(`ngrids>1`, CP2K NGRIDS 4 twin; `reg_gpw-vdw-multigrid`, Γ RKS).
**Periodic Ewald + k-mesh MBD@rsSCS**
productizes free-atom / Hirshfeld MBD for periodic cells
(`reg_gpw-mbd-periodic`). **XDM** free-atom pairwise dispersion ships on
GPW (`GpwDft::xdm` / `GPWEvaluator::xdm`; `reg_gpw-xdm-ice`). **Hirshfeld**
volumes productize for **closed-shell RKS** (Γ + multi-k), **Γ-UKS**
(`reg_gpw-hirshfeld-uks`), **and multi-k UKS**
(`reg_gpw-hirshfeld-uks-kpoints`; `fold_dispersion_k_uks`). **optB88-vdW**
(optB88 exchange + DRSLL nonlocal)
ships on GPW Γ RKS (`reg_gpw-optb88-vdw`), **Γ-UKS** (`reg_gpw-optb88-uks`),
**and multi-k RKS** (`reg_gpw-optb88-kpoints`). Stock **VV10** (distinct from rVV10;
CP2K 2026.1 has no `TYPE VV10`) is **blocked** — there is no offline twin to
pin it against, so the shipped VV-family member remains rVV10.
**DFT-D4 analytic fixed-\(P\) stress** productizes (`reg_gpw-d4-stress`);
**D4 analytic ionic forces** productize (`reg_gpw-d4-forces`; dftd4
gradient pin + product wiring) and **MBD@rsSCS analytic ionic forces**
productize at **Γ** (`reg_gpw-mbd-forces`; free-atom / periodic path)
**and multi-k** (`reg_gpw-mbd-kpoint-forces`; k-path force assembly).
Force assembly folds free-atom **TS** / **XDM** on Γ + multi-k with
dedicated offline force pins (`reg_gpw-ts-forces` / `reg_gpw-xdm-forces`
+ multi-k XDM `reg_gpw-xdm-kpoint-forces` + multi-k free-atom **TS forces**
`reg_gpw-ts-kpoint-forces`).
**Three-body ATM** energy productizes on GPW: D3 C9 (`GpwDft::d3_atm`;
`reg_gpw-d3-atm`) and D4 \(s_9=1\) (`GpwDft::d4_atm`; `reg_gpw-d4-atm`).
Two-body D3/D3(BJ) **energy and analytic ionic forces** are product
(`reg_gpw-d3-forces` / `reg_gpw-d3bj-forces`; CP2K-gated, unified
`2 × R_CUTOFF` pair cutoff). **D3(BJ) analytic fixed-\(P\) stress** is now
CP2K-pinned (`reg_gpw-d3bj-stress`; numerical-stress *difference* pin —
the real oracle stress pin for the whole dispersion stress fold, Δσ parity
3.9e−14 Ha/Bohr³), and **D3 zero-damping stress** carries its own dedicated
pin (`reg_gpw-d3-stress`; CP2K 2026.1 NUMERICAL stress-**difference** pin —
D3-minus-plain isolates the dispersion stress — plus `gpw_stress_fd`
dual-gate). **ATM forces**
are product on both families: D3 C9 carries a dedicated CP2K force pin
(`reg_gpw-d3-atm-forces`; periodic image-triple overcount + pbc anchoring
fixed in `accumulate_atm_c9`) and `cpu::scf::dftd4` carries the fully
analytic \(s_9\) gradient (triple derivatives + C9→neutral-C6→CN chain;
`reg_gpw-d4-atm-forces`, dftd4 3.7.0 `--grad` pin at 1e−9 Ha/Bohr).
**MBD stress** ships at **Γ** (`reg_gpw-mbd-stress`) **and multi-k**
(`reg_gpw-mbd-kpoint-stress`; k-summed fixed-\(P_k\) Lagrangian folds
periodic MBD, strained-cell pymbd pins). Dedicated **TS / XDM stress** pins are product
(`reg_gpw-ts-stress` / `reg_gpw-xdm-stress`; energy folds product).
Multi-k dedicated stress pins for **D3 / D3(BJ) / D3 ATM (C9)** are product
(`reg_gpw-d3-kpoint-stress` / `reg_gpw-d3bj-kpoint-stress` /
`reg_gpw-d3-atm-kpoint-stress`; CP2K 2026.1 NUMERICAL stress-difference
twins on MP meshes). Multi-k **D4** and **XDM** stress pins are product
(`reg_gpw-d4-kpoint-stress` CP2K libdftd4 NUMERICAL Δσ;
`reg_gpw-xdm-kpoint-stress` dual pure-Python strain); multi-k free-atom
**TS** stress is product (`reg_gpw-ts-kpoint-stress`; libMBD strain
dual-gate). The **D3 ATM stress** pin is product
(`reg_gpw-d3-atm-stress`: CP2K 2026.1 `STRESS_TENSOR NUMERICAL`
difference C9 on−off isolates the pure ATM stress, Δσ parity 1.1e−13
Ha/Bohr³ at CP2K's own `DX 0.001` FD step — the hard-gated triple sum is
step-sensitive, unlike the smooth-cutoff two-body fold — plus the
configurable `d3_rcut_bohr` member mirroring CP2K `R_CUTOFF`); the **D4
ATM stress** pin shipped too (`reg_gpw-d4-atm-stress`:
dftd4 3.7.0 strained-cell energies gate the s9 fold — σ^ATM parity
3.7e-16 Ha/Bohr³ vs the ~7.5e-10 signal, plus `gpw_stress_fd` dual gate).
Nonlocal vdW × REL_CUTOFF multigrid (`ngrids>1`) is
product at Γ RKS (`reg_gpw-vdw-multigrid`; SC V_nl on the finest grid
before per-level restriction). Nonlocal vdW on the **k-point path** is product for closed-shell
RKS (`reg_gpw-vdw-kpoints`; single-grid and multigrid) and at **Γ UKS**
(`reg_gpw-vdw-uks`). Γ-UKS + multigrid + nonlocal is product
(`reg_gpw-vdw-uks-multigrid`; matched CP2K NGRIDS 4 / DRSLL / UKS twin).
Multi-k **UKS** + nonlocal is product (`reg_gpw-vdw-uks-kpoints`; dual-spin
\(V_{\mathrm{nl}}[\rho]\) fold on the k path). Multi-k UKS × nonlocal ×
REL_CUTOFF multigrid is product (`reg_gpw-vdw-uks-kpoint-multigrid`;
matched CP2K NGRIDS 4 / DRSLL / UKS / MP twin + multi-k UKS sg↔mg
consistency). Γ-UKS **optB88-vdW** is product (`reg_gpw-optb88-uks`; CP2K
UKS OptB88+DRSLL twin); multi-k RKS optB88-vdW is product
(`reg_gpw-optb88-kpoints`); multi-k UKS optB88-vdW is product
(`reg_gpw-optb88-uks-kpoints`; CP2K KPOINTS + UKS + OptB88+DRSLL twin).
Multi-k **D4 ATM forces** are product (`reg_gpw-d4-atm-kpoint-forces`;
CP2K libdftd4 multi-k ENERGY_FORCE); multi-k **D3 ATM forces** are
product (`reg_gpw-d3-atm-kpoint-forces`; CP2K multi-k ENERGY_FORCE C9
on−off ΔF). **Hirshfeld-scaled TS/MBD forces** (and stress fold) are
product (`reg_gpw-hirshfeld-forces`; fixed-\(v\) dual-gate + product
wiring; energy and force/stress both use post-SCF volumes). Multi-k
dedicated **two-body D3 / D3(BJ) / D4 force** pins are product
(`reg_gpw-d3-kpoint-forces` / `reg_gpw-d3bj-kpoint-forces` /
`reg_gpw-d4-kpoint-forces`). Multi-k free-atom **XDM forces** are
product (`reg_gpw-xdm-kpoint-forces`; dual pure-Python / FD dual-gate).
Multi-k **Hirshfeld-scaled TS/MBD forces** are product
(`reg_gpw-hirshfeld-kpoint-forces`; fixed-\(v\) dual-gate + multi-k
wiring). Multi-k free-atom **TS forces** are product
(`reg_gpw-ts-kpoint-forces`). Multi-k **D4 ATM stress** is product
(`reg_gpw-d4-atm-kpoint-stress`; CP2K libdftd4 multi-k NUMERICAL ATM−2b
Δσ). Multi-k free-atom **TS stress** is product
(`reg_gpw-ts-kpoint-stress`). Multi-k **Hirshfeld-scaled stress** is
product (`reg_gpw-hirshfeld-kpoint-stress`; fixed-\(v\) dual-gate). Multi-k
**nonlocal vdW forces** and **optB88 ENERGY_FORCE** pins are product
(`reg_gpw-vdw-kpoint-forces` / `reg_gpw-optb88-forces`). Analytic cell
stress folds nonlocal \(E_{\mathrm{c}}^{\mathrm{nl}}\) under strain
(`reg_gpw-vdw-stress`).

## 9. Composite DFT Methods

| Method    | CPU | CUDA | Kokkos |
| --------- | :-: | :--: | :---: |
| HF-3c     |  ○  |   ○  |   ○  |
| PBEh-3c   |  ○  |   ○  |   ○  |
| B97-3c    |  ○  |   ○  |   ○  |
| r2SCAN-3c |  ○  |   ○  |   ○  |

---

## 10. Periodic Kohn–Sham Methods

Two periodic DFT engines exist, with disjoint strengths:

- **Plane-wave DFT** (`cpu::pw::PW`, `cuda::pw`) — k-points, norm-conserving
  UPF pseudopotentials, block Davidson. **Ionic forces via analytic local HF +
  Ewald + Kleinman–Bylander nonlocal** (`pw_forces_analytic`, default for
  `PWEvaluator::with_gradient(true)`); FD (`pw_forces_fd`) is the oracle.
  **Cell stress** is **Nielsen–Martin analytic** (`pw_stress` → `kStress`;
  multi-SCF strain-FD oracle `pw_stress_fd`; `reg_pw-analytic-stress-si`).
  XC is **true GGA PBE** on the FFT grid (density gradients enter Vxc; default
  `xc = PBE`) or LSDA for **closed-shell**; **collinear spin**
  (`spin_polarized`) uses **true collinear spin-GGA** on the PW grid
  (`XCBuilder::step_spin`; `reg_pw-spin-gga`), including **k-mesh UKS**
  (`reg_pw-uks-kmesh-fe`) and **metallic smearing** (Fermi–Dirac /
  Methfessel–Paxton; Al pin + **bulk Fe UKS+smear**
  `reg_pw-uks-fe-smearing`) on CPU. **Bands / DOS** post-process
  (`cpu::pw::Bands` / `Dos`; `reg_bands-dos-si` + spin
  `reg_bands-spin-si` + metallic DOS `reg_dos-metallic-si` + metallic
  bands `reg_bands-metallic-si` + spin DOS `reg_dos-spin-si`). **PW
  CELL_OPT**
  (`IsotropicCellRelax` + intensive `kStress`; `reg_pw-variable-cell-si`).
  CUDA is `cuda::pw::PW` — a strict subset of the CPU method: **LSDA**
  (Slater + VWN5 device kernel) or **PBE** (device closed-shell GGA **and**
  collinear spin-GGA energy, `internal/xc_pbe.cu` — table **Periodic GGA**
  CUDA ✓ is this energy+F/σ slice; collinear spin×smear **energy and
  analytic forces** compose on CPU and CUDA — shared μ, \(f\in[0,1]\);
  dual-gate `reg_cuda-pw-spin-pbe`; spin×smear **stress** still closed), **energy** (`run()`),
  **analytic ionic forces** (closed-shell **LSDA + PBE**;
  `cuda::pw::pw_forces_analytic` — local HF + Ewald + KB nonlocal; GH200
  dual-gate), **analytic cell stress** (closed-shell **LSDA + PBE** dual-gate
  pins; `cuda::pw::pw_stress` — `reg_cuda-pw-stress` + `reg_cuda-pw-stress-pbe`;
  GH200 PBE max‖Δσ‖ **5.585e-12** eV/Å³ job **1123959**), **collinear spin**
  energy (**LSDA** + **spin-PBE**; fixed occupation; GH200 spin-PBE
  **1124645**) + **collinear LSDA spin forces** (`cuda-pw-spin-forces` /
  GH200 **1130196**) + **collinear LSDA spin stress** (`cuda-pw-spin-stress` /
  GH200 **1135756** max‖Δσ‖=**1.435e-11** eV/Å³; **spin-PBE F/σ landed**
  GH200 **1335945**; **collinear spin×smear energy** GH200 **1336653**
  \|ΔE\|=1.10e-13 Ha, mag=1.6682; **spin×smear forces** GH200 **1357414**
  max\|ΔF\|=**1.92e-12** eV/Å, force-path SCF energy_ry=**−57.57643373**
  smear=1), and **restricted metallic smear** energy (FD/Gaussian/MP +
  `extra_bands` + Mermin free energy) **+ restricted metallic-smear analytic
  ionic forces** (`cuda-pw-smear-forces` / GH200 **1130427** max\|ΔF\|=
  **1.36e-11** eV/Å) **+ restricted metallic-smear cell stress**
  (`cuda-pw-smear-stress` / GH200 **1133111** max‖Δσ‖=**7.25e-11** eV/Å³,
  Al FCC 1-atom; production multi-atom is still open, and spin+smear still
  rejects on the base path). Large-cell wall evidence exists on Si supercells
  (`benchmarks/cuda/bench_pw_large_cell.cu`).
  **CUDA GPW first product slice
  landed:** public `cuda::gpw::GpwDft` (Γ RKS DIIS PADE/PBE + **Γ UKS DIIS
  PADE** + **multigrid energy** + **multi-k RKS DIIS energy H₂ foothold** +
  **multi-k device χ^k collocate + V→H_k integrate + H_k fuse + KDiis/zheevd + phased STVnl + H residual residency** mid-SCF;
  residual host χ^k + host Löwdin from S + **device** product collocate/KS/integrate via
  `KsChainDevice::run` / `run_uks` + `KsMultiGridChainDevice` on **non-default
  stream** + **device RKS mid-SCF Pulay DIIS + cuSOLVER eigen**
  (`AoDiisEigenDevice` RKS + **dual-spin UKS**) + **device weight pack**
  (`ProductWeightPackDevice`) — collocate GH200 **1125709**, MG **1126433**,
  UKS **1126430** / O₂ gate **1127173**, PBE H₂ **1127166**, stream
  **1128196**, DIIS/eigen **1129719**, multi-k **1130141**, multi-k device
  collocate **1130716** / integrate **1130717**, multi-k STVnl
  **1133851**, UKS-DIIS **1130402**, weight-pack **1130344**; free-fn forces = **host AO
  Pulay + device grid HF** **1127167**; free-fn stress = **CUDA density +
  host fixed-\(P\)** **1128195** H₂ dual-gate; water Γ RKS **E+F** dual-gate
  **1130195**; water **σ** + Γ UKS **F/σ** + multi-k **F/σ** GH200
  **1363408**, **◐** CPU-twin). On base also: **Si Γ ngrids=1 product-si-gate** (**1167099**),
  **device H assemble / UKS+multi-k P residency / multi-k device e_kin/e_nl /
  force-stream** (**1158949**), **sphere-window PBC product collocate + SG
  device product KS** (`product-collocate-pbc` **1173564**/`cfb246d1`). Depth
  residual (**`multigrid-si-energy-path` landed** host
  collint multi-level GH200 **1180883**; **`device-payload` landed** GH200
  **1182615**/`d476ee7a` Γ RKS final device eigen; iron-law monom restore →
  **`multigrid-device-ks`**; Si MG energy path **landed**; water σ / UKS F/σ
  / multi-k F/σ **landed** **1363408**; residual: water×PBE; device-cold-start;
  multi-k payload/χ/orth/vnl/h0; Si/water multi-k production; MG F/σ; UKS-PBE
  joints; device AO-Pulay / device-stress) —
  table CUDA for full GPW remains **◐** not ✓ (CPU-twin gates, no CP2K pin). Stage units under
  `cuda/gpw/internal/`; multi-path DIIS CPU Driver + public-shape freeze
  **green**.
  **Loop shape:** `GpwDft` is a `cpu::Driver<gpw::Phase>` for its multi-path
  DIIS routes (Γ and multi-k, RKS and UKS); the OT path is still a free loop
  rather than staged. `cpu::pw::PW` is a `cpu::Driver<pw::Phase>` covering
  restricted fixed-occupation, collinear spin and metallic smearing on typed
  stages. The capability tables above describe physics, not loop shape.
- **GPW**, Gaussian and plane waves (`cpu::gpw::GpwDft` / `GPWEvaluator`) —
  a Gaussian basis collocated on a uniform grid with GTH pseudopotentials.
  Every capability below is available for **restricted (RKS) and collinear-spin
  (UKS)**, at **Γ and on a Monkhorst–Pack k-mesh** (phased lattice sums, phased
  KB projectors, per-k DIIS), and via either diagonalization or the
  **orbital-transformation (OT)** minimizer:

    - **Functionals:** LDA (Padé), PBE, PBEsol, revPBE, RPBE, BLYP, optB88.
    - **Corrections:** Dudarev **DFT+U**, metallic occupation **smearing**
      (free-energy functional), and **multigrid** collocation.
    - **Dispersion / vdW:** D3-zero, D3(BJ), D4 — each with ATM three-body —
      TS, MBD@rsSCS (free-atom, Hirshfeld-scaled, and periodic-Ewald), XDM,
      and the nonlocal functionals vdW-DF (DRSLL), vdW-DF2 (LMKLL), rVV10 and
      optB88-vdW with a self-consistent nonlocal potential.
    - **Derivatives:** analytic ionic forces and analytic cell stress for every
      combination above, plus CELL_OPT and FD phonons.
    - **Output:** bands and DOS (including the metallic and spin cases),
      Berry-phase polarization, and maximally localized Wannier functions.
    - **Dynamics:** BOMD, CPMD (with an electronic Nosé–Hoover thermostat),
      second-generation CPMD, and XL-BOMD including its linearized form.
    - **Correlated:** RI-MP2/UMP2, SCS-MP2/SCS-UMP2, dRPA/URPA and spectral
      G0W0/UGW, each at Γ, on a k-mesh, and (RPA/GW) at finite q.

  Hybrids, meta-GGA, GAPW and PAW remain **blocked**, as does stock VV10 — no
  CP2K twin exists to pin it. The GPW *method* is CPU-only; `cuda/gpw/` holds
  device kernel footholds (collocation, integration, H assembly, DIIS/eigen,
  forces), not a second method product.

| Method                           | CPU | CUDA | Kokkos |
| -------------------------------- | :-: | :--: | :---: |
| Γ-point periodic RKS             |  ✓  |   ◐  |   ○  |
| k-point periodic RKS             |  ✓  |   ✓  |   ○  |
| Γ-point periodic UKS             |  ✓  |   ◐  |   ○  |
| k-point periodic UKS             |  ✓  |   ○  |   ○  |
| Plane-wave DFT                   |  ✓  |   ✓  |   ○  |
| Gaussian and plane-wave DFT, GPW |  ✓  |   ◐  |   ○  |
| GAPW DFT                         |  ○  |   ○  |   ○  |
| Periodic Gaussian-basis DFT      |  ✓  |   ○  |   ○  |
| Periodic LDA                     |  ✓  |   ✓  |   ○  |
| Periodic GGA                     |  ✓  |   ✓  |   ○  |
| Periodic meta-GGA                |  ○  |   ○  |   ○  |
| Periodic hybrid DFT              |  ○  |   ○  |   ○  |
| Periodic range-separated hybrid  |  ○  |   ○  |   ○  |
| Periodic DFT+U                   |  ✓  |   ○  |   ○  |
| Periodic dispersion-corrected DFT |  ✓  |  ○  |   ○  |
| Periodic MP2                     |  ✓  |   ○  |   ○  |
| Periodic RPA                     |  ✓  |   ○  |   ○  |
| Periodic GW                      |  ✓  |   ○  |   ○  |

Which engine backs which ✓:

- **Γ-point periodic RKS** is GPW (`reg_gpw-water-pbe-mt`) on CPU **and** a
  **first-slice CUDA product** (`cuda::gpw::GpwDft`; host STVnl + weight pack +
  device product collocate/KS/integrate via `KsChainDevice::run` + non-default
  stream + device RKS DIIS/eigen; PADE dual-gate `reg_cuda-gpw-full-product` /
  collocate GH200 **1125709** / PBE H₂ **1127166** / forces **1127167** /
  stress **1128195** / DIIS-eigen **1129719** / water E+F **1130195** — table
  ◐ until water σ + multi-k depth + force-stream/AO-Pulay land).
  **Γ-point periodic UKS** is GPW PADE LSD on CPU (`reg_gpw-uks-o2-gamma`)
  **and** a **first-cut CUDA energy product** (`cuda-gpw-uks-gamma` /
  GH200 **1126430** H₂ + **1127173** O₂; device `KsChainDevice::run_uks` Padé
  LSD only; table **◐** — UKS forces/stress, UKS-PBE and UKS-MG are still
  open).
  **k-point periodic RKS** is the plane-wave solver (Monkhorst–Pack mesh) on
  both CPU and CUDA, **and** GPW on CPU — the Gaussian-basis k-point path
  (phased lattice sums, KB nonlocal, per-k DIIS; `reg_gpw-kpoints-si`, diamond
  Si vs CP2K Quickstep, 2026-07-25); CUDA GPW multi-k has **H₂ energy
  foothold** (`cuda-gpw-kpoints` GH200 **1130141**; multi-k device
  collocate+integrate+DIIS+STVnl+H-residency landed GH200
  **1130716**/**1130717**/**1133110**/**1133851**/**1142864** —
  residual energy / payload / cold-start / χ / orth / vnl / h0 / Si-water F·σ) —
  Si/water production + host residual spine + F/σ still open (table PW k-RKS ✓
  is plane-wave, not GPW depth; table CUDA GPW stays **◐**).
  **k-point periodic UKS** is CPU PW collinear spin on a multi-k mesh
  (`reg_pw-uks-kmesh-fe`; O₂ twin) plus **bulk Fe UKS + metallic smearing**
  (`reg_pw-uks-fe-smearing`) **and** GPW collinear multi-k UKS
  (`reg_gpw-kpoints-uks`). **Γ UKS PBE-GGA** on GPW: `reg_gpw-uks-pbe`
  (forces: `reg_gpw-uks-forces`); **Γ-UKS BLYP**: `reg_gpw-uks-blyp`;
  **Γ-UKS revPBE**: `reg_gpw-uks-revpbe`; **Γ-UKS RPBE**: `reg_gpw-uks-rpbe`;
  **Γ-UKS PBEsol**: `reg_gpw-uks-pbesol`.
- **Periodic LDA** is provided twice: Padé-LDA in GPW, and Slater + VWN5
  (`XCType::LSDA`) on the plane-wave grid (the latter is the CUDA ✓; CPU PW
  also has LDA). **Periodic
  GGA** is GPW PBE / **PBEsol** (`reg_gpw-pbesol-si` + UKS
  `reg_gpw-uks-pbesol`) / **revPBE**
  (`reg_gpw-revpbe-si` + UKS `reg_gpw-uks-revpbe`) / **RPBE**
  (`reg_gpw-rpbe-water` + UKS `reg_gpw-uks-rpbe`) / **BLYP**
  (`reg_gpw-blyp-water`) **and** CPU PW true PBE (`reg_pw-pbe-gga-si`) +
  **true collinear spin-GGA** (`reg_pw-spin-gga`) **and** CUDA PW
  closed-shell device PBE (`cuda::pw::PW` / `reg_cuda-pw-pbe-device`) **and**
  CUDA collinear spin-PBE energy (`reg_cuda-pw-spin-pbe` — table CUDA ✓ for
  PBE energy; spin F/σ + PBE-family variants still open on device).
- **Periodic DFT+U** is GPW Dudarev PLUS_U energy (`reg_gpw-dftu-nio`) **and
  analytic stress** (`reg_gpw-dftu-stress`) **and ionic forces**
  (`reg_gpw-dftu-forces`; full-6N FD dual-gate + CP2K ENERGY_FORCE energy
  pin) on the **Γ** path **and multi-k SCF** (`reg_gpw-dftu-kpoints`)
  **and multi-k analytic stress** (`reg_gpw-dftu-kpoint-stress`) **and
  multi-k ionic forces** (`reg_gpw-dftu-kpoint-forces`; variational
  ½(V̂S+SV̂) +U scatter, full-6N FD dual gate on the k path). Multi-k
  **multigrid** (`ngrids>1`) is product (`reg_gpw-multigrid-kpoints`), and
  nonlocal vdW × multigrid is product at Γ RKS (`reg_gpw-vdw-multigrid`).
  Molecular Gaussian DFT+U remains ○.
- **Periodic dispersion-corrected DFT** is GPW + DFT-D3 zero damping,
  **D3(BJ)** (§8; `reg_gpw-pbe-d3bj-ice`), **D4** energy (`reg_gpw-d4-water`)
  **+ analytic stress** (`reg_gpw-d4-stress`) **+ ionic forces**
  (`reg_gpw-d4-forces`), **TS** (`reg_gpw-ts-ice`), free-atom **MBD@rsSCS**
  (`reg_gpw-mbd-ice`) **+ ionic forces** (Γ `reg_gpw-mbd-forces` + multi-k
  `reg_gpw-mbd-kpoint-forces`),
  **Hirshfeld-scaled MBD** (Γ `reg_gpw-mbd-hirshfeld` + multi-k energy
  `reg_gpw-hirshfeld-kpoints` + **Γ-UKS** `reg_gpw-hirshfeld-uks` +
  **multi-k UKS** `reg_gpw-hirshfeld-uks-kpoints` + Γ forces
  `reg_gpw-hirshfeld-forces` + multi-k forces
  `reg_gpw-hirshfeld-kpoint-forces`),
  **periodic Ewald MBD** (`reg_gpw-mbd-periodic`), **XDM**
  (`reg_gpw-xdm-ice` + multi-k forces `reg_gpw-xdm-kpoint-forces`),
  **vdW-DF DRSLL** (`reg_gpw-vdw-df-ice`), **vdW-DF2
  LMKLL** (`reg_gpw-vdw-df2-ice`), **rVV10** (`reg_gpw-rvv10-ice`),
  **optB88-vdW** (`reg_gpw-optb88-vdw`), and **nonlocal SCF
  potential/forces** (`reg_gpw-vdw-scf-potential`).
- **Periodic MP2** is **Γ-point RI-GPW product** (`GpwDft::mp2` with
  `mp2_ri` default true; Del Ben et al. 2012 layout;
  `reg_periodic-ri-mp2-si` vs CP2K 2026.1; Poisson foothold
  `reg_periodic-mp2-si`) **and multi-k RI-GPW** (`reg_periodic-mp2-kpoints`)
  **and Γ UMP2@UKS** (`reg_periodic-ump2-si`) **and multi-k UMP2**
  (`reg_periodic-ump2-kpoints`) **and Γ SCS-MP2** (`mp2_os_scale` /
  `mp2_ss_scale`; `reg_periodic-scs-mp2-si` vs CP2K SCALE_S/SCALE_T)
  **and multi-k SCS-MP2** (`reg_periodic-scs-mp2-kpoints`; Grimme algebraic
  dual — CP2K 2026.1 aborts MP2_GPW under DFT KPOINTS) **and Γ SCS-UMP2@UKS**
  (`reg_periodic-scs-ump2-si`; `mp2_os_scale`/`mp2_ss_scale` on product UMP2
  + CP2K UKS SCALE wall twin) **and multi-k SCS-UMP2**
  (`reg_periodic-scs-ump2-kpoints`; Grimme algebraic dual — same CP2K
  multi-k MP2_GPW abort as RKS SCS multi-k).
  **Periodic RPA** is **Γ-point dRPA@KS** product (`GpwDft::direct_rpa`;
  plasmon formula; `reg_periodic-rpa-si`), **multi-k k-diagonal dRPA**
  (`reg_periodic-rpa-kpoints`), **finite-q inter-k dRPA**
  (`reg_periodic-rpa-finite-q`), **Γ URPA@UKS** (`reg_periodic-urpa-si`),
  **and multi-k URPA** (`reg_periodic-urpa-kpoints`).
  **Periodic GW** is **Γ-point spectral G0W0@KS** product (`GpwDft::g0w0`;
  `reg_periodic-gw-si`), **multi-k k-diagonal spectral G0W0**
  (`reg_periodic-gw-kpoints`), **finite-q inter-\(k\) \(W\)**
  (`reg_periodic-gw-finite-q`), **Γ UGW@UKS** (`reg_periodic-ugw-si`),
  **multi-k UGW** (`reg_periodic-ugw-kpoints`), **and UGW@UKS-RPBE**
  spin XC (`reg_ugw-uks-rpbe`; PBE/revPBE/BLYP/RPBE UKS UGW product).
- Periodic hybrids are ○ for the same reason periodic HF is (see
  [Hartree–Fock](hartree-fock.md#2-periodic-hartreefock)): no periodic exact
  exchange.

## 11. Periodic Response and Analysis

| Method                                 | CPU | CUDA | Kokkos |
| -------------------------------------- | :-: | :--: | :---: |
| Periodic analytic forces               |  ✓  |   ◐  |   ○  |
| Periodic analytic stress               |  ✓  |   ◐  |   ○  |
| Density-functional perturbation theory |  ✓  |   ○  |   ○  |
| Finite-displacement phonons            |  ✓  |   ○  |   ○  |
| Berry-phase polarization               |  ✓  |   ○  |   ○  |
| Maximally localized Wannier functions  |  ✓  |   ○  |   ○  |

- **Periodic analytic forces** are GPW (Hellmann–Feynman + Pulay; Γ RKS,
  Γ-UKS `reg_gpw-uks-forces`, multi-k **RKS** `reg_gpw-kpoint-forces`,
  multi-k **UKS** `reg_gpw-kpoint-uks-forces`) and plane-wave **local HF +
  Ewald + KB nonlocal** (`pw_forces_analytic`). PW FD remains as oracle.
  **CUDA ◐** = plane-wave closed-shell LSDA+PBE (`cuda::pw::pw_forces_analytic`)
  **+ collinear LSDA spin forces** (`cuda-pw-spin-forces`) **+ restricted
  metallic-smear forces** (`cuda-pw-smear-forces`) **and** CUDA GPW free-fn
  first slice (host AO Pulay + device grid HF; `reg_cuda-gpw-device-forces`
  H₂; UKS/MG free-fn still reject); spin-PBE F landed; spin×smear F
  landed GH200 **1357414** (`reg_cuda-pw-spin-pbe`).
- **PW stress** is **Nielsen–Martin analytic** (`pw_stress` → Frame `kStress`;
  multi-SCF strain-FD oracle `pw_stress_fd`; `reg_pw-analytic-stress-si`).
  **CUDA ◐** = plane-wave closed-shell **LSDA + PBE** (`cuda::pw::pw_stress`;
  GH200 dual-gates `reg_cuda-pw-stress` + `reg_cuda-pw-stress-pbe`) **+
  collinear LSDA spin stress** (`cuda-pw-spin-stress`; GH200 **1135756**) **+
  restricted metallic-smear stress** (`cuda-pw-smear-stress`; Al 1-atom;
  production multi-atom residual) **and** CUDA GPW free-fn first slice
  (CUDA density + host fixed-\(P\); `reg_cuda-gpw-stress` H₂); spin-PBE σ /
  spin+smear σ still reject on PW; GPW multi-k / UKS / water σ depth is still
  open.
  **GPW stress** is **analytic fixed-\(P\)+Pulay** product at Γ **RKS and
  UKS**, **k-points for closed-shell RKS**, **multi-k UKS**
  (`reg_gpw-kpoint-uks-stress`), and **Γ DFT+U** (`reg_gpw-dftu-stress`)
  (`gpw_stress` / `GPWEvaluator::compute_stress`, `stress_use_fd=false`;
  multi-SCF oracle `gpw_stress_fd`; `reg_gpw-analytic-stress` dual-gate +
  `reg_gpw-uks-stress` + `reg_gpw-kpoint-stress` +
  `reg_gpw-stress-water` vs CP2K NUMERICAL) **and multi-k DFT+U**
  (`reg_gpw-dftu-kpoint-stress`). **GPW variable-cell relax ships**
  (evaluator re-target, stress axes mask, `StressConverged`;
  `reg_gpw-variable-cell-ice` GTH-chain + `reg_gpw-variable-cell-molopt`
  MOLOPT re-twin vs CP2K CELL_OPT — see [Optimization](optimization.md)).
  **PW variable-cell relax** ships (`reg_pw-variable-cell-si`).
- **Finite-displacement phonons** at Γ: `cpu::vib::Phonon` expands a supercell,
  runs 6N force evaluations, and diagonalizes the mass-weighted Hessian
  (`reg_fd-phonon-supercell`; SW 64-atom Si twin) **and on GPW analytic
  forces** (`reg_gpw-fd-phonon-si`; Si₂ UKS triplet dual-gate). Molecular
  `cpu::vib::Hessian` remains the non-periodic path. **Phonon dispersions
  ω(q)** productize supercell IFC Fourier on the same SW twin
  (`cpu::vib::PhononDispersion`; `reg_phonon-dispersion-si` Γ/X/L windows).
- **DFPT** ships **Γ-point (q = 0)**, **finite-q**, and **incommensurate-q**
  for the PW method: `cpu::pw::DfptPhonon` — self-consistent Sternheimer
  response (conduction CG, Hartree + LDA or **pure GGA** kernel), analytic
  Ewald/KB second derivatives (`regressions/response/dfpt-phonon-si`;
  energy-second-difference twin to 0.07 %; **finite-q at X** commensurate
  supercell path `reg_dfpt-finite-q-si`; **incommensurate** direct Sternheimer
  `reg_dfpt-incommensurate-si`; **GGA kernel** `reg_dfpt-gga-si`;
  **collinear spin** `reg_dfpt-spin-si`; **collinear spin at incommensurate
  \(q\)** `reg_dfpt-spin-incommensurate`; **metallic occupation smearing**
  `reg_dfpt-metallic-si`; **finite-q + metallic smear**
  `reg_dfpt-finite-q-smearing`; **collinear spin + metallic smear** at
  Γ `reg_dfpt-spin-smearing`; **finite-\(q\) collinear spin + metallic
  smear** `reg_dfpt-finite-q-spin-smearing`; **joint collinear spin × pure
  GGA (PBE)** `reg_dfpt-gga-spin-si` — spin-resolved Sternheimer with
  \(K_{\mathrm{xc}}^{\sigma\sigma'}\) from FD of the spin-GGA operator,
  dual-gated by a spin-PBE energy-second-difference pin **and** a CP2K
  2026.1 UKS-PBE k-point `VIBRATIONAL_ANALYSIS` physics anchor, fair wall
  ratio 0.20; **finite-q × pure GGA (PBE)** `reg_dfpt-gga-finite-q-si` —
  X commensurate supercell Γ-Sternheimer with the GGA kernel, dual-gated by
  a supercell PBE energy-second-difference pin (21.2006 vs DFPT 21.1974
  eV/Å², 0.015 %) **and** a CP2K 2026.1 supercell `VIBRATIONAL_ANALYSIS`
  X-branch anchor, fair wall ratio 0.17). NC-UPF; fixed-occupation and
  metallic-smear Sternheimer paths (Γ / commensurate / direct finite-\(q\);
  collinear spin × smear at Γ and finite-\(q\)) productized; the GGA kernel
  is now pinned at Γ (restricted + spin) and finite \(q\) (restricted +
  **joint spin × GGA** `reg_dfpt-gga-spin-finite-q-si` — X supercell
  spin-resolved Sternheimer + spin-GGA FD kernel, dual-gated by a supercell
  energy-second-difference pin (21.2006 vs DFPT 21.1911 eV/Å²) **and** a
  CP2K 2026.1 supercell UKS-PBE `VIBRATIONAL_ANALYSIS` X-branch anchor,
  fair wall ratio 0.19; M=0 identity to the restricted finite-q GGA parent).
  **Joint Γ collinear spin × pure GGA × metallic smear**
  `reg_dfpt-gga-spin-smearing` — spin-resolved Sternheimer with spin-GGA FD
  kernel + FD occupation weights, dual-gated by a free-energy second-difference
  pin (17.3454 vs DFPT 17.3711 eV/Å²) **and** a CP2K 2026.1 UKS-PBE+SMEAR
  k-point `VIBRATIONAL_ANALYSIS` physics anchor, fair wall ratio 0.21;
  M=0 gapped-FD identity to the fixed-occ spin×GGA parent; **four-way
  finite-q × collinear spin × pure GGA × metallic smear**
  `reg_dfpt-gga-spin-finite-q-smearing` — X supercell spin-resolved
  Sternheimer + spin-GGA FD kernel + FD smear weights, dual-gated by a
  free-energy second-difference pin **and** a CP2K 2026.1 UKS-PBE+SMEAR
  supercell `VIBRATIONAL_ANALYSIS` physics anchor, fair wall ratio 0.25;
  M=0 gapped-FD identity to the fixed-occ finite-q spin×GGA parent).
- **Berry-phase polarization**: King-Smith–Vanderbilt (KSV) electronic
  polarization on the PW method (`cpu::pw::BerryPolarization`; string-averaged
  Berry phases over occupied Bloch manifolds; symmetry-exact Si pins,
  `regressions/pw/berry-polarization`) **and collinear spin**
  (`reg_berry-spin-si`) **and GPW-grid KSV** (`reg_periodic-berry-gpw`;
  symmetry-exact Si pins + CP2K Quickstep `&MOMENTS PERIODIC` fair twin)
  **and collinear UKS GPW KSV** (`reg_periodic-berry-gpw-uks`; dual-channel
  string phases + CP2K Quickstep `&MOMENTS PERIODIC` UKS fair twin).
- **Maximally localized Wannier functions**: molecular **Foster–Boys** on
  closed-shell Gaussian RHF/RKS (`cpu::scf::WannierCenters`;
  `regressions/scf/wannier-centers`) **and** periodic multi-k
  **Marzari–Vanderbilt** on PW (`cpu::pw::PeriodicWannier`; bond-midpoint Si
  pin; `reg_periodic-wannier-si`) **and collinear-spin**
  (`reg_periodic-wannier-uks`) **and GPW-grid MLWF** (multi-k
  Marzari–Vanderbilt on the Gaussian-basis path;
  `reg_periodic-wannier-gpw`) **and collinear UKS GPW MLWF**
  (`reg_periodic-wannier-gpw-uks`).

## 12. Periodic infrastructure

| Capability | CPU | CUDA | Kokkos |
| ---------- | :-: | :--: | :---: |
| Monkhorst–Pack k-mesh | ✓ | ✓ |   ○  |
| Block Davidson band solver | ✓ | ✓ |   ○  |
| Norm-conserving UPF pseudopotentials (KB projectors) | ✓ | ✓ |   ○  |
| GTH pseudopotentials (local + nonlocal) | ✓ | ○ |   ○  |
| FFT Poisson solver | ✓ | ✓ |   ○  |
| Martyna–Tuckerman non-periodic Poisson | ✓ | ○ |   ○  |
| Ewald ion–ion energy | ✓ | ✓ |   ○  |
| Pulay/DIIS density mixing | ✓ | ○ |   ○  |
| Orbital-transformation (OT) minimizer | ✓ | ○ |   ○  |
| Real-space multigrid collocation | ✓ | ○ |   ○  |
| Ultrasoft / PAW | ○ | ○ |   ○  |
| Smearing for metals | ✓ | ◐ |   ○  |
| Band structure / DOS post-processing | ✓ | ○ |   ○  |

**Smearing for metals** on CPU PW: `PW::smearing` (Fermi–Dirac / Methfessel–Paxton)
+ `degauss` [Ry] + `extra_bands` (`reg_pw-smearing-al`) **and GPW** multi-k RKS
FD/MP (`reg_gpw-metallic-smearing`) **and multi-k UKS** (`reg_gpw-uks-smearing`).
**CUDA ◐** = restricted PW path only (`cuda::pw::PW` FD/Gaussian/MP +
`extra_bands` + Mermin free energy; `reg_cuda-pw-metallic-smear`) **+
restricted smear analytic ionic forces** (`reg_cuda-pw-smear-forces`; GH200
**1130427**) **+ restricted smear cell stress** (`reg_cuda-pw-smear-stress`;
GH200 **1133111** Al 1-atom; multi-atom residual still open);
spin×smear **energy + forces** landed (`reg_cuda-pw-spin-pbe`; GH200
**1357414**); spin×smear **stress** still reject; no CUDA GPW smear product.
GPW free-energy **analytic ionic forces under smear** are product for multi-k
RKS and UKS (`reg_gpw-metallic-forces` / `reg_gpw-uks-metallic-forces`;
fractional \(P_k/W_k\) payload). Multi-k free-energy **stress under smear**
is product for RKS and UKS (`reg_gpw-metallic-stress` /
`reg_gpw-uks-metallic-stress`). Metallic **BOMD** under FD/MP smear is
product (`reg_gpw-bomd-smearing`; fractional \(P_k\) warm start). GPW
**bands/DOS** product covers closed-shell fixed-occ (`reg_gpw-bands-si` /
`reg_gpw-dos-si`), collinear **UKS** bands/DOS (`reg_gpw-bands-uks` /
`reg_gpw-dos-uks`), and **metallic** FD bands/DOS
(`reg_gpw-bands-metallic` / `reg_gpw-dos-metallic`). Analytic stress folds
nonlocal \(E_{\mathrm{c}}^{\mathrm{nl}}\) under strain
(`reg_gpw-vdw-stress`). OT on
GPW is a product member (`use_ot`; RKS `reg_ot-minimizer-water` +
**Γ-UKS** `reg_ot-uks-o2` + **multi-k RKS** `reg_ot-kpoints` + **multi-k
UKS** `reg_ot-uks-kpoints` + **Γ OT+DFT+U** `reg_ot-dftu` + **multi-k
OT+DFT+U** `reg_ot-dftu-kpoints` + **OT+smear** RKS `reg_ot-smearing` +
**UKS OT+smear** `reg_ot-uks-smearing` + **multi-k UKS OT+U**
`reg_ot-uks-dftu` + **OT+U+smear** `reg_ot-dftu-smearing` + **multi-k
UKS OT+U+smear** `reg_ot-uks-dftu-smearing`). **UKS multigrid**
product collocation ships (`reg_gpw-uks-multigrid`) **and multi-k RKS
multigrid** (Bloch-phased product collocation;
`reg_gpw-multigrid-kpoints`) **and multi-k UKS multigrid**
(`reg_gpw-uks-multigrid-kpoints`).
**Band structure / DOS** on CPU PW: `cpu::pw::Bands`
(high-symmetry path) + `cpu::pw::Dos` (MP mesh + Gaussian smearing;
`reg_bands-dos-si`) **plus collinear-spin bands** (`reg_bands-spin-si`)
**and metallic occupation DOS** (`reg_dos-metallic-si`) **and metallic
occupation bands** (`reg_bands-metallic-si`) **and collinear-spin DOS**
(`reg_dos-spin-si`) **and collinear spin + metallic smear bands**
(`reg_bands-spin-metallic-si`) **and collinear spin + metallic smear DOS**
(`reg_dos-spin-metallic-si`). **GPW-grid bands/DOS** (Bloch
\(C_k/\varepsilon_k\) path) are product for closed-shell fixed-occ
(`reg_gpw-bands-si` / `reg_gpw-dos-si`; CP2K Quickstep BAND_STRUCTURE /
DOS twins) **plus collinear UKS bands/DOS** (`reg_gpw-bands-uks` /
`reg_gpw-dos-uks`) **and metallic FD bands/DOS**
(`reg_gpw-bands-metallic` / `reg_gpw-dos-metallic`).

On CUDA the plane-wave hot path (Davidson H·ψ, cuFFT, Poisson, V_eff, LDA,
density accumulation) is device-resident; Rayleigh–Ritz subspace bookkeeping and
the SCF control loop stay on the host.
