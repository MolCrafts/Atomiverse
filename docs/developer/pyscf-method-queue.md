# PySCF method queue

Durable worklist for workflow `pyscf-method-loop`. Do not delete done rows.
The census phase rewrites **pending** from `docs/roadmap/` ○ rows that PySCF
can oracle offline; **done** / **blocked** history is preserved across runs.

Census date: 2026-07-24 (harness wave 3 post re-census merge). Roadmap CPU ○ ∩
PySCF oracle; excludes classical MD / FF / sampling / CUDA-Kokkos / ML /
PW-GPW-without-Gaussian-twin / parked hf-parity product-depth / SE tables
without PySCF twin / meta-GGA (no τ) / range-separated (no erf ERI) /
CASPT2 (no stock PySCF). **Wave 1 merged (12):** eom-ea-ccsd, adc2x-h2co-sto3g,
ip-adc2-h2o-sto3g, ea-adc2-h2-sto3g, sf-tddft-o2, mp4-h2-sto3g, cc2-h2-sto3g,
direct-rpa-h2-sto3g, bse-h2-sto3g, dft-d4-h2o-pbe, eom-sf-ccsd-o2, gf2-h2-sto3g.
**Wave 2 merged (7):** eom-cc2-h2, eom-ip-ccsd, uks-full-lr-o2-lsda,
casscf-h2-sto3g, ccsd-grad-h2o-sto3g, rks-hessian-h2o-sto3g, ump2-grad-o2-sto3g.
**Wave 3 merged (2):** nevpt2-h2-sto3g, uks-hessian-o2-sto3g.
**pending:** empty after wave 3. Historical blocked list unchanged.

## pending

_(empty — wave 3 closed all assignable rows)_

## in_progress

_(empty)_

## blocked

PySCF *can* pin these; Atomiverse is missing a prerequisite engine (not a
missing method flag). Dropped from this queue (not PySCF twins): `td-dftb`,
`gfn2-xtb`, `pm6-pm7`, `caspt2` — revive under DFTB+/xTB/MOPAC/OpenMolcas if
needed.

Deferred specs (approved, implement later — not auto-impl):
`.claude/specs/periodic-rhf-gamma/`.

- [x] `cam-b3lyp` | done — `rys_eri_erf_block` + `XCType::CAMB3LYP` + RSH Fock mix; `regressions/scf/cam-b3lyp-h2-sto3g` (composed PySCF 2.14.0 pin, |dE|=2.5e-11 Ha)
- [!] `periodic-rhf-gamma` | blocked — no periodic exact exchange; PW/GPW are semi-local DFT only; PySCF `pbc.scf` can oracle once four-index PBC HF-X exists — **spec:** `.claude/specs/periodic-rhf-gamma/`

## done

- [x] `cam-b3lyp` | CAM-B3LYP RSH | regressions/scf/cam-b3lyp-h2-sto3g | H2/STO-3G cart; composed PySCF 2.14.0 E=-1.3067931936345114 Ha; `RKS{CAMB3LYP}` + erf K; tol 1e-6 Ha
- [x] `nevpt2-h2-sto3g` | SC-NEVPT2 | regressions/scf/nevpt2-h2-sto3g | H2 CASSCF(2,2)/STO-3G cart SC-NEVPT2; PySCF 2.14.0 mrpt.NEVPT e_corr=0 (ncore=nvirt=0); `RHF::nevpt2`; tol 1e-6 Ha; ratio 0.072; tutorial: methods/nevpt2-h2-sto3g.md
- [x] `uks-hessian-o2-sto3g` | UKS nuclear Hessian | regressions/scf/uks-hessian-o2-sto3g | O2 UKS-LSDA/STO-3G FD-of-forces Hessian; PySCF 2.14.0 FD twin H_zz~128.46 eV/Å²; `UKS::hessian`; tol 2e-2 eV/Å²; ratio 2.13 (vs FD twin); tutorial: methods/uks-hessian-o2-sto3g.md
- [x] `eom-cc2-h2` | EOM-CC2 | regressions/scf/eom-cc2-h2 | H2/STO-3G cart EOM-CC2; PySCF 2.14.0 S1=26.164970973312 eV; `RHF::eom_cc2`; tol 1e-4 eV; ratio 0.300
- [x] `eom-ip-ccsd` | EOM-IP-CCSD | regressions/scf/eom-ip-ccsd | H2O/STO-3G cart EOM-IP; PySCF 2.14.0 IP0=8.415727034062 eV; `RHF::eom_ip_ccsd`; tol 1e-4 eV; ratio 0.754
- [x] `uks-full-lr-o2-lsda` | UKS full-LR TDDFT | regressions/scf/uks-full-lr-o2-lsda | O2 UKS-LSDA/STO-3G full RPA; PySCF 2.14.0 roots 7.6209/7.6424/7.6424/8.3863/8.3863 eV; `UKS::tddft`; tol 1e-4 eV; ratio 2.937
- [x] `casscf-h2-sto3g` | CASSCF | regressions/scf/casscf-h2-sto3g | H2 CASSCF(2,2)/STO-3G cart; PySCF 2.14.0 e_tot=-1.137283834489 Ha; `RHF::casscf`; tol 1e-6 Ha; ratio 0.0435
- [x] `ccsd-grad-h2o-sto3g` | CCSD nuclear gradient | regressions/scf/ccsd-grad-h2o-sto3g | H2O/STO-3G cart CCSD forces; PySCF 2.14.0 analytic g_O,z=-1.10696956e-01 Ha/Bohr; `RHF::ccsd_forces` (FD of CCSD energy); tol 1e-5 Ha/Bohr; ratio 1.665
- [x] `rks-hessian-h2o-sto3g` | RKS nuclear Hessian | regressions/scf/rks-hessian-h2o-sto3g | H2O RKS-LSDA/STO-3G FD-of-forces Hessian; PySCF 2.14.0 FD twin H_yy~89.53 eV/Å²; `RKS::hessian`; tol 2e-2 eV/Å²; ratio 3.08 (vs FD twin)
- [x] `ump2-grad-o2-sto3g` | UMP2 analytic gradient | regressions/scf/ump2-grad-o2-sto3g | O2 UHF→UMP2/STO-3G cart; PySCF 2.14.0 g_z=±1.66513266e-01 Ha/Bohr; `UHF::mp2_forces`; tol 1e-5 Ha/Bohr; ratio 1.37
- [x] `eom-ea-ccsd` | EOM-EA-CCSD | regressions/scf/eom-ea-ccsd | H2/STO-3G cart EOM-EA; PySCF 2.14.0 EA0=18.821245619413 eV; `RHF::eom_ea_ccsd`; tol 1e-4 eV; ratio 0.068
- [x] `adc2x-h2co-sto3g` | ADC(2)-x | regressions/scf/adc2x-h2co-sto3g | H2CO/STO-3G cart ADC(2)-x EE; PySCF 2.14.0 S1=3.318442496747 eV; `RHF::adc2x`; tol 1e-4 eV; ratio 0.240
- [x] `ip-adc2-h2o-sto3g` | IP-ADC(2) | regressions/scf/ip-adc2-h2o-sto3g | H2O/STO-3G cart IP-ADC(2); PySCF 2.14.0 IP1=8.277183500364 eV; `RHF::ip_adc2`; tol 1e-4 eV; ratio 0.191
- [x] `ea-adc2-h2-sto3g` | EA-ADC(2) | regressions/scf/ea-adc2-h2-sto3g | H2/STO-3G cart EA-ADC(2); PySCF 2.14.0 EA0=18.620249979933 eV; `RHF::ea_adc2`; tol 1e-4 eV; ratio 0.213
- [x] `sf-tddft-o2` | Spin-flip TDDFT | regressions/scf/sf-tddft-o2 | O2 UHF/STO-3G collinear SF-RPA; offline twin roots 4.3347/7.1554/7.1554/8.1814/8.1814 eV; `UHF::sf_tddft`; tol 1e-4 eV; ratio 9.51
- [x] `mp4-h2-sto3g` | MP4(SDTQ) | regressions/scf/mp4-h2-sto3g | H2/STO-3G cart RMP4; offline det-space RSPT-4 pin e_corr=-0.019685225019 Ha; `RHF::mp4`; tol 1e-6 Ha; ratio 0.031
- [x] `cc2-h2-sto3g` | CC2 | regressions/scf/cc2-h2-sto3g | H2/STO-3G cart CC2; PySCF RCCSD(cc2=True) e_corr=-0.013138073590 Ha; `RHF::cc2`; tol 1e-6 Ha; ratio 0.392
- [x] `direct-rpa-h2-sto3g` | Direct RPA | regressions/scf/direct-rpa-h2-sto3g | H2 RHF/STO-3G plasmon dRPA; e_corr=-0.020633073749 Ha (exact-ERI twin); `RHF::direct_rpa`; tol 1e-6 Ha; ratio 0.760
- [x] `bse-h2-sto3g` | Bethe–Salpeter BSE | regressions/scf/bse-h2-sto3g | H2 BSE@G0W0/STO-3G S1=25.956271542931 eV; `RHF::bse`; tol 1e-4 eV; ratio 0.060
- [x] `dft-d4-h2o-pbe` | DFT-D4 | regressions/scf/dft-d4-h2o-pbe | H2O molecular PBE-D4 E_disp=-0.000195303719 Ha (PySCF 2.14.0+pyscf-dispersion); `cpu::scf::D4` + `MeanFieldMethod::d4`; tol 1e-6 Ha; ratio 2.17
- [x] `eom-sf-ccsd-o2` | EOM-SF-CCSD | regressions/scf/eom-sf-ccsd-o2 | O2 UHF/UCCSD/STO-3G SF; PySCF 2.14.0 ω0..4=0.00415/1.0240/1.0240/1.5864/6.1844 eV; `UHF::eom_sf_ccsd`; tol 1e-4 eV; ratio 3.12
- [x] `gf2-h2-sto3g` | GF2 | regressions/scf/gf2-h2-sto3g | H2 RHF/STO-3G one-shot G0F2; QP gap=34.717288266166 eV (PySCF 2.14.0 AGF2 twin); `RHF::gf2`; tol 1e-4 eV; ratio 0.071
- [x] `eom-ee-ccsd` | EOM-EE-CCSD | regressions/scf/eom-ee-ccsd | H2/STO-3G cart singlet EOM-EE; PySCF 2.14.0 S1=26.365966612792 eV; `RHF::eom_ee_ccsd`; tol 1e-4 eV; ratio 0.193
- [x] `delta-scf-n2` | Delta-SCF | regressions/scf/delta-scf-n2 | N2/STO-3G α HOMO→LUMO UHF-IMOM; PySCF 2.14.0 mom_occ_ ΔE=8.24694070 eV; `UHF::delta_scf` + `MomStyle::IMOM`; tol 1e-4 eV; ratio 7.83
- [x] `casci-h2-sto3g` | CASCI | regressions/scf/casci-h2-sto3g | H2 CASCI(2,2)/STO-3G cart; PySCF 2.14.0 e_tot=-1.137283834489 Ha; `RHF::casci(frame,2,2)`; tol 1e-6 Ha; ratio 0.076
- [x] `adc2-h2co-sto3g` | ADC(2) | regressions/scf/adc2-h2co-sto3g | H2CO/STO-3G cart strict ADC(2) EE; PySCF 2.14.0 S1=4.535317180049 eV; `RHF::adc2`; tol 1e-4 eV; ratio 0.267
- [x] `mp3-h2-sto3g` | MP3 | regressions/scf/mp3-h2-sto3g | H2/STO-3G cart RMP3; offline pin e_corr=-0.017974146227 Ha (Szabo–Ostlund on PySCF RHF orbs); `RHF::mp3`; tol 1e-6 Ha; ratio 0.774
- [x] `ccsd-t-h2o-sto3g` | CCSD(T) | regressions/scf/ccsd-t-h2o-sto3g | H2O/STO-3G cart CCSD+(T); PySCF 2.14.0 e_corr=-0.049505972715 Ha, e_t=-6.740968e-05 Ha; `RHF::ccsd_t`; tol 1e-6 Ha; ratio 0.333
- [x] `ccsd-h2-sto3g` | CCSD | regressions/scf/ccsd-h2-sto3g | H2/STO-3G cart CCSD; PySCF 2.14.0 e_corr=-0.020524527092 Ha, e_tot=-1.137283834489 Ha; `RHF::ccsd`; tol 1e-6 Ha; ratio 0.317
- [x] `g0w0-h2-sto3g` | G0W0 | regressions/scf/g0w0-h2-sto3g | H2 RHF/STO-3G spectral G0W0 QP gap=35.007583892680 eV (PySCF 2.14.0 GWExact+dRPA); `RHF::g0w0`; tol 1e-4 eV; ratio 0.109
- [x] `b2plyp-h2o` | B2PLYP double hybrid | regressions/scf/b2plyp-h2o | H2O/STO-3G cart; PySCF 2.14.0 composed E_KS+a_c·MP2 e_tot=-75.210908483134 Ha; `XCType::B2PLYP` a_x=0.53 a_c=0.27 + `RKS::mp2`; tol 1e-6 Ha; ratio 1.75
- [x] `romp2-o2-sto3g` | ROMP2 | regressions/scf/romp2-o2-sto3g | O2 triplet ROHF→ROMP2/STO-3G cart; PySCF 2.14.0 UMP2(ROHF) e_corr=-0.096718716392 Ha, e_tot=-147.728885707074 Ha; `ROHF::mp2`; tol 1e-6 Ha; ratio 1.37
- [x] `roks-grad-o2-sto3g` | ROKS analytic gradient | regressions/scf/roks-grad-o2-sto3g | O2 ROKS-LSDA/STO-3G cart forces; PySCF 2.14.0 g_z=±0.162926534046 Ha/Bohr; `ROKS::forces` spin-Fock W; tol 2e-5 Ha/Bohr; ratio 1.117
- [x] `h2-b3lyp-rpa-tddft` | Hybrid full-LR TDDFT (B3LYP RPA) | regressions/scf/h2-b3lyp-rpa-tddft | H2 RKS-B3LYP/STO-3G Casida A/B; PySCF 2.14.0 S1=25.307797345333 eV; `RKS::tddft`; tol 1e-4 eV; ratio 0.430
- [x] `spin-flip-tda-o2` | Spin-flip TDA | regressions/scf/spin-flip-tda-o2 | O2 UHF/STO-3G collinear SF-CIS; offline pin roots 0.0451/1.5677/1.5677/3.0384/7.8486 eV; `UHF::sf_tda`; tol 1e-4 eV; ratio 9.56
- [x] `rohf-hessian-o2` | ROHF nuclear Hessian | regressions/scf/rohf-hessian-o2 | O2 ROHF/STO-3G FD of analytic forces; stretch H_zz~135.57 eV/Å²; `ROHF::hessian`; tol 1e-5 eV/Å²; perf_exempt (no PySCF hessian.ROHF)
- [x] `mp2-grad-h2o-sto3g` | MP2 analytic gradient | regressions/scf/mp2-grad-h2o-sto3g | H2O/STO-3G cart RMP2; PySCF 2.14.0 g_O,z=-9.528798e-02 Ha/Bohr; `RHF::mp2_forces`; tol 1e-5 Ha/Bohr
- [x] `sos-mp2-h2o-sto3g` | SOS-MP2 | regressions/scf/sos-mp2-h2o-sto3g | H2O/STO-3G cart Jung c_OS=1.3 c_SS=0; PySCF 2.14.0 e_corr=-0.043570566111 Ha, e_tot=-75.006593704574 Ha; `mp2_os_scale`/`mp2_ss_scale` members; tol 1e-6 Ha
- [x] `uks-tda-o2-lsda` | UKS TDA | regressions/scf/uks-tda-o2-lsda | O2 UKS-LSDA/STO-3G cart spin-conserving TDA; PySCF 2.14.0 roots 7.6209/7.6424/7.6425/8.4033/8.4033 eV; `UKS::tda`; tol 1e-4 eV
- [x] `d3bj-molecular-pbe` | DFT-D3 BJ damping | regressions/scf/d3bj-molecular-pbe | H2O molecular PBE-D3(BJ) E_disp=-0.000359468656 Ha (PySCF 2.14.0+pyscf-dispersion); `cpu::scf::D3BJ` + `MeanFieldMethod::d3bj`; tol 1e-6 Ha
- [x] `uhf-hessian-o2` | UHF analytic Hessian | regressions/scf/uhf-hessian-o2 | O2 UHF/STO-3G cart CPHF Hessian; PySCF 2.14.0 `hessian.UHF`; `UHF::hessian`; tol 1e-5 eV/Å²
- [x] `scs-mp2-h2o-sto3g` | SCS-MP2 | regressions/scf/scs-mp2-h2o-sto3g | H2O/STO-3G cart Grimme c_OS=1.2 c_SS=1/3; PySCF 2.14.0 e_corr=-0.040895594623 Ha, e_tot=-75.003918733086 Ha; `mp2_os_scale`/`mp2_ss_scale` members; tol 1e-6 Ha
- [x] `full-lr-tddft-h2-pbe` | Full linear-response TDDFT | regressions/scf/full-lr-tddft-h2-pbe | RKS::tddft Casida A/B PBE; S1=25.519450223124 eV (PySCF 2.14.0)
- [x] `roks-o2-sto3g` | ROKS | regressions/scf/roks-o2-sto3g | O2 triplet ROKS/LSDA/STO-3G cart e_tot=-147.193715448518 Ha (PySCF 2.14.0 dft.ROKS xc=LDA,VWN); tol 1e-6 Ha; spin purity ⟨S²⟩=2
- [x] `static-polariz-h2o` | RHF static polarizability (CPHF) | regressions/scf/static-polariz-h2o | H2O/STO-3G α_ii=(0.04190863, 5.20973255, 2.12715820) a.u. (PySCF 2.14.0 CPHF/FD); `RHF::polarizability`; tol 1e-5 a.u.
- [x] `ief-pcm-h2o` | IEF-PCM | regressions/scf/ief-pcm-h2o | H2O RHF/6-31G* cart ε=78.4 Bondi×1.2 Lebedev 302; PySCF 2.14.0 e_solv=-0.01265803490853 Ha; `solvent_model=IEFPCM`; tol 1e-4 Ha (hard bury-delete vs SWIG)
- [x] `pcm-grad-h2o` | Implicit-solvent analytic gradient | regressions/scf/pcm-grad-h2o | FD of free energy ≤1e-5 Ha/Bohr; PySCF pin in REFERENCE
- [x] `ghf-grad-h3-sto3g` | GHF analytic gradient | regressions/scf/ghf-grad-h3-sto3g | noncollinear H3 forces 1e-5 Ha/Bohr
- [x] `blyp-water-sto3g` | BLYP GGA | regressions/scf/blyp-water-sto3g | H2O/STO-3G cart e_tot=-75.277447482665 Ha (PySCF 2.14.0 xc=blyp); tol 1e-6 Ha
- [x] `cisd-h2co-sto3g` | CIS(D) | regressions/scf/cisd-h2co-sto3g | H2CO/STO-3G S1 CIS(D)=4.54457685 eV (PySCF+Head-Gordon U+V); tol 1e-4 eV
- [x] `frozen-core-mp2-h2o` | Frozen-core RMP2 | regressions/scf/frozen-core-mp2-h2o | H2O/6-31G* cart frozen=1 (O 1s); PySCF 2.14.0 e_corr=-0.186119540118 Ha, e_tot=-76.196624528375 Ha; `rhf.frozen=1`; tol 1e-6 Ha
- [x] `rhf-analytic-hessian-h2o` | RHF analytic Hessian (CPHF) | regressions/scf/h2o_rhf_freq | PySCF analytic Hessian freq ≤1 cm⁻¹
- [x] `lowdin-mk-chelpg-h2o` | Löwdin + MK + CHELPG charges | regressions/scf/h2o_charges_pyscf | PySCF 2.14.0 pins in REFERENCE.md
- [x] `rhf-dipole-h2o` | RHF dipole | regressions/scf/h2o_rhf_dipole | PySCF dipole pin
- [x] `rmp2-h2o-sto3g` | RMP2 | regressions/scf/h2o_mp2_sto3g | PySCF mp.MP2
- [x] `ri-mp2-h2o` | RI-MP2 | regressions/scf/h2o_ri_mp2 | density-fitted MP2
- [x] `ump2-o2-sto3g` | UMP2 | regressions/scf/ump2-o2-sto3g | UMP2/O2/STO-3G triplet cart conv_tol=1e-12 PySCF 2.14.0: e_corr=-0.096393144130 Ha, e_tot=-147.730339929580 Ha (UHF e_hf=-147.633946785450 Ha); tol 1e-6 Ha
- [x] `rhf-tda-tdhf-ch2o` | RHF TDA/TDHF | regressions/scf/ch2o_tdhf | CIS/TDHF vs PySCF
- [x] `rks-tda-alda` | RKS TDA full-grid ALDA | regressions/scf/h2_full_grid_alda | LSDA ALDA kernel
- [x] `hybrid-tda-b3lyp-h2` | Hybrid TDDFT-TDA | regressions/scf/h2-b3lyp-hybrid-tda | B3LYP exact-exchange kernel; S1=25.928364822972 eV (PySCF 2.14.0)
- [x] `analytic-tda-force-h2` | CIS/TDA analytic forces | regressions/scf/h2_analytic_tda_force | analytic default
- [x] `tda-nac-h2` | TDA NAC vectors | regressions/scf/h2_nac_vectors | true NAC
- [x] `rohf-o2-energy-grad` | ROHF energy + gradient | regressions/scf/o2_rohf_sto3g, o2_rohf_grad | PySCF ROHF
- [x] `ghf-h3-energy` | GHF energy | regressions/scf/h3_ghf_sto3g | energy-only
- [x] `cpcm-h2o` | C-PCM/COSMO energy | regressions/scf/h2o_rhf_631gs_cpcm | conductor-like solvent
