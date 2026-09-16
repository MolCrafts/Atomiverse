# Excited States and Many-Body Methods

Legend and caveats: [Roadmap index](index.md).

**Ground-state engine with closed- and open-shell response footholds.**
`RHF::tda` (CIS / TDA) and `RHF::tdhf` (full RPA) return excitation energies and
length-gauge transition dipoles on a converged RHF reference
(`regressions/scf/ch2o_tdhf`). **`RKS::tda`** on a converged KS reference uses
the **full MO-product LSDA ALDA** kernel (`reg_h2_full_grid_alda`;
`ExcitedStatesResult::alda`) and **hybrid TDDFT-TDA** with the exact-exchange
kernel on B3LYP/PBE0 (`reg_h2-b3lyp-hybrid-tda`). **`UKS::tda`** is spin-conserving
open-shell TDA on pure LSDA (`reg_uks-tda-o2-lsda`). **Full linear-response
TDDFT** (`RKS::tddft`, Casida A/B) ships for pure KS (PBE;
`reg_full-lr-tddft-h2-pbe`) and **hybrid B3LYP** (`reg_h2-b3lyp-rpa-tddft`).
**Collinear spin-flip TDA** on UHF ships (`UHF::sf_tda`; `reg_spin-flip-tda-o2`).
**Analytic CIS/TDA forces** (default) and **true NAC vectors** ship
(`RHF::tda_forces` / `tda_nac_vectors` / `tda_nac_strength`;
`reg_h2_analytic_tda_force`, `reg_h2_nac_vectors`); FD forces remain as
`analytic=false`. **CIS(D)** second-order doubles correction on RHF TDA ships
(`RHF::cisd`; `reg_cisd-h2co-sto3g` H2CO/STO-3G S1 vs PySCF).
**Strict ADC(2)** electronic excitation energies on RHF ship
(`RHF::adc2`; `reg_adc2-h2co-sto3g` H2CO/STO-3G S1 vs PySCF).
**Spectral G0W0** quasiparticle energies on RHF ship
(`RHF::g0w0`; `reg_g0w0-h2-sto3g` H2/STO-3G QP gap vs PySCF
`gw.gw_exact.GWExact` + dRPA).
**ΔSCF / MOM / IMOM** non-Aufbau re-SCF ships on UHF
(`UHF::delta_scf`; `MomStyle::{MOM,IMOM}`; `reg_delta-scf-n2` N₂/STO-3G
α HOMO→LUMO vs PySCF `scf.addons.mom_occ_`).
**EOM-EE-CCSD** singlet excitation energies ship on closed-shell RHF+CCSD
(`RHF::eom_ee_ccsd`; `reg_eom-ee-ccsd` H₂/STO-3G S1 vs PySCF
`cc.CCSD(...).eomee_ccsd_singlet`).
**EOM-CC2** singlet roots ship on closed-shell RHF+CC2
(`RHF::eom_cc2`; `reg_eom-cc2-h2` vs PySCF `RCCSD(cc2=True).eomee_ccsd_singlet`).
**EOM-IP-CCSD** ionization roots ship (`RHF::eom_ip_ccsd`; `reg_eom-ip-ccsd`
H₂O/STO-3G IP0 vs PySCF `cc.CCSD(...).ipccsd`).
**EOM-EA-CCSD** attachment roots ship (`RHF::eom_ea_ccsd`; `reg_eom-ea-ccsd`).
**EOM-SF-CCSD** spin-flip roots ship on UHF+UCCSD (`UHF::eom_sf_ccsd`;
`reg_eom-sf-ccsd-o2`).
**ADC(2)-x** EE and **IP/EA-ADC(2)** ship (`RHF::adc2x` / `ip_adc2` / `ea_adc2`;
`reg_adc2x-h2co-sto3g`, `reg_ip-adc2-h2o-sto3g`, `reg_ea-adc2-h2-sto3g`).
**Collinear spin-flip full TDDFT** (SF-RPA) ships on UHF (`UHF::sf_tddft`;
`reg_sf-tddft-o2`).
**UKS full-LR TDDFT** (Casida A/B, pure LSDA) ships (`UKS::tddft`;
`reg_uks-full-lr-o2-lsda` O₂ triplet vs PySCF `tdscf.TDDFT(UKS)`).
**Direct RPA** correlation, **BSE@G0W0**, and **one-shot GF2** ship on RHF
(`RHF::direct_rpa` / `bse` / `gf2`; `reg_direct-rpa-h2-sto3g`,
`reg_bse-h2-sto3g`, `reg_gf2-h2-sto3g`).

The same orbital-Hessian / A,B machinery also feeds Seeger–Pople stability
analysis on the ground-state HF path (see [Hartree–Fock](hartree-fock.md)).

## 1. SCF and Linear-Response Methods

| Method                             | CPU | CUDA | Kokkos |
| ---------------------------------- | :-: | :--: | :---: |
| Delta-SCF excited states           |  ✓  |   ○  |   ○  |
| MOM excited states                 |  ✓  |   ○  |   ○  |
| IMOM excited states                |  ✓  |   ○  |   ○  |
| ROKS excited states                |  ○  |   ○  |   ○  |
| CIS excitation energies            |  ✓  |   ○  |   ○  |
| CIS(D)                             |  ✓  |   ○  |   ○  |
| SCS-CIS(D)                         |  ○  |   ○  |   ○  |
| SOS-CIS(D)                         |  ○  |   ○  |   ○  |
| Time-dependent Hartree–Fock, TDHF  |  ✓  |   ○  |   ○  |
| Tamm–Dancoff TDDFT                 |  ✓  |   ○  |   ○  |
| Full linear-response TDDFT         |  ✓  |   ○  |   ○  |
| Spin-flip TDA                      |  ✓  |   ○  |   ○  |
| Spin-flip TDDFT                    |  ✓  |   ○  |   ○  |
| Restricted excitation-window TDDFT |  ○  |   ○  |   ○  |
| Core–valence-separation TDDFT      |  ○  |   ○  |   ○  |
| Real-time TDDFT                    |  ○  |   ○  |   ○  |
| TDDFT analytic gradient            |  ✓  |   ○  |   ○  |
| TDDFT nonadiabatic coupling        |  ✓  |   ○  |   ○  |

TDA-TDDFT is the LSDA ALDA path and the global-hybrid exact-exchange path on
`RKS::tda` (B3LYP/PBE0 singlet), plus **open-shell UKS-TDA** spin-conserving
response on pure LSDA (`UKS::tda`; `reg_uks-tda-o2-lsda` O₂ triplet) and
**collinear spin-flip TDA** on UHF (`UHF::sf_tda`; `reg_spin-flip-tda-o2`
O₂ triplet, exchange-only SF-CIS) and **collinear spin-flip full TDDFT**
(SF-RPA; `UHF::sf_tddft`; `reg_sf-tddft-o2`). Full RPA-style Casida A/B is
`RKS::tddft` for pure KS (`reg_full-lr-tddft-h2-pbe`) and **global hybrid**
B3LYP (`reg_h2-b3lyp-rpa-tddft`), plus **open-shell UKS full-LR** on pure LSDA
(`UKS::tddft`; `reg_uks-full-lr-o2-lsda`). Analytic gradient and NAC rows are
the CIS/TDA members on `RHF` (analytic default); full KS TDDFT gradients
beyond CIS@RKS are not a separate product path yet.

## 2. Algebraic-Diagrammatic Construction

| Method                | CPU | CUDA | Kokkos |
| --------------------- | :-: | :--: | :---: |
| ADC(1)                |  ○  |   ○  |   ○  |
| ADC(2)                |  ✓  |   ○  |   ○  |
| ADC(2)-x              |  ✓  |   ○  |   ○  |
| ADC(3)                |  ○  |   ○  |   ○  |
| CVS-ADC(2)            |  ○  |   ○  |   ○  |
| CVS-ADC(2)-x          |  ○  |   ○  |   ○  |
| IP-ADC(2)             |  ✓  |   ○  |   ○  |
| EA-ADC(2)             |  ✓  |   ○  |   ○  |
| ADC analytic gradient |  ○  |   ○  |   ○  |

## 3. Equation-of-Motion Coupled Cluster

| Method                     | CPU | CUDA | Kokkos |
| -------------------------- | :-: | :--: | :---: |
| EOM-EE-CCSD                |  ✓  |   ○  |   ○  |
| EOM-IP-CCSD                |  ✓  |   ○  |   ○  |
| EOM-EA-CCSD                |  ✓  |   ○  |   ○  |
| EOM-SF-CCSD                |  ✓  |   ○  |   ○  |
| EOM-DIP-CCSD               |  ○  |   ○  |   ○  |
| EOM-DEA-CCSD               |  ○  |   ○  |   ○  |
| CVS-EOM-CCSD               |  ○  |   ○  |   ○  |
| EOM-CC2                    |  ✓  |   ○  |   ○  |
| EOM-MP2                    |  ○  |   ○  |   ○  |
| STEOM-CCSD                 |  ○  |   ○  |   ○  |
| EOM-CCSD analytic gradient |  ○  |   ○  |   ○  |

**EOM-EE-CCSD** (`RHF::eom_ee_ccsd`) builds the spin-orbital EE Jacobian on a
converged closed-shell CCSD reference and returns singlet roots
(`reg_eom-ee-ccsd` vs PySCF). **EOM-CC2** (`RHF::eom_cc2`) is the same EE
Jacobian at Christiansen–Koch–Jørgensen CC2 amplitudes
(`reg_eom-cc2-h2` vs PySCF `RCCSD(cc2=True).eomee_ccsd_singlet`).
**EOM-IP-CCSD** (`RHF::eom_ip_ccsd`; `reg_eom-ip-ccsd`) returns ionization
roots on the same closed-shell CCSD reference. **EOM-EA-CCSD**
(`RHF::eom_ea_ccsd`; `reg_eom-ea-ccsd`) and **EOM-SF-CCSD**
(`UHF::eom_sf_ccsd`; `reg_eom-sf-ccsd-o2`) ship. DIP/DEA, CVS, STEOM, and EOM
gradients remain open. See also
[Post-Hartree–Fock](post-hf.md#3-coupled-cluster) for ground-state CCSD /
CCSD(T) / CC2.

## 4. Simplified Excited-State Methods

| Method                       | CPU | CUDA | Kokkos |
| ---------------------------- | :-: | :--: | :---: |
| Simplified TDA, sTDA         |  ○  |   ○  |   ○  |
| Simplified TDDFT, sTDDFT     |  ○  |   ○  |   ○  |
| sTDA-xTB                     |  ○  |   ○  |   ○  |
| TD-DFTB                      |  ✓  |   ○  |   ○  |
| Long-range-corrected TD-DFTB |  ○  |   ○  |   ○  |

**TD-DFTB** ships as `cpu::semiempirical::DFTB2::td_casida` — the tight-binding
Casida linear response of Niehaus, Suhai, Della Sala, Lugli, Elstner, Seifert,
Frauenheim, *Phys. Rev. B* **63**, 085108 (2001): Mulliken transition charges
contract γ_AB (singlet) or the on-site spin constants W_A (triplet) into
ΩF = ω²F. **Energies only** — no oscillator strengths, no transition dipoles,
no excited-state gradients or NAC. Restricted closed shell, molecular Γ point,
MioSketch Slater–Koster, dense solver (≤ 512 pairs). It carries **no absolute
excitation-energy oracle** (no `dftb+` binary on the pin host; no published
H₂/mio-1-1 value exists) and is gated on structural limits — see
`regressions/semiempirical/td-dftb-h2/REFERENCE.md`.

The remaining rows are blocked upstream: no xTB response, no long-range
correction (see [Semiempirical and Tight Binding](semiempirical.md)).

## 5. Green's-Function and Many-Body Methods

| Method                                        | CPU | CUDA | Kokkos |
| --------------------------------------------- | :-: | :--: | :---: |
| Second-order Green's function, GF2            |  ✓  |   ○  |   ○  |
| Self-consistent GF2                           |  ○  |   ○  |   ○  |
| G0W0                                          |  ✓  |   ○  |   ○  |
| Eigenvalue-self-consistent GW                 |  ○  |   ○  |   ○  |
| Quasiparticle self-consistent GW              |  ○  |   ○  |   ○  |
| Fully self-consistent GW                      |  ○  |   ○  |   ○  |
| Bethe–Salpeter equation, BSE                  |  ✓  |   ○  |   ○  |
| Direct random-phase approximation             |  ✓  |   ○  |   ○  |
| RPA with exchange                             |  ○  |   ○  |   ○  |
| SOSEX                                         |  ○  |   ○  |   ○  |
| Renormalized second-order perturbation theory |  ○  |   ○  |   ○  |
| ACFDT-RPA                                     |  ○  |   ○  |   ○  |

**G0W0** (`RHF::g0w0`) is spectral (frequency-exact) quasiparticle energies with
direct-RPA screening of \(W\) on a closed-shell RHF reference
(`reg_g0w0-h2-sto3g` vs PySCF `gw.gw_exact.GWExact` + dRPA). **Direct RPA**
correlation energy ships standalone (`RHF::direct_rpa`;
`reg_direct-rpa-h2-sto3g`). **BSE@G0W0** singlet roots ship (`RHF::bse`;
`reg_bse-h2-sto3g`). **One-shot GF2** (G0F2) QP gaps ship (`RHF::gf2`;
`reg_gf2-h2-sto3g`). No self-consistent GW/AGF2, no RPAX/SOSEX product path.
