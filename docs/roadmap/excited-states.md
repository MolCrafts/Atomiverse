# Excited States and Many-Body Methods

Tutorial: [Electronic excitations](../tutorial/electronic-excitations.md).

Response and many-body methods on HF/KS references. Legend:
[Roadmap index](index.md).

## 1. SCF and linear response

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| ΔSCF / MOM / IMOM | ✓ | ○ | `UHF::delta_scf` |
| CIS / TDA (RHF) | ✓ | ○ | `TDA<RHF>::run` after `RHF::run` |
| TDHF / RPA (RHF) | ✓ | ○ | `TDHF<RHF>::run` after `RHF::run` (`pyscf.tdscf.rhf.TDHF`) |
| CIS(D) | ✓ | ○ | `CISD<RHF>::run` after `RHF::run`. Not an `RHF` member |
| TDA-TDDFT (RKS) | ✓ | ○ | `TDA<RKS>::run` (`pyscf.tdscf.rks.TDA`). LDA/GGA/global hybrids, singlet + triplet; range-separated hybrids throw |
| TDA-TDDFT (UKS) | ✓ | ○ | `TDA<UKS>::run` (`pyscf.tdscf.uks.TDA`). Spin-conserving LSDA |
| Full LR-TDDFT (RKS) | ✓ | ○ | `TDDFT<RKS>::run` (`pyscf.tdscf.rks.TDDFT`). Casida A/B, singlet + triplet; range-separated hybrids throw |
| Full LR-TDDFT (UKS) | ✓ | ○ | `TDDFT<UKS>::run` (`pyscf.tdscf.uks.TDDFT`). Pure LSDA |
| Spin-flip TDA / TDDFT | ✓ | ○ | `SfTDA<UHF>::run` / `SfTDHF<UHF>::run` |
| CIS/TDA analytic forces | ◐ | ○ | `TDA<RHF>::forces`. Default analytic; FD fallback. No external pin; self-pinned cases removed |
| TDA NAC vectors | ◐ | ○ | `TDA<RHF>::nac_vectors`. No FSSH consumer yet. No external pin; self-pinned cases removed |
| Real-time / CVS / REW TDDFT | ○ | ○ | |

## 2. ADC

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| ADC(2) | ✓ | ○ | `ADC<RHF>::adc2` after `RHF::run` |
| ADC(2)-x | ✓ | ○ | `ADC<RHF>::adc2x` |
| IP-ADC(2) / EA-ADC(2) | ✓ | ○ | `ADC<RHF>::ip_adc2` / `ea_adc2` |
| ADC(3) / CVS-ADC / ADC gradient | ○ | ○ | |

## 3. EOM coupled cluster

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| EOM-EE-CCSD | ✓ | ○ | `CCSD<RHF>::eom_ee_ccsd` (singlet). Not an `RHF` member |
| EOM-IP / EA-CCSD | ✓ | ○ | `CCSD<RHF>::eom_ip_ccsd` / `eom_ea_ccsd` |
| EOM-SF-CCSD | ✓ | ○ | `CCSD<UHF>::eom_sf_ccsd` |
| EOM-CC2 | ✓ | ○ | `CC2<RHF>::eom_cc2` |
| EOM-DIP / DEA / CVS / STEOM / gradients | ○ | ○ | |

## 4. Simplified / tight-binding response

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| TD-DFTB | ✓ | ○ | `DFTB2::td_casida`; energies + length oscillator strengths; Γ, closed-shell MioSketch. 3ob and collinear-reference Casida still ○. No absolute oracle |
| TD-DFTB excited gradient / NAC | ○ | ○ | |
| sTDA / sTDDFT / sTDA-xTB | ○ | ○ | |

## 5. Green’s function and many-body

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| G0W0 (spectral) | ✓ | ○ | `G0W0<RHF>::run` (`pyscf.gw.gw_exact.GWExact`). Not an `RHF` member. Periodic GPW is `GpwDft::g0w0` |
| Direct RPA | ✓ | ○ | `DirectRPA<RHF>::run`; periodic on GPW |
| BSE@G0W0 | ✓ | ○ | `BSE<RHF>::run` |
| One-shot GF2 | ✓ | ○ | `GF2<RHF>::run` |
| Self-consistent GW / RPAX / SOSEX | ○ | ○ | |

Periodic MP2 / RPA / GW depth is tabulated under
[DFT § Periodic](dft.md#4-periodic-kohnsham).
