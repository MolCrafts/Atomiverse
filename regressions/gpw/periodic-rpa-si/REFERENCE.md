# Reference — Γ-point GPW direct RPA on periodic Si₂

## Method

**Direct RPA (dRPA)** correlation energy on top of a converged **Γ-point
GPW PBE** SCF, with the occupied–virtual MO repulsion integrals \((ia|jb)\)
from the same product **RI-GPW** path as `periodic-ri-mp2-si` (Del Ben,
Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)).

Science matches molecular `RHF::direct_rpa` (Furche plasmon formula):

1. pair density \(\rho_{ia}=\psi_i\psi_a\) on the SCF density grid;
2. plane-wave Coulomb intermediate → \((ia|jb)\);
3. singlet Coulomb-only Casida
   \(A=\Delta\varepsilon\,\delta+2V\), \(B=2V\) (\(\alpha_{\mathrm{hf}}=0\));
4. Hermitian reduction \((A-B)^{1/2}(A+B)(A-B)^{1/2}\);
5. \(E_c=\tfrac12\bigl(\sum_s\Omega_s-\mathrm{Tr}\,A\bigr)\);
6. \(E_{\mathrm{total}}=E_{\mathrm{SCF}}+E_c\) (**dRPA@PBE** — no EXX
   replacement; periodic HF remains blocked).

```cpp
cpu::gpw::GpwDft m;
// ... members; build(); run(atoms);
const cpu::gpw::RpaResult& r = m.direct_rpa(atoms);
// r.e_corr, r.e_total [Ha]; r.omega_ha; r.ri
```

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-ri-mp2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE** |
| Cutoff | **30 Ry** |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| RPA | all kept virtuals; `mp2_ri = true` |

## First-green values (Atomiverse, this deck, 2026-07-25)

| Quantity | Value [Ha] |
|----------|------------|
| E_SCF (PBE, periodic Γ) | **−7.5955445070** |
| **E_corr (dRPA plasmon)** | **−0.2658539776** |
| Tr A | 70.122806 |
| E_total | −7.8613984846 |
| n_occ / n_virt | 4 / 22 (n_ov = 88) |
| Ω_min / Ω_max | 0.029896 / 3.347032 |

Wall (4 OpenMP threads, EPYC login): SCF ≈ 20 s, RI-RPA ≈ 1 s,
Poisson cross-check ≈ 0.3 s → **~22 s** total (≤ 60 s).

## Offline CP2K status

Input twin: `tools/cp2k_oracle/inputs/periodic-rpa-si.inp` (RI-RPA +
minimax quadrature + Gaussian `RI_AUX`). CP2K's default path is
**frequency-integrated RI-RPA with Gaussian RI_AUX and EXX total-energy
assembly**, not the exact plasmon formula on PW-RI \((ia|jb)\). Absolute
CP2K \(E_c\) digits are therefore **not invented** as a hard twin of the
plasmon value.

Pin strategy (binding, mirrors `periodic-mp2-si` foothold):

1. **Hard internal gates:** SCF converged; \(E_c\) finite and strictly
   negative; all \(\Omega_s>0\); \(E_{\mathrm{total}}\equiv E_{\mathrm{SCF}}+E_c\);
   \(|E_c^{\mathrm{RI}}-E_c^{\mathrm{Poisson}}|\le 10^{-10}\) Ha.
2. **Frozen first-green band** \(|E_c-(-0.2658539776)|\le 10^{-3}\) Ha.
3. Unit tier pins the plasmon kernel vs an independent dense A/B path
   (`tests/cpu/gpw/test_cpu_gpw_rpa.cpp`).
4. Offline re-pin: when a fair plasmon-equivalent CP2K path exists,
   replace the band with cited CP2K correlation digits.

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-RPA ≠ plasmon algorithm; EXX
block also changes the total-energy path). Regression wall ≈ 22 s single
process, well under the ≤ 60 s cap. Never launch CP2K from ctest.
