# Reference — Γ-point GPW spectral G0W0 on periodic Si₂

## Method

**Spectral G0W0** quasiparticle energies on top of a converged **Γ-point
GPW PBE** SCF. Screening is **direct RPA** (Coulomb-only Casida /
Furche reduction) on the same product **RI-GPW** `(ia|jb)` path as
`periodic-rpa-si` / `periodic-ri-mp2-si`. The correlation self-energy is
the frequency-exact spectral sum over RPA poles (same form as molecular
`RHF::g0w0` / PySCF `GWExact`). Exact-exchange diagonals use **truncated
(MT) Coulomb** so the Γ-only foothold is G=0-safe without a periodic-HF
product path. Local \(v_{\mathrm{xc}}\) is PBE on the closed-shell density.

```cpp
cpu::gpw::GpwDft m;
// ... members; build(); run(atoms);
const cpu::gpw::G0W0Result& gw = m.g0w0(atoms);
// gw.gap_ev [eV]; gw.qp_energy_ha [Ha]; gw.converged
```

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-rpa-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE** |
| Cutoff | **30 Ry** |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| G0W0 | all kept virtuals for dRPA screening; `mp2_ri = true` |

## First-green values (Atomiverse, this deck, 2026-07-25)

| Quantity | Value |
|----------|--------|
| E_SCF (PBE, periodic Γ) | **−7.5955445070** Ha |
| KS gap | **0.410147** eV |
| **QP gap** | **5.719545** eV |
| ε_HOMO^QP | −0.13897686 Ha (Δ vs KS +0.071804 Ha) |
| ε_LUMO^QP | +0.07121256 Ha (Δ vs KS +0.123312 Ha) |
| IP / EA (QP) | 3.7818 / 1.9378 eV |
| n_occ / n_virt | 4 / 22 |

Wall (4 OpenMP threads, EPYC login): SCF ≈ 23 s, G0W0 ≈ 0.6 s →
**~24 s** total (≤ 60 s).

## Offline CP2K status

Input twin: `tools/cp2k_oracle/inputs/periodic-gw-si.inp` (RI-G0W0 +
minimax quadrature + Gaussian `RI_AUX`). CP2K's default path is
**frequency-integrated RI-GW**, not spectral plasmon G0W0 with MT
exchange. Absolute CP2K QP digits are therefore **not invented** as a
hard twin of the spectral value.

Pin strategy (binding, mirrors `periodic-rpa-si`):

1. **Hard internal gates:** SCF converged; Newton QP converged; QP gap
   finite and positive; HOMO QP below LUMO QP; finite HOMO/LUMO shifts.
2. **Frozen first-green band** \(|\mathrm{gap}-5.719545|\le 0.05\) eV.
3. Unit tier pins dRPA amplitudes vs energy path
   (`tests/cpu/gpw/test_cpu_gpw_g0w0.cpp`).
4. Offline re-pin: when a fair spectral-equivalent CP2K path exists,
   replace the band with cited CP2K QP gap digits.

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-G0W0 ≠ spectral algorithm;
MT exchange ≠ periodic HF). Regression wall ≈ 24 s single process, well
under the ≤ 60 s cap. Never launch CP2K from ctest.
