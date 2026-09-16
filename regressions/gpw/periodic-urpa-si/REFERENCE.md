# Reference — Γ-point unrestricted direct RPA (URPA@UKS) on periodic Si₂

## Method

**Direct RPA (dRPA)** correlation energy on top of a converged **Γ-point
UKS GPW PBE** SCF with closed-shell occupations \(n_\alpha = n_\beta = 4\).
Screening is **spin-conserving direct RPA** (Coulomb-only Casida over
α→α ⊕ β→β transitions; spins explicit — no closed-shell factor 2).
\((ia|jb)\) from the product **RI-GPW** path (same as `periodic-rpa-si` /
`periodic-ump2-si`). Furche plasmon formula:

\[
E_c=\tfrac12\Bigl(\sum_s\Omega_s-\mathrm{Tr}\,A\Bigr).
\]

When \(C^\alpha \approx C^\beta\), **URPA ≡ RKS dRPA** — dual-gated against
the `periodic-rpa-si` first-green \(E_c\).

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 4; m.n_beta = 4;   // UKS
// ... members; build(); run(atoms);
const cpu::gpw::RpaResult& r = m.direct_rpa(atoms);
// r.e_corr, r.e_total [Ha]; r.omega_ha (composite α⊕β OV)
```

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-rpa-si` / `periodic-ump2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, UKS \(n_\alpha=n_\beta=4\) |
| Cutoff | **30 Ry** |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| RPA | all kept virtuals per spin; `mp2_ri = true` |

## Dual-gate / first-green band

| Quantity | Value | Notes |
|----------|--------|--------|
| E_corr (RKS dRPA first-green, `periodic-rpa-si`) | **−0.2658539776 Ha** | closed-shell twin |
| E_corr (URPA first-green, this deck, 2026-07-26) | **−0.2658538251 Ha** | Δ vs RKS ≈ 1.5e-7 Ha |
| E_SCF (UKS-PBE) | **−7.5955445033 Ha** | matches RKS to ~4e-9 Ha |
| Tr A (URPA composite) | 136.534225 | ≈ 2 × RKS Tr A (α⊕β) |
| n_occ / n_virt / n_Ω | 8 / 44 / 176 | spin-orbital counts; composite OV |
| Regression band | \|E_c(URPA) − (−0.2658539776)\| ≤ **2×10⁻³ Ha** | dual-gate vs RKS pin |
| Internal | SCF converged; \(E_c<0\); all \(\Omega>0\); \(E_{\mathrm{tot}}=E_{\mathrm{SCF}}+E_c\); RI≡Poisson ≤ 1e-8 Ha | hard |

Wall (8 OpenMP threads, EPYC login): SCF ≈ 14 s, RI-URPA ≈ 0.7 s,
Poisson cross-check ≈ 2.7 s → **~18 s** total (≤ 60 s).

Unit tier (`tests/cpu/gpw/test_cpu_gpw_rpa.cpp`): `direct_rpa_uhf` E_c dual-gates
`direct_rpa_rhf` on identical α=β tensors; UKS unconverged guard.

## Offline CP2K status

Input twin: `tools/cp2k_oracle/inputs/periodic-urpa-si.inp` (UKS MULT 1 +
RI-RPA). CP2K's default path is **frequency-integrated RI-RPA with Gaussian
RI_AUX and EXX total-energy assembly**, not the exact plasmon formula on
PW-RI \((ia|jb)\). Absolute CP2K \(E_c\) digits are **not invented** as a
hard twin of the plasmon value.

Pin strategy (binding, mirrors `periodic-rpa-si` / `periodic-ump2-si`):

1. **Hard internal gates** (convergence, \(E_c\) sign, \(\Omega>0\), energy identity).
2. **Dual-gate band** vs RKS dRPA first-green.
3. Offline re-pin: when a fair plasmon-equivalent CP2K URPA path exists,
   replace the band with cited CP2K correlation digits.

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-RPA ≠ plasmon algorithm;
EXX block also changes the total-energy path; no CP2K binary required
on the pin host). Wall target ≤ 60 s (SCF + URPA). Never launch CP2K from ctest.
