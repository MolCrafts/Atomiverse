# Reference — Γ-point unrestricted spectral G0W0 (UGW@UKS) on periodic Si₂

## Method

**Spectral G0W0** quasiparticle energies on top of a converged **Γ-point
UKS GPW PBE** SCF with closed-shell occupations
\(n_\alpha = n_\beta = 4\). Screening is **spin-conserving direct RPA**
(Coulomb-only Casida over α→α ⊕ β→β transitions; spins explicit — no
closed-shell factor 2). Correlation self-energy is the frequency-exact
spectral sum over RPA poles with same-spin \(G_\sigma W\). Exact-exchange
diagonals use **truncated (MT) Coulomb** per spin; local
\(v_{\mathrm{xc}}^\sigma\) is spin-resolved PBE on the UKS density.

When \(C^\alpha \approx C^\beta\), **UGW ≡ RGW** — dual-gated against the
`periodic-gw-si` first-green QP gap.

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 4; m.n_beta = 4;   // UKS
// ... members; build(); run(atoms);
const cpu::gpw::G0W0Result& gw = m.g0w0(atoms);
// gw.gap_ev [eV] = α HOMO→LUMO QP gap; gw.qp_energy_ha = α spectrum
```

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-gw-si` / `periodic-ump2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, UKS \(n_\alpha=n_\beta=4\) |
| Cutoff | **30 Ry** |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| G0W0 | all kept virtuals per spin for dRPA; `mp2_ri = true` |

## Dual-gate / first-green band

| Quantity | Value | Notes |
|----------|--------|--------|
| QP gap (RKS G0W0 first-green, `periodic-gw-si`) | **5.719545 eV** | closed-shell twin |
| QP gap (UGW first-green, this deck, 2026-07-26) | **5.719564 eV** | α channel; Δ vs RKS ≈ 2e-5 eV |
| E_SCF (UKS-PBE) | **−7.5955445033 Ha** | matches RKS to ~4e-9 Ha |
| KS gap α | **0.410157 eV** | |
| Regression band | \|gap(UGW) − 5.719545\| ≤ **0.05 eV** | dual-gate vs RKS pin |
| Internal | SCF+Newton converged; QP gap > 0; HOMO_QP < LUMO_QP; finite shifts | hard |

Unit tier (`tests/cpu/gpw/test_cpu_gpw_g0w0.cpp`): `direct_rpa_uhf` E_c dual-gates
`direct_rpa_rhf` on identical α=β tensors; UKS unconverged guard.

## Offline CP2K status

Input twin: `tools/cp2k_oracle/inputs/periodic-ugw-si.inp` (UKS MULT 1 +
RI-G0W0). CP2K's default path is **frequency-integrated RI-GW**, not spectral
plasmon G0W0 with MT exchange. Absolute CP2K QP digits are **not invented**
as a hard twin of the spectral value.

Pin strategy (binding, mirrors `periodic-gw-si` / `periodic-ump2-si`):

1. **Hard internal gates** (convergence, gap sign, HOMO/LUMO order).
2. **Dual-gate band** vs RKS spectral first-green.
3. Offline re-pin: when a fair spectral-equivalent CP2K UGW path exists,
   replace the band with cited CP2K QP gap digits.

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-G0W0 ≠ spectral algorithm;
MT exchange ≠ periodic HF; no CP2K binary required on the pin host).
Wall (8 OpenMP threads, EPYC login): SCF ≈ 13 s, UGW ≈ 1 s → **~14 s**
total (≤ 60 s). Never launch CP2K from ctest.
