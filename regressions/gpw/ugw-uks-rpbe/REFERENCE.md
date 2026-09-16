# Reference — Γ-point unrestricted spectral G0W0 (UGW@UKS) with **RPBE**

## Method

**Spectral G0W0** quasiparticle energies on top of a converged **Γ-point
UKS GPW RPBE** SCF with closed-shell occupations
\(n_\alpha = n_\beta = 4\). Screening is **spin-conserving direct RPA**
(Coulomb-only Casida over α→α ⊕ β→β; spins explicit). Correlation
self-energy is the frequency-exact spectral sum over RPA poles with
same-spin \(G_\sigma W\). Exact-exchange diagonals use **truncated (MT)
Coulomb** per spin; local \(v_{\mathrm{xc}}^\sigma\) is spin-resolved
**RPBE** (Hammer, Hansen, Nørskov 1999: exponential exchange enhancement
\(F_x = 1 + \kappa(1 - e^{-\mu s^2/\kappa})\) with PBE \(\kappa/\mu\) and
**PBE correlation**; libxc `GGA_X_RPBE` + `GGA_C_PBE`) via
`xc_grid_rpbe_spin` (same kernel as UKS SCF).

**Not** revPBE (Zhang–Yang κ-revision).

When \(C^\alpha \approx C^\beta\), **UGW ≡ RGW** — dual-gated against
closed-shell `GpwDft::g0w0` on the **same RPBE** deck in this binary.

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 4; m.n_beta = 4;   // UKS
m.xc = cpu::gpw::XcKind::RPBE;
// ... members; build(); run(atoms);
const cpu::gpw::G0W0Result& gw = m.g0w0(atoms);
// gw.gap_ev [eV] = α HOMO→LUMO QP gap
```

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same geometry as `periodic-ugw-si` / `periodic-gw-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **RPBE**, UKS \(n_\alpha=n_\beta=4\) (dual-gate RKS \(n_{\mathrm{occ}}=4\)) |
| Cutoff | **30 Ry** |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| G0W0 | all kept virtuals; `mp2_ri = true` |

## Dual-gate / first-green band

| Quantity | Value | Notes |
|----------|--------|--------|
| Regression band | \|gap(UGW@RPBE) − gap(RKS G0W0@RPBE)\| ≤ **0.05 eV** | dual-gate in-binary |
| Functional integrity | \|gap(UGW@RPBE) − 5.719545 eV\| ≥ **1e-3 eV** | vs PBE UGW pin (`periodic-ugw-si`) |
| Internal | SCF+Newton converged; QP gap > 0; HOMO_QP < LUMO_QP; finite shifts | hard |

First-green digits (Atomiverse Release, 2026-07-26, EPYC login 8 OMP;
not a CP2K absolute twin):

| Quantity | First-green (this deck) | Notes |
|----------|-------------------------|--------|
| QP gap UGW@RPBE | **5.674272 eV** | α channel |
| QP gap RKS G0W0@RPBE | **5.674251 eV** | closed-shell twin; \|Δ\| ≈ 2e-5 eV |
| E_SCF (UKS-RPBE) | **−7.6173910139 Ha** | matches RKS to ~7e-9 Ha |
| KS gap α | **0.384145 eV** | |
| Wall (UKS SCF + UGW + RKS SCF + G0W0) | **≈ 22 s** | ≤ 60 s |

## Offline CP2K status

Input twin: `tools/cp2k_oracle/inputs/ugw-uks-rpbe.inp` (UKS MULT 1 +
RPBE nested XC + RI-G0W0). CP2K's default path is **frequency-integrated
RI-GW**, not spectral plasmon G0W0 with MT exchange. Absolute CP2K QP
digits are **not invented** as a hard twin of the spectral value.

Pin strategy (binding, mirrors `periodic-ugw-si`):

1. **Hard internal gates** (convergence, gap sign, HOMO/LUMO order).
2. **Dual-gate band** vs RKS spectral G0W0@RPBE on the same deck.
3. **Functional integrity** vs PBE UGW first-green (RPBE must differ).
4. Offline re-pin: when a fair spectral-equivalent CP2K UGW path exists,
   replace the dual-gate band with cited CP2K QP gap digits.

CP2K XC form (2026.1): nested `&XC_FUNCTIONAL` / `&GGA_X_RPBE` +
`&GGA_C_PBE` (short name `XC_FUNCTIONAL RPBE` is **not** a valid
enumeration in CP2K 2026.1 — same as `gpw-uks-rpbe`).

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-G0W0 ≠ spectral algorithm;
MT exchange ≠ periodic HF; no CP2K binary required on the pin host).
Measured wall (8 OpenMP threads, EPYC login): two SCFs (UKS + RKS) + two
G0W0 ≈ **22 s** (≤ 60 s). Never launch CP2K from ctest.
