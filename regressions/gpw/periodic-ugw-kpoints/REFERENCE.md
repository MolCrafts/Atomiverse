# Reference — multi-k unrestricted spectral G0W0 (UGW@UKS) on diamond Si

## Method

**Spectral G0W0** quasiparticle energies on top of a multi-k **GPW UKS PBE**
SCF with closed-shell occupations \(n_\alpha = n_\beta = 8\). Screening is
**spin-conserving direct RPA** over α→α ⊕ β→β transitions with
**finite-\(q\) inter-\(k\)** momentum conservation (product multi-k \(W\),
same dielectric path as `periodic-gw-kpoints` / `periodic-gw-finite-q`).
Correlation self-energy is the frequency-exact spectral sum over RPA poles
with same-spin \(G_\alpha W\). Exact-exchange diagonals use **truncated (MT)
Coulomb** per spin; local \(v_{\mathrm{xc}}^\alpha\) is spin-resolved PBE on
the BZ-weighted UKS density.

When \(C^\alpha \approx C^\beta\), **multi-k UGW ≡ multi-k RGW** — dual-gated
against the `periodic-gw-kpoints` first-green min-direct QP gap. Kernel unit
test: \(N_k=1\) UHF α=β recovers RHF spectral QP to 1e-8 Ha.

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 8; m.n_beta = 8;
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = false;
m.mp2_ri = true; m.mp2_n_virt = 2;
// build(); run(atoms);
const cpu::gpw::G0W0Result& gw = m.g0w0(atoms);  // multi-k UGW, no throw
// gw.n_kpts == 2; gw.gap_ev = min_k direct α QP gap [eV]
```

## System

| Item | Value |
|------|--------|
| Crystal | diamond Si, **bct 4-atom** (same as `periodic-gw-kpoints`) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE UKS** \(n_\alpha=n_\beta=8\), cutoff **80 Ry** |
| Valence | 16 e⁻ → n_occ_spin = 8 |
| k | **MP 2×1×1**, `k_time_reversal=false` → 2 k, w = ½ |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| G0W0 | finite-q inter-k spectral UGW, **n_virt = 2 per spin** |

## Offline oracle status

**No CP2K binary / module on the worker host at pin time.** Absolute CP2K
digits are **not invented**. Input twin:
`tools/cp2k_oracle/inputs/periodic-ugw-kpoints.inp`.

CP2K's default path is frequency-integrated RI-GW with Gaussian `RI_AUX` —
not multi-k spectral UGW with MT exchange. Absolute CP2K QP gaps are
therefore not a hard twin.

### Binding gates

1. **Hard multi-k UGW:** SCF converged; `C_k.size()==C_k_beta.size()==n_kpts`;
   `g0w0` no throw; `n_kpts==2`; `ri`; Newton converged; finite positive
   min-direct α QP/KS gaps; HOMO_QP < LUMO_QP on the reported spectrum;
   `n_occ==8`, `n_virt==4` (2+2).
2. **Kernel unit:** multi-k UHF at \(N_k=1\) with \(C^\alpha=C^\beta\) ≡
   multi-k RHF spectral QP (`test_cpu_gpw_g0w0`).
3. **Dual-gate band** vs RKS multi-k first-green min-direct QP gap
   (`periodic-gw-kpoints` **7.529834 eV**): \|gap − ref\| ≤ **0.5 eV**.

### Atomiverse numbers (first green, 2026-07-26, OMP=8)

| Quantity | Value |
|----------|--------|
| E_SCF (2k UKS-PBE, 80 Ry) | **−15.5751782413** Ha (7 iters; ≡ RKS multi-k twin) |
| KS min-direct gap α | **1.561168** eV (≡ RKS multi-k twin) |
| **min-direct QP gap α** | **7.698404** eV (Δ vs RKS multi-k G0W0 ≈ 0.17 eV) |
| ε_HOMO^QP / ε_LUMO^QP (report k) | +0.158037 / +0.440948 Ha |
| Wall (SCF + UGW, OMP=8) | **~16 s** |
| Dual-gate ref (RKS multi-k G0W0) | **7.529834 eV** |
| Regression band | \|gap − 7.529834\| ≤ **0.5 eV** |

Unit tier (`tests/cpu/gpw/test_cpu_gpw_g0w0.cpp`): `UhfKpointsNk1DualGatesRhf`
recovers RHF multi-k spectral QP; `DirectRpaUhfDualGatesRhf` E_c; UKS
unconverged guard.

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-G0W0 ≠ spectral + MT; algorithm
differs). Never launch CP2K from ctest. Wall budget ≤ 60 s.
