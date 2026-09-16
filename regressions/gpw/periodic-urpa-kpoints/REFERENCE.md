# Reference — multi-k unrestricted direct RPA (URPA@UKS) on diamond Si

## Method

**Direct RPA (dRPA)** correlation on top of a multi-k **GPW UKS PBE** SCF
with closed-shell occupations \(n_\alpha = n_\beta = 8\). Screening is
**spin-conserving direct RPA** over α→α ⊕ β→β with **finite-\(q\)
inter-\(k\)** momentum conservation (product multi-k path after
`periodic-rpa-finite-q` / multi-k UGW dielectric). Furche plasmon:

\[
E_c=\sum_q w_q\,\tfrac12\Bigl(\sum_s\Omega_{q,s}-\mathrm{Tr}\,A_q\Bigr).
\]

When \(C^\alpha \approx C^\beta\), **multi-k URPA ≡ multi-k RKS dRPA** —
dual-gated against the `periodic-rpa-kpoints` first-green \(E_c\).

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 8; m.n_beta = 8;
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = false;
m.mp2_ri = true; m.mp2_n_virt = 2;
// build(); run(atoms);
const cpu::gpw::RpaResult& r = m.direct_rpa(atoms);  // multi-k URPA, no throw
```

Kernel unit: \(N_k=1\) UHF α=β recovers real closed-shell / Γ URPA plasmon
(`test_cpu_gpw_rpa`).

## System

| Item | Value |
|------|--------|
| Crystal | diamond Si, **bct 4-atom** (same as `periodic-rpa-kpoints`) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE UKS** \(n_\alpha=n_\beta=8\), cutoff **80 Ry** |
| Valence | 16 e⁻ → n_occ_spin = 8 |
| k | **MP 2×1×1**, `k_time_reversal=false` → 2 k, w = ½ |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| RPA | finite-q inter-k RI-GPW URPA, **n_virt = 2 per spin** |

## Offline oracle status

**No CP2K binary / module on the worker host at pin time.** Absolute CP2K
digits are **not invented**. Input twin:
`tools/cp2k_oracle/inputs/periodic-urpa-kpoints.inp`.

CP2K's default path is frequency-integrated RI-RPA with Gaussian `RI_AUX`
and EXX total-energy assembly — not the exact finite-\(q\) plasmon formula
on PW-RI \((ia|jb)\). Absolute CP2K \(E_c\) is therefore not a hard twin.

### Binding gates

1. **Hard multi-k URPA:** SCF converged; `C_k.size()==C_k_beta.size()==n_kpts`;
   `direct_rpa` no throw; `n_kpts==2`; `ri`; \(E_c\) finite and **strictly
   negative**; all \(\Omega>0\); e_total ≡ e_scf+e_corr; n_occ=16, n_virt=4;
   composite omega length \(N_k^2\cdot(n_{\mathrm{ov}}^\alpha+n_{\mathrm{ov}}^\beta)\).
2. **Kernel unit:** multi-k UHF at \(N_k=1\) with \(C^\alpha=C^\beta\) ≡
   real plasmon / multi-k RHF (`test_cpu_gpw_rpa`).
3. **Dual-gate band** vs RKS multi-k first-green \(E_c\)
   (`periodic-rpa-kpoints` **−0.0736273682 Ha**): \|E_c − ref\| ≤ **5e-3 Ha**.

### Atomiverse numbers (first green, 2026-07-26, OMP=8)

| Quantity | Value |
|----------|--------|
| E_SCF (2k UKS-PBE, 80 Ry) | **−15.5751782413** Ha (7 iters; ≡ RKS multi-k twin; ⟨S²⟩=8) |
| **E_corr (2k URPA, n_virt=2/spin)** | **−0.0736273682** Ha (bit-match RKS multi-k twin) |
| Tr A (composite) | 17.508325 |
| n_occ / n_virt / n_Ω | 16 / 4 / 128 |
| Wall (SCF + URPA) | **~2.7 s** |
| Dual-gate ref (RKS multi-k dRPA) | **−0.0736273682 Ha** |
| Regression band | \|E_c − (−0.0736273682)\| ≤ **5e-3 Ha** |

Unit tier (`tests/cpu/gpw/test_cpu_gpw_rpa.cpp`): `UhfKpointsNk1DualGatesRhf`
recovers real plasmon; `UhfFiniteQTwoKSpectrumShape` dual-gates multi-k RHF.

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-RPA ≠ plasmon algorithm;
EXX block also changes the total-energy path; no CP2K binary required
on the pin host). Never launch CP2K from ctest.
