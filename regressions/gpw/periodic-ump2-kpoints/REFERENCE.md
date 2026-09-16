# Reference — multi-k unrestricted RI-GPW MP2 on diamond Si (bct 4-atom)

## Method

Unrestricted **momentum-conserving** second-order Møller–Plesset correlation
on top of a multi-k **GPW UKS PBE** SCF (closed-shell occupations
\(n_\alpha = n_\beta = 8\)). Same-spin αα/ββ and opposite-spin αβ doubles use
the molecular / PySCF UMP2 convention on complex Bloch MOs; Coulomb via
linear real Poisson on Re/Im (RI-GPW / Del Ben 2012; CP2K `METHOD MP2_GPW` +
`&KPOINTS` + `UKS`). Public path:

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 8; m.n_beta = 8;
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = false;
m.mp2_ri = true;
m.mp2_n_virt = 2;
// build(); run(atoms);
const cpu::gpw::Mp2Result& r = m.mp2(atoms);  // multi-k UMP2, no throw
```

When \(C^\alpha \approx C^\beta\), **UMP2 ≡ RMP2** — dual-gated against the
`periodic-mp2-kpoints` first-green correlation energy.

## System

| Item | Value |
|------|--------|
| Crystal | diamond Si, **bct 4-atom** (same as `periodic-mp2-kpoints`) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE UKS** \(n_\alpha=n_\beta=8\), cutoff **80 Ry** |
| Valence | 16 e⁻ → n_occ_spin = 8 |
| k | **MP 2×1×1**, `k_time_reversal=false` → 2 k, w = ½ |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| MP2 | RI-GPW, **n_virt = 2 per spin** (e2e ≤ 60 s) |

## Offline oracle status

**No CP2K binary / module on the worker host at pin time.** Absolute CP2K
digits are **not invented**. Input twin:
`tools/cp2k_oracle/inputs/periodic-ump2-kpoints.inp`.

### Binding gates

1. **Hard multi-k UMP2:** SCF converged; `C_k.size()==C_k_beta.size()==n_kpts`;
   `e_corr` finite and **strictly negative**; OS+SS ≡ e_corr;
   e_total ≡ e_scf+e_corr; n_occ=16, n_virt=4, ri, n_kpts=2.
2. **Dual-gate vs RKS multi-k first-green**
   (`periodic-mp2-kpoints` E_corr = **−0.0657637511** Ha): band
   \|E_corr(UMP2) − ref\| ≤ **2e-3** Ha (UKS numerical drift).
3. **Kernel unit:** multi-k UMP2 α=β recovers multi-k RMP2 OS/SS
   (`test_cpu_gpw_mp2`).

### Atomiverse numbers (first green, 2026-07-26, OMP=4)

| Quantity | Value |
|----------|--------|
| E_SCF (2k UKS PBE, 80 Ry) | **−15.5751782413** Ha (7 iters; matches RKS multi-k) |
| E_corr (2k UMP2, n_virt=2/spin) | **−0.0657637511** Ha |
| E_corr OS | −0.0598553603 Ha |
| E_corr SS | −0.0059083908 Ha |
| E_total | −15.6409419924 Ha |
| Wall (SCF + UMP2) | **3.7 s** |
| Dual-gate ref (RMP2 multi-k) | −0.0657637511 Ha |
| Regression band | ± **2e-3** Ha (setup integrity; exact match on first green) |

## Perf

Fair wall vs CP2K when binary exists: target
\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\). Without CP2K binary:
**perf_exempt** (oracle unavailable at pin host). Never launch CP2K from ctest.
