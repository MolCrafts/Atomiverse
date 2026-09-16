# Reference — finite-q inter-k spectral G0W0 on diamond Si (bct 4-atom, MP 2×1×1)

## Method

Closed-shell **finite-\(q\) inter-\(k\) spectral G0W0** quasiparticle energies
on top of a multi-k **GPW PBE** SCF. For each crystal-momentum transfer \(q\)
on the same closed Monkhorst–Pack mesh as \(k\), the Coulomb-only Casida
spectrum is built from momentum-conserving transitions \(i_k\to a_{k+q}\) via
RI-GPW Poisson (complex Bloch; same dielectric as multi-k dRPA), \(X+Y\)
amplitudes form spectral transition potentials \(W_s^q\), and the correlation
self-energy at target \(k\) contracts intermediate states at \(k-q\) with
weight \(1/N_k\). Exact-exchange diagonals use truncated (MT) Coulomb. Local
\(v_{\mathrm{xc}}\) is PBE on the BZ-weighted density.

This supersedes the multi-k **k-diagonal** foothold of `periodic-gw-kpoints`
as the product multi-k path.

Public path:

```cpp
cpu::gpw::GpwDft m;
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = false;
m.mp2_ri = true;
m.mp2_n_virt = 2;
// build(); run(atoms);
const cpu::gpw::G0W0Result& gw = m.g0w0(atoms);  // finite-q multi-k
// gw.n_kpts == 2; gw.gap_ev = min_k direct QP gap [eV]
```

Kernel Γ recovery (\(N_k=1\), real-valued complex MOs ≡ real spectral path)
is unit-tested in `test_cpu_gpw_g0w0` (no full SCF).

## System

| Item | Value |
|------|--------|
| Crystal | diamond Si, **bct 4-atom** cell (same as `periodic-gw-kpoints`) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, cutoff **80 Ry**, eps_default 1e-7 |
| Valence | 16 e⁻ → n_occ = 8 |
| k / q | **MP 2×1×1**, `k_time_reversal=false` → 2 k = 2 q |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| G0W0 | finite-q inter-k spectral, **n_virt = 2** screening (e2e ≤ 60 s) |

## Offline oracle status

**No CP2K binary / module on the worker host at pin time** (same status as
`periodic-gw-kpoints` / `periodic-rpa-finite-q`). Absolute CP2K digits are
**not invented**. Input twin:
`tools/cp2k_oracle/inputs/periodic-gw-finite-q.inp`.

CP2K's default path is frequency-integrated RI-GW with Gaussian `RI_AUX` —
not finite-q spectral G0W0 with MT exchange. Absolute CP2K QP gaps are
therefore not a hard twin.

### Binding gates

1. **Hard multi-k finite-q:** SCF converged; `C_k.size()==n_kpts`; `g0w0` no
   throw; `n_kpts==2`; `ri`; Newton converged; finite positive min-direct
   QP/KS gaps; HOMO QP < LUMO QP on the reported spectrum.
2. **Kernel Γ recovery (unit):** multi-k kernel at \(N_k=1\) ≡ real spectral
   QP on the same orbitals.
3. **First-green absolute band** (setup integrity) — freeze ATV digits below;
   replace with fair CP2K when available.

### Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
|----------|--------|
| E_SCF (2k, PBE, 80 Ry) | **−15.5751782413** Ha (7 iters; same deck as multi-k RPA) |
| KS min-direct gap | **1.561168** eV |
| **min-direct QP gap** | **7.529834** eV |
| ε_HOMO^QP / ε_LUMO^QP (report k) | +0.171486 / +0.448203 Ha |
| Wall (SCF + G0W0, OMP=4) | **~10 s** |

Regression absolute tol on `gap_ev`: **0.5 eV** about 7.529834 (setup band
covering OMP orbital scatter; not a CP2K digit twin). Note: MT exchange +
tiny virt space can place absolute QP levels above zero — gap is the hard
product observable. Finite-\(q\) \(W\) softens the gap relative to the
historical k-diagonal foothold (~8.01 eV).

## Perf

`perf_exempt` — no fair wall twin (CP2K RI-G0W0 ≠ finite-q spectral + MT).
Never launch CP2K from ctest.
