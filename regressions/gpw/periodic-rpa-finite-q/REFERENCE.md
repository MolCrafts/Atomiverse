# Reference — finite-q inter-k RI-GPW dRPA on diamond Si (bct 4-atom, MP 2×1×1)

## Method

Closed-shell **finite-\(q\) inter-\(k\) direct RPA (dRPA)** correlation on top
of a multi-k **GPW PBE** SCF. For each crystal-momentum transfer \(q\) on the
same closed Monkhorst–Pack mesh as \(k\), the Coulomb-only Casida spectrum is
built from momentum-conserving transitions \(i_k\to a_{k+q}\) via RI-GPW
Poisson at transfer \(q\) (complex Bloch products; linear real Poisson on
Re/Im; \(G=0\) dropped only at \(q=0\)), reduced by the Furche plasmon
formula, and Brillouin-zone averaged:

\[
E_c=\sum_q w_q\,\tfrac12\Bigl(\sum_s\Omega_{q,s}-\mathrm{Tr}\,A_q\Bigr).
\]

Total energy is **dRPA@KS** \(E_{\mathrm{SCF}}+E_c\) (no EXX — periodic HF
blocked). This supersedes the multi-k **k-diagonal** foothold of
`periodic-rpa-kpoints` as the product multi-k path.

Public path:

```cpp
cpu::gpw::GpwDft m;
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = false;
m.mp2_ri = true;
m.mp2_n_virt = 2;
// build(); run(atoms);
const cpu::gpw::RpaResult& r = m.direct_rpa(atoms);
```

Kernel Γ recovery (\(N_k=1\), real-valued complex MOs ≡ real plasmon) is
unit-tested in `test_cpu_gpw_rpa` (no full SCF).

## System

| Item | Value |
|------|--------|
| Crystal | diamond Si, **bct 4-atom** cell (same as `gpw-kpoints-si`) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, cutoff **80 Ry**, eps_default 1e-7 |
| Valence | 16 e⁻ → n_occ = 8 |
| k / q | **MP 2×1×1**, `k_time_reversal=false` → 2 k = 2 q, w = ½ |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| RPA | RI-GPW finite-q inter-k dRPA, **n_virt = 2** (e2e ≤ 60 s) |
| Spectrum | \(n_\Omega = N_q N_k n_{\mathrm{occ}} n_{\mathrm{virt}} = 64\) |

## Offline oracle status

**No CP2K binary / module on the worker host at pin time** (same status as
`periodic-rpa-kpoints` / `periodic-rpa-si`). Absolute CP2K digits are **not
invented**. Input twin:
`tools/cp2k_oracle/inputs/periodic-rpa-finite-q.inp`.

CP2K's default path is frequency-integrated RI-RPA with Gaussian `RI_AUX`
and EXX total-energy assembly — not the exact q-resolved plasmon formula on
PW-RI \((ia|jb)\). Absolute CP2K \(E_c\) is therefore not a hard twin.

### Binding gates

1. **Hard multi-k finite-q:** SCF converged; `C_k.size()==n_kpts`; `e_corr`
   finite and **strictly negative**; all \(\Omega>0\); e_total ≡ e_scf+e_corr;
   n_occ/n_virt; `ri` and `n_kpts` flags; \(\lvert\omega\rvert=N_k^2 n_{\mathrm{ov}}\).
2. **Kernel Γ recovery (unit):** multi-k kernel at \(N_k=1\) ≡ real
   `direct_rpa_rhf_energy`.
3. **First-green absolute band** (setup integrity) — freeze ATV digits below;
   replace with fair CP2K when available.

### Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
|----------|--------|
| E_SCF (2k, PBE, 80 Ry) | **−15.5751782413** Ha (7 iters) |
| **E_corr (2k, finite-q, n_virt=2)** | **−0.0736273682** Ha |
| Wall (SCF + RPA, OMP=4) | **~3 s** |

Regression absolute tol on E_corr: **5e-3 Ha** about −0.0736273682 (setup
band covering OMP orbital scatter; not a CP2K digit twin).

## Perf

Fair wall vs CP2K when binary exists: target
\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\). Without CP2K binary / without
fair algorithm twin (CP2K RI-RPA ≠ finite-q plasmon on PW-RI): **perf_exempt**.
Never launch CP2K from ctest.
