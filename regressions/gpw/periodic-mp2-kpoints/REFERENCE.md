# Reference — multi-k RI-GPW MP2 on diamond Si (bct 4-atom, MP 2×1×1)

## Method

Closed-shell **momentum-conserving** second-order Møller–Plesset correlation
on top of a multi-k **GPW PBE** SCF. Occupied–virtual pair densities are
complex Bloch products; Coulomb via linear real Poisson on Re/Im (RI-GPW /
Del Ben 2012; CP2K `METHOD MP2_GPW` + `&KPOINTS`). Public path:

```cpp
cpu::gpw::GpwDft m;
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = false;   // equal weights
m.mp2_ri = true;
m.mp2_n_virt = 2;
// build(); run(atoms);
const cpu::gpw::Mp2Result& r = m.mp2(atoms);  // multi-k, no throw
```

Multi-k SCF stores complex Bloch `GpwResult::C_k`. Kernel Γ recovery
(complex `mo_on_grid_k` ≡ real `mo_on_grid` at κ=0; complex OS/SS ≡ real
doubles sum) is unit-tested in `test_cpu_gpw_mp2` (no full SCF).

## System

| Item | Value |
|------|--------|
| Crystal | diamond Si, **bct 4-atom** cell (same as `gpw-kpoints-si`) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, cutoff **80 Ry**, eps_default 1e-7 |
| Valence | 16 e⁻ → n_occ = 8 |
| k | **MP 2×1×1**, `k_time_reversal=false` → 2 k, w = ½ |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** (k-path rejects it) |
| MP2 | RI-GPW, **n_virt = 2** (e2e ≤ 60 s) |

**Note:** artificial sc-Si₂ (Γ RI-MP2 deck) needs `level_shift` at Γ and does
not converge on the k-path without it — product multi-k MP2 uses the physical
diamond bct cell instead.

## Offline oracle status

**No CP2K binary / module on the worker host at pin time** (same status as
`gpw-kpoints-si` / `periodic-mp2-si`). Absolute CP2K digits are **not invented**.
Input twin: `tools/cp2k_oracle/inputs/periodic-mp2-kpoints.inp`.

### Binding gates

1. **Hard multi-k:** SCF converged; `C_k.size()==n_kpts`; `e_corr` finite and
   **strictly negative**; OS+SS ≡ e_corr; e_total ≡ e_scf+e_corr; n_occ/n_virt.
2. **Kernel Γ recovery (unit):** `mo_on_grid_k` at κ=0 ≡ real `mo_on_grid`;
   complex OS/SS formula ≡ real `mp2_rhf_energy`.
3. **First-green absolute band** (setup integrity) — freeze ATV digits below;
   replace with CP2K when available.

### Atomiverse numbers (first green, 2026-07-25, OMP=4)

| Quantity | Value |
|----------|--------|
| E_SCF (2k, PBE, 80 Ry) | **−15.5751782413** Ha (7 iters) |
| E_corr (2k, n_virt=2) | **−0.0657637511** Ha |
| E_corr OS | −0.0598553603 Ha |
| E_corr SS | −0.0059083908 Ha |
| E_total | −15.6409419924 Ha |
| Wall (SCF + MP2) | **3.6 s** |

Regression absolute tol on E_corr: **1e-4 Ha** (setup band; widen only if
re-pin requires).

## Perf

Fair wall vs CP2K when binary exists: target
\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\). Without CP2K binary:
**perf_exempt** (oracle unavailable at pin host). Never launch CP2K from ctest.
