# Reference — multi-k RI-GPW SCS-UMP2 on diamond Si (bct 4-atom, MP 2×1×1)

## Method

Spin-component-scaled unrestricted second-order Møller–Plesset
(**SCS-UMP2**) on top of a multi-k **GPW UKS PBE** SCF with equal-spin
occupations \(n_\alpha = n_\beta = 8\). Occupied–virtual pair densities are
complex Bloch products; Coulomb via the product **RI-GPW** assembly of
spin-resolved \((ia|jb)\) (Del Ben, Hutter, VandeVondele, *J. Chem. Theory
Comput.* **8**, 4177 (2012)). Momentum-conserving multi-k UMP2 doubles sum is
the same product path as `periodic-ump2-kpoints`; this slug only reweights the
unscaled OS/SS channels via `mp2_os_scale` / `mp2_ss_scale`.

Grimme scales (*J. Chem. Phys.* **118**, 9095 (2003)):

\[
E_{\mathrm{corr}}^{\mathrm{SCS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}}
  + c_{\mathrm{SS}}\,E_{\mathrm{SS}},\quad
  c_{\mathrm{OS}}=1.2,\; c_{\mathrm{SS}}=1/3.
\]

Public path:

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 8; m.n_beta = 8;   // UKS
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = false;
m.mp2_ri = true;
m.mp2_n_virt = 2;
m.mp2_os_scale = 1.2;
m.mp2_ss_scale = 1.0 / 3.0;
// build(); run(atoms);
const cpu::gpw::Mp2Result& r = m.mp2(atoms);
// r.e_corr_os / r.e_corr_ss remain **unscaled**;
// r.e_corr = 1.2 * OS + (1/3) * SS
```

CP2K twin keywords: `UKS` + `MULTIPLICITY 1`, `SCALE_S 1.2` (singlet / OS),
`SCALE_T 1/3` (triplet / SS) under `&WF_CORRELATION`, plus DFT `&KPOINTS`.

## System

| Item | Value |
|------|--------|
| Crystal | diamond Si, **bct 4-atom** cell (same as `periodic-ump2-kpoints`) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE UKS** \(n_\alpha=n_\beta=8\), cutoff **80 Ry**, eps_default 1e-7 |
| Valence | 16 e⁻ → n_occ_spin = 8 |
| k | **MP 2×1×1**, `k_time_reversal=false` → 2 k, w = ½ |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| MP2 | RI-GPW, **n_virt = 2 per spin** (e2e ≤ 60 s); SCS scales as above |

## Oracle status (literature — CP2K has no multi-k MP2_GPW)

**Honest gate:** CP2K 2026.1 (`cp2k.psmp`, git:**5e54ba2**) was **actually run**
offline with the matched twin input. The multi-k **UKS-PBE SCF converges**
(closed-shell occupations, integrated |spin dens.| = 0), but `METHOD MP2_GPW`
under DFT `&KPOINTS` aborts:

```
 * [ABORT]
 *  \___/                   Method not implemented for k-points
 *                                                kpoint_transitional.F:51
 ===== Routine Calling Stack =====
            3 mp2_main
            2 qs_energies
            1 CP2K
```

| Item | Value |
|------|--------|
| Binary | CP2K **2026.1**, `cp2k.psmp`, source revision **git:5e54ba2** |
| Env | `source …/cp2k/install/env.sh` |
| Input | `tools/cp2k_oracle/inputs/periodic-scs-ump2-kpoints.inp` |
| Host | arrhenius1 (EPYC), **1 MPI × OMP=4** |
| Command | `mpirun -np 1 cp2k.psmp -i periodic-scs-ump2-kpoints.inp -o …` |
| Work dir | `/tmp/cp2k_oracle_periodic-scs-ump2-kpoints` (offline; never from ctest) |
| E_SCF (CP2K, multi-k UKS-PBE) | **−15.5750245898** Ha (13 iters; before MP2 abort) |
| Integrated \|spin dens.\| | **0** (no spin contamination on this deck) |
| Wall to abort | **2.92 s** (SCF only — MP2 never started) |
| MP2 / SCS result | **none** — algorithm not implemented for k-points |

Therefore the correlation pin is **`oracle: literature`** (Grimme SCS formula +
Del Ben RI-GPW method class + algebraic dual against product multi-k UMP2 /
closed-shell multi-k SCS), **not** a fake `oracle: cp2k` energy. The twin
`.inp` is retained as the unexecutable sketch and as proof of the offline run.

CP2K multi-k UKS SCF parity with ATV (`periodic-ump2-kpoints` E_SCF ≈
−15.575178 Ha) is ~1.5×10⁻⁴ Ha — setup integrity only, not an SCS pin. Same
class of exclusion as `periodic-scs-mp2-kpoints` / `periodic-ump2-kpoints`
(no multi-k `MP2_GPW` in CP2K 2026.1).

## Binding gates

1. **Hard multi-k UKS:** SCF converged; `C_k.size()==C_k_beta.size()==n_kpts==2`;
   `e_corr` finite and **strictly negative**; `ri`; n_occ/n_virt = 16/4.
2. **Algebraic dual (Grimme):** unscaled OS/SS invariant under scale members
   vs plain multi-k UMP2 on the same orbitals; plain OS+SS ≡ plain e_corr;
   SCS e_corr ≡ 1.2·OS + (1/3)·SS; e_total ≡ e_scf+e_corr (all 1e-10…1e-12).
3. **Absolute dual-gate vs closed-shell multi-k SCS**
   (`periodic-scs-mp2-kpoints` E_corr = **−0.0737958960** Ha): band
   \|E_corr(SCS-UMP2) − ref\| ≤ **2e-3** Ha (UKS numerical drift). When
   \(C^\alpha \approx C^\beta\), **SCS-UMP2 ≡ SCS-RMP2**.

## Atomiverse numbers (first green, 2026-08-01, OMP=8)

| Quantity | Value |
|----------|--------|
| E_SCF (2k UKS PBE, 80 Ry) | **−15.5751782413** Ha (7 iters; matches `periodic-ump2-kpoints`) |
| E_corr plain (OS+SS) | **−0.0657637511** Ha (OS=−0.0598553603, SS=−0.0059083908) |
| **E_corr SCS** | **−0.0737958960** Ha (= 1.2·OS + (1/3)·SS; matches RKS multi-k SCS) |
| E_total | −15.6489741373 Ha |
| Wall twin (SCF + SCS only) | **2.79 s** (reg total with plain dual ~3.7 s) |
| Dual-gate ref (RKS multi-k SCS) | −0.0737958960 Ha |
| Regression absolute tol | **2e-3 Ha** |

Algebra check: \(1.2\times(-0.0598553603)+(1/3)\times(-0.0059083908)=-0.0737958960\).

## Perf

| Engine | Wall |
|--------|------|
| CP2K 2026.1 | **no fair twin** — MP2_GPW aborts under DFT KPOINTS after SCF (2.92 s to abort) |
| Atomiverse twin (SCF + SCS) | **2.79 s** |

**perf_exempt:** CP2K has no multi-k `MP2_GPW` / SCS-UMP2 algorithm (real abort
on the offline twin; same exclusion as `periodic-scs-mp2-kpoints`). Never
launch CP2K from ctest. Do **not** claim `within_10x` against the SCF-only
abort wall.

## Literature citations

1. S. Grimme, *J. Chem. Phys.* **118**, 9095 (2003) — SCS-MP2 scales
   \(c_{\mathrm{OS}}=1.2\), \(c_{\mathrm{SS}}=1/3\).
2. M. Del Ben, J. Hutter, J. VandeVondele, *J. Chem. Theory Comput.* **8**,
   4177 (2012) — RI-GPW MP2 (CP2K `METHOD MP2_GPW` method class; Γ-point in
   CP2K product; multi-k UMP2 is Atomiverse product).
