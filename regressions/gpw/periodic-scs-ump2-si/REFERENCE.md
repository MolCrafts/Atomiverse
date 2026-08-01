# Reference — Γ-point RI-GPW SCS-UMP2 on periodic Si₂

## Method

Spin-component-scaled second-order Møller–Plesset (**SCS-UMP2**) on top of a
converged **Γ-point UKS GPW PBE** SCF with equal-spin occupations
\(n_\alpha = n_\beta = 4\), using the product **RI-GPW** assembly of
spin-resolved \((ia|jb)\) (Del Ben, Hutter, VandeVondele, *J. Chem. Theory
Comput.* **8**, 4177 (2012); CP2K `&WF_CORRELATION` `METHOD MP2_GPW` +
`&WFC_GPW`).

Grimme scales (J. Chem. Phys. **118**, 9095 (2003)):

\[
E_{\mathrm{corr}}^{\mathrm{SCS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}}
  + c_{\mathrm{SS}}\,E_{\mathrm{SS}},\quad
  c_{\mathrm{OS}}=1.2,\; c_{\mathrm{SS}}=1/3.
\]

Product API (same member names as molecular `UHF` and closed-shell
`GpwDft` SCS-MP2):

```cpp
cpu::gpw::GpwDft m;
m.n_alpha = 4; m.n_beta = 4;   // UKS
// ... PBE GPW members; build(); run(atoms);
m.mp2_ri = true;
m.mp2_os_scale = 1.2;
m.mp2_ss_scale = 1.0 / 3.0;
const cpu::gpw::Mp2Result& r = m.mp2(atoms);
// r.e_corr_os / r.e_corr_ss remain **unscaled**;
// r.e_corr = 1.2 * OS + (1/3) * SS
```

CP2K twin keywords: `UKS` + `MULTIPLICITY 1`, `SCALE_S 1.2` (singlet / OS),
`SCALE_T 1/3` (triplet / SS) under `&WF_CORRELATION`.

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-ump2-si` / `periodic-scs-mp2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, UKS \(n_\alpha=n_\beta=4\) |
| Cutoff | **30 Ry** (SCF + WFC_GPW) |
| SCF (ATV) | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| MP2 | all kept virtuals; `mp2_ri = true`; SCS scales as above |

## Absolute correctness dual-gate

When the UKS solution stays near closed-shell (\(C^\alpha\approx C^\beta\),
low \(\langle S^2\rangle\)), **SCS-UMP2 ≡ SCS-RMP2**. Absolute `e_corr` is
therefore dual-gated against the product closed-shell SCS pin
(`periodic-scs-mp2-si`, offline CP2K RKS `SCALE_S`/`SCALE_T`):

| Quantity | Pin / rule | Tol |
|----------|------------|-----|
| E_corr (SCS, RI-GPW UMP2@UKS) | **−0.297941206394** Ha (RKS SCS twin) | **2e-3 Ha** |
| e_corr ≡ 1.2·OS + (1/3)·SS | exact | 1e-12 |
| unscaled OS/SS invariant under scale members | vs plain UMP2 on same orbitals | 1e-10 |
| plain OS+SS ≡ plain e_corr; e_total ≡ e_scf+e_corr | exact | 1e-12 |
| `Mp2Result::ri`, `n_occ == 8` | true / 8 | — |

Hard gates: UKS SCF converged with β orbitals; E_corr finite and **strictly
negative**.

## Offline CP2K UKS twin (2026-08-01) — provenance + wall

| Item | Value |
|------|--------|
| Binary | CP2K **2026.1**, `cp2k.psmp`, source revision **git:5e54ba2** |
| Env | `source …/cp2k/install/env.sh` (sets `CP2K_DATA_DIR`) |
| Input | `tools/cp2k_oracle/inputs/periodic-scs-ump2-si.inp` |
| Host | arrhenius1 (EPYC), **1 MPI × OMP=4** |
| Command | `mpirun -np 1 cp2k.psmp -i periodic-scs-ump2-si.inp -o …` |
| Work dir | `/tmp/cp2k_scs_ump2_si` (offline; never from ctest) |
| SCF note | `EPS_SCF 1e-4`, Broyden α=0.2, LEVEL_SHIFT 0.5, ADDED_MOS 22; UKS multiplicity 1 |
| E_SCF (UKS-PBE) | **−7.621755519053** Ha |
| \(\langle S^2\rangle\) (single det.) | **1.009** (ideal 0 — **spin contamination**) |
| Integrated \|spin dens.\| | 1.478 |
| E_corr SO (unscaled) | −0.197005609586 Ha |
| E_corr SS (unscaled) | −0.037886308179 Ha |
| SCALE_S / SCALE_T | 1.2 / 0.33333333333333 |
| **E_corr (SCS total)** | **−0.249035500896** Ha |
| E_total (QS) | −7.870791019948 Ha |
| Wall (full twin job) | **3.83 s** (`/usr/bin/time`; SCF + MP2) |

Pin line (stdout):

```
  MP2 Energy SO component (singlet) =                          -0.19700560958573
  MP2 Energy SS component (triplet) =                          -0.03788630817892
  Scaling factor SO                 =                           1.20000000000000
  Scaling factor SS                 =                           0.33333333333333
  Second order perturbation energy  =                          -0.24903550089585
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]             -7.870791019948481
```

Algebra check:
\(1.2\times(-0.19700560958573)+(1/3)\times(-0.03788630817892)=-0.24903550089585\).

**Why not absolute gate on CP2K UKS digits:** on this small-gap sc-Si₂ deck
CP2K UKS (multiplicity 1) **spin-breaks** (\(\langle S^2\rangle\approx 1\),
finite absolute spin density), so the UKS MP2 residual is not the
closed-shell-equivalent twin that Atomiverse recovers with \(n_\alpha=n_\beta\)
and low spin contamination. The matched **algorithm** twin (UKS + MP2_GPW +
SCALE_S/SCALE_T) is still the fair **wall** twin; absolute `e_corr` is dual-
gated against closed-shell SCS (same method class when the UKS solution stays
near RKS), plus exact product SCS algebra. RKS SCS offline twin
(`periodic-scs-mp2-si.inp`) remains the high-precision absolute SCS pin.

Cross-check plain UKS MP2 on the same SCF deck (no scales): SO+SS =
−0.234869646921 Ha — consistent with the unscaled sum above.

## Perf (fair twin wall)

| Engine | Wall (this deck, 1 MPI × OMP=4 / ATV 8t) |
|--------|------------------------------------------|
| CP2K 2026.1 full UKS twin (SCF + SCS-UMP2) | **3.83 s** (repeat median ≈ 3.6 s) |
| Atomiverse twin (SCF + SCS-UMP2) | **15.79 s** (warm `twin_wall`; SCF 14.94 + SCS 0.85) |
| Ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **4.12** → **within_10x** |

Fair ratio uses **T_atv(SCF+SCS)** / **T_cp2k(full UKS twin)** — not the
multi-gate regression wall that also runs plain UMP2 for the algebraic dual.
Target: \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\).

ATV product numbers on the same host (arrhenius1, 2026-08-01):
`E_corr(SCS-UMP2) = −0.2982928914` Ha (`|Δ|` vs closed-shell SCS pin =
3.5e-4 Ha < 2e-3); plain UMP2 `E_corr = −0.2764033032` Ha (matches
`periodic-ump2-si` dual-gate); \(\langle S^2\rangle = 0\).

Never launch CP2K from ctest.
