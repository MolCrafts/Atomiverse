# Reference — Γ-point RI-GPW SCS-MP2 on periodic Si₂

## Method

Spin-component-scaled second-order Møller–Plesset (SCS-MP2) on top of a
converged **Γ-point GPW PBE** SCF, with the product **RI-GPW** assembly of
\((ia|jb)\) (Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**,
4177 (2012); CP2K `&WF_CORRELATION` `METHOD MP2_GPW` + `&WFC_GPW`).

Grimme scales (J. Chem. Phys. **118**, 9095 (2003)):

\[
E_{\mathrm{corr}}^{\mathrm{SCS}} = c_{\mathrm{OS}}\,E_{\mathrm{OS}}
  + c_{\mathrm{SS}}\,E_{\mathrm{SS}},\quad
  c_{\mathrm{OS}}=1.2,\; c_{\mathrm{SS}}=1/3.
\]

Product API (same member names as molecular `RHF` / `UHF`):

```cpp
cpu::gpw::GpwDft m;
// ... PBE GPW members; build(); run(atoms);
m.mp2_ri = true;
m.mp2_os_scale = 1.2;
m.mp2_ss_scale = 1.0 / 3.0;
const cpu::gpw::Mp2Result& r = m.mp2(atoms);
// r.e_corr_os / r.e_corr_ss remain **unscaled**;
// r.e_corr = 1.2 * OS + (1/3) * SS
```

CP2K twin keywords: `SCALE_S 1.2` (singlet / OS), `SCALE_T 1/3` (triplet / SS)
under `&WF_CORRELATION`.

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-ri-mp2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE** |
| Cutoff | **30 Ry** (SCF + WFC_GPW) |
| SCF (ATV) | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| MP2 | all kept virtuals; `mp2_ri = true`; SCS scales as above |

## Offline CP2K pin (2026-08-01)

| Item | Value |
|------|--------|
| Binary | CP2K **2026.1**, `cp2k.psmp`, source revision **git:5e54ba2** |
| Env | `source …/cp2k/install/env.sh` (sets `CP2K_DATA_DIR`) |
| Input | `tools/cp2k_oracle/inputs/periodic-scs-mp2-si.inp` |
| Host | arrhenius1 (EPYC), **1 MPI × OMP=4** |
| Command | `mpirun -np 1 cp2k.psmp -i periodic-scs-mp2-si.inp -o …` |
| Work dir | `/tmp/cp2k_scs3` (offline; never from ctest) |
| SCF note | `EPS_SCF 1e-4`, Broyden α=0.2, LEVEL_SHIFT 0.5, ADDED_MOS 22 — at 30 Ry this small-gap sc-Si₂ residual plateaus ~2–4e-5 under diagonalization (virtuals required for MP2; OT lacks them). Energy stable to ~1e-5 Ha vs tighter attempts. |
| E_SCF (PBE) | **−7.596029572060** Ha (after SCF, before MP2) |
| E_corr SO (unscaled) | −0.237587967176 Ha |
| E_corr SS (unscaled) | −0.038506937350 Ha |
| SCALE_S / SCALE_T | 1.2 / 0.33333333333333 |
| **E_corr (SCS total)** | **−0.297941206394** Ha (`Second order perturbation energy`) |
| E_total (QS) | −7.893970778455 Ha |
| Wall (full twin job) | **17.81 s** (`/usr/bin/time`; SCF ~14.7 s + MP2 1.12 s) |

Pin line (stdout):

```
  MP2 Energy SO component (singlet) =                          -0.23758796717559
  MP2 Energy SS component (triplet) =                          -0.03850693735043
  Scaling factor SO                 =                           1.20000000000000
  Scaling factor SS                 =                           0.33333333333333
  Second order perturbation energy  =                          -0.29794120639418
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]             -7.893970778454608
```

Algebra check: \(1.2\times(-0.23758796717559)+(1/3)\times(-0.03850693735043)=-0.29794120639418\).

Unscaled SO+SS = −0.276094904526 Ha, matching the plain RI-MP2 pin
(`periodic-ri-mp2-si`, −0.276087740659 Ha) to ~7e-6 Ha (SCF residual / mixer).

## Atomiverse gates

| Quantity | Pin / rule | Tol |
|----------|------------|-----|
| E_corr (SCS, RI-GPW) | **−0.297941206394** Ha (CP2K) | **1e-3 Ha** |
| e_corr ≡ 1.2·OS + (1/3)·SS | exact | 1e-12 |
| unscaled OS/SS invariant under scale members | vs plain MP2 on same orbitals | 1e-10 |
| plain OS+SS ≡ plain e_corr; e_total ≡ e_scf+e_corr | exact | 1e-12 |
| `Mp2Result::ri` | true | — |

Hard gates: SCF converged; E_corr finite and **strictly negative**.

## Perf (fair twin wall)

| Engine | Wall (this deck, 1 MPI × OMP=4 / ATV 8t) |
|--------|------------------------------------------|
| CP2K 2026.1 full twin (SCF + SCS-MP2) | **17.81 s** |
| Atomiverse twin (SCF + SCS-MP2) | **7.55 s** (`twin_wall`; SCF 7.34 + SCS 0.22) |
| Ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **0.424** → **within_10x** |

Fair ratio uses **T_atv(SCF+SCS)** / **T_cp2k(full twin)** — not the multi-gate
regression wall that also runs plain MP2 for the algebraic dual gate.
Target: \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\).

ATV product numbers on the same host (arrhenius1, 2026-08-01):
`E_corr(SCS) = −0.2982932565` Ha (`|Δ|` vs CP2K pin = 3.5e-4 Ha < 1e-3).

Never launch CP2K from ctest.
