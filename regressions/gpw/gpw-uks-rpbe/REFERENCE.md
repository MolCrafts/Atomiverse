# Reference — Γ-point UKS GPW O₂ (RPBE / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham GPW (Quickstep LSD) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **RPBE GGA** (Hammer, Hansen, Nørskov
1999: exponential exchange enhancement
\(F_x = 1 + \kappa(1 - e^{-\mu s^2/\kappa})\) with PBE \(\kappa/\mu\) and
**PBE correlation**; libxc `GGA_X_RPBE` + `GGA_C_PBE`), GTH-PBE-q6
pseudopotential, DZVP-MOLOPT-GTH Gaussian basis.

**Not** the same as **revPBE** (Zhang & Yang κ-revision of the rational PBE
form — see `gpw-uks-revpbe`).

Atomiverse path: `cpu::gpw::GpwDft` / `GPWEvaluator` with `n_alpha`/`n_beta`,
`XcKind::RPBE`, dual KS matrices, `xc_grid_rpbe_spin`, Löwdin ⟨S²⟩.

## System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell (avoid MT edge truncation) |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (6 valence e⁻ / O → 12 total) |
| XC | **RPBE** (GGA, spin-resolved) |
| Cutoff | 400 Ry (Atomiverse single density grid; CP2K twin also NGRIDS 4 / REL_CUTOFF 60) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), modules
  `buildtool-easybuild/5.2.1-hpca3ef7d197` + `GCC/14.3.0` + `MPICH/4.3.2` +
  `CP2K/2026.1` on Arrhenius.
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-rpbe/o2_uks_rpbe.inp`
- **XC input form:** nested `&XC_FUNCTIONAL` / `&GGA_X_RPBE` + `&GGA_C_PBE`
  (short name `XC_FUNCTIONAL RPBE` and `PARAMETRIZATION RPBE` are **not** valid
  enumerations in CP2K 2026.1).
- **SCF:** `EPS_SCF 1e-6`, Broyden α=0.4, DIAGONALIZATION STANDARD.
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  cp2k.psmp -i o2_uks_rpbe.inp -o o2_uks_rpbe.out
  ```
- **Date:** 2026-07-26
- **Wall (CP2K, 1 process × 4 OMP):** ≈ **30.4 s** total (`CP2K` timer MAXIMUM)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.97338355451434** | **1e-3 Ha** (first green; single-grid vs CP2K multigrid) |
| ⟨S²⟩ (single det.) | **2.005630** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines:

```
  *** SCF run converged in    15 steps ***
  Total energy:                                               -31.97338355451434
  Ideal and single determinant S**2 :                    2.000000       2.005630
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.973383554514335
 FUNCTIONAL| GGA_X_RPBE: … {scale=1.000, spin-polarized}
 FUNCTIONAL| GGA_C_PBE: … {scale=1.000, spin-polarized}
```

### Atomiverse first-green (2026-07-26, OMP=4 Release)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{RPBE}}\) | **−31.973422072918 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(3.85 \times 10^{-5}\) Ha |
| ⟨S²⟩ | 2.00563766 |
| \(E_{\mathrm{PBE}}\) twin | −31.928854728710 Ha |
| \(\lvert E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}\rvert\) | \(4.46 \times 10^{-2}\) Ha (\(> 10^{-4}\)) |
| SCF iters (RPBE) | 59 |
| Wall RPBE SCF | ~82.0 s |

## Dual-gate

\(|E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha on the same geometry
(hard activation check; PBE twin mirrors `gpw-uks-pbe`).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 30.4 | 1 process, OMP=4, NGRIDS 4 multigrid |
| Atomiverse | **82.0** | Release, single density grid, OMP=4, 2026-07-26 |

`ratio = T_atv / T_cp2k = 82.0/30.4 ≈ **2.70**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true.

## Citations

- B. Hammer, L. B. Hansen, J. K. Nørskov, *Phys. Rev. B* **59**, 7413 (1999) — RPBE.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE correlation.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — ⟨S²⟩ for unrestricted single dets.

## Provenance

- Spec: `.claude/specs/gpw-uks-rpbe/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-uks-rpbe/o2_uks_rpbe.inp`
- Sibling pins: `gpw-uks-pbe`, `gpw-uks-blyp`, `gpw-uks-revpbe`, `gpw-rpbe-water`
