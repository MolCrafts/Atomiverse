# Reference — Γ-point UKS GPW O₂ (revPBE / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham GPW (Quickstep LSD) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **revPBE GGA** (Zhang & Yang 1998:
PBE form with revised exchange enhancement \(\kappa = 1.245\); libxc
`GGA_X_PBE_R` + `GGA_C_PBE` / CP2K nested `&PBE` / `PARAMETRIZATION revPBE`),
GTH-PBE-q6 pseudopotential, DZVP-MOLOPT-GTH Gaussian basis.

Atomiverse path: `cpu::gpw::GpwDft` / `GPWEvaluator` with `n_alpha`/`n_beta`,
`XcKind::RevPBE`, dual KS matrices, `xc_grid_revpbe_spin`, Löwdin ⟨S²⟩.

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
| XC | **revPBE** (GGA, spin-resolved) |
| Cutoff | 400 Ry (Atomiverse single density grid; CP2K twin also NGRIDS 4 / REL_CUTOFF 60) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), module `CP2K/2026.1` on Arrhenius
  (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-revpbe/o2_uks_revpbe.inp`
- **XC input form:** nested `&XC_FUNCTIONAL` / `&PBE` / `PARAMETRIZATION revPBE`
  (short name `XC_FUNCTIONAL revPBE` is **not** a valid enumeration in CP2K 2026.1).
- **SCF:** `EPS_SCF 1e-6`, Broyden α=0.4, DIAGONALIZATION STANDARD (matches
  energy to ~1e-4 Ha of a 1e-7 twin; used for fair wall under login load).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  cp2k.psmp -i o2_uks_revpbe.inp -o o2_uks_revpbe.out
  ```
- **Date:** 2026-07-26
- **Wall (CP2K, 1 process × 4 OMP):** ≈ **25.8 s** total (`CP2K` timer MAXIMUM)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.96512377015027** | **1e-3 Ha** (first green; single-grid vs CP2K multigrid) |
| ⟨S²⟩ (single det.) | **2.005611** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines:

```
  *** SCF run converged in    15 steps ***
  Total energy:                                               -31.96512377015027
  Ideal and single determinant S**2 :                    2.000000       2.005611
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.965123770150271
 FUNCTIONAL| revPBE, Yingkai Zhang and Weitao Yang, … {spin polarized}
```

### Atomiverse first-green (2026-07-26, OMP=4 Release)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{revPBE}}\) | **−31.965167155386 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(4.34 \times 10^{-5}\) Ha |
| ⟨S²⟩ | 2.00561329 |
| \(E_{\mathrm{PBE}}\) twin | −31.928854728432 Ha |
| \(\lvert E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}\rvert\) | \(3.63 \times 10^{-2}\) Ha (\(> 10^{-4}\)) |
| SCF iters (revPBE) | 59 |
| Wall revPBE SCF | ~83.1 s |

## Dual-gate

\(|E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha on the same geometry
(hard activation check; PBE twin mirrors `gpw-uks-pbe`).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 25.8 | 1 process, OMP=4, NGRIDS 4 multigrid |
| Atomiverse | **83.1** | Release, single density grid, OMP=4, 2026-07-26 |

`ratio = T_atv / T_cp2k = 83.1/25.8 ≈ **3.22**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true.

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- Y. Zhang, W. Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — revPBE.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE base.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — ⟨S²⟩ for unrestricted single dets.

## Provenance

- Spec: `.claude/specs/gpw-uks-revpbe/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-uks-revpbe/o2_uks_revpbe.inp`
- Sibling pins: `gpw-uks-pbe`, `gpw-uks-blyp`, `gpw-uks-o2-gamma`, `gpw-revpbe-si`
