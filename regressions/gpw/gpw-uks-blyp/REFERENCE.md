# Reference — Γ-point UKS GPW O₂ (BLYP / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham GPW (Quickstep LSD) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **BLYP GGA** (Becke 1988 exchange +
Lee–Yang–Parr correlation; libxc `GGA_X_B88` + `GGA_C_LYP` / CP2K
`XC_FUNCTIONAL BLYP`), GTH-PBE-q6 pseudopotential, DZVP-MOLOPT-GTH Gaussian
basis.

Atomiverse path: `cpu::gpw::GpwDft` / `GPWEvaluator` with `n_alpha`/`n_beta`,
`XcKind::BLYP`, dual KS matrices, `xc_grid_blyp_spin`, Löwdin ⟨S²⟩.

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
| XC | **BLYP** (GGA, spin-resolved) |
| Cutoff | 400 Ry (Atomiverse single density grid; CP2K twin also NGRIDS 4 / REL_CUTOFF 60) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), module `CP2K/2026.1` on Arrhenius
  (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-blyp/o2_uks_blyp.inp`
- **SCF:** `EPS_SCF 1e-6`, Broyden α=0.4, DIAGONALIZATION STANDARD (matches
  energy to ~1e-4 Ha of a 1e-7 twin; used for fair wall under login load).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  cp2k.psmp -i o2_uks_blyp.inp -o o2_uks_blyp.out
  ```
- **Date:** 2026-07-26
- **Wall (CP2K, 1 process × 4 OMP):** ≈ **31.7 s** total (`CP2K` timer MAXIMUM)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.90397685949710** | **1e-3 Ha** (first green; single-grid vs CP2K multigrid) |
| ⟨S²⟩ (single det.) | **2.006656** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines:

```
  *** SCF run converged in    19 steps ***
  Total energy:                                               -31.90397685949710
  Ideal and single determinant S**2 :                    2.000000       2.006656
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.903976859497096
```

### Atomiverse first-green (2026-07-26, OMP=4 Release)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{BLYP}}\) | **−31.903662478892 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(3.14 \times 10^{-4}\) Ha |
| ⟨S²⟩ | 2.00658851 |
| \(E_{\mathrm{PBE}}\) twin | −31.928854728464 Ha |
| \(\lvert E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}\rvert\) | \(2.52 \times 10^{-2}\) Ha (\(> 10^{-4}\)) |
| SCF iters (BLYP) | 60 |
| Wall BLYP SCF | ~78.6 s |

## Dual-gate

\(|E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha on the same geometry
(hard activation check; PBE twin mirrors `gpw-uks-pbe`).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 31.7 | 1 process, OMP=4, NGRIDS 4 multigrid |
| Atomiverse | **78.6** | Release, single density grid, OMP=4, 2026-07-26 |

`ratio = T_atv / T_cp2k = 78.6/31.7 ≈ **2.48**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true.

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) — B88 exchange.
- C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) — LYP correlation.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — ⟨S²⟩ for unrestricted single dets.

## Provenance

- Spec: `.claude/specs/gpw-uks-blyp/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-uks-blyp/o2_uks_blyp.inp`
- Sibling pins: `gpw-uks-pbe`, `gpw-uks-o2-gamma`, `gpw-blyp-water`
