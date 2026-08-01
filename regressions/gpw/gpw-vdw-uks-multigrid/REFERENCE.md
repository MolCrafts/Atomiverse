# Reference — Γ collinear-UKS GPW PBE + vdW-DF (DRSLL) on REL_CUTOFF multigrid

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Γ collinear UKS**
(LSD) with **CP2K-style CUTOFF/REL_CUTOFF multigrid product collocation**
(`ngrids > 1 && rel_cutoff > 0`).

Each axis is already product alone:

| Axis | Slug |
|------|------|
| Γ-UKS + DRSLL (single-grid ATV) | `gpw-vdw-uks` |
| Γ-RKS + DRSLL + multigrid | `gpw-vdw-multigrid` |
| Γ-UKS + multigrid (no nonlocal) | `gpw-uks-multigrid` |

This slug pins the **triple**: \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\)
of the **total** density \(\rho=\rho_\alpha+\rho_\beta\) is spin-independent
and is added identically to both spin potentials on the **fine** grid every
SCF iteration (`build_uks_H`) **before** the per-level restriction — so the
integrated dual KS matrices stay the exact gradient of the collocated
\(E_H + E_{xc} + E_c^{\mathrm{nl}}\) (variational SCF). \(E_c^{\mathrm{nl}}\)
lands on `GpwResult::e_disp` [Hartree].

Public API: `n_alpha`/`n_beta` + `vdw_df` + `ngrids` + `rel_cutoff` on
`cpu::gpw::GpwDft` / `GPWEvaluator`. Multi-k UKS + nonlocal remains
fail-fast (`gpw-vdw-uks-kpoints`).

## Offline CP2K pin

| Item | Value |
|------|-------|
| Molecule | O₂, R(O–O) = 1.2075 Å, centered along *z* |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, **PERIODIC XYZ** |
| Basis / PP | DZVP-MOLOPT-GTH + GTH-PBE-q6 (12 valence e⁻) |
| XC | PBE + `&NON_LOCAL TYPE DRSLL` (`vdW_kernel_table.dat`, SCALE 1.0) |
| Cutoff | **400 Ry / REL_CUTOFF 60 / NGRIDS 4** (matched on both sides) |
| Spin | LSD, MULTIPLICITY 3 → Nα = 7, Nβ = 5 |
| SCF | CP2K: diagonalization + Broyden, EPS_SCF 1e−6; ATV: DIIS, tol 1e−7 |

- **Engine:** CP2K **2026.1** (Development Version, source revision
  **git:5e54ba2**, `cp2k.psmp`,
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`,
  GCC 14.3 / MPICH; libxc + fftw3 + elpa).
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-uks-multigrid/o2_uks_pbe_drsll_mg.inp`
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  cd /tmp/run && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 mpirun -np 1 cp2k.psmp -i o2_uks_pbe_drsll_mg.inp \
      -o o2_uks_pbe_drsll_mg.out
  ```
- **Date:** 2026-08-01. SCF converged in 14 steps.
- **Wall (CP2K, 1 rank × 8 OMP):** **34.247 s** (`CP2K` timer TOTAL TIME).

| Quantity | CP2K pin | Regression tolerance |
|----------|----------|----------------------|
| Total energy [Ha] | **−31.815654475057819** | **1e−3 Ha** |
| Dispersion energy \(E_c^{\mathrm{nl}}\) [Ha] | **+0.11311984819209** | **1e−4 Ha** |
| ⟨S²⟩ (single det.) | 2.005576 | (not gated; `gpw-uks-pbe` pin) |

Key log lines:

```
 Dispersion energy:                                            0.11311984819209
 Total energy:                                               -31.81565447505782
 Ideal and single determinant S**2 :                    2.000000       2.005576
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.815654475057819
```

## Gates (measured 2026-08-01, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| \|E_tot(mg) − CP2K\| | **7.2e−5 Ha** | **1e−3 Ha** |
| \|E_disp(mg) − CP2K\| | **1.3e−5 Ha** | **1e−4 Ha** |
| \(E_c^{\mathrm{nl}} \neq 0\) on `e_disp` | +0.113106 Ha | hard |

ATV values: E_tot = −31.815726663481 Ha; E_disp = +0.113106483338 Ha;
SCF converged in 63 DIIS iters.

## Perf gate

CP2K QS wall **34.247 s** (1 × 8 OMP, NGRIDS 4). Atomiverse multigrid wall
**152.0 s** (OMP=8, `OMP_PROC_BIND=close`, same host). Ratio
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 4.44\) (within_10x). Primary
deliverable is pin parity of the triple combo; SC RPS costs
\(O(N_{\mathrm{SCF}}\times n_q)\) FFTs every SCF iteration.

## Citations

- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF (DRSLL).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) —
  RPS FFT evaluation.
- T. Thonhauser et al., *Phys. Rev. B* **76**, 125112 (2007) — SC V_nl.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
