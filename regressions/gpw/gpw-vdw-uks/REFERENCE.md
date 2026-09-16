# Reference — Γ collinear-UKS GPW PBE + nonlocal vdW-DF (DRSLL)

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Γ collinear UKS**
(LSD). \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) is a functional
of the **total** density \(\rho=\rho_\alpha+\rho_\beta\) and is
spin-independent, so it is added identically to both spin potentials every
SCF iteration (`build_uks_H`; CP2K `qs_dispersion_nonloc`). The
self-consistent \(E_c^{\mathrm{nl}}\) is folded onto `GpwResult::e_disp`
[Hartree] at every UKS exit (DIIS, OT, fixed-density, budget-exhaust).

Public API: `n_alpha`/`n_beta` + `vdw_df` (also `vdw_df2` / `rvv10`) on
`cpu::gpw::GpwDft` / `GPWEvaluator`. This removes the former
`GpwDft::build` Γ-UKS fail-fast ("E_c^nl would be silently dropped");
the **multi-k** + nonlocal-vdW reject remains (no nonlocal fold on the k
path — `gpw-vdw-kpoints` gate).

Extends `gpw-vdw-scf-potential` (SC V_nl, Γ RKS) × `gpw-uks-pbe`
(O₂ triplet Γ-UKS PBE twin).

## Part A — O₂ triplet LSD PBE + DRSLL vs offline CP2K

| Item | Value |
|------|-------|
| Molecule | O₂, R(O–O) = 1.2075 Å, centered along *z* |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, **PERIODIC XYZ** (RPS kernel needs periodic Poisson) |
| Basis / PP | DZVP-MOLOPT-GTH + GTH-PBE-q6 (12 valence e⁻) |
| XC | PBE + `&NON_LOCAL TYPE DRSLL` (vdW_kernel_table.dat, SCALE 1.0) |
| Cutoff | CP2K: 400 Ry / REL_CUTOFF 60 / NGRIDS 4; Atomiverse: 400 Ry single grid |
| Spin | LSD, MULTIPLICITY 3 → Nα = 7, Nβ = 5 |
| SCF | diagonalization + Broyden (CP2K), EPS_SCF 1e−6; ATV DIIS, tol 1e−7 |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (Development Version, source revision
  **85b8a9b49c**, `cp2k.psmp`, `~/work/cp2k/install` on Arrhenius epyc,
  GCC 11.5 / MPICH; libxc + fftw3 flags).
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-uks/o2_uks_pbe_drsll.inp`
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2
  export LD_LIBRARY_PATH=$HOME/work/cp2k/install/lib:$LD_LIBRARY_PATH
  cd /tmp/run && ln -sf $DATA/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 ~/work/cp2k/install/bin/cp2k.psmp -i o2_uks_pbe_drsll.inp \
      -o o2_uks_pbe_drsll.out
  ```
- **Date:** 2026-07-31. SCF converged in 14 steps.
- **Wall (CP2K, 1 rank × 8 OMP):** **159.5 s** (`CP2K` timer).

| Quantity | CP2K pin | Regression tolerance |
|----------|----------|----------------------|
| Total energy [Ha] | **−31.815658522210054** | **1e−3 Ha** (single-grid vs NGRIDS-4 twin; `gpw-uks-pbe` precedent) |
| Dispersion energy \(E_c^{\mathrm{nl}}\) [Ha] | **+0.11311983030413** | **1e−4 Ha** (vdW-family precedent; observed 1e−5 class) |
| ⟨S²⟩ (single det.) | 2.005576 | (not gated here; pinned by `gpw-uks-pbe`) |

Key log lines:

```
 Dispersion energy:                                            0.11311983030413
 Total energy:                                               -31.81565852221005
 Ideal and single determinant S**2 :                    2.000000       2.005576
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.815658522210054
```

## Part B — closed-shell spin-degeneracy identity

Water dimer (same geometry as `gpw-vdw-df-ice` / `gpw-vdw-scf-potential`;
12 Å cubic periodic cell, DZVP-MOLOPT-SR-GTH + GTH-PBE-q1/q6, 200 Ry,
PBE + `vdw_df`, tol 1e−7): 16 valence e⁻; UKS with nα = nβ = 8 must
reproduce RKS with n_occ = 8 (double-occupied).

| Gate | Tolerance |
|------|-----------|
| \(\lvert E_{\mathrm{UKS}}-E_{\mathrm{RKS}}\rvert\) | **3e−4 eV** |
| \(\lvert E_{c,\mathrm{UKS}}^{\mathrm{nl}}-E_{c,\mathrm{RKS}}^{\mathrm{nl}}\rvert\) | **1e−5 Ha** |
| \(E_c^{\mathrm{nl}} \neq 0\) folded on `e_disp` | hard (the old fail-fast's silent-drop scenario) |

### Atomiverse numbers (verified first green 2026-07-31, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| \|E_tot − CP2K\| (Part A) | **6.8e−5 Ha** | 1e−3 Ha |
| \|E_disp − CP2K\| (Part A) | **1.3e−5 Ha** | 1e−4 Ha |
| \|E_UKS − E_RKS\| (Part B) | **2.6e−8 eV** | 3e−4 eV |
| \|E_disp UKS − RKS\| (Part B) | **1.9e−7 Ha** | 1e−5 Ha |
| SCF | A: 55 iters, B: both converged | |

## Perf gate

CP2K QS wall **159.5 s** (1 × 8 OMP, NGRIDS 4). Atomiverse Part A wall
**~196 s** (OMP=8, single-grid 400 Ry; Part B +21 s). Ratio
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 1.23\) (within_10x).
vdW-family policy (`gpw-vdw-scf-potential` / `gpw-uks-pbe` 177 s twin):
primary deliverable is pin parity; SC RPS costs
\(O(N_{\mathrm{SCF}}\times n_q)\) FFTs every SCF iteration.

## Citations

- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF (DRSLL).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) —
  RPS FFT evaluation.
- T. Thonhauser et al., *Phys. Rev. B* **76**, 125112 (2007) — SC V_nl.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
