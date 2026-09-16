# Reference — Γ-point UKS GPW multigrid O₂ (PADE / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham GPW at Γ with Martyna–Tuckerman isolated electrostatics
and **CP2K-style multigrid product collocation** (`ngrids=4`, `rel_cutoff=60`).
Spin-resolved PADE (GTH Padé LDA), GTH-PBE-q6, DZVP-MOLOPT-GTH.

Atomiverse path: `cpu::gpw::GPWEvaluator` / `GpwDft` with `n_alpha`/`n_beta`,
`ngrids`, `rel_cutoff` — dual KS matrices via product collocation +
`prolong_fft` / `restrict_fft` (same machinery as RKS multigrid).

## System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (6 valence e⁻ / O → 12 total) |
| XC | PADE (LDA LSD) |
| Cutoff | **400 Ry** |
| Multigrid | **NGRIDS 4**, **REL_CUTOFF 60** Ry (ATV + CP2K twin) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1`
  on Arrhenius (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-multigrid/o2_uks_pade_mg.inp`
  (identical physics twin to `gpw-uks-o2-gamma/o2_uks_pade.inp`: same geometry,
  CUTOFF 400 / NGRIDS 4 / REL_CUTOFF 60 / PADE / LSD / MT).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i o2_uks_pade_mg.inp -o o2_uks_pade_mg.out
  ```
- **Date:** 2026-07-23 (centered geometry; pin shared with `gpw-uks-o2-gamma`)
- **Wall (CP2K, 1 MPI × 4 OMP):** ≈ **21.3 s** total (`CP2K` timer MAXIMUM)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.76980349931425** | **1e-3 Ha** |
| ⟨S²⟩ (single det.) | **2.004659** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines (CP2K 2026.1):

```
Total energy:                                               -31.76980349931425
Ideal and single determinant S**2 :                    2.000000       2.004659
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314250
```

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 21.3 | 1 rank, OMP=4, NGRIDS 4 |
| Atomiverse | **32.8** | Release, ngrids=4, rel_cutoff=60, OMP=4 (2026-07-26) |

`ratio = T_atv / T_cp2k = 32.8/21.3 ≈ **1.54**` (≤ 10 ✓).

## Implementation note

ATV multigrid product collocation uses **spectral (FFT) inter-level transfer**
even under Martyna–Tuckerman nonperiodic Poisson: the density lives on a
uniform box grid and is well-contained for centered O₂, so FFT prolong/restrict
(the RKS product path) is the correct adjoint transfer; trilinear fallback on
`periodic=false` left an O(0.4 Ha) Hartree residual.

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- CP2K Quickstep `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS multigrid collocation.
