# Reference — Γ-UKS OT GPW O₂ (PADE / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham **GPW** at Γ with **Orbital Transformation (OT)** SCF
minimizer (`use_ot = true`), Martyna–Tuckerman isolated electrostatics,
spin-resolved PADE LSD, GTH-PBE-q6 / DZVP-MOLOPT-GTH.

Product path: `cpu::gpw::GPWEvaluator` with `n_alpha`/`n_beta` + `use_ot`.
Open-shell OT gradient uses occupation 1 per spin; joint PR+ CG + Armijo on
the rebuilt total energy (same fixed point as UKS DIIS).

OT formulation: VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003);
Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C.

Public Frame energy [eV] via `kEnergy`. Method channel:
`last_energy_hartree()` / `GpwResult::e_total` [Hartree].

## System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q6 |
| XC | PADE (LDA LSD) |
| Cutoff | **250 Ry** (regression budget; 400 Ry is `gpw-uks-o2-gamma` DIIS twin) |
| Spin | UKS, multiplicity 3 → **Nα=7, Nβ=5** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1`
  on Arrhenius (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/ot-uks-o2/o2_uks_ot_pade.inp`
  (`&OT` + `PRECONDITIONER FULL_SINGLE_INVERSE`, `MINIMIZER DIIS`, OUTER_SCF,
  CUTOFF 250).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  mpirun -np 1 cp2k.psmp -i o2_uks_ot_pade.inp -o o2_uks_ot_pade.out
  ```
- **Date:** 2026-07-26
- **Wall (CP2K, 1 MPI × 4 OMP):** **8.68 s** (`CP2K` timer MAXIMUM)

| Quantity | CP2K OT pin | Tolerance (regression) |
|----------|------------:|------------------------:|
| E_tot [Ha] | **−31.770029989852723** | **2.5×10⁻³ Ha** |
| ⟨S²⟩ (single det.) | **2.004666** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines:

```
  *** SCF run converged in    30 steps ***
  Total energy:                                               -31.77002998985272
  Ideal and single determinant S**2 :                    2.000000       2.004666
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.770029989852723
```

Cross-check: CP2K DIAG at 400 Ry (centered O₂) was −31.7698035 Ha
(`gpw-uks-o2-gamma`) — same energy class; 250 Ry OT pin is the regression twin.

## Regression gates

| Check | Tolerance |
| ----- | --------: |
| \|E_ATV,OT − E_CP2K,OT\| | **2.5×10⁻³ Ha** |
| \|E_ATV,OT − E_ATV,DIIS\| | **1×10⁻⁴ Ha** |
| \|⟨S²⟩_ATV − ⟨S²⟩_CP2K\| | **0.15** |

Milli-Ha pin tol at 250 Ry absorbs single-grid vs CP2K multigrid (NGRIDS/REL_CUTOFF)
differences; OT/DIIS in-tree is the tight fixed-point gate.

## Fair wall (perf gate)

Measured 2026-08-05 (Release, OMP=4, `OMP_PROC_BIND=close`), **MINIMIZER
DIIS** (OtOrbitalDiis SAFE directions + Armijo on total E; FULL_SINGLE-style
precond; OUTER precond refresh):

| Twin | Wall [s] | Notes |
| ---- | -------: | ----- |
| CP2K OT | **8.68** | `cp2k.psmp`, MINIMIZER DIIS + FULL_SINGLE_INVERSE + OUTER_SCF |
| Atomiverse OT only | **~62** | 180 outer iters; DIIS directions + Armijo |
| Atomiverse DIIS (parity check) | **~11** | 30 diagonalization DIIS iters |

Fair OT-only ratio: 62 / 8.68 ≈ **7.1**. Full regression (OT+DIIS) wall ≈ 73 s.
Fixed-STEPSIZE pure DIIS (no Armijo) diverges on this micro-case.

`within_10x=true` (fair OT, OMP=4).

In-tree OT/DIIS parity: |E_OT − E_DIIS| ≈ **5×10⁻¹⁰ Ha** (same grids).

## Citations

- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT.
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) — CP2K OT.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE/GTH.
