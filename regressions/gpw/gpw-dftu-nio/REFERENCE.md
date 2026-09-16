# Reference — NiO molecule Γ-UKS GPW + Dudarev DFT+U (PADE / SZV-GTH-PADE)

## Method

Unrestricted Kohn–Sham GPW at the Γ point with Martyna–Tuckerman isolated
electrostatics, PADE LSD, and **Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken) on
Ni \(d\) (\(L=2\), \(U-J=6.0\) eV).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `n_alpha`/`n_beta`, `plus_u`,
single density grid.

**Note:** solid rocksalt AFM NiO (2 f.u.) did **not** converge under SZV /
Γ-only in CP2K (diag+Broyden or no-U restart) within 200–300 steps. The
**molecule** is the first-green PLUS_U foothold; solid AFM is follow-on
(k-points / larger basis).

## System

| Item | Value |
|------|--------|
| Molecule | NiO, **centered** in cell |
| Bond length | R(Ni–O) = **1.668 Å** (along *z*) |
| Coordinates (Å) | Ni (5, 5, 4.166), O (5, 5, 5.834) |
| Cell | **10 × 10 × 10 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | Ni `SZV-GTH-PADE-q18`, O `SZV-GTH-PADE` |
| PP | Ni `GTH-PADE-q18` (18e), O `GTH-PADE-q6` (6e) → **24** valence e⁻ |
| XC | PADE (LDA LSD) |
| DFT+U | Ni \(L=2\), \(U-J = 6.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| Cutoff | 250 Ry (Atomiverse single grid; CP2K NGRIDS 4 / REL_CUTOFF 40) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=13, Nβ=11** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, Source code revision 85b8a9b49c),
  local install under `/nobackup/.../work/cp2k/install` (GCC 11.5, OpenMP).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-nio/nio_mol_pade_u.inp`
- **Run (offline only; never in ctest):**
  ```bash
  export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
  export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
  OMP_NUM_THREADS=8 cp2k.psmp -i nio_mol_pade_u.inp -o nio_mol.out
  ```
- **Date:** 2026-07-24
- **CP2K SCF:** OT `FULL_SINGLE_INVERSE` + OUTER_SCF (diag+Broyden did not
  converge this twin within budget).
- **Wall (CP2K, OMP=8):** ≈ **15.79 s**

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−184.92466076474312** | **5e-2 Ha** (single-grid vs multigrid+OT) |
| DFT+U energy [Ha] | **0.05230865222345** | diagnostic (not asserted) |
| ⟨S²⟩ (single det.) | **2.019198** | **0.25** |
| Ideal S(S+1) | 2.000000 | — |
| Mulliken spin m_Ni | **1.177632** | **0.5** (and m_Ni > 0.3) |
| Mulliken spin m_O | 0.822368 | — |

Key log lines:

```
  DFT+U energy:                                                 0.05230865222345
  Total energy:                                              -184.92466076474312
  Ideal and single determinant S**2 :                    2.000000       2.019198
  Integrated absolute spin density:                                 2.0728941742
       1     Ni       1         9.343192     8.165560     0.491248     1.177632
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.924660764743123
```

## Atomiverse (2026-07-24, Release, OMP=8)

| Quantity | Value |
|----------|--------|
| E_tot [Ha] | ≈ −184.94637 |
| \|ΔE\| vs CP2K | ≈ **2.2e-2 Ha** (within 5e-2) |
| ⟨S²⟩ | ≈ 2.016 |
| m_Ni (Mulliken) | ≈ 1.07 |
| SCF iters | ≈ 56 |
| Wall | ≈ **3.3 s** |

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 15.79 | OMP=8, OT + OUTER_SCF, multigrid |
| Atomiverse | 3.29 | Release, single density grid, OMP=8 |

`ratio = T_atv / T_cp2k = 3.29/15.79 ≈ **0.21**` (≤ 10 ✓).

## Citations

- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U.
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified rotationally invariant U.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
