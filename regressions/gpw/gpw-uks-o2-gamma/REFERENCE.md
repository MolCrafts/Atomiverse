# Reference — Γ-point UKS GPW O₂ (PADE / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham GPW (Quickstep LSD) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved PADE (Goedecker–Teter–Hutter Padé LDA),
GTH-PBE-q6 pseudopotential, DZVP-MOLOPT-GTH Gaussian basis.

Atomiverse path: `cpu::gpw::GpwDft` with `n_alpha`/`n_beta`, dual KS matrices,
`xc_grid_pade_lsd`, Löwdin ⟨S²⟩.

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
| XC | PADE (LDA LSD) |
| Cutoff | 400 Ry (Atomiverse single density grid; CP2K twin also NGRIDS 4 / REL_CUTOFF 60) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1`
  on Arrhenius (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-o2-gamma/o2_uks_pade.inp`
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i o2_uks_pade.inp -o o2_uks_pade.out
  ```
- **Date:** 2026-07-23 (centered geometry re-pin)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.76980349931425** | **1e-3 Ha** (first green; single-grid vs CP2K multigrid) |
| ⟨S²⟩ (single det.) | **2.004659** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |
| Integrated spin density | 2.000000 | — |

Key log lines:

```
Total energy:                                               -31.76980349931425
Ideal and single determinant S**2 :                    2.000000       2.004659
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314250
```

## Perf gate

This file carries no wall numbers. The single binding fair-wall source for
this twin — host, CP2K binary provenance, PRINT_LEVEL/timer convention,
reproduction recipe, and the current \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) —
is [`.claude/notes/gpw-perf-baseline.md`](../../../.claude/notes/gpw-perf-baseline.md).
Case-local walls were last measured here 2026-07-23 and are superseded by
that note.

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — ⟨S²⟩ for unrestricted single dets.
