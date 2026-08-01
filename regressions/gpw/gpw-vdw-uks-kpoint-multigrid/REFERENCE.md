# Reference — multi-k collinear UKS GPW PBE + vdW-DF (DRSLL) on REL_CUTOFF multigrid

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Monkhorst–Pack
k-point collinear UKS** GPW PBE with **CP2K-style CUTOFF/REL_CUTOFF multigrid
product collocation** (`ngrids > 1 && rel_cutoff > 0`).

Each pairwise axis is already product:

| Axis | Slug |
|------|------|
| multi-k UKS + DRSLL | `gpw-vdw-uks-kpoints` |
| Γ UKS + DRSLL + multigrid | `gpw-vdw-uks-multigrid` |
| multi-k UKS + multigrid (no nonlocal) | `gpw-uks-multigrid-kpoints` |
| multi-k RKS + DRSLL + multigrid | `gpw-vdw-kpoints` / `gpw-vdw-multigrid` |

This slug pins the **triple**: \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\)
of the k-summed **total** density \(\rho=\rho_\alpha+\rho_\beta\) is
spin-independent and is added identically to both spin potentials on the
**fine** grid every SCF iteration (`build_uks_k_energy` + DIIS twin)
**before** per-level restriction / phase-summed \(V_k^\sigma\) integration —
so the dual KS matrices stay the exact gradient of the collocated
\(E_H + E_{xc} + E_c^{\mathrm{nl}}\) (variational SCF). Post-SCF
\(E_c^{\mathrm{nl}}\) lands on `GpwResult::e_disp` via
`fold_dispersion_k_uks` (multigrid product density through
`collocate_rho_spin_k`).

Public API: `n_alpha`/`n_beta` + `vdw_df` (also `vdw_df2` / `rvv10`) +
`k_mesh` / explicit `k_points` + `ngrids` + `rel_cutoff` on
`cpu::gpw::GpwDft` / `GPWEvaluator`.

CP2K twin surface: `LSD` + `&KPOINTS SCHEME MONKHORST-PACK 2 1 1` +
`&MGRID NGRIDS 4 / REL_CUTOFF 40` + `&VDW_POTENTIAL NON_LOCAL TYPE DRSLL`.
KPOINTS forces diagonalization SCF (no OT) on the CP2K side; ATV uses the
multi-k UKS DIIS path.

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision **git:5e54ba2**),
  Arrhenius local install via `…/cp2k/install/env.sh`.
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-uks-kpoint-multigrid/water_dimer_uks_vdw_df_kp_mg.inp`
  (water dimer, 12 Å cell, LSD MULTIPLICITY 1, PBE, DZVP-MOLOPT-SR-GTH +
  GTH-PBE, CUTOFF 200, NGRIDS 4 / REL_CUTOFF 40, EPS_SCF 1e-7, EPS_DEFAULT
  1e-10, DIAGONALIZATION + DIRECT_P_MIXING; DRSLL kernel
  `vdW_kernel_table.dat`; MP 2×1×1).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  cd /tmp/run && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 mpirun -np 1 cp2k.psmp \
      -i water_dimer_uks_vdw_df_kp_mg.inp -o water_dimer_uks_vdw_df_kp_mg.out
  ```
- **Date:** 2026-08-01. Host `arrhenius1`. Key lines:

| Quantity | CP2K MP 2×1×1 NGRIDS 4 LSD MULT 1 |
|----------|-------------------------------------|
| E_tot [Ha] | **−34.186333514352619** |
| Dispersion energy [Ha] | **+0.15193559105517** |
| SCF | converged |
| wall (1 MPI × 8 OMP, CP2K timer TOTAL TIME) | **13.703 s** (`/usr/bin/time` REAL_SEC 14.09 s) |

```
 Dispersion energy:                                            0.15193559105517
 Total energy:                                               -34.18633351435262
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -34.186333514352619
 CP2K                                 1  1.0    0.008    0.008   13.703   13.703
```

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell **ABC 12.0 12.0 12.0 Å**, PERIODIC XYZ. Basis `DZVP-MOLOPT-SR-GTH`,
PP `GTH-PBE-q1`/`GTH-PBE-q6`. ATV: CUTOFF 200 Ry, DIIS, tol 1e-7 Ha;
G1 single-grid multi-k UKS; G2 `ngrids=4`, `rel_cutoff=40`, `k_mesh={2,1,1}`,
`k_time_reversal=false`, `n_alpha=n_beta=8`.

## Gates (measured 2026-08-01, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| G1 \|E_tot(mg) − E_tot(sg)\| multi-k UKS | **4.27e-6 Ha** | ≤ 2e-5 Ha |
| G1 \|E_disp(mg) − E_disp(sg)\| multi-k UKS | **1.55e-7 Ha** | ≤ 2e-5 Ha |
| G2 \|E_disp(MP UKS mg) − CP2K\| | **4.00e-5 Ha** | ≤ 1e-4 Ha |
| G2 \|E_tot(MP UKS mg) − CP2K\| | **7.93e-3 Ha** | ≤ 1e-2 Ha |

ATV G2 values: E_tot = −34.178407376349 Ha, E_disp = +0.151895614211 Ha.
The absolute E_tot offset is the **pre-existing Γ-dimer twin-family systematic**
(identical envelope to `gpw-vdw-kpoints` / `gpw-vdw-uks-kpoints` /
`gpw-vdw-multigrid` Δ ≈ 7.9e-3 Ha).

## Out of scope here

- Open-shell multi-k UKS O₂ absolute ATV re-pin (see `gpw-vdw-uks-kpoints`
  secondary pin; wall ~2–3 min class).
- Analytic k-point UKS vdW force/stress gates.
- CUDA.

## Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K MP 2×1×1 NGRIDS 4 LSD MULT 1 (twin) | **13.703 s** |
| ATV G2 only (MP multigrid UKS) | **11.6 s** |
| ATV full regression (G1 + G2) | **47.5 s** |

\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.85\) on the G2 parity twin (≤ 10×;
no exemption needed — CP2K implements the matched algorithm).

## Citations

- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF (DRSLL).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) —
  RPS FFT evaluation.
- T. Thonhauser et al., *Phys. Rev. B* **76**, 125112 (2007) — SC V_nl.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
