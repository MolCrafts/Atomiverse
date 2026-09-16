# Reference — multi-k collinear UKS GPW PBE + vdW-DF (DRSLL)

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Monkhorst-Pack
k-point collinear UKS** GPW PBE. \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\)
is evaluated on the finest-grid k-summed **total** density
\(\rho=\rho_\alpha+\rho_\beta\) and added identically to both spin
potentials every SCF iteration (`build_uks_k_energy` + DIIS twin) **before**
`last_veff_ab` / multigrid restriction; post-SCF \(E_c^{\mathrm{nl}}\) is
reported via `e_disp` (`fold_dispersion_k_uks`). Placement mirrors Γ UKS
(`gpw-vdw-uks`) and multi-k RKS (`gpw-vdw-kpoints`).

Public API: `n_alpha`/`n_beta` + `vdw_df` (also `vdw_df2` / `rvv10`) together
with `k_mesh` / explicit `k_points` become legal for collinear UKS
(single-grid or REL_CUTOFF multigrid).

CP2K twin surface: `LSD` + `&KPOINTS SCHEME MONKHORST-PACK 2 1 1` +
`&MGRID NGRIDS 4 / REL_CUTOFF 40` + `&VDW_POTENTIAL NON_LOCAL TYPE DRSLL`.
KPOINTS forces diagonalization SCF (no OT) on the CP2K side; ATV uses the
multi-k UKS DIIS path.

## Offline CP2K pin (primary — closed-shell LSD MULT 1 water dimer)

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision **git:5e54ba2**),
  Arrhenius local install via `…/cp2k/install/env.sh`.
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-uks-kpoints/water_dimer_uks_vdw_df_kp.inp`
  (water dimer, 12 Å cell, LSD MULTIPLICITY 1, PBE, DZVP-MOLOPT-SR-GTH +
  GTH-PBE, CUTOFF 200, NGRIDS 4 / REL_CUTOFF 40, EPS_SCF 1e-7, EPS_DEFAULT
  1e-10, DIAGONALIZATION + DIRECT_P_MIXING; DRSLL kernel
  `vdW_kernel_table.dat`; MP 2×1×1).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  cd /tmp/run && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 mpirun -np 1 cp2k.psmp \
      -i water_dimer_uks_vdw_df_kp.inp -o water_dimer_uks_vdw_df_kp.out
  ```
- **Date:** 2026-08-01. Host `arrhenius1`. Key lines:

| Quantity | CP2K MP 2×1×1 NGRIDS 4 LSD MULT 1 |
|----------|-------------------------------------|
| E_tot [Ha] | **−34.186333514352576** |
| Dispersion energy [Ha] | **+0.15193559059130** |
| SCF | converged, 14 steps |
| wall (1 MPI × 8 OMP, CP2K timer) | **60.2 s** (`/usr/bin/time` 60.17 s) |

Context: the closed-shell RKS twin (`gpw-vdw-kpoints`) is E_tot
−34.186333514352548 Ha / E_disp +0.15193559113836 Ha — LSD MULT 1 matches
RKS to ~1e-16 Ha on total energy (spin-degeneracy).

## Secondary offline pin — O₂ triplet multi-k UKS + DRSLL

Open-shell product documentation pin (not gated in the regression wall
budget; same dual-spin V_nl fold as the primary).

- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-uks-kpoints/o2_uks_vdw_df_kp.inp`
  (O₂ R=1.2075 Å, 12 Å cell, LSD MULT 3 → Nα=7 Nβ=5, CUTOFF 200, NGRIDS 4 /
  REL_CUTOFF 40, MP 2×1×1, DZVP-MOLOPT-GTH + GTH-PBE-q6).
- **Date:** 2026-08-01. Same binary `git:5e54ba2`.

| Quantity | CP2K |
|----------|------|
| E_tot [Ha] | **−31.815384846475904** |
| Dispersion energy [Ha] | **+0.11306034917458** |
| SCF | converged, 45 steps |
| wall (1 × 8 OMP) | **152.6 s** (`CP2K` timer 151.7 s) |

## Geometry (primary)

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
G1 single-grid; G2 `ngrids=4`, `rel_cutoff=40`, `k_mesh={2,1,1}`,
`k_time_reversal=false`, `n_alpha=n_beta=8`.

## Gates (measured 2026-08-01, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| G1 \|E_tot(kΓ-UKS) − E_tot(Γ-UKS)\| | **3.1e-13 Ha** | ≤ 1e-8 Ha |
| G1 \|E_disp(kΓ-UKS) − E_disp(Γ-UKS)\| | **5.2e-15 Ha** | ≤ 1e-8 Ha |
| G2 \|E_disp(MP UKS mg) − CP2K\| | **4.00e-5 Ha** | ≤ 1e-4 Ha |
| G2 \|E_tot(MP UKS mg) − CP2K\| | **7.93e-3 Ha** | ≤ 1e-2 Ha |
| G3 \|E_UKS − E_RKS\| multi-k | **5.2e-8 eV** | ≤ 3e-4 eV |
| G3 \|E_disp UKS − RKS\| multi-k | **1.9e-7 Ha** | ≤ 1e-5 Ha |

ATV G2 values: E_tot = −34.178407376349 Ha, E_disp = +0.151895614211 Ha.
The absolute E_tot offset is the **pre-existing Γ-dimer twin-family systematic**
(identical to `gpw-vdw-kpoints` / `gpw-vdw-multigrid` Δ ≈ 7.93e-3 Ha).

## Out of scope here

- Dedicated multi-k UKS open-shell O₂ absolute ATV re-pin in ctest (see
  secondary pin above; wall ~2–3 min class).
- Analytic k-point UKS vdW force/stress gates.
- CUDA.

## Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K MP 2×1×1 NGRIDS 4 LSD MULT 1 (primary twin) | **60.2 s** |
| ATV G2 only (MP multigrid UKS) | **21.2 s** |
| ATV full regression (G1 + G2 + G3) | **173.1 s** |

\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.35\) on the G2 parity twin (≤ 10×;
no exemption needed).

## Citations

- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF (DRSLL).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) —
  RPS FFT evaluation.
- T. Thonhauser et al., *Phys. Rev. B* **76**, 125112 (2007) — SC V_nl.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
