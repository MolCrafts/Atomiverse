# Reference — multi-k collinear UKS GPW **optB88-vdW** (OptB88 + DRSLL)

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Monkhorst–Pack
k-point collinear UKS** with semi-local **optB88-vdW** exchange–correlation:

| Piece | Implementation |
| ----- | -------------- |
| Exchange | Optimized B88 (`libxc` `GGA_X_OPTB88_VDW`) via `XcKind::OptB88VdW` |
| Local correlation | PW92 LDA (`xc_grid_optb88_vdw_spin`) |
| Nonlocal correlation | DRSLL / vdW-DF DF1 self-consistent (`GpwDft::vdw_df`) |

\(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) is evaluated on the
finest-grid k-summed **total** density \(\rho=\rho_\alpha+\rho_\beta\) and
added identically to both spin potentials every SCF iteration
(`build_uks_k_energy` + DIIS twin) **before** `last_veff_ab` / multigrid
restriction; post-SCF \(E_c^{\mathrm{nl}}\) is reported via `e_disp`
(`fold_dispersion_k_uks`). Placement mirrors Γ UKS optB88 (`gpw-optb88-uks`)
and multi-k RKS optB88 (`gpw-optb88-kpoints`).

Public API:

```cpp
gpw.xc = cpu::gpw::XcKind::OptB88VdW;
gpw.vdw_df = true;
gpw.n_alpha = 8;  // closed-shell water dimer (16 valence e⁻)
gpw.n_beta  = 8;
gpw.k_mesh = {2, 1, 1};   // or explicit k_points
```

This slug pins the **OptB88 × multi-k UKS** product surface that previously
only had Γ UKS (`reg_gpw-optb88-uks`) and multi-k RKS (`reg_gpw-optb88-kpoints`)
pins; the dual-spin k-path + spin XC + DRSLL wiring was already product.

CP2K twin surface (2026.1 modern libxc subsections):

```
LSD
MULTIPLICITY 1
&KPOINTS SCHEME MONKHORST-PACK 2 1 1
&XC_FUNCTIONAL
  &GGA_X_OPTB88_VDW
  &END GGA_X_OPTB88_VDW
  &PW92
  &END PW92
&END XC_FUNCTIONAL
&VDW_POTENTIAL POTENTIAL_TYPE NON_LOCAL
  &NON_LOCAL TYPE DRSLL KERNEL_FILE_NAME vdW_kernel_table.dat
```

Input: `tools/cp2k_oracle/inputs/gpw-optb88-uks-kpoints/water_dimer_uks_optb88_vdw_kp.inp`.

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision `git:5e54ba2`,
  `cp2kflags: … libxc …`, linked **libxc 7.0.0**). Arrhenius source install:
  `source …/cp2k/install/env.sh`.
- **Input:** `tools/cp2k_oracle/inputs/gpw-optb88-uks-kpoints/water_dimer_uks_optb88_vdw_kp.inp`
  (water dimer, 12 Å cell, LSD MULTIPLICITY 1, OptB88+PW92+DRSLL,
  DZVP-MOLOPT-SR-GTH + GTH-PBE, CUTOFF 200, NGRIDS 4 / REL_CUTOFF 40,
  EPS_SCF 1e-7, DIAGONALIZATION + DIRECT_P_MIXING, MP 2×1×1).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  export OMP_NUM_THREADS=8
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  mpirun -np 1 cp2k.psmp -i water_dimer_uks_optb88_vdw_kp.inp \
      -o water_dimer_uks_optb88_vdw_kp.out
  ```
- **Date:** 2026-08-01. Host `arrhenius1`. Key lines
  (`grep 'ENERGY| Total\|Dispersion energy'`):

| Quantity | CP2K MP 2×1×1 NGRIDS 4 LSD MULT 1 |
|----------|-------------------------------------|
| E_tot [Ha] | **−34.487914403936173** |
| Dispersion energy [Ha] | **+0.15206184094414** |
| SCF | converged, 15 steps |
| wall (1 MPI × OMP=8) | **13.29 s** (`/usr/bin/time`; CP2K timer 12.867 s) |

Log: `/tmp/atv-cp2k-gpw-optb88-uks-kpoints/water_dimer_uks_optb88_vdw_kp.out`.

Context: the closed-shell multi-k RKS twin (`gpw-optb88-kpoints`) is E_tot
−34.487914403936173 Ha / E_disp +0.15206184081319 Ha — LSD MULT 1 matches
RKS to printed total-energy digits (spin-degeneracy); E_disp differs by
~1e−10 Ha (harmless float noise).

```
 Dispersion energy:                                            0.15206184094414
 Total energy:                                               -34.48791440393617
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -34.487914403936173
 CP2K                                 1  1.0    0.009    0.009   12.867   12.867
```

## Geometry

Same water dimer as `gpw-optb88-kpoints` / `gpw-vdw-uks-kpoints`:

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
`k_time_reversal=false`, `n_alpha=n_beta=8`, `XcKind::OptB88VdW`.

## Gates (measured 2026-08-01, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| G1 \|E_tot(kΓ-UKS) − E_tot(Γ-UKS)\| | **1.4e-14 Ha** | ≤ 1e-8 Ha |
| G1 \|E_disp(kΓ-UKS) − E_disp(Γ-UKS)\| | **1.1e-14 Ha** | ≤ 1e-8 Ha |
| G1 dual \|E_optB88 − E_PBE\| (kΓ-UKS) | **0.303 Ha** | > 1e-4 Ha |
| G2 \|E_disp(MP UKS mg) − CP2K\| | **3.77e-5 Ha** | ≤ 1e-4 Ha |
| G2 \|E_tot(MP UKS mg) − CP2K\| | **6.43e-3 Ha** | ≤ 1e-2 Ha |
| G3 \|E_UKS − E_RKS\| multi-k OptB88 | **1.9e-13 eV** | ≤ 3e-4 eV |
| G3 \|E_disp UKS − RKS\| multi-k | **5.3e-16 Ha** | ≤ 1e-5 Ha |

ATV G2 values: E_tot = −34.481480313875 Ha, E_disp = +0.152024156312 Ha.
The absolute E_tot offset is the **pre-existing Γ-dimer twin-family systematic**
(same class as `gpw-optb88-kpoints` / `gpw-vdw-uks-kpoints` ~6–8e-3 Ha); the
capability statements are the Γ-limit identity, OptB88 dual gate, E_disp pin,
and multi-k UKS≡RKS identity.

## Out of scope here

- Open-shell multi-k UKS O₂ absolute ATV re-pin in ctest (Γ O₂ pin lives in
  `gpw-optb88-uks`; multi-k open-shell wall is multi-minute class).
- Analytic k-point UKS vdW force/stress gates.
- CUDA.

## Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K MP 2×1×1 NGRIDS 4 LSD MULT 1 (primary twin) | **13.29 s** |
| ATV G2 only (MP multigrid UKS OptB88) | **9.8 s** |
| ATV full regression (G1×3 + G2 + G3) | **212.7 s** |

\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.73\) on the G2 parity twin (≤ 10×;
no exemption needed).

## Citations

- J. Klimeš, D. R. Bowler, A. Michaelides, *J. Phys.: Condens. Matter*
  **22**, 022201 (2010) — optB88-vdW.
- M. Dion et al., *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF (DRSLL).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) — RPS.
- T. Thonhauser et al., *Phys. Rev. B* **76**, 125112 (2007) — SC V_nl.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- CP2K stock section: `data/xc_section/vdW-DF-optB88.sec`.
