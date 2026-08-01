# Reference — multi-k GPW RKS **optB88-vdW** (OptB88 + DRSLL)

## Method

**optB88-vdW** (Klimeš, Bowler, Michaelides, *J. Phys.: Condens. Matter*
**22**, 022201 (2010)) on the **Monkhorst–Pack multi-k RKS** GPW path:

| Piece | Implementation |
| ----- | -------------- |
| Exchange | Optimized B88 (`libxc` `GGA_X_OPTB88_VDW`) via `XcKind::OptB88VdW` |
| Local correlation | PW92 LDA (`xc_grid_optb88_vdw`) |
| Nonlocal correlation | DRSLL / vdW-DF DF1 self-consistent (`GpwDft::vdw_df`) |

Public API:

```cpp
gpw.xc = cpu::gpw::XcKind::OptB88VdW;
gpw.vdw_df = true;
gpw.k_mesh = {2, 1, 1};   // or explicit k_points
```

The multi-k DIIS runner already dispatches `xc_grid_optb88_vdw` and folds
self-consistent \(V_{\mathrm{nl}}\) on the k-summed density (same placement as
`gpw-vdw-kpoints`). This slug pins the **OptB88 × multi-k** product surface
that previously only had a Γ RKS pin (`reg_gpw-optb88-vdw`).

CP2K twin surface (2026.1 modern libxc subsections):

```
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

Input: `tools/cp2k_oracle/inputs/gpw-optb88-kpoints/water_dimer_optb88_vdw_kp.inp`.

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision `git:5e54ba2`,
  `cp2kflags: … libxc …`, linked **libxc 7.0.0**). Arrhenius source install:
  `source …/cp2k/install/env.sh`.
- **Input:** `tools/cp2k_oracle/inputs/gpw-optb88-kpoints/water_dimer_optb88_vdw_kp.inp`
  (water dimer, 12 Å cell, OptB88+PW92+DRSLL, DZVP-MOLOPT-SR-GTH + GTH-PBE,
  CUTOFF 200, NGRIDS 4 / REL_CUTOFF 40, EPS_SCF 1e-7, DIAGONALIZATION +
  DIRECT_P_MIXING, MP 2×1×1).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  export OMP_NUM_THREADS=8
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  mpirun -np 1 cp2k.psmp -i water_dimer_optb88_vdw_kp.inp \
      -o water_dimer_optb88_vdw_kp.out
  ```
- **Date:** 2026-08-01. Key lines
  (`grep 'ENERGY| Total\|Dispersion energy'`):

| Quantity | CP2K MP 2×1×1 NGRIDS 4 |
|----------|------------------------|
| E_tot [Ha] | **−34.487914403936173** |
| Dispersion energy [Ha] | **+0.15206184081319** |
| SCF | converged, 17 steps |
| wall (1 MPI × OMP=8) | **12.59 s** (`/usr/bin/time`; CP2K timer 12.225 s) |

Log: `/tmp/atv-cp2k-gpw-optb88-kpoints/water_dimer_optb88_vdw_kp.out`.

## Geometry

Same water dimer as `gpw-optb88-vdw` / `gpw-vdw-kpoints`:

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
`k_time_reversal=false`.

## Gates (measured 2026-08-01, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| G1 \|E_tot(kΓ) − E_tot(Γ)\| | **5.8e-10 Ha** | ≤ 1e-8 Ha |
| G1 \|E_disp(kΓ) − E_disp(Γ)\| | **5.8e-10 Ha** | ≤ 1e-8 Ha |
| G1 dual \|E_optB88 − E_PBE\| (kΓ) | **0.303 Ha** | > 1e-4 Ha |
| G2 \|E_disp(MP mg) − CP2K\| | **3.77e-5 Ha** | ≤ 1e-4 Ha |
| G2 \|E_tot(MP mg) − CP2K\| | **6.43e-3 Ha** | ≤ 1e-2 Ha |

ATV G2 values: E_tot = −34.481480310607 Ha, E_disp = +0.152024156621 Ha.
The absolute E_tot offset is the **pre-existing Γ-dimer twin-family systematic**
(same class as `gpw-vdw-kpoints` / `gpw-vdw-multigrid` ~8e-3 Ha); the capability
statements are the Γ-limit identity, OptB88 dual gate, and E_disp pin.

## Out of scope here

- Γ-UKS optB88-vdW (`gpw-optb88-uks`).
- Multi-k UKS optB88 (hold behind UKS pin).
- Analytic k-point vdW force/stress gates.
- CUDA.

## Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K MP 2×1×1 NGRIDS 4 | **12.59 s** |
| ATV G2 only (MP multigrid) | **8.4 s** |
| ATV full regression (G1×3 + G2) | **58.4 s** (≤ 60 s budget) |

\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.67\) on the G2 parity twin (≤ 10×;
no exemption needed).
