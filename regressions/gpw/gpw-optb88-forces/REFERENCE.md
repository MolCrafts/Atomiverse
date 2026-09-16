# Reference — water-dimer GPW **optB88-vdW** analytic ionic forces

## Method

**optB88-vdW** (Klimeš, Bowler, Michaelides, *J. Phys.: Condens. Matter*
**22**, 022201 (2010)) on Γ-point **GPW** with **analytic ionic forces**:

| Piece | Form |
| ----- | ---- |
| Exchange | optimized B88 (`GGA_X_OPTB88_VDW`) |
| Local correlation | PW92 LDA only |
| Nonlocal correlation | DRSLL / DF1 via SC \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) |
| Forces | Hellmann–Feynman + Pulay + grid/core; \(V_{\mathrm{nl}}\) in `veff` inside `assemble_gpw_forces` |

Public API:

```cpp
cpu::gpw::GPWEvaluator eval({O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80,
                            /*tol*/ 1e-7, /*mix*/ 0.4,
                            cpu::gpw::XcKind::OptB88VdW,
                            /*nonperiodic*/ false, /*compute_forces*/ true);
eval.ngrids = 4;
eval.rel_cutoff = 40.0;
eval.vdw_df = true;
// Frame forces [eV/Å]; last_result().e_disp [Ha]
```

Extends energy product `gpw-optb88-vdw` × SC force path `gpw-vdw-scf-potential`.

## Geometry

Same ice-relevant water dimer as `gpw-optb88-vdw` / `gpw-vdw-df-ice`:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).  
CUTOFF **200 Ry**, REL_CUTOFF **40 Ry**, NGRIDS **4**, Γ-point, \(n_{\mathrm{occ}}=8\).

## Offline CP2K ENERGY_FORCE pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:**5e54ba2**, `cp2kflags: … libxc …`) |
| Host | Arrhenius epyc (offline oracle; **not** invoked from ctest) |
| Install | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install` + `env.sh` |
| Input | `tools/cp2k_oracle/inputs/gpw-optb88-forces/water_dimer_optb88_vdw_force.inp` |
| RUN_TYPE | ENERGY_FORCE |
| XC | `&GGA_X_OPTB88_VDW` + `&PW92` + `&NON_LOCAL TYPE DRSLL` |
| SCF | OT + OUTER_SCF, EPS_SCF 1e−7 (converged 1 outer / 29 OT steps) |
| Date | 2026-08-01 |
| Wall (CP2K timer TOTAL) | **17.509 s** (1 MPI × `OMP_NUM_THREADS=8`) |

### Energies

| Quantity | CP2K pin |
| -------- | -------- |
| Dispersion \(E_c^{\mathrm{nl}}\) | **+0.15206184251169 Ha** |
| Total FORCE_EVAL (QS) | **−34.487914413113714 Ha** |

Key log lines:

```
 Dispersion energy:                                            0.15206184251169
 Total energy:                                               -34.48791441311332
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -34.487914413113714
 CP2K                                 1  1.0    0.009    0.009   17.509   17.509
```

### Atomic forces [Ha/Bohr] (CP2K `FORCES|`)

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −7.33537750×10⁻² | +4.42763551×10⁻⁸ | −3.00361565×10⁻⁸ |
| H1 | +3.93194959×10⁻² | −3.97828002×10⁻⁹ | +1.31800024×10⁻¹ |
| H2 | +3.93194286×10⁻² | −3.13656238×10⁻⁸ | −1.31799911×10⁻¹ |
| O1 | −6.85304811×10⁻² | −2.97383508×10⁻⁸ | +1.84114929×10⁻⁸ |
| H3 | +3.65105370×10⁻² | +3.88727769×10⁻⁸ | +1.35580659×10⁻¹ |
| H4 | +3.65106052×10⁻² | −1.71698035×10⁻⁸ | −1.35581298×10⁻¹ |

CP2K \(\lvert\sum F\rvert\) residual on this twin: **9.78×10⁻³** Ha/Bohr
(grid / multigrid residual — not a physical free-molecule zero).

### Dual-gate companion (OptB88 plain, no DRSLL)

Input: `water_dimer_optb88_plain_force.inp` (same geometry/cutoff, no
`VDW_POTENTIAL`). Offline 2026-08-01:

| Quantity | CP2K |
| -------- | ---- |
| \(E_{\mathrm{tot}}\) | −34.639998219383145 Ha |
| \(\lvert E_{\mathrm{opt+DRSLL}}-E_{\mathrm{opt}}\rvert\) | **0.152 Ha** (≈ \(E_c^{\mathrm{nl}}\)) |
| max\|ΔF\| (O0 \(F_x\)) | ~1.4×10⁻³ Ha/Bohr |

## Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − CP2K\| | **1×10⁻⁴** Ha |
| \|E_tot − CP2K\| | **2.5×10⁻²** Ha (OptB88 native vs CP2K/libxc multigrid residual class; family `gpw-optb88-uks` uses 2.5e−2) |
| max\|F_ATV − F_CP2K\| | **1×10⁻²** Ha/Bohr (bounded by CP2K ∑F residual ~1e−2) |
| Dual \|ΔE\| OptB88+DRSLL vs plain OptB88 | **> 1×10⁻⁴** Ha (hard floor) |
| \|∑ F_ATV\| | **≤ 2×10⁻²** Ha/Bohr (oracle residual class) |

## Perf gate

Fair wall on the **same** ENERGY_FORCE twin job only (1 MPI × 8 OMP):

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) | **17.509 s** (CP2K timer TOTAL TIME) |
| \(T_{\mathrm{atv}}\) | **19.403 s** (OptB88+DRSLL SCF + forces only, OMP=8) |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **1.108** |
| perf_exempt | **false** |
| within_10x | **true** (≤ 10) |

ATV wall is the OptB88+DRSLL SCF + force pass only (not the dual plain second SCF).

### Atomiverse first-green (2026-08-01, OMP=8)

| Gate | Measured | Tolerance |
| ---- | -------- | --------- |
| \|E_disp − CP2K\| | **3.77×10⁻⁵ Ha** | 1×10⁻⁴ Ha |
| \|E_tot − CP2K\| | **6.43×10⁻³ Ha** | 2.5×10⁻² Ha |
| max\|F − CP2K\| | **6.58×10⁻³ Ha/Bohr** | 1×10⁻² Ha/Bohr |
| \|∑ F\| | **8.0×10⁻⁴ Ha/Bohr** | 2×10⁻² Ha/Bohr |
| dual \|E_vdw − E_plain\| | **0.152 Ha** | > 1×10⁻⁴ Ha |
| SCF iters (vdw) | 12 | converged |

## Oracle provenance

```bash
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close
cd /tmp/atv-gpw-optb88-forces
ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
mpirun -np 1 cp2k.psmp -i water_dimer_optb88_vdw_force.inp \
  -o water_dimer_optb88_vdw_force.out
```

- **Binary:** `cp2k.psmp` 2026.1 git:5e54ba2  
- **Input:** `water_dimer_optb88_vdw_force.inp`  
- **Output lines:** Dispersion energy / ENERGY\| Total FORCE_EVAL / FORCES\| / CP2K timer  
