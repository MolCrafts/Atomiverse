# Reference — multi-k GPW PBE + vdW-DF (DRSLL) analytic ionic forces

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Monkhorst–Pack
k-point** **GPW PBE** (closed-shell RKS) with **analytic ionic forces**.
SCF \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) is evaluated on the
finest-grid k-summed density and added to `veff` every iteration; the same
\(V_{\mathrm{nl}}\) is folded into the multi-k force assembly
(`assemble_gpw_forces_kpoints` grid-term `veff` in `gpw_forces.cpp`).

Public API: `cpu::gpw::GpwDft` with `vdw_df=true` + `k_mesh` (or explicit
`k_points`) + `gpw_forces_from_scf` / `gpw_forces_analytic`. No public
`*Config`, no factories. Energy product already closed by
`reg_gpw-vdw-kpoints`; Γ SC potential + FD by `reg_gpw-vdw-scf-potential`.

## System

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ=8`) |
| XC | PBE + NON_LOCAL **DRSLL** (kernel `vdW_kernel_table.dat`) |
| Grid | CUTOFF **200** Ry, NGRIDS **4**, REL_CUTOFF **40** |
| k mesh | MONKHORST-PACK **2×1×1** (ATV `k_time_reversal=false` for 1:1 twin) |
| SCF | EPS_SCF / tol **1e-7** Ha; DIAGONALIZATION + DIRECT_P_MIXING |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, source build `…/cp2k/install/env.sh`, git:**5e54ba2**) |
| Host | Arrhenius login, offline only |
| Parallelism | **1 MPI × OMP 8** (`mpirun -np 1`, `OMP_NUM_THREADS=8`) |
| Input | `tools/cp2k_oracle/inputs/gpw-vdw-kpoint-forces/water_dimer_vdw_df_kp_forces.inp` |
| Wall (ENERGY_FORCE twin) | **13.216 s** (CP2K internal `CP2K` timer TOTAL TIME maximum) |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| **Dispersion energy** | **0.15193559149613** |
| **Total** | **−34.186333514352626** |

(Matches the energy-only multi-k twin of `gpw-vdw-kpoints` to print residual.)

### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` table)

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.35844769E−02 | 1.49028331E−09 | 2.51422839E−10 |
| H0 | 4.16583157E−02 | −5.35060271E−09 | 1.31835454E−01 |
| H1 | 4.16583337E−02 | 4.10205702E−09 | −1.31835315E−01 |
| O1 | −7.89346446E−02 | 2.67651054E−11 | 6.49135595E−09 |
| H2 | 3.76866359E−02 | −4.43451105E−09 | 1.35635778E−01 |
| H3 | 3.76865671E−02 | 4.54720810E−09 | −1.35635393E−01 |

CP2K `FORCES| Sum` residual ≈ 3.8×10⁻³ Ha/Bohr (finite basis / grid).

### Re-run (offline only — never ctest)

```bash
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
cd /tmp && mkdir -p atv-pin && cd atv-pin
ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
export OMP_NUM_THREADS=8
mpirun -np 1 cp2k.psmp \
  -i …/tools/cp2k_oracle/inputs/gpw-vdw-kpoint-forces/water_dimer_vdw_df_kp_forces.inp \
  -o water_dimer_vdw_df_kp_forces.out
# Record: ENERGY| Total; Dispersion energy; FORCES|; CP2K timer TOTAL TIME
```

## Offline FD identity (decisive correctness)

On the **single-grid** multi-k product path (`ngrids=1`, same mesh / DRSLL /
CUTOFF 200, Release, OMP=8, 2026-08-01), central FD of \(E_{\mathrm{total}}\)
w.r.t. O0 \(x\) (\(h=5\times10^{-4}\) Bohr):

| Quantity | Value |
| -------- | ----- |
| \(F_{\mathrm{ana},O0,x}\) | −8.976047×10⁻² Ha/Bohr |
| \(F_{\mathrm{FD},O0,x}\) | −8.975976×10⁻² Ha/Bohr |
| \(\|F_{\mathrm{ana}}-F_{\mathrm{FD}}\|\) | **7.07×10⁻⁷ Ha/Bohr** |

Proves multi-k analytic forces with SC \(V_{\mathrm{nl}}\) are \(-\nabla E\) of
the multi-k energy. Not re-run in ctest: three multi-k SCFs exceed the ≤ 60 s
regression budget (same policy as freezing a first-green band after offline FD
on `gpw-kpoint-forces`).

## Atomiverse twin (first green, 2026-08-01, OMP=8)

| Quantity | Value |
| -------- | ----- |
| E_tot | −34.17840737459292 Ha (Δ vs CP2K **7.926e−3** — pre-existing Γ-dimer systematic) |
| E_disp | +0.15189542233538 Ha (Δ **4.02e−5**) |
| n_kpts | 2 |
| max\|F_atv − F_cp2k\| | **7.25e−3** Ha/Bohr (signal ~0.136) |
| F_O0_x (ATV pin) | **−8.976000e−02** Ha/Bohr |
| \|∑F\| | ~1.5e−2 Ha/Bohr |
| Twin wall | **20.8–44 s** (SCF + force; host load) |
| Fair ratio | **1.6–3.3** (≤ 10) |

## Gates / tolerances

| Gate | Tol | Why |
| ---- | --- | --- |
| \|E_disp − CP2K\| | **1×10⁻⁴ Ha** | Same class as `gpw-vdw-kpoints` E_disp pin |
| \|E_tot − CP2K\| | **1×10⁻² Ha** | Twin-family systematic ~8e−3 Ha (not a force defect) |
| max\|F − CP2K\| | **1×10⁻² Ha/Bohr** | Same residual class on absolute forces; decisive correctness is offline FD |
| \|F_O0_x − ATV pin\| | **5×10⁻⁵ Ha/Bohr** | Frozen product reproducibility |
| fair wall ratio | **≤ 10** | vs real CP2K ENERGY_FORCE twin only |
| product wall | **≤ 60 s** | e2e tier budget |

## Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K ENERGY_FORCE twin | **13.216 s** |
| ATV twin SCF + forces | **~21–44 s** |
| \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **~1.6–3.3** (≤ 10; no exemption) |

## Out of scope

- UKS multi-k vdW forces (`gpw-vdw-uks-kpoints` energy only)
- Analytic stress with nonlocal vdW
- CUDA / cross-representation (PW NC-UPF) twins

## Provenance

- Spec: `.claude/specs/gpw-vdw-kpoint-forces/`
- Deps: `gpw-vdw-kpoints` ✓, `gpw-vdw-scf-potential` ✓
- Oracle family: **cp2k** (matched GTH-GPW + NON_LOCAL + KPOINTS ENERGY_FORCE)
