# Reference — Γ collinear-UKS GPW **optB88-vdW** (OptB88 + DRSLL)

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Γ collinear UKS**
with semi-local **optB88-vdW** exchange–correlation:

| Piece | Form |
| ----- | ---- |
| Exchange | optimized B88 (`GGA_X_OPTB88_VDW`; Klimeš et al.) |
| Local correlation | PW92 LDA only |
| Nonlocal correlation | DRSLL / DF1 \(E_c^{\mathrm{nl}}\) via SC \(V_{\mathrm{nl}}\) |

\(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\) is a functional of the
**total** density \(\rho=\rho_\alpha+\rho_\beta\) and is spin-independent, so
it is added identically to both spin potentials every SCF iteration
(`build_uks_H`; CP2K `qs_dispersion_nonloc`). Semi-local spin XC uses
`xc_grid_optb88_vdw_spin` (`XcKind::OptB88VdW`). Self-consistent
\(E_c^{\mathrm{nl}}\) lands on `GpwResult::e_disp` [Hartree].

Public API:

```cpp
cpu::gpw::GPWEvaluator eval({O});
eval.xc = cpu::gpw::XcKind::OptB88VdW;
eval.n_alpha = 7; // O₂ triplet, GTH-q6
eval.n_beta = 5;
eval.vdw_df = true;
cpu::single_point(frame, eval);
// Frame kEnergy [eV]; last_result().e_disp [Ha]
```

Extends `gpw-optb88-vdw` (RKS water-dimer pin) × `gpw-vdw-uks` (Γ-UKS PBE +
DRSLL) — same UKS-GGA expand pattern as revPBE / RPBE / BLYP.

## Part A — O₂ triplet LSD optB88-vdW vs offline CP2K

| Item | Value |
|------|-------|
| Molecule | O₂, R(O–O) = 1.2075 Å, centered along *z* |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, **PERIODIC XYZ** (RPS kernel needs periodic Poisson) |
| Basis / PP | DZVP-MOLOPT-GTH + GTH-PBE-q6 (12 valence e⁻) |
| XC | `GGA_X_OPTB88_VDW` + `PW92` + `&NON_LOCAL TYPE DRSLL` |
| Cutoff | CP2K + ATV: 400 Ry / REL_CUTOFF 60 / NGRIDS 4 (matched multigrid twin) |
| Spin | LSD, MULTIPLICITY 3 → Nα = 7, Nβ = 5 |
| SCF | CP2K OT + OUTER_SCF (EPS_SCF 1e−6); ATV DIIS, tol 1e−7 |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (Development Version, source revision
  **git:5e54ba2**, `cp2k.psmp`,
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`,
  GCC 14.3 / MPICH; libxc + fftw3 flags).
- **Input:** `tools/cp2k_oracle/inputs/gpw-optb88-uks/o2_uks_optb88_vdw.inp`
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  export OMP_NUM_THREADS=8
  cd /tmp/run && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  mpirun -np 1 cp2k.psmp -i o2_uks_optb88_vdw.inp -o o2_uks_optb88_vdw.out
  ```
- **Date:** 2026-08-01. Outer SCF converged in 1 iteration / 27 OT steps.
- **Wall (CP2K, 1 rank × 8 OMP):** **66.011 s** (`CP2K` timer TOTAL TIME).

| Quantity | CP2K pin | Regression tolerance |
|----------|----------|----------------------|
| Total energy [Ha] | **−32.054900934599011** | **2.5e−2 Ha** (OptB88 native vs CP2K/libxc grid twin; family `gpw-optb88-kpoints` uses 1e−2 Ha on RKS water dimer) |
| Dispersion energy \(E_c^{\mathrm{nl}}\) [Ha] | **+0.11322988837478** | **1e−4 Ha** (tight product pin) |
| ⟨S²⟩ (single det.) | 2.010347 | (not gated; info) |

Key log lines:

```
 Dispersion energy:                                            0.11322988837478
 Total energy:                                               -32.05490093459901
 Ideal and single determinant S**2 :                    2.000000       2.010347
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -32.054900934599011
 CP2K                                 1  1.0    0.019    0.019   66.011   66.011
```

**Note:** Broyden + diagonalization diverged on this LSD × OptB88 × DRSLL combo
in CP2K 2026.1; OT (same recipe as the RKS `gpw-optb88-vdw` water-dimer twin)
converges cleanly. Atomiverse stays on product DIIS/diag UKS — energies are
compared at functional minimum, not SCF path.

## Dual gate — OptB88 vs PBE activation

Same O₂ geometry / spin / DRSLL / 400 Ry:  
\(|E_{\mathrm{OptB88+DRSLL}} - E_{\mathrm{PBE+DRSLL}}| > 10^{-4}\) Ha  
(ensures semi-local optB88 exchange is live, not a silent PBE alias).

## Part B — closed-shell spin-degeneracy identity

Water dimer (same geometry as `gpw-optb88-vdw` / `gpw-vdw-uks`; 12 Å cubic
periodic cell, DZVP-MOLOPT-SR-GTH + GTH-PBE-q1/q6, 200 Ry, OptB88VdW +
`vdw_df`, tol 1e−7): 16 valence e⁻; UKS with nα = nβ = 8 must reproduce RKS
with n_occ = 8 (double-occupied).

| Gate | Tolerance |
|------|-----------|
| \(\lvert E_{\mathrm{UKS}}-E_{\mathrm{RKS}}\rvert\) | **3e−4 eV** |
| \(\lvert E_{c,\mathrm{UKS}}^{\mathrm{nl}}-E_{c,\mathrm{RKS}}^{\mathrm{nl}}\rvert\) | **1e−5 Ha** |
| \(E_c^{\mathrm{nl}} \neq 0\) folded on `e_disp` | hard |

### Atomiverse numbers (verified green 2026-08-01, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| \|E_tot − CP2K\| (Part A) | **1.90e−2 Ha** | 2.5e−2 Ha |
| \|E_disp − CP2K\| (Part A) | **1.5e−5 Ha** | 1e−4 Ha |
| \|E_OptB88 − E_PBE\| (dual) | **0.220 Ha** | > 1e−4 Ha |
| \|E_UKS − E_RKS\| (Part B) | **1.6e−8 eV** | 3e−4 eV |
| \|E_disp UKS − RKS\| (Part B) | **5.8e−10 Ha** | 1e−5 Ha |
| SCF | A: 55 iters (OptB88), B: both converged | |

## Perf gate

CP2K QS wall **66.011 s** (1 × 8 OMP, NGRIDS 4, OT). Atomiverse Part A wall
**220.9 s** (OMP=8, matched NGRIDS 4 / REL_CUTOFF 60 OptB88+DRSLL SCF only —
not the multi-gate reg wall that also runs PBE dual + water-dimer identity).
Ratio \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 3.35\) (**within_10x**).

## Citations

- J. Klimeš, D. R. Bowler, A. Michaelides, *J. Phys.: Condens. Matter*
  **22**, 022201 (2010) — optB88-vdW.
- M. Dion et al., *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF (DRSLL).
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) — RPS.
- T. Thonhauser et al., *Phys. Rev. B* **76**, 125112 (2007) — SC V_nl.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- CP2K stock section: `data/xc_section/vdW-DF-optB88.sec`.
