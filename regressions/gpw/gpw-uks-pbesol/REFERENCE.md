# Reference — Γ-point UKS GPW O₂ (PBEsol / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham GPW (Quickstep LSD) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **PBEsol GGA** (Perdew et al. 2008 solid
exchange \(\mu\) revision + PBEsol correlation; libxc `GGA_X_PBE_SOL` +
`GGA_C_PBE_SOL` / CP2K `PARAMETRIZATION PBESOL`), GTH-PBE-q6 pseudopotential,
DZVP-MOLOPT-GTH Gaussian basis.

Atomiverse path: `cpu::gpw::GpwDft` / `GPWEvaluator` with `n_alpha`/`n_beta`,
`XcKind::PBEsol`, dual KS matrices, `xc_grid_pbesol_spin`, Löwdin ⟨S²⟩.

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
| XC | **PBEsol** (GGA, spin-resolved) |
| Cutoff | 400 Ry (Atomiverse single density grid; CP2K twin also NGRIDS 4 / REL_CUTOFF 60) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), source install
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`
  (`env.sh` → binary; git **5e54ba2**; `cp2kflags: omp libint fftw3 libxc elpa
  parallel scalapack mpi_f08 spglib`; GCC 14.3.0).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-pbesol/o2_uks_pbesol.inp`
- **XC input form:** nested `&XC_FUNCTIONAL` / `&PBE` / `PARAMETRIZATION PBESOL`
  (matches RKS `gpw-pbesol-si` twin; libxc-equivalent `GGA_X_PBE_SOL` +
  `GGA_C_PBE_SOL`).
- **SCF:** `EPS_SCF 1e-6`, Broyden α=0.4, DIAGONALIZATION STANDARD.
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i o2_uks_pbesol.inp -o o2_uks_pbesol.out
  ```
- **Date:** 2026-08-01
- **Host:** arrhenius1.hpc.arrhenius.naiss
- **Wall (CP2K, 1 MPI × 4 OMP):** ≈ **18.52 s** total (`CP2K` timer MAXIMUM)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.82107945887769** | **1e-3 Ha** (first green; single-grid vs CP2K multigrid) |
| ⟨S²⟩ (single det.) | **2.004962** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines:

```
 FUNCTIONAL| PBE:
 FUNCTIONAL| PBEsol, J.P. Perdew et al., Phys. Rev. Letter, vol 100,n 13, p. 136
 FUNCTIONAL| 406, (2008) {spin polarized}
  *** SCF run converged in    14 steps ***
  Total energy:                                               -31.82107945887769
  Ideal and single determinant S**2 :                    2.000000       2.004962
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.821079458877691
 CP2K                                 1  1.0    0.014    0.014   18.524   18.524
```

### Atomiverse first-green (2026-08-01, OMP=4 Release)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{PBEsol}}\) | **−31.821113232325 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(3.38 \times 10^{-5}\) Ha |
| ⟨S²⟩ | 2.00496250 |
| \(E_{\mathrm{PBE}}\) twin | −31.928854734085 Ha |
| \(\lvert E_{\mathrm{PBEsol}}-E_{\mathrm{PBE}}\rvert\) | \(1.08 \times 10^{-1}\) Ha (\(> 10^{-4}\)) |
| SCF iters (PBEsol) | 49 |
| Wall PBEsol SCF | **62.331 s** |

## Dual-gate

\(|E_{\mathrm{PBEsol}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha on the same geometry
(hard activation check; PBE twin mirrors `gpw-uks-pbe`).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | **18.52** | 1 MPI × OMP=4, NGRIDS 4 multigrid |
| Atomiverse | **62.33** | Release, single density grid, OMP=4, 2026-08-01 |

`ratio = T_atv / T_cp2k = 62.33/18.52 ≈ **3.37**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true; fair twin only.

## Citations

- J. P. Perdew et al., *Phys. Rev. Lett.* **100**, 136406 (2008) — PBEsol.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE family.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — ⟨S²⟩ for unrestricted single dets.

## Provenance

- Spec: `.claude/specs/gpw-uks-pbesol/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-uks-pbesol/o2_uks_pbesol.inp`
- Sibling pins: `gpw-uks-pbe`, `gpw-pbesol-si`, `gpw-uks-rpbe`, `gpw-uks-revpbe`
