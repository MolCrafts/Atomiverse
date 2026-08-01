# Reference — GPW band structure on diamond Si (bct 4-atom cell)

## Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack SCF** followed by
**non-self-consistent band structure** on a high-symmetry fractional path:
freeze \(V_\mathrm{eff}\) from the SCF density \(\rho=\sum_k w_k\rho_k\), rebuild
\(H_k=T_k+V_\mathrm{nl}^k+V^k[V_\mathrm{eff}]\) at each path k, diagonalize.

Public API:

```cpp
cpu::gpw::GpwDft dft;   // members: k_mesh, cutoff_ry, n_occ, xc, …
dft.build(atoms);
cpu::gpw::Bands bands;  // members: n_bands, kfrac / set_diamond_path
auto br = bands.compute(dft, atoms);
// br.eigenvalues_ha[ik][ib], gap_ha, e_vbm_ha, e_cbm_ha [Hartree]
```

No public `*Config`, no factories. Twin class: CP2K Quickstep
`&PRINT &BAND_STRUCTURE` on GTH-GPW Si.

## System

| | |
| --- | --- |
| Lattice | Diamond Si, bct 4-atom cell: \(a=5.43\) Å; cell \((a/\sqrt2,a/\sqrt2,a)\) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → `n_occ = 8` |
| XC | PBE |
| Density cutoff | 150 Ry, NGRIDS 1 |
| SCF k mesh | MONKHORST-PACK 2×2×2, time-reversal fold |
| Path | Γ–X–W–L–Γ–K fractional, `n_seg=4` → 16 k |
| SCF | DIIS, tol \(10^{-8}\) Ha, EPS_DEFAULT \(10^{-7}\) |

**Note:** SPECIAL_POINT labels match the FCC diamond path list; on the **bct**
reciprocal lattice they are not the FCC absolute X/W/L. The twin compares ATV
and CP2K on the **same fractional list**. On this cell both codes place VBM and
CBM at Γ (direct gap on the path).

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), source build
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`
  (`env.sh` → git:5e54ba2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-bands-si.inp`
  (CUTOFF 150, NGRIDS 1, MP 2 2 2, PBE, EPS_SCF 1e-8, BAND_STRUCTURE NPOINTS 4).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  OMP_NUM_THREADS=8 mpirun -np 1 cp2k.psmp -i gpw-bands-si.inp -o gpw-bands-si.out
  ```
- **Date:** 2026-08-01.

| Quantity | CP2K pin | ATV first green | Tolerance |
| --- | --- | --- | --- |
| \(E_\mathrm{tot}\) [Ha] | **−15.802416412655166** | −15.8023941333 | **1e-3 Ha** |
| Path gap [eV] | **0.893516** (Γ→Γ) | 0.894761 | **50 meV** |
| Γ local gap [eV] | 0.893516 | 0.894761 | 50 meV |

Key log lines:

- `ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -15.802416412655166`
- Band file `si_gpw_bands`: Point 1 (Γ) band 8 = 5.172733 eV (occ), band 9 = 6.066249 eV (empty).

CP2K BAND_STRUCTURE used NPOINTS 4 (21 path k); ATV `n_seg=4` uses 16 path k with
the same special points. Gap is fixed by the special-point spectrum (Γ).

## Fair wall (matched twin job)

| | Wall [s] | Threads |
| --- | --- | --- |
| ATV (`reg_gpw-bands-si`, SCF + NSCF path) | **6.66** | 1 MPI × OMP 8 |
| CP2K (`gpw-bands-si.inp`, ENERGY + BAND_STRUCTURE) | **1.90** | 1 MPI × OMP 8 |

**Ratio \(T_\mathrm{ATV}/T_\mathrm{CP2K} = 3.51\)** (≤ 10). Warm re-runs; same
binary pins as above.

## Acceptance gates (regression)

1. SCF converges; \(E\) finite; \(|E-E_\mathrm{CP2K}|\le 10^{-3}\) Ha.
2. Path has 16 k (`n_seg=4`), eigenvalues sorted, `n_bands` per k.
3. Path includes Γ and X; min local gap > 0.
4. Path gap and Γ local gap within 50 meV of CP2K 0.893516 eV.
5. VBM and CBM at Γ (twin character on this bct path).
