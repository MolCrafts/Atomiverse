# Reference — Multi-k GPW collinear UKS density of states (diamond Si BCT)

## Method

Collinear unrestricted Kohn–Sham GPW DFT (PBE LSD) with **Monkhorst–Pack
multi-k sampling** and a **Gaussian-broadened dual-spin density of states**
on the SCF mesh eigenvalues \(\varepsilon_{nk}^\sigma\). Matched algorithm
class: CP2K Quickstep GPW + `LSD` + `MULTIPLICITY 1` +
`&KPOINTS SCHEME MONKHORST-PACK 2 2 2` + `&PRINT &DOS` (histogram of the
same multi-k dual-spin spectrum class; ATV uses continuous Gaussian
broadening rather than a raw histogram — absolute \(D(E)\) is not a pin).

Public API: `cpu::gpw::GpwDft` with `n_alpha`/`n_beta` (UKS) +
`cpu::gpw::Dos` members (`k_mesh`, `sigma_ha`, `n_bins`, …) and
`DosResult compute(GpwDft&, atoms)` filling `nspin=2`, `dos_alpha`,
`dos_beta`, total `dos`. Free math helpers `gap_from_spectrum`,
`gaussian_dos` (\(g=1\) per spin-orbital), `integrate_dos_to`. No public
`*Config`, no factories. Lookup constructors
`GaussianBasis(Element::Si, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::Si, "GTH-PBE-q4")`.

Product surface landed with `gpw-dos-si` (RKS foothold); this slug is the
**dedicated UKS pin** on that path.

## System

| | |
| --- | --- |
| Lattice | a = 5.43 Å (10.26120 Bohr); cell = (a/√2, a/√2, a) BCT |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → \(N_\alpha = N_\beta = 8\) (multiplicity 1) |
| XC | PBE (spin-resolved LSD path) |
| Density cutoff | 150 Ry, NGRIDS 1 |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal folded → 4 k |
| SCF | DIIS depth 6, tol 1e-8 Ha, EPS_DEFAULT 1e-7 |
| DOS | Gaussian \(\sigma = 0.025\) Ha (~0.68 eV), 400 bins |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), source install
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`
  (`env.sh` → binary; git **5e54ba2**; `cp2kflags: omp libint fftw3 libxc elpa
  parallel scalapack mpi_f08 spglib`; GCC 14.3.0).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dos-uks/si_gpw_dos_uks.inp`
  (CUTOFF 150, NGRIDS 1, MONKHORST-PACK 2 2 2, LSD, MULTIPLICITY 1, PBE,
  EPS_SCF 1e-8, `&PRINT &DOS`).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i si_gpw_dos_uks.inp -o si_gpw_dos_uks.out
  ```
- **Date:** 2026-08-01
- **Host:** arrhenius1 (login node; 1 MPI × OMP=4)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−15.802416412655198** | **1e-3 Ha** |
| DOS character | nspin=2; open gap; occupied integral ≈ 16 | gap ≥ 0.1 eV; n_elec ± 1.5; dual channels |

Key log lines:

```
  *** SCF run converged in    13 steps ***
  Total energy:                                               -15.80241641265520
   Calculate DOS at iteration step                                   0
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -15.802416412655198
 CP2K                                 1  1.0    0.004    0.004    5.164    5.164
```

Note: CP2K may warn that **PDOS is not implemented for k points**; total DOS
print (`&PRINT &DOS`) still runs and writes `*.dos`. Absolute \(D(E)\) is
not compared (histogram vs Gaussian). Closed-shell M=1 energy matches the
RKS twin of `gpw-dos-si` to machine noise (dual-spin path exercised).

### Atomiverse first-green (2026-08-01, OMP=4 Release)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{tot}}\) | **−15.802394133257 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(2.23 \times 10^{-5}\) Ha |
| gap (integer-fill VBM–CBM) | **3.953 eV** |
| \(\int^{E_{\mathrm{vbm}}} D(E)\,dE\) | **14.84** (target 16 ± 1.5; Gaussian tails) |
| per-spin \(\int D_\sigma\) | \(n_a = n_b = 7.42\) |
| nspin / n_kpts (TR-folded) | 2 / 4 |
| n_bands (AO spectrum) | 52 |
| Wall SCF+DOS | **13.648 s** |

## Dual gate

1. **Energy** vs offline CP2K twin (\(|\Delta E| \le 10^{-3}\) Ha).
2. **DOS character:** `nspin == 2`, multi-k payload (`n_kpts ≥ 2`), open gap
   ≥ 0.1 eV, dual-channel grids, \(D(E) \ge 0\) with a positive peak,
   occupied integral ≈ \(n_e\), \(\int D_\alpha + \int D_\beta \approx
   \int D\), each spin channel ≈ \(n_e/2\).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | **5.164** | 1 MPI × OMP=4, `CP2K` timer MAXIMUM |
| Atomiverse | **13.648** | Release, OMP=4, UKS SCF + Gaussian DOS |

`ratio = T_atv / T_cp2k = 13.648 / 5.164 ≈ **2.64**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true; fair twin only (same GTH-GPW
multi-k collinear UKS SCF job class).

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh.
- CP2K Quickstep `&PRINT &DOS` / multi-k UKS MO eigenvalues (qs_dos.F).

## Provenance

- Spec: `.claude/specs/gpw-dos-uks/acceptance.md`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dos-uks/si_gpw_dos_uks.inp`
- Sibling RKS DOS: `gpw-dos-si`
- Sibling multi-k UKS energy: `gpw-kpoints-uks`
- PW pattern: `cpu::pw::Dos` / `reg_dos-spin-si`
