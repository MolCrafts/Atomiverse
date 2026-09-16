# Reference — Multi-k GPW density of states on diamond Si (BCT 4-atom)

## Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack multi-k sampling**
and a **Gaussian-broadened density of states** on the SCF mesh eigenvalues
\(\varepsilon_{nk}\). Matched algorithm class: CP2K Quickstep GPW +
`&KPOINTS SCHEME MONKHORST-PACK 2 2 2` + `&PRINT &DOS` (histogram of the
same multi-k spectrum class; ATV uses continuous Gaussian broadening rather
than a raw histogram — absolute \(D(E)\) is not a pin).

Public API: `cpu::gpw::Dos` members (`k_mesh`, `sigma_ha`, `n_bins`, …) and
`DosResult compute(GpwDft&, atoms)`. Free math helpers `gap_from_spectrum`,
`gaussian_dos`, `integrate_dos_to`. No public `*Config`, no factories.
Lookup constructors `GaussianBasis(Element::Si, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::Si, "GTH-PBE-q4")`.

## System

| | |
| --- | --- |
| Lattice | a = 5.43 Å (10.26120 Bohr); cell = (a/√2, a/√2, a) BCT |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → n_occ = 8 |
| XC | PBE |
| Density cutoff | 150 Ry, NGRIDS 1 |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal folded → 4 k |
| SCF | DIIS depth 6, tol 1e-8 Ha, EPS_DEFAULT 1e-7 |
| DOS | Gaussian \(\sigma = 0.025\) Ha (~0.68 eV), 400 bins |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), source install
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`
  (`env.sh` → binary; git **5e54ba2**; `cp2kflags: omp libint fftw3 libxc elpa
  parallel scalapack mpi_f08 spglib`; GCC 14.3.0).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dos-si/si_gpw_dos.inp`
  (CUTOFF 150, NGRIDS 1, MONKHORST-PACK 2 2 2, PBE, EPS_SCF 1e-8, `&PRINT &DOS`).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i si_gpw_dos.inp -o si_gpw_dos.out
  ```
- **Date:** 2026-08-01
- **Host:** arrhenius (login node; 1 MPI × OMP=4)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−15.802416412655166** | **1e-3 Ha** |
| DOS character | open gap; occupied integral ≈ 16 | gap ≥ 0.1 eV; n_elec ± 1.5 |

Key log lines:

```
  *** SCF run converged in    13 steps ***
  Total energy:                                               -15.80241641265517
   Calculate DOS at iteration step                                   0
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -15.802416412655166
 CP2K                                 1  1.0    0.004    0.004    2.595    2.595
```

Note: CP2K warns that **PDOS is not implemented for k points**; total DOS
print (`&PRINT &DOS`) still runs and writes `*.dos`. Absolute \(D(E)\) is
not compared (histogram vs Gaussian).

### Atomiverse first-green (2026-08-01, OMP=4 Release)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{tot}}\) | **−15.802394133256 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(2.23 \times 10^{-5}\) Ha |
| gap (integer-fill VBM–CBM) | **3.953 eV** |
| \(\int^{E_{\mathrm{vbm}}} D(E)\,dE\) | **14.84** (target 16 ± 1.5; Gaussian tails) |
| n_kpts (TR-folded) | 4 |
| n_bands (AO spectrum) | 52 |
| Wall SCF+DOS | **4.209 s** |

## Dual gate

1. **Energy** vs offline CP2K twin (\(|\Delta E| \le 10^{-3}\) Ha).
2. **DOS character:** multi-k payload (`n_kpts ≥ 2`), open gap ≥ 0.1 eV,
   \(D(E) \ge 0\) with a positive peak, occupied integral ≈ \(n_e\).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | **2.595** | 1 MPI × OMP=4, `CP2K` timer MAXIMUM |
| Atomiverse | **4.209** | Release, OMP=4, SCF + Gaussian DOS |

`ratio = T_atv / T_cp2k = 4.209 / 2.595 ≈ **1.62**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true; fair twin only (same GTH-GPW
Si multi-k SCF job class).

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh.
- CP2K Quickstep `&PRINT &DOS` / multi-k MO eigenvalues (qs_dos.F).

## Provenance

- Spec: `.claude/specs/gpw-dos-si/acceptance.md`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dos-si/si_gpw_dos.inp`
- Sibling energy twin: `gpw-kpoints-si` (same geometry / cutoff / mesh)
- PW pattern: `cpu::pw::Dos` / `reg_bands-dos-si`
