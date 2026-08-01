# Reference — multi-k GPW metallic DOS (BCC Na, FD smear)

## Method

Closed-shell **GPW** Kohn–Sham free-energy SCF with **Fermi–Dirac metallic
occupation smearing** and a **Gaussian-broadened density of states** on the
SCF multi-k eigenvalues \(\varepsilon_{nk}\). Occupied integral uses the SCF
Fermi level:

\[
n_{\mathrm{elec\_to\_EF}} = \int_{-\infty}^{E_F} D(E)\,dE
\qquad\text{(target }\approx N_e\text{)}
\]

Matched algorithm class: CP2K Quickstep GPW +
`&KPOINTS SCHEME MONKHORST-PACK 2 2 2` + `&SMEAR FERMI_DIRAC` +
`&PRINT &DOS` (histogram of the same multi-k spectrum class; ATV uses
continuous Gaussian broadening — absolute \(D(E)\) is not a pin).

Public API: `GpwDft` members `smearing` / `degauss` / `extra_bands` and
`cpu::gpw::Dos` (`k_mesh`, `sigma_ha`, `n_bins`) → `DosResult` with
`e_fermi_ha` and `n_elec_to_vbm` (cut at \(E_F\) under smear). No public
`*Config`, no factories. Lookup constructors
`GaussianBasis(Element::Na, "DZVP-MOLOPT-SR-GTH")` /
`GthPotential(Element::Na, "GTH-PBE-q9")`.

## System

| | |
| --- | --- |
| Lattice | BCC conventional cubic \(a = 4.225\) Å (2 Na) |
| Positions | (0,0,0), (½,½,½) |
| Basis | DZVP-MOLOPT-SR-GTH (Na) |
| Pseudopotential | GTH-PBE-q9 |
| Valence | 18 e⁻ → `n_occ = 9` seed |
| XC | PADE (LDA) |
| Density cutoff | 150 Ry, NGRIDS 1 (ATV single grid) |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal folded → 4 k |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry (\(T_e \approx 3157.75\) K) |
| Extra bands | 4 |
| SCF | DIIS depth 6, tol 1e-6 Ha, EPS_DEFAULT 1e-7, mix 0.4 |
| DOS | Gaussian \(\sigma = 0.025\) Ha, 400 bins |

## Offline CP2K pin (algorithm twin)

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), source install
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`
  (`env.sh` → binary; git **5e54ba2**; `cp2kflags: omp libint fftw3 libxc elpa
  parallel scalapack mpi_f08 spglib`; GCC 14.3.0).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dos-metallic/na_bcc_fd_dos.inp`
  (CUTOFF 150, NGRIDS 1, MP 2×2×2, PADE, FERMI_DIRAC \(T_e=3157.75\) K,
  ADDED_MOS 4, EPS_SCF 1e-6, `&PRINT &DOS`).
- **Run (offline only; never in ctest):**
  ```bash
  source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
  export OMP_NUM_THREADS=4
  cd /tmp/atv-gpw-dos-metallic-cp2k
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i na_bcc_fd_dos.inp -o na_bcc_fd_dos.out
  ```
- **Date:** 2026-08-01
- **Host:** arrhenius (login node; 1 MPI × OMP=4)

| Quantity | CP2K pin | Role in regression |
|----------|----------|--------------------|
| \(F\) / \(E_{\mathrm{tot}}\) [Ha] | **−92.766060953857320** | Algorithm twin energy (not absolute dual gate) |
| \(E_F\) [Ha] | **0.05447820003643** | Character pin \(\lvert\Delta E_F\rvert \le 0.02\) Ha |
| DOS write | `na_bcc_fd_dos-1.dos` (histogram) | Absolute \(D(E)\) not compared |
| SCF steps | 13 | — |

Key log lines:

```
  *** SCF run converged in    13 steps ***
  Fermi energy:                                                 0.05447820003643
  Total energy:                                               -92.76606095385732
   Calculate DOS at iteration step                                   0
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -92.766060953857320
 CP2K                                 1  1.0    0.004    0.004    0.499    0.499
```

Absolute free energy is **not** dual-gated vs CP2K (same residual class as
`reg_gpw-metallic-smearing`: ~0.64 Ha from grid/Poisson/table details at
matched CUTOFF / NGRIDS 1). Correctness gates are free-energy freeze +
Fermi character + \(\int^{E_F} D \approx N_e\).

### Atomiverse free-energy freeze + first-green DOS (2026-08-01)

| Quantity | Value |
| --- | --- |
| Free energy \(F\) | **−92.126330608378 Ha** (freeze; tol \(10^{-4}\) Ha) |
| \(E_F\) | **0.055826 Ha** (\(\lvert\Delta E_F\rvert\) vs CP2K \(1.35\times 10^{-3}\) Ha) |
| \(\int^{E_F} D(E)\,dE\) | **17.999** (target 18 ± 2) |
| n_kpts (TR-folded) | 4 |
| n_bands (AO spectrum) | 18 |
| Wall SCF+DOS (OMP=4) | **2.736 s** |

## Dual gate

1. **Free energy** vs ATV freeze \(\lvert\Delta F\rvert \le 10^{-4}\) Ha.
2. **Metallic DOS character:** multi-k (`n_kpts ≥ 2`), finite \(E_F\) within
   0.02 Ha of the CP2K twin, \(D(E) \ge 0\) with a positive peak,
   \(\int^{E_F} D \approx 18\).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | **0.499** | 1 MPI × OMP=4, `CP2K` timer MAXIMUM |
| Atomiverse | **2.736** | Release, OMP=4, SCF + Gaussian DOS |

`ratio = T_atv / T_cp2k = 2.736 / 0.499 ≈ **5.48**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true; fair twin only (same GTH-GPW
Na multi-k FD SCF + DOS job class).

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) — free-energy occupations.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh.
- CP2K Quickstep `&SMEAR` / `&PRINT &DOS` / multi-k MO eigenvalues.

## Provenance

- Spec: `.claude/specs/gpw-dos-metallic/` (deleted on close)
- Twin input: `tools/cp2k_oracle/inputs/gpw-dos-metallic/na_bcc_fd_dos.inp`
- Sibling free-energy SCF: `reg_gpw-metallic-smearing`
- Sibling fixed-occ DOS: `reg_gpw-dos-si`
- PW metallic DOS pattern: `reg_dos-metallic-si`
