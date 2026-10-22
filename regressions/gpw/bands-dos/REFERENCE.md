# gpw/bands-dos

One case; every section below is run by `atv_regressions --case gpw/bands-dos` and keeps its own oracle record.


## Reference — GPW band structure on diamond Si (bct 4-atom cell)

### Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack SCF** followed by
**non-self-consistent band structure** on a high-symmetry fractional path:
freeze \(V_\mathrm{eff}\) from the SCF density \(\rho=\sum_k w_k\rho_k\), rebuild
\(H_k=T_k+V_\mathrm{nl}^k+V^k[V_\mathrm{eff}]\) at each path k, diagonalize.

Public API:

```cpp
cpu::gpw::GpwDft dft{{cpu::gpw::Kind{Si, basis, pot}}, /*cutoff_ry=*/150.0,
                     cpu::gpw::XcKind::PBE};  // + knobs: k_mesh, n_occ, …
dft.build(frame);       // positions [Å], Z in kAtom_type, orthorhombic box [Å]
cpu::gpw::Bands bands;  // members: n_bands, kfrac / set_diamond_path
auto br = bands.compute(dft, frame);
// br.eigenvalues_ha[ik][ib], gap_ha, e_vbm_ha, e_cbm_ha [Hartree]
```

No public `*Config`, no factories. Twin class: CP2K Quickstep
`&PRINT &BAND_STRUCTURE` on GTH-GPW Si.

### System

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

### Offline CP2K pin

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

### Fair wall (matched twin job)

| | Wall [s] | Threads |
| --- | --- | --- |
| ATV (`reg_gpw-bands-si`, SCF + NSCF path) | **6.66** | 1 MPI × OMP 8 |
| CP2K (`gpw-bands-si.inp`, ENERGY + BAND_STRUCTURE) | **1.90** | 1 MPI × OMP 8 |

**Ratio \(T_\mathrm{ATV}/T_\mathrm{CP2K} = 3.51\)** (≤ 10). Warm re-runs; same
binary pins as above.

### Acceptance gates (regression)

1. SCF converges; \(E\) finite; \(|E-E_\mathrm{CP2K}|\le 10^{-3}\) Ha.
2. Path has 16 k (`n_seg=4`), eigenvalues sorted, `n_bands` per k.
3. Path includes Γ and X; min local gap > 0.
4. Path gap and Γ local gap within 50 meV of CP2K 0.893516 eV.
5. VBM and CBM at Γ (twin character on this bct path).


## REFERENCE — gpw-bands-metallic

### Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-bands-metallic/na_bcc_smear_bands.inp` |
| Class | Quickstep GTH-GPW RKS + SMEAR FERMI_DIRAC + `&PRINT &BAND_STRUCTURE` |
| System | BCC Na conventional 2-atom, DZVP-MOLOPT-SR-GTH + GTH-PBE-q9, PADE, MP 2×2×2 |

### Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| Free energy \(F\) | −92.126330608378 Ha (ATV freeze class) | \(10^{-3}\) Ha |
| \(E_F\) class | 0.054478 Ha (CP2K dos-metallic twin) | 0.05 Ha |
| Metallic character | path spectrum straddles \(E_F\) | product |

### Fair wall

| Side | Wall |
| --- | --- |
| ATV (OMP=8, first green 2026-08-04) | 1.77 s (`ATV_WALL_S`) |
| CP2K twin | run offline; gate \(r\le 10\) |


## REFERENCE — gpw-bands-uks

### Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-bands-uks/si_uks_bands.inp` |
| Class | Quickstep GTH-GPW collinear UKS + `&PRINT &BAND_STRUCTURE` |
| System | Diamond Si BCT 4-atom, DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE, MP 2×2×2 |

### Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| \(E_{\mathrm{tot}}\) | −15.802416412655198 Ha | \(10^{-3}\) Ha |
| Path gap (closed-shell UKS ≡ RKS) | 0.893516 eV class | 80 meV |
| \(n_{\mathrm{spin}}\) | 2 | exact |
| α≡β spectrum (closed shell) | max\|Δ\| < 1e-6 Ha | product |

### Fair wall

| Side | Wall |
| --- | --- |
| ATV (OMP=8, first green 2026-08-04) | 7.56 s (`ATV_WALL_S`) |
| CP2K twin | run offline; gate \(r\le 10\) |


## Reference — Multi-k GPW density of states on diamond Si (BCT 4-atom)

### Method

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

### System

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

### Offline CP2K pin

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

#### Atomiverse first-green (2026-08-01, OMP=4 Release)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{tot}}\) | **−15.802394133256 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(2.23 \times 10^{-5}\) Ha |
| gap (integer-fill VBM–CBM) | **3.953 eV** |
| \(\int^{E_{\mathrm{vbm}}} D(E)\,dE\) | **14.84** (target 16 ± 1.5; Gaussian tails) |
| n_kpts (TR-folded) | 4 |
| n_bands (AO spectrum) | 52 |
| Wall SCF+DOS | **4.209 s** |

### Dual gate

1. **Energy** vs offline CP2K twin (\(|\Delta E| \le 10^{-3}\) Ha).
2. **DOS character:** multi-k payload (`n_kpts ≥ 2`), open gap ≥ 0.1 eV,
   \(D(E) \ge 0\) with a positive peak, occupied integral ≈ \(n_e\).

### Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | **2.595** | 1 MPI × OMP=4, `CP2K` timer MAXIMUM |
| Atomiverse | **4.209** | Release, OMP=4, SCF + Gaussian DOS |

`ratio = T_atv / T_cp2k = 4.209 / 2.595 ≈ **1.62**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true; fair twin only (same GTH-GPW
Si multi-k SCF job class).

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh.
- CP2K Quickstep `&PRINT &DOS` / multi-k MO eigenvalues (qs_dos.F).

### Provenance

- Spec: `.claude/specs/gpw-dos-si/acceptance.md`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dos-si/si_gpw_dos.inp`
- Sibling energy twin: `gpw-kpoints-si` (same geometry / cutoff / mesh)
- PW pattern: `cpu::pw::Dos` / `reg_bands-dos-si`


## Reference — multi-k GPW metallic DOS (BCC Na, FD smear)

### Method

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

### System

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

### Offline CP2K pin (algorithm twin)

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
the metallic free-energy SCF: ~0.64 Ha from grid/Poisson/table details at
matched CUTOFF / NGRIDS 1). Correctness gates are free-energy freeze +
Fermi character + \(\int^{E_F} D \approx N_e\).

#### Atomiverse free-energy freeze + first-green DOS (2026-08-01)

| Quantity | Value |
| --- | --- |
| Free energy \(F\) | **−92.126330608378 Ha** (freeze; tol \(10^{-4}\) Ha) |
| \(E_F\) | **0.055826 Ha** (\(\lvert\Delta E_F\rvert\) vs CP2K \(1.35\times 10^{-3}\) Ha) |
| \(\int^{E_F} D(E)\,dE\) | **17.999** (target 18 ± 2) |
| n_kpts (TR-folded) | 4 |
| n_bands (AO spectrum) | 18 |
| Wall SCF+DOS (OMP=4) | **2.736 s** |

### Dual gate

1. **Free energy** vs ATV freeze \(\lvert\Delta F\rvert \le 10^{-4}\) Ha.
2. **Metallic DOS character:** multi-k (`n_kpts ≥ 2`), finite \(E_F\) within
   0.02 Ha of the CP2K twin, \(D(E) \ge 0\) with a positive peak,
   \(\int^{E_F} D \approx 18\).

### Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | **0.499** | 1 MPI × OMP=4, `CP2K` timer MAXIMUM |
| Atomiverse | **2.736** | Release, OMP=4, SCF + Gaussian DOS |

`ratio = T_atv / T_cp2k = 2.736 / 0.499 ≈ **5.48**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true; fair twin only (same GTH-GPW
Na multi-k FD SCF + DOS job class).

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) — free-energy occupations.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh.
- CP2K Quickstep `&SMEAR` / `&PRINT &DOS` / multi-k MO eigenvalues.

### Provenance

- Spec: `.claude/specs/gpw-dos-metallic/` (deleted on close)
- Twin input: `tools/cp2k_oracle/inputs/gpw-dos-metallic/na_bcc_fd_dos.inp`
- Sibling metallic forces: `reg_gpw-metallic-forces`
- Sibling fixed-occ DOS: `reg_gpw-dos-si`
- PW metallic DOS pattern: `reg_dos-metallic-si`


## Reference — Multi-k GPW collinear UKS density of states (diamond Si BCT)

### Method

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

### System

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

### Offline CP2K pin

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

#### Atomiverse first-green (2026-08-01, OMP=4 Release)

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

### Dual gate

1. **Energy** vs offline CP2K twin (\(|\Delta E| \le 10^{-3}\) Ha).
2. **DOS character:** `nspin == 2`, multi-k payload (`n_kpts ≥ 2`), open gap
   ≥ 0.1 eV, dual-channel grids, \(D(E) \ge 0\) with a positive peak,
   occupied integral ≈ \(n_e\), \(\int D_\alpha + \int D_\beta \approx
   \int D\), each spin channel ≈ \(n_e/2\).

### Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | **5.164** | 1 MPI × OMP=4, `CP2K` timer MAXIMUM |
| Atomiverse | **13.648** | Release, OMP=4, UKS SCF + Gaussian DOS |

`ratio = T_atv / T_cp2k = 13.648 / 5.164 ≈ **2.64**` (≤ 10 ✓).

`perf_exempt` = false; `within_10x` = true; fair twin only (same GTH-GPW
multi-k collinear UKS SCF job class).

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh.
- CP2K Quickstep `&PRINT &DOS` / multi-k UKS MO eigenvalues (qs_dos.F).

### Provenance

- Spec: `.claude/specs/gpw-dos-uks/acceptance.md`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dos-uks/si_gpw_dos_uks.inp`
- Sibling RKS DOS: `gpw-dos-si`
- Sibling multi-k UKS energy: `gpw-hirshfeld-uks-kpoints`
- PW pattern: `cpu::pw::Dos` / `reg_dos-spin-si`
