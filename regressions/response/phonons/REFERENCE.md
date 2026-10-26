# response/phonons

One case; every section below is run by `atv_regressions --case response/phonons` and keeps its own oracle record.


## Reference — phonon dispersion q-path (diamond Si, SW IFC Fourier)

### Method

Interatomic force constants Φ of a **2×2×2 primitive supercell** of diamond
Si are assembled by finite-displacement central differences
(`cpu::vib::Phonon` + Stillinger–Weber). The mass-weighted dynamical matrix
of the **2-atom primitive cell** at fractional wavevector **q** is the
Fourier sum

\[
D(\mathbf{q})
  = \frac{1}{\sqrt{m_i m_j}}
    \sum_{\mathbf{R}}
    \Phi(0i,\,\mathbf{R}j)\,e^{i\mathbf{q}\cdot\mathbf{R}},
\]

with Φ converted eV/Å² → Hartree/Bohr² using the same NIST factors as
`MassWeighting` / `PhononDiagonalizer`. Eigenvalues → ω [cm⁻¹]. Public path:
`cpu::vib::PhononDispersion::compute(hessian, unit, na, nb, nc)`.

This is the phonopy-style **supercell IFC Fourier** product surface for
full dispersions (q ≠ 0). Finite-q DFPT Sternheimer remains a follow-on.

Cite: Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) (potential);
Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001) (DFPT context);
phonopy / Togo–Tanaka for the Fourier IFC construction.

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, **conventional cubic 8-atom** cell, a = **5.431 Å** |
| Force engine | `StillingerWeber` / `SWPotential(Si, "stillinger-weber")` |
| Supercell | **2×2×2** conventional (64 atoms) for IFCs — same SW twin as `fd-phonon-supercell` |
| FD step | \(h = 10^{-3}\) Å (`kHessianStepAngstrom`) |
| q-path | conventional-cubic **Γ → X → L** (fractional): (0,0,0), (½,0,0), (½,½,½) + midpoints |
| Masses | IUPAC-2013 via `cpu::vib::mass_amu(Z)` |
| Branches | \(3N = 24\) (conventional cell) |

**Why conventional, not 2-atom primitive:** the classical neighbor cell list
takes `Box::length(d) = cell(d,d)`, which is **0** for the diamond primitive
triclinic cell and can infinite-loop in orthorhombic min-image helpers. The
Fourier API itself is cell-agnostic; cubic conventional is the proven SW
geometry from `fd-phonon-supercell`. (Triclinic neighbor fix is out of scope
for this slug.)

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + phonon band structure on the same SW (or GTH) cell |
| CP2K binary this host | **not available** (no CP2K module on the pin host this shift) |
| Pin 1 (Γ acoustic) | \(\max|\omega_\mathrm{ac}(\Gamma)| < 30\) cm⁻¹ (no ASR projection) — same bound as `fd-phonon-supercell` / `dfpt-phonon-si` |
| Pin 2 (Γ optical) | \(\omega_\mathrm{max}(\Gamma) \in [500, 900]\) cm⁻¹ — SW Si optical band (same window as `reg_fd-phonon-supercell`) |
| Pin 3 (X, L) | \(\omega_\mathrm{max}(X),\omega_\mathrm{max}(L) \in [300, 900]\) cm⁻¹ — finite optical/acoustic zone-boundary band for SW Si |

#### Measured pins (this host, 2026-07-25, Release, OMP=4)

| Quantity | Value |
| -------- | ----- |
| Γ \|acoustic\|_max | **0.00** cm⁻¹ |
| Γ ω_max (optical) | **594.82** cm⁻¹ |
| X ω_max | **562.56** cm⁻¹ |
| L ω_max | **559.27** cm⁻¹ |
| Branches / path length | 24 / 5 |
| Regression wall | **0.07 s** (≤ 60 s budget) |

#### Physics notes

- SW is short-ranged (cutoff \(a\sigma \approx 3.77\) Å). The 2×2×2
  conventional supercell (L = 10.862 Å > 2×cutoff) is the same geometry as
  `reg_fd-phonon-supercell`; IFC support is complete for this twin.
- True experimental Si LO(Γ) ≈ 520 cm⁻¹ is **not** the pin — SW is classical
  and systematically stiffer; the window is the same SW band used by the FD
  Γ foothold.
- Γ acoustic near-zero is a hard correctness gate (broken ASR → broken Φ).
- At Γ, the Fourier spectrum of the conventional cell must recover the same
  acoustic/optical physics as the supercell-Γ FD spectrum of the 64-atom cell
  (zone-folded branches sit among the 24 modes).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair VIBRATIONAL_ANALYSIS wall twin this shift |
| ATV wall (informative) | same FD cost class as `fd-phonon-supercell` (64-atom SW, 6N displacements) + cheap Fourier path; unit test baseline ~30 ms for FD alone; expect ≪ 60 s |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |

### CP2K twin (offline only)

When a CP2K binary is available: run Quickstep / classical SW vibrational
analysis for diamond Si at a = 5.431 Å, record ω(Γ/X/L) [cm⁻¹] + wall with
the CP2K version, and optionally the fair wall ratio
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) (require ≤ 10).


## Reference — FD supercell phonons (diamond Si, Γ)

### Method

Finite-displacement (central difference, \(h = 10^{-3}\) Å) Cartesian force
constants on a periodic supercell, mass-weighted dynamical matrix at **Γ**,
full \(3N\) spectrum in cm⁻¹. Public path:

`cpu::vib::Phonon` → `expand_supercell` (when \(n_a n_b n_c > 1\)) →
`DisplaceStep` / `HessianAccumulator` / `Symmetrize` / `MassWeighting` /
`PhononDiagonalizer`.

This is **not** molecular `cpu::vib::Hessian` (which projects free-molecule
translations+rotations and has no supercell).

Cite: standard FD phonon machinery (e.g. phonopy-style finite displacements;
CP2K `VIBRATIONAL_ANALYSIS` is the intended Quickstep twin when a binary is
available offline).

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, **2×2×2 conventional** cubic supercell, **64 atoms** |
| Lattice constant \(a\) | **5.431 Å** (SW fit target); box \(L = 2a = 10.862\) Å |
| Why 2×2×2 | SW cutoff \(a\sigma = 3.7712\) Å requires \(L > 2\times\) cutoff (same gate as `reg_sw_si`) |
| `Phonon` factors | \(n_a=n_b=n_c=1\) on the already-built supercell Frame (Γ of that cell). `expand_supercell` is unit-tested separately |
| Force engine | Stillinger–Weber Si (`cpu::StillingerWeber`) |
| SW parameters | Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) |
| FD step \(h\) | \(10^{-3}\) Å (`kHessianStepAngstrom`) |
| Neighbor | always-rebuild (`check=false`), cutoff \(a\sigma\) |

Basis: orthorhombic cubic cell; fractional diamond sites tiled over \(2^3\)
conventional cells (see `reg_fd-phonon-supercell.cpp`).

### Offline pin (no live CP2K in ctest)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` on the same geometry (Quickstep GPW or classical) |
| CP2K binary this host | **not available** (no EasyBuild CP2K module on the pin host this shift) |
| Surrogate pin | Physical acoustic sum rule + published SW optical scale |

#### Acceptance numbers (hard-coded in the regression)

1. **Mode count:** \(3N = 192\) (64 atoms).
2. **Acoustic sum rule (Γ):** three lowest \(|\omega| < 30\) cm⁻¹.
   At a stress-free crystal minimum the acoustic branches vanish at \(\mathbf{q}=0\);
   residual is FD / force noise.
3. **Highest optical wavenumber** in **[500, 900]** cm⁻¹.
   Experimental Si TO(Γ) ≈ 520 cm⁻¹; the original SW potential is stiffer and
   typically overshoots (several hundred cm⁻¹). The band rejects unit disasters
   (Hz vs cm⁻¹, free-atom zero spectrum, NaN) without freezing Atomiverse’s own
   first-run output as a golden snapshot.

### CP2K twin (offline only)

When `cp2k.psmp` is available:

1. Run a fixed `VIBRATIONAL_ANALYSIS` input on diamond Si (same \(a\), comparable
   force model or Quickstep).
2. Record \(\omega\) [cm⁻¹] and wall time in this file with CP2K version.
3. Optional fair wall: \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\); require \(r \le 10\).

Skeleton inputs may be added under `tools/cp2k_oracle/inputs/`.

### Measured twin (this host, 2026-07-23)

Release build, `OMP_NUM_THREADS=8`, Stillinger–Weber 64-atom Γ FD:

| Quantity | Value |
| -------- | ----- |
| Acoustic \(\|\omega\|_\max\) (3 lowest) | **0.000 cm⁻¹** |
| Highest optical \(\omega\) | **594.822 cm⁻¹** (inside [500, 900]) |
| Wall | ≈ **0.06 s** |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair `VIBRATIONAL_ANALYSIS` wall twin this shift |
| ATV twin wall (informative) | SW 64-atom Γ FD ≈ 0.06 s (Release, 8 threads) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

### Why SW not PW for the first pin

A full PW/GPW FD Hessian is \(6N\) SCF+force evaluations. The **machinery**
under test is supercell expansion + crystal \(3N\) diagonalization + public
`Phonon` API. SW provides periodic many-body forces with a literature potential
and finishes in seconds. Binding `PWEvaluator::with_gradient(true)` (or GPW)
on `vib::Phase::force` is the same public composition path for a later CP2K
energy/frequency twin.


## Reference — GPW Si₂ vibrational spectrum

### Oracle and physical inputs

CP2K **2026.2**, source revision **git:c92cc08**, actually executed
2026-10-25 using the native Homebrew binary `/opt/homebrew/bin/cp2k.psmp`.
The three canonical `si2_matched_{energy_force,geo_opt,vib}.inp` inputs,
complete outputs, hashes, exact commands and version are recorded in
[`tools/cp2k_oracle/inputs/gpw-fd-phonon-si/reference.json`](../../../tools/cp2k_oracle/inputs/gpw-fd-phonon-si/reference.json)
and its neighboring files. The matching CP2K data files are `BASIS_MOLOPT`
and `GTH_POTENTIALS`; their hashes are also recorded.

| Item | Value |
| --- | --- |
| Hamiltonian | PADE / DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Local pseudopotential | CP2K `CORE_PPL GRID`, native `local_pp_on_grid=true` |
| Spin | UKS, Nα=5, Nβ=3 |
| Cell | 8 Å cube, isolated Martyna–Tuckerman |
| Density grid | 150 Ry, one level, 60³ points |
| MT auxiliary grid | Twice each density-grid dimension, 120³ points; CP2K `MT REL_CUTOFF 4`, `ALPHA 7` |
| Gaussian screening | EPS_DEFAULT 1e-10 |
| Mass | 28.0855 amu |
| Actual stationary vibration-input bond | 2.1450905162 Å |
| Final GRID optimization bond | 2.1450907568000073 Å |
| Cartesian displacement | 0.001 Å on both engines |

### Printed energy, force and frequency pins

At the deliberately nonstationary bond 2.246 Å, centred at (4,4,4) Å,
`si2_matched_energy_force.cp2k.txt` lines 1069–1074 print:

```text
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -7.600561394706659
FORCES| 1  2.72740080E-02  6.32154382E-09  3.51118623E-09
FORCES| 2 -2.72740080E-02 -6.32099739E-09 -3.51163041E-09
```

The GRID optimization completed with energy -7.603345177586373 Ha and
bond 2.1450907568000073 Å, including tiny transverse coordinate shifts.
The vibration input deliberately preserves its actual bond 2.1450905162 Å
and exact centring. Its bond differs from the GRID optimum by
-2.4060000747e-7 Å; its frequency is pinned at that actual input rather than
at substituted coordinates.

The last, thirteenth evaluation in `si2_matched_vib-r-0.cp2k.txt` is the
restored **undisplaced vibration input**. The preceding 12 evaluations are
Cartesian plus/minus samples. Lines 2202 and 2206–2207 print:

```text
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -7.603345177586478
FORCES| 1 -7.09733727E-08 -3.89216218E-09  3.95186186E-09
FORCES| 2  7.09720992E-08  3.89282473E-09 -3.95119886E-09
```

Its largest absolute force component is 7.09733727e-8 Ha/Bohr and largest
force norm is 7.11897870e-8 Ha/Bohr, independently satisfying the unchanged
1e-5 Ha/Bohr stationarity requirement. The main vibration output line 568
prints:

```text
VIB|Frequency (cm^-1) 5.7365790135E+02
```

CP2K reports this internal stretch after removing rotations and translations.
The native phonon output remains the unprojected six Cartesian modes; its
five near-zero modes have their separate physical-symmetry acceptance bound.
GEO_OPT generates the restart explicitly read by VIBRATIONAL_ANALYSIS.
The native regression likewise continues the same method instance from the
nonstationary input through the stationary geometry and displacements.
An arbitrary cold guess need not find the same nonlinear SCF root.

### Acceptance

The section mounts `GpwDft` directly in the existing `Evaluate` and `Phonon`
Drivers. It introduces no extra method Driver or wrapper. Every energy and
force component of the nonstationary input is compared with CP2K, within
1e-5 Ha and 1e-4 Ha/Bohr. At the actual vibration input, force components
must be below 1e-5 Ha/Bohr and energy within 1e-5 Ha of the center-point pin.

Three translations and two rotations must satisfy the original |ω|<80 cm⁻¹
bound. The stretch must match the independent CP2K frequency within 100 cm⁻¹,
retaining the original 100 cm⁻¹ energy-curvature consistency bound as an
additional check. The whole section, including the external-vector preflight
and energy-curvature solves, must take at most 60 seconds at its registered
eight-thread width. No acceptance bound has been relaxed.

### Performance boundaries and numerical primitives

Actual CP2K process walls were 70.622467s (energy/force), 118.210183s
(optimization), and 133.218425s (vibration), one MPI rank, with environment
requests `OMP_NUM_THREADS=8` / `OPENBLAS_NUM_THREADS=1`, binding unset.
CP2K's actual internal BLAS width was not queried; those environment requests
do not establish single-thread BLAS or a fair matched-width wall ratio.
The native complete section includes energy/force preflight
and energy-curvature checks beyond the vibration itself. Distinguish that
whole-section boundary from the external vibration-only process in wall
comparisons; the section's 60s budget applies to all its work.

The CPU operator uses the erf-screened MT kernel, and the isolated density
is a periodized Gaussian **product** expansion. Clipping individual orbital
values at cell edges loses charge; summing orbitals before multiplication
adds cross-image products. The force contracts center derivatives of the
same product expansion and uses the adjoint grid transfer. Closed-form
Gaussian electrostatics and s/p/d Gaussian Fourier-moment gradients
independently test these primitives.
