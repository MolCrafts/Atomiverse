# Reference — FD supercell phonons (diamond Si, Γ)

## Method

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

## System (regression twin)

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

## Offline pin (no live CP2K in ctest)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` on the same geometry (Quickstep GPW or classical) |
| CP2K binary this host | **not available** (no EasyBuild CP2K module on the pin host this shift) |
| Surrogate pin | Physical acoustic sum rule + published SW optical scale |

### Acceptance numbers (hard-coded in the regression)

1. **Mode count:** \(3N = 192\) (64 atoms).
2. **Acoustic sum rule (Γ):** three lowest \(|\omega| < 30\) cm⁻¹.
   At a stress-free crystal minimum the acoustic branches vanish at \(\mathbf{q}=0\);
   residual is FD / force noise.
3. **Highest optical wavenumber** in **[500, 900]** cm⁻¹.
   Experimental Si TO(Γ) ≈ 520 cm⁻¹; the original SW potential is stiffer and
   typically overshoots (several hundred cm⁻¹). The band rejects unit disasters
   (Hz vs cm⁻¹, free-atom zero spectrum, NaN) without freezing Atomiverse’s own
   first-run output as a golden snapshot.

## CP2K twin (offline only)

When `cp2k.psmp` is available:

1. Run a fixed `VIBRATIONAL_ANALYSIS` input on diamond Si (same \(a\), comparable
   force model or Quickstep).
2. Record \(\omega\) [cm⁻¹] and wall time in this file with CP2K version.
3. Optional fair wall: \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\); require \(r \le 10\).

Skeleton inputs may be added under `tools/cp2k_oracle/inputs/`.

## Measured twin (this host, 2026-07-23)

Release build, `OMP_NUM_THREADS=8`, Stillinger–Weber 64-atom Γ FD:

| Quantity | Value |
| -------- | ----- |
| Acoustic \(\|\omega\|_\max\) (3 lowest) | **0.000 cm⁻¹** |
| Highest optical \(\omega\) | **594.822 cm⁻¹** (inside [500, 900]) |
| Wall | ≈ **0.06 s** |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair `VIBRATIONAL_ANALYSIS` wall twin this shift |
| ATV twin wall (informative) | SW 64-atom Γ FD ≈ 0.06 s (Release, 8 threads) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

## Why SW not PW for the first pin

A full PW/GPW FD Hessian is \(6N\) SCF+force evaluations. The **machinery**
under test is supercell expansion + crystal \(3N\) diagonalization + public
`Phonon` API. SW provides periodic many-body forces with a literature potential
and finishes in seconds. Binding `PWEvaluator::with_gradient(true)` (or GPW)
on `vib::Phase::force` is the same public composition path for a later CP2K
energy/frequency twin.
