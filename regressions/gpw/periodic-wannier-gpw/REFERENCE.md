# Reference — Periodic multi-k MLWF (Marzari–Vanderbilt) on GPW, bulk Si

## Method

Maximally localized Wannier functions on the CPU **GPW** method: closed-shell
RKS multi-k SCF (GTH + DZVP-MOLOPT, PADE), then Marzari–Vanderbilt
localization on the occupied AO Bloch manifold
(`cpu::gpw::PeriodicWannier`). Neighbor overlaps use phase-weighted Gaussian
cross-AO matrices; projection init onto bond-centered Gaussians; optional
steepest-descent polish; centers from the Berry-phase formula.

Public API:

```cpp
cpu::gpw::PeriodicWannier mlwf;   // members: k_mesh, max_iter, tol, …
auto r = mlwf.compute(method, atoms);
// r.centers[i] [Å], r.spreads[i] [Å²]
```

No public `*Config`, no factories.

## System

Diamond Si in the **body-centred-tetragonal 4-atom** orthorhombic cell
(GPW is orthorhombic-only; same setting as `gpw-kpoints-si`):

| | |
| --- | --- |
| Lattice | a = 5.43 Å; cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → n_occ = 8 → **8 MLWFs** |
| XC | PADE (LDA) |
| Density cutoff | 100 Ry |
| MLWF k-mesh | 2×2×2, time-reversal **off** (full neighbor graph) |
| SCF | DIIS, tol 1e-7 Ha, EPS_DEFAULT 1e-7 |

Literature geometry pin (not an ATV snapshot): diamond / zincblende valence
MLWFs sit on the **Si–Si bond midpoints** (Marzari & Vanderbilt 1997 §IV;
standard Wannier90 Si demo). In the BCT 4-atom cell there are **eight**
unique tetrahedral bonds.

## Acceptance (hard-coded in `reg_periodic-wannier-gpw.cpp`)

1. SCF converges; `n_wann == 8`.
2. After greedy bipartite assignment and min-image under the BCT cell, every
   Wannier center lies within **0.90 Å** of a unique bond midpoint (same
   coarse 2×2×2 SMV window as `reg_periodic-wannier-si`).
3. Spread reduction: \(\Omega_\mathrm{final} < 0.5\,\Omega_\mathrm{init}\).
4. Finite spreads; no external process; ≤ 60 s.

## Offline oracle status

**No CP2K binary / module on the worker host at pin time** (same as
`gpw-kpoints-si` / `periodic-wannier-si`). Absolute CP2K Wannier coordinates
are **not invented**. Shared physical content with CP2K Quickstep
`&LOCALIZE` / print Wannier centers is the **bond-midpoint geometry** of the
eight valence MLWFs. Twin input (human/agent pin host only, never ctest):

`tools/cp2k_oracle/inputs/si_periodic_wannier_gpw.inp`

### Atomiverse numbers (first green)

Recorded after first green run on the worker host (Release). Replace the
absolute energy band with a cited CP2K total when an offline twin is available.

| Quantity | Value |
| --- | --- |
| Literature pin | bond midpoints (MV 1997) |
| SCF energy | ≈ −15.77 Ha (PADE, 100 Ry, 2³ k) |
| Midpoint tol | 0.90 Å |
| Observed dmax | ≈ 0.012 Å |
| Spread ratio floor | 0.50 |
| Observed Ω/Ω₀ | ≈ 0.09 |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on the pin host for a fair wall twin. |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |
