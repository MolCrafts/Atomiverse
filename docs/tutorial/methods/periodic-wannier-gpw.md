# Maximally localized Wannier functions on GPW (multi-k Marzari–Vanderbilt)

*(Technical-report page; CP2K capability loop wave 14.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-wannier-gpw` |
| Regression | `regressions/gpw/periodic-wannier-gpw/` |
| Public API | `cpu::gpw::PeriodicWannier` → `centers[i]` [Å], `spreads[i]` [Å²] |
| Roadmap / queue | `docs/roadmap/dft.md` (response/localization); queue done 2026-07-26 |

## Summary

**Marzari–Vanderbilt** maximally localized Wannier functions from a multi-k GPW
calculation: overlap matrices between neighbouring \(k\) points, spread
functional \(\Omega\) minimized by steepest descent, and centers from the
Berry-phase (diagonal-phase) formula.

## System

| | |
| --- | --- |
| Lattice | \(a = 5.43\) Å; cell \((a/\sqrt2, a/\sqrt2, a)\) — body-centred tetragonal Si |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Valence | 16 e⁻ → \(n_{\mathrm{occ}} = 8\) → **8 MLWFs** |
| XC / cutoff | PADE, 100 Ry |
| MLWF k-mesh | 2×2×2, time reversal off (full neighbour graph) |
| SCF | DIIS, tol 1e−7 Ha |

## Gate

The physical anchor is the **literature result** (Marzari & Vanderbilt 1997):
in a covalent crystal the MLWF centers sit on the **bond midpoints**. The
regression asserts SCF convergence, `n_wann == 8` with 8 centers, every center
within **0.90 Å** of its assigned bond midpoint under minimum image (observed
\(d_{\max} \approx 0.012\) Å), and a spread reduction
\(\Omega/\Omega_0 < 0.5\) (observed **≈ 0.09**) with finite spreads. SCF energy
≈ −15.77 Ha. Wall ≤ 60 s, no external process.

Unit gates cover the bond-midpoint generator, the cubic SMV weights, the
center-from-diagonal-phases formula, and the rejected inputs (`k_mesh < 1`,
`tol <= 0`, metallic smearing, UKS).

Ratio: **perf_exempt** — CP2K's `&LOCALIZE` prints centers, and the geometric
bond-midpoint criterion rather than a wall is the meaningful twin here.

See `regressions/gpw/periodic-wannier-gpw/REFERENCE.md`.
