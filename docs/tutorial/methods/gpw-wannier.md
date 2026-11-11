# GPW maximally localized Wannier functions

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Marzari–Vanderbilt maximally localized Wannier functions (MLWFs) from a
multi-k GPW calculation: overlap matrices between neighbouring \(k\) points,
the spread functional \(\Omega\) minimized by steepest descent after a
projection start, and centers from the Berry-phase (diagonal-phase) formula.
`cpu::gpw::PeriodicWannier` (public members `k_mesh`, `max_iter`, `tol`,
`step`, `use_projection`, `proj_alpha_ang`) takes a built `cpu::gpw::GpwDft`
and the Frame and returns `centers[i]` [Å] and `spreads[i]` [Å²].

The case runs bulk diamond Si in two sections: `periodic-wannier-gpw`
(closed-shell RKS, below) and `periodic-wannier-gpw-uks` (collinear UKS,
\(N_\uparrow = N_\downarrow = 8\), per-spin MLWFs; record in the case
`REFERENCE.md`). The gate is the literature geometry: valence MLWFs of a
covalent crystal sit on the bond midpoints (Marzari & Vanderbilt 1997).

Case: `ctest -R '^gpw/wannier$'` — sources and oracle record in [`regressions/gpw/wannier/`](../../../regressions/gpw/wannier/). One section of the case per `##` heading below: `atv_regressions --case gpw/wannier --section <name>`.

## periodic-wannier-gpw — Multi-k Marzari–Vanderbilt MLWFs, bulk Si {#periodic-wannier-gpw}

### System

| | |
| --- | --- |
| Lattice | \(a = 5.43\) Å; cell \((a/\sqrt2, a/\sqrt2, a)\) — body-centred tetragonal Si |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Valence | 16 e⁻ → \(n_{\mathrm{occ}} = 8\) → **8 MLWFs** |
| XC / cutoff | PADE, 100 Ry |
| MLWF k-mesh | 2×2×2, time reversal off (full neighbour graph) |
| SCF | DIIS, tol 1e−7 Ha |

### Gate

The regression asserts SCF convergence, `n_wann == 8` with 8 centers, every
center within **0.90 Å** of its assigned bond midpoint under minimum image
(observed \(d_{\max} \approx 0.012\) Å), and a spread reduction
\(\Omega/\Omega_0 < 0.5\) (observed **≈ 0.09**) with finite spreads. SCF
energy ≈ −15.77 Ha. Wall ≤ 60 s, no external process.

Unit gates cover the bond-midpoint generator, the cubic SMV weights, the
center-from-diagonal-phases formula, and the rejected inputs (`k_mesh < 1`,
`tol <= 0`, metallic smearing).

Ratio: **perf_exempt** — CP2K's `&LOCALIZE` prints centers, and the geometric
bond-midpoint criterion rather than a wall is the meaningful twin here.
