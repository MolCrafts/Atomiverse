<!-- mol:note:topic:es-optimization-order -->
## ES optimization order — wave state

Binding long-term law: **CLAUDE.md → Electronic structure optimization order**.
This page holds only **current-wave** focus (rewrite in place; no diary stack).

### Active first cut (CPU GPW structure)

Strict sequence on the existing gpw-perf queue — no mega-spec spanning CUDA:

1. ~~`p1-gpw-workspace`~~ **landed** — private `GpwWorkspace` owns grid / mg /
   products / cbg / kbg / `last_veff` on Γ + multi-k RKS/UKS runners
   (`internal/gpw_workspace.h`). DIIS objects still stack-local. Lambda aliases
   keep control flow unchanged.
2. ~~`p1-ks-grid-build`~~ **landed** — `internal::ks_grid_build_rks/uks` +
   `evaluate_xc_*` share ρ→Poisson→XC→V_eff [+ nonlocal vdW] for Γ RKS/UKS,
   multi-k RKS, multi-k UKS (OT + DIIS). Collocate / H-integrate stay path-local.
3. ~~`p1-ot-uks-k`~~ **landed** — OT line-search economics on Γ UKS + multi-k
   RKS/UKS integer paths (carried E/H, quadratic Armijo retry, budget reuse).
4. ~~`p1-integrate-windows` / `p2-simd-pointfill-2`~~ **landed** — local \|V\|
   windows + 1D poly xsimd.
5. ~~`p2-cxyz-collint` + lp-spec + sphere cache~~ **landed**
6. ~~`p2-stvnl-setup` + collint group-parallel~~ **landed** — Hermite
   memoization + fused `prim_ST`; group-idx collocate.
7. ~~`p2-collint-polish` (residual)~~ **landed** — Hermite gen-stamp + s–s
   fast path (STVnl ~1500→520 ms); integrate group-idx + segment SIMD.
   MG MAIN **~2.1 s** (~**1.6×** CP2K quiet ~1.28 s; was ~42 s). Pins green.
8. ~~`cpu-xc-spectral-fft`~~ **landed** (es-cpu-cuda-opt wave 1) — fused
   multi-out FFT + MEASURE plans for GGA spectral grad/div; XC ~108→95 ms/iter.
9. ~~`cpu-ot-diis-outer`~~ **landed** (es-cpu-cuda-opt wave 2) — g·e B-matrix +
   CG fallback; water **93→32** it; UKS O₂ **180→~90** it / ratio **~7.1→~4.4×**.
10. ~~`cpu-ot-residual`~~ **landed** (es-cpu-cuda-opt wave 3) — OUTER_SCF polish
    (fixed-H eigen+mix); UKS O₂ **~90→27** it / OT **~13 s** / ratio **~1.5×**.
11. ~~`cpu-stvnl-residual`~~ **landed** (wave 3) — Hermite geometry-key reuse +
    phase flatten; STVnl ~507→435 ms; MG MAIN **~2.3 s / ~1.8×**.
12. ~~`cpu-collint-residual`~~ **landed** (wave 4) — hybrid xsimd FMA + 4-wide
    residual, lp_z≤9, cab-style alpha, TLS multi-imr sphere, gamma group_idx
    nthread grids; coll **~63** / int **~76** ms; MAIN remeasure **~2.4 s /
    ~1.9×** (setup residual dominated remaining).
13. ~~`cpu-products-mg-setup`~~ **landed** (wave 5) — geometry-batch
    `build_products` (shared K/pc per collint geometry, flat exclusive-scan
    packing); products_mg **~285→~18 ms**; MAIN **~1.9 s / ~1.5×** CP2K.
14. ~~`cpu-collint-shell-tasks`~~ **landed** (wave 6) — `CollintTask` ranges
    once in `build_products`; collocate/integrate over contiguous monom runs;
    MAIN **~1.6 s / ~1.25×**; coll **~30** / int **~58** ms.
15. ~~`cpu-collint-integrate-residual`~~ **landed** (wave 7) — width-4 integrate
    accumulate on AVX-512, short-segment scalar, Ltot triangular skip, zero-pol
    reverse, cab n≥4; int **~58→~45** ms; MAIN **~1.6 s / ~1.25×**.

> **Wall numbers in the wave list above are as-claimed-at-the-time and are
> superseded.** [`gpw-perf-baseline.md`](gpw-perf-baseline.md) is the single
> authority: it re-measured MG MAIN at **1.6 s ATV / 0.688 s CP2K = 2.33×**
> (worst_ratio **3.02**, `ot-minimizer-water`) and states explicitly that that
> row *replaces* the earlier ~1.25× figures, which came from an older quiet
> definition. Do not quote ~1.25× as current.

**Active next (`cp2k-perf-loop` — CP2K hard anchor, CPU only):**

| Priority | Slug | Target |
| --- | --- | --- |
| P0 | `p0-mg-main-structure` | MG MAIN **2.33× → ~1×** non-kernel structure |
| P1 | `p1-products-default-k` | fresh products vs dense A/B only |

Merged, no longer pending: `p0-ot-cp2k-match` (`cfd75024`),
`p0-stvnl-setup-residual` (`d9e1b0aa`).

**Forbidden this loop:** thin `product_collint` width-N re-polish (wave-8
exhausted); CUDA staff; hybrid/GAPW capability; global `XTENSOR_USE_XSIMD`.

**Worker law:** branches `wf/perf-<slug>` off **`dev`**; no push; wipe
`build` after every cmake/ctest. Metric = absolute `T_atv/T_cp2k` vs pinned
real `cp2k.psmp`.

Queue: [gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md).
Architecture: [gpw-perf-architecture.md](gpw-perf-architecture.md).
Baseline: [gpw-perf-baseline.md](gpw-perf-baseline.md).

### CUDA ES this wave

**CPU-first.** Footholds already landed; `cp2k-perf-loop` does **not** staff
CUDA. Full `cuda-gpw-full-product` still blocked. **Forbidden:** hybrid / GAPW /
PAW / meta-GGA product; claiming CUDA wins on small-molecule correctness twins;
classical MD CUDA (separate OpenMM track); full CUDA GPW method product before
architecture freeze.
