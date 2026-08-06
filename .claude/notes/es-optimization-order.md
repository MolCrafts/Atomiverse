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

**Active next (`es-cpu-cuda-opt` post wave-5 land):**

| Track | Focus | Anchor |
| --- | --- | --- |
| **cpu** | `cpu-collint-shell-tasks` (then optional `cpu-simd-2center-setup`) | `T_atv/T_cp2k` MG MAIN ~1.5× → ~1× |
| **cuda** | open-gate only after next GapCensus; no full GPW product; no higher-L RHF | same-settings CPU oracle; device-only; GH200 |

1. **CPU wall** — MG MAIN **~1.5×** is the worst fair-wall class (OT worst
   **~1.4×**, ≤3× aim met). Remaining lever: shell-level collint tasks toward
   CP2K ~20 ms/iter class. Energy-path change ⇒ re-run FD force oracle.
2. **CUDA ES** — RHF device s/p energy+forces+level_shift+path-select+**CDIIS**
   **landed**; Γ GPW Poisson/FFT + Padé LDA XC + **ρ collocate** footholds
   **landed** (not full method). Full CUDA GPW product remains blocked.

**Worker law:** branches `wf/escuda-<slug>` off **`dev`**; no push; no
`reset --hard`. Disk: wipe `build`/`build-*` after every cmake/ctest. CUDA
build/run only via `sbatch -p gpu` (login has no toolkit). GH200 cmake needs
`-DCMAKE_CUDA_ARCHITECTURES=90` and must not link x86 `libxtb` (SE OFF or
null `ATV_XTB_ROOT`).

Queue: [es-cpu-cuda-queue.md](../../docs/developer/es-cpu-cuda-queue.md) +
[gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md).

Architecture: [gpw-perf-architecture.md](gpw-perf-architecture.md).
Baseline walls: [gpw-perf-baseline.md](gpw-perf-baseline.md).
CUDA RHF: [cuda-rhf-device-only.md](cuda-rhf-device-only.md).

### CUDA ES this wave

**Wave 5 landed** device CDIIS (Pulay) on s/p RHF + thin product ρ collocate
foothold (plus CPU products_mg setup). **Forbidden:** hybrid / GAPW / PAW /
meta-GGA product; claiming CUDA wins on small-molecule correctness twins;
classical MD CUDA (separate OpenMM track); full CUDA GPW method product before
architecture freeze.
