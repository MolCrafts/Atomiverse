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
> authority: MG MAIN **1.6 s ATV / 0.688 s CP2K = 2.33×** (worst fair ratio
> after OT water closed ≤1.1×). Do not quote older ~1.25× MAIN figures.

### Active next — `es-cpu-cuda-opt` product depth (2026-08-10 wave 6)

Arch residual **closed**; public-shape freeze **green**. First-slice product
**landed** on `dev` (feat `31d24207`; GH200 **1120817**). Device mid-SCF
collocate/integrate **landed** (`cuda-gpw-device-collocate` merge `69296751` /
GH200 **1125709**). Closed-shell PBE stress dual-gate **landed**
(`cuda-pw-pbe-stress` / `18663d13`; GH200 **1123959**). Collinear **spin-PBE
energy** **landed** (`cuda-pw-spin-pbe` / `118d3003`; GH200 **1124645**).
GapCensus **wave 6** re-diff held wave-5 depth + opened **10** deeper rows:
GPW `pbe-forces-gate`, `uks-multigrid-forces`, `multigrid-kpoint-forces`,
`multigrid-kpoint-stress`, `uks-multigrid-kpoints`, `kpoint-uks-forces`,
`kpoint-uks-stress`; PW `spin-kmesh`, `spin-smear-pbe-forces`,
`spin-smear-pbe-stress`. Staff **P1 product depth** next; do not re-merge
closed product / collocate rows.

| Priority | slug | Goal |
| --- | --- | --- |
| P1 | `pbe-product-gate` / `device-forces` / `multigrid-run` / `uks-gamma` | GPW product depth (PBE gate + real force path + MG run + UKS energy) |
| P2 | `nondefault-stream` / `device-diis-eigen` / `product-h2o-gate` / `kpoints` / `stress` + PW smear/spin F/σ/smear + **`spin-kmesh`** | Stream + host-SCF residual + water dual-gate + multi-k + stress + PW depth |
| P3 | wave-4/5/6 joint paths (UKS/MG/k F+σ + multi-k MG/UKS + triple UKS-MG-k + kpoint-UKS F/σ + pbe-forces-gate + device-stvnl + spin-pbe/smear F/σ + spin-smear-pbe F/σ) | after parent energy rows |

**cpu_pending=0** — MG MAIN **2.33×** remains blocked pin-unsafe residual
(not soft-tol). **pending_count=39**; **in_progress=0**; **cuda_pending=39**;
**gaps_remaining≈40**. Contract:
[cuda-gpw-public-shape.md](cuda-gpw-public-shape.md). Queue:
[es-cpu-cuda-queue.md](../../docs/developer/es-cpu-cuda-queue.md).

**Forbidden:** soft tols; invent hybrid/GAPW; staff product with non-CPU public
shape; leave `build*`.

### Superseded focus — dual track `es-cpu-cuda-opt` (foothold set closed 2026-08-10)

**Two tracks, one loop.** No new electronic methods (no hybrid / GAPW / PAW /
meta-GGA product). No classical MD CUDA. Parent **`dev`**. Workers **only** on
`wf/escuda-<slug>`; no push; no `reset --hard`. Wipe `build` / `build-*` after
every cmake/ctest and before every worker/clearup exit (`builds_wiped=false`
fails the wave).

| Track | Goal | Anchor / gate |
| --- | --- | --- |
| **cpu** | Cut wall on **landed** GPW/SCF vs real CP2K | \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) on matched twin, **1 MPI × OMP=4**; \(T_{\mathrm{cp2k}}\) from real `cp2k.psmp` (`source …/cp2k/install/env.sh`); provenance in [gpw-perf-baseline.md](gpw-perf-baseline.md) / [oracle-pin-provenance.md](oracle-pin-provenance.md). Pins green; FD-force if energy path moves. |
| **cuda** | **Corresponding** CUDA for landed ES — PW method depth + GPW compose | CPU twin green; public `run`/`forces`/`stress` stable where owned; CUDA↔CPU tols **stated**; device-only kernels. Same-settings CPU is correctness oracle. Login has **no** toolkit — `sbatch -p gpu`, module `GPU/buildenv-gcccuda/2026.03-cu13.0`; never claim green without GPU log evidence. |

**CPU rank (fair walls — baseline authority):**

| Twin | ratio | Residual / staff note |
| --- | ---: | --- |
| `gpw-multigrid-kpoints` MAIN | **2.33** | worst_ratio — pin-safe collint/structure/sphere **exhausted**; blocked as `mg-main-structural` |
| `ot-uks-o2` | **~1.4** | ≤3× aim met; not primary |
| `ot-minimizer-water` | OT ~**1.20** / SCF body **≤1.0** | control + residual closed |
| `gpw-uks-o2-gamma` | **1.077** | gate closed — do not re-open dual-spin fuse |
| `gpw-kpoints-si` MAIN | **1.18** | hold products vs dense A/B until MG moves |

**CUDA open-gate (`dev` @ `2958f949`):** Landed: device s/p RHF + CDIIS +
non-default stream D2H; Γ GPW footholds + LDA + closed-shell PBE `ks_chain` +
**multi-grid product KS stage** + **first public `cuda::gpw::GpwDft`** +
**device mid-SCF collocate/integrate** (Γ RKS DIIS PADE/PBE; host STVnl/DIIS/
eigen + weight pack; device product KS stream-null; PADE H₂ dual-gate only);
`cuda::pw::PW` LSDA + closed-shell PBE + residency + stream + large-cell +
forces (LSDA+PBE) + cell stress (**LSDA + PBE dual-gate**) + collinear **LSDA +
spin-PBE energy** + **restricted metallic smear energy**. **Pending:** GPW
product depth (PBE + H₂O gates / forces + PBE F gate / MG run + forces / UKS Γ
+ UKS F/σ + UKS-PBE / multi-k + kpoint F/σ / UKS multi-k + kpoint-UKS F/σ /
UKS MG + UKS-MG F / multi-k MG + MG-k F/σ + UKS-MG-k triple / stress / stream /
DIIS+eigen / device STVnl); PW spin + smear F/σ depth (spin F/σ + spin-PBE F/σ
+ **spin multi-k dual-gate** + spin-smear energy + spin-smear F/σ +
spin-smear-pbe + spin-smear-pbe F/σ — see queue). **No open tips.**
**Blocked structural:** `mg-main-structural`. Public-shape freeze **green**
(`0147fe02`). **cpu_pending=0** — no pin-safe wall lever.

**Queues (binding worklists):**

- Dual-track queue: [es-cpu-cuda-queue.md](../../docs/developer/es-cpu-cuda-queue.md)
- CPU GPW wall detail: [gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md)
- Architecture: [gpw-perf-architecture.md](gpw-perf-architecture.md)
- CUDA RHF / PW envelope: [cuda-rhf-device-only.md](cuda-rhf-device-only.md)

**Forbidden:** hybrid/GAPW/PAW/meta-GGA product; classical MD CUDA; global
`XTENSOR_USE_XSIMD`; arch intrinsics; soften tols / self-golden pins; leave
multi-GB builds; nested long-lived subagents; invent capability gaps here
(use method loops). SIMD = portable xsimd only.
