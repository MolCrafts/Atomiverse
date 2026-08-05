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
   multi-out FFT + MEASURE plans for GGA spectral grad/div; XC ~108→95 ms/iter;
   OT water quiet ~16.5→16.2 s (~5.99× vs 2.7 s CP2K note). Pins + FD forces
   green. Non-bit-exact MEASURE FP with stated tol.

**Active next (`es-cpu-cuda-opt` post wave-1 land):**

| Track | Focus | Anchor |
| --- | --- | --- |
| **cpu** | OT outer count (`cpu-ot-diis-outer` still open on branch) + residual STVnl | `T_atv/T_cp2k`, 1 MPI × OMP=4 |
| **cuda** | Device-only polish on landed s/p RHF; GPW foothold only under open-gate | same-settings CPU oracle; device-only kernels |

1. **CPU wall** — finish/merge `cpu-ot-diis-outer` (Γ OT MINIMIZER DIIS outer
   count toward CP2K ~30 class; branch claims 93→32 / 180→94 / ~3.9×). Then
   residual STVnl/setup on MG twin (~1.6×). Energy-path change ⇒ re-run FD
   force oracle.
2. **CUDA ES** — `cuda-rhf-pshell-forces` **landed** (wave 6 device s/p forces;
   GH200 unit green). Next: `cuda-rhf-device-only-polish` →
   `cuda-rhf-level-shift` → optional thin Γ GPW foothold (`cuda-gpw-foothold`)
   **only if** open-gate holds.

**Worker law:** branches `wf/escuda-<slug>` off **`dev`**; no push; no
`reset --hard`. Disk: wipe `build`/`build-*` after every cmake/ctest. CUDA
build/run only via `sbatch -p gpu` (login has no toolkit).

Queue: [es-cpu-cuda-queue.md](../../docs/developer/es-cpu-cuda-queue.md) +
[gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md).

Architecture: [gpw-perf-architecture.md](gpw-perf-architecture.md).
Baseline walls: [gpw-perf-baseline.md](gpw-perf-baseline.md).
CUDA RHF: [cuda-rhf-device-only.md](cuda-rhf-device-only.md).

### CUDA ES this wave

**Re-opened and partially landed.** Device s/p energy + device s/p forces on
`cuda::scf::RHF` (wave 6). **Forbidden:** hybrid / GAPW / PAW / meta-GGA
product; claiming CUDA wins on small-molecule correctness twins; classical MD
CUDA (separate OpenMM track).
