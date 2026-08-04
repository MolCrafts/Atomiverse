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
5. ~~`p2-cxyz-collint` + lp-spec + sphere cache~~ **landed** — CP2K ortho
   collint + geometry pre-sort at build + process-wide sphere cache +
   `lp_z` 0..6 specialized collocate. MG MAIN **~2.6 s** (~**2.0×** CP2K
   quiet ~1.28 s; was ~42 s). Energy pins green. **≤2× gate met.**

**Active next:** STVnl_k setup (~1.3 s once) and residual kernel polish;
products-default-k re-gate when setup+collint both quiet.

**Out of this wave:** Gaussian/PW cross-method DIIS·LA extraction; CUDA ES
coverage freeze still holds.

Architecture detail: [gpw-perf-architecture.md](gpw-perf-architecture.md).
Queue: [gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md).

### CUDA ES this wave

**Freeze new capability** (methods, bases, shell coverage). Keep existing RHF
device-only foothold and its regressions green. Classical MD CUDA is
unaffected. Re-open a CUDA ES slice only when the CLAUDE open-gate bullets
hold for that method.
