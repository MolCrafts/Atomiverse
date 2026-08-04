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
5. ~~`p2-cxyz-collint` (geometry batch)~~ **landed** — `product_collint.{h,cpp}`
   (CP2K collint algorithm, BSD-3): cxyz + pol + monom integrate + geometry
   group batch. MG MAIN quiet **~27 s** (~30× CP2K 0.885 s); integrate
   ~0.75 s/iter (was ~4.3 s). Energy pins green.

**Active:** collint polish — reflection pairs, sphere-bounds cache, tighter
task locality (still ~30× vs CP2K). Fair wall law: re-measure CP2K on this host.

**Out of this wave:** Gaussian/PW cross-method DIIS·LA extraction; CUDA ES
coverage freeze still holds.

Architecture detail: [gpw-perf-architecture.md](gpw-perf-architecture.md).
Queue: [gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md).

### CUDA ES this wave

**Freeze new capability** (methods, bases, shell coverage). Keep existing RHF
device-only foothold and its regressions green. Classical MD CUDA is
unaffected. Re-open a CUDA ES slice only when the CLAUDE open-gate bullets
hold for that method.
