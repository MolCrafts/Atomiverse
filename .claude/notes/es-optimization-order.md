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
2. `p1-ks-grid-build` — **next** — one internal `KsGridBuild` (ρ→Poisson→XC→V_eff)
   for Γ/k × RKS/UKS DIIS+OT on top of workspace.

**Out of this wave:** Gaussian/PW cross-method DIIS·LA extraction; CUDA ES
coverage expansion; collocation rewrite (`p2-cxyz-reflection`).

**Serial with solver structure:** `p1-ot-uks-k` and other `gpw_solver.cpp`
control-flow edits should not fight KsGridBuild ownership of that file.
`multigrid.cpp` SIMD rows may run in parallel (different main battlefield).

Architecture detail: [gpw-perf-architecture.md](gpw-perf-architecture.md).
Queue: [gpw-perf-queue.md](../../docs/developer/gpw-perf-queue.md).

### CUDA ES this wave

**Freeze new capability** (methods, bases, shell coverage). Keep existing RHF
device-only foothold and its regressions green. Classical MD CUDA is
unaffected. Re-open a CUDA ES slice only when the CLAUDE open-gate bullets
hold for that method.
