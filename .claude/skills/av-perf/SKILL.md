---
name: av-perf
description: Performance review for Atomiverse code.
argument-hint: "[path or module]"
user-invocable: true
---

# Performance Review

Delegate to the `av-optimizer` agent with the user's argument (or all source files if none given).

The agent will:

## CUDA backend (`src/cuda/`, `include/atomiverse/cuda/`)
1. Read CLAUDE.md for GPU data residency and lifecycle rules
2. Scan hot-path code for anti-patterns (D2H in launch, malloc in launch, etc.)
3. Check grid/block sizing, memory access patterns, atomic contention

## CPU backend (`src/cpu/`, `include/atomiverse/cpu/`)
1. Check xtensor usage patterns (unnecessary copies, missing move semantics, redundant evaluations)
2. Look for cache-unfriendly access patterns in SoA data
3. Identify vectorization blockers and missing SIMD-friendly idioms
4. Check for unnecessary allocations in hot loops
5. **ERI performance baseline:** benchmark `BM_GBuild_Block_Aspirin` (shell-block STO-3G) vs `BM_GBuild_AO_Aspirin` (AO-level) in `bench_native`. Target: Fock build wall time ≤ PySCF direct SCF (~950 ms for aspirin STO-3G; Phase-3 native is ~250-500 ms). `BM_GBuild_Block_Aspirin_ccPVDZ` covers the d-polarized (6-31G*) Fock build — exercises the Rys d-shell ERI path (~1.7 s, 211 AOs; no PySCF cc-pVDZ baseline is documented in the spec).
6. **Build-flag check:** confirm `-march=native` is set (language-guarded generator expression). `XTENSOR_USE_XSIMD` must **NOT** be set globally — it breaks `reshape_view` in `xc_grid_engine.h`; xsimd 11.2.0 is linked directly and `#include`d only in hot `.cpp` files.
7. **ERI vectorization blockers:** the `(ss|ss)` kernel `eri_blk_ss_ss` is the only hand-xsimd'd ERI path (Boys-SIMD ket loop); all l≥1 quartets go through `rys_eri_block`, which relies on `-march=native -O3` auto-vectorizing the 2D-VRR recurrences. When profiling d-shell SCF, time `build_g_rhf_direct_block` and check the Rys roots generator (`rys_roots.cpp`) and the `g0_2e_2d` recurrence are not the bottleneck.

## Cross-cutting
1. Review transfer functions (`to_cuda`, `to_cpu`) for redundant copies or missing async overlap
2. Check build-time vs launch-time work partitioning (expensive setup should be in build, not launch)
3. Report findings with impact estimates in `[SEVERITY] file:line` format
