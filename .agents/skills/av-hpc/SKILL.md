---
name: av-hpc
description: CPU HPC review and optimization -- OpenMP, MPI, SIMD, cache optimization for the CPU backend.
argument-hint: "[path or module]"
user-invocable: true
---

# CPU HPC Review & Optimization

Delegate to the `av-hpc` agent with the user's argument (or all CPU backend files if none given).

The agent operates in two modes:

## Review mode (default)
1. Discover CPU backend files (`src/cpu/`, `include/atomiverse/cpu/`)
2. Identify hot paths (force computation, neighbor list, integrators)
3. Audit parallelism: OpenMP usage, race conditions, reduction patterns
4. Audit vectorization: xtensor ops vs scalar loops, SoA access, SIMD blockers
5. Audit memory: allocation timing, cache friendliness, false sharing
6. Report findings in severity-sorted `[SEVERITY] file:line` format

## Implementation mode (when asked to optimize or implement)
1. Profile before optimizing
2. Add OpenMP / SIMD / MPI as appropriate:
   - **`-march=native` is required for CPU SCF code** — language-guarded
     generator expression so NVCC is not poisoned:
     `target_compile_options(atomiverse PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)`.
   - **Do NOT set `XTENSOR_USE_XSIMD` globally.** It activates SIMD-backed
     xtensor storage which breaks `reshape_view` in `xc_grid_engine.h`.
     Use **xsimd 11.2.0** (xsimd 13.x is incompatible with xtensor 0.25.0)
     *directly* via `#include <xsimd/xsimd.hpp>` in hot `.cpp` files.
   - Use `xsimd::batch<double>` for primitive loop vectorization;
     `xsimd::select` for branch-free logic; horizontal reduce via
     store-to-array + scalar sum (`xsimd::hadd` is not a free function
     in 11.2.0).
   - **Boys-SIMD pattern** (used in the `(ss|ss)` ERI kernel): guard the
     `T < 1e-12` case before `sqrt`/`erf` with `safe_T = select(small, 1, T)`,
     then `F0 = select(small, 1, 0.5*sqrt(PI/safe_T)*erf(sqrt(safe_T)))`.
   - **ERI `eri_block()` dispatch** (`native_integrals.cpp`): `(ss|ss)` →
     the xsimd ket-loop kernel `eri_blk_ss_ss`; every other quartet (s/p/d,
     l ≤ 2) → the native Rys-quadrature engine `rys_eri_block`
     (`rys_eri.cpp`), the libcint-equivalent 2D-VRR + HRR path that already
     handles d-shells (cc-pVDZ). Do not add an MD-BLAS angular-contraction
     path for l≥2 — Rys is the d-shell path. See `.Codex/specs/eri-hpc-opt.md`.
3. Validate numerical correctness against serial baseline (ERI: cross-check
   vs libcint via `tests/cpu/scf/test_rys_eri.cpp`, d-shells to 1e-7)
4. Report speedup measurements
