# Spec — `cpu-simd-2center-setup`

| Field | Value |
| --- | --- |
| slug | `cpu-simd-2center-setup` |
| method | cpu |
| branch | `wf/cpu-simd-2center-setup` |
| oracle | literature closed-form pure-s S/T (Helgaker et al. molecular-integral forms); dual-check fused path vs image sum of `prim_ST` |
| status | implement |

## Problem

GPW setup ST_k on **slab-class** cells (many live lattice images) still walks
pure s–s contracted pairs with a scalar monom-pair / image loop. The four fair
twins use tiny image windows (scalar residual secondary after Hermite
`p0-stvnl-setup-residual`); slab systems pay O(nimg · na · nb) without portable
xsimd. Queue row promoted from gpw-perf `p2-simd-2center-setup`.

## Lever

| | |
| --- | --- |
| Class | Portable **xsimd** analytic 2-center S/T (raw spans, scalar tail) |
| Files | `src/cpu/gpw/internal/one_electron.cpp`, `include/.../one_electron.h`; call sites fuse Γ S+T via `contracted_ST_periodic` in `gpw_solver.cpp` / forces / stress |
| Functions | `contracted_ST_ss_simd` (flattened monom pairs, fixed r²); `contracted_ST_ss_images_simd` + `pack_live_images_r2` (image-axis SIMD when nimg ≥ native batch width); `contracted_ST_periodic` public fused entry |
| Metric | Slab pure-s multi-image fragment wall vs scalar closed form (~1.2–2× target); four fair twins MAIN no regression |
| Non-goal | Arch intrinsics; global `XTENSOR_USE_XSIMD`; p/d Hermite SIMD; MG MAIN structural rewrite |

## Bit-exact plan

Default gate prefers bit-identical pins. This path uses `xsimd::exp` + lane
reduce → **not bit-exact**. Land with:

- unit / microbench tol **≤ 1e-13** relative vs closed form / prim sum;
- existing energy pins (MG MAIN \|ΔE\| ≤ pin budget) must hold;
- energy-path change ⇒ re-run FD `gpw-forces-water` if forces gate is in scope
  for the dual check (or confirm MAIN pin bit-stable enough that forces
  residual is unchanged within published FD tol).

## Acceptance

1. Unit `test_cpu_gpw_one_electron` covers multi-prim pure-s + multi-image
   fused `contracted_ST_periodic` vs closed form (tol 1e-13 class).
2. Regression `reg_cpu-simd-2center-setup` PASS — slab-class multi-image pure-s
   S/T pin vs literature closed form; fused ≡ separate image sum within tol.
3. Four fair twins not required to re-measure wall this slug (lever invisible
   on tiny image windows); no intentional pin soften.
4. SIMD only when work fills ≥1 native vector; scalar fallback otherwise.
5. Build dirs wiped after worker exit.

## Dual gate

| Gate | How |
| --- | --- |
| Correctness | unit + `reg_cpu-simd-2center-setup` literature closed form |
| Wall | microbench speedup printed; **perf_exempt** for fair-twin ratio (slab-class only; four twins use scalar fallback) — documented |

## Evidence

Fill after dual gate (unit PASS, reg PASS, optional MAIN quiet remeasure).
