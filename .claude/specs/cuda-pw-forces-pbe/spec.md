---
slug: cuda-pw-forces-pbe
status: in_progress
created: 2026-08-10
domain: cuda / planewave
queue: es-cpu-cuda
method: cuda
depends_on: [cuda-pw-forces, cuda-pw-pbe-device]
---

# Spec: `cuda-pw-forces-pbe` — PBE dual-gate for analytic ionic forces

## Lever (research)

| Item | Value |
| --- | --- |
| **Gap** | `cuda-pw-forces` dual-gates **LSDA only**; device PBE energy is green (`cuda-pw-pbe-device`) but force dual-gate under PBE is open. |
| **Lever** | Force-path SCF + `pw_forces_analytic_impl` already construct `DeviceDensity(..., method.xc)` and rebuild V_eff via `den.exchange_correlation()` (PBE spectral ∇ρ / −∇·flux). **No energy-path change.** Deliver dual-gate regression + GH200 evidence. |
| **Files** | `regressions/pw/cuda-pw-forces-pbe/` (primary); CMake + GPU job; unit: public API accepts PBE without spin/smear reject. |
| **Metric** | \(\max_{i\alpha}\|F^{\mathrm{cuda}}_{i\alpha}-F^{\mathrm{cpu}}_{i\alpha}\|\) [eV/Å] |
| **Oracle** | Same-settings `cpu::pw::pw_forces_analytic` with `XCType::PBE` (not CP2K). Optional class: CPU FD already covers analytic↔FD. |

## System / settings

Stretched Si dimer (same geometry as `cuda-pw-forces`), `GTH-PBE-q4`,
`ecut_ry=8`, Γ (`k_mesh=1`), `tol=1e-6`, `max_iter=40`, **`xc=XCType::PBE`**.

## Acceptance

1. GH200: unit `test_cuda_pw_force_local` green; `reg_cuda-pw-forces-pbe --device=both` green.
2. \(\max|F_{\mathrm{cpu}}| > 0.05\) eV/Å (non-vacuous).
3. \(\max|\Delta F| < 5\times 10^{-3}\) eV/Å (design-time; re-pin from log if residual is noise).
4. Energy path unchanged; **not** stress / spin / smear / PBE-family variants.
5. **perf_exempt** — small-cell correctness foothold (same class as LSDA forces).

## Non-goals

Stress product, collinear spin forces, metallic smear, PBEsol/revPBE/RPBE device kernels.
