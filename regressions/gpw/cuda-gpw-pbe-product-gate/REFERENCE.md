# `cuda-gpw-pbe-product-gate`

## What

Public product dual-gate under **PBE-GGA**: `ATV::cuda::gpw::GpwDft` with
`xc = XcKind::PBE` vs same-settings CPU `GpwDft`. Composes the landed
`KsChainDevice` PBE stage (`cuda-gpw-ks-gga`) into the public method energy
path. Not a new kernel foothold.

## Twin

Same-settings CPU `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / **GTH-PBE-q1**),
Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry, closed-shell RKS, **PBE**,
DIIS. Oracle: **CPU twin** (not CP2K for this correctness gate).

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |

## Perf

**perf_exempt** — small cell correctness-only (same class as
`cuda-gpw-full-product` / other CUDA ES dual-gates).

## Non-goals (this pin)

UKS-PBE, multigrid×PBE, multi-k×PBE, H₂O production cell, OT, MT, vdW,
device DIIS/eigen residual, stress.

## GH200 evidence (2026-08-10)

Job **1127166** (`wf/cuda-gpw-pbe-product-gate`): unit
`test_cuda_gpw_full_product` PASS (1/1, 2.70 s); dual-gate
CPU E=−1.076079182475 Ha, CUDA E=−1.076079016645 Ha,
\|ΔE\|=**1.658e-07** Ha (tol 5e-6), max\|ΔF\|=**5.232e-07** Ha/Bohr
(tol 5e-5). **perf_exempt** small cell.
