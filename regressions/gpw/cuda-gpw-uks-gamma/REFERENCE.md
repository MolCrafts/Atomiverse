# `cuda-gpw-uks-gamma`

## What

Lift collinear UKS reject on public `ATV::cuda::gpw::GpwDft` — Γ DIIS UKS
**energy** (Padé LSD first cut). Device mid-SCF uses
`KsChainDevice::run_uks` (dual product collocate → Poisson(total) → Padé LSD
→ dual V_eff → dual integrate) with host dual DIIS/eigen.

## Twin

Same-settings CPU `cpu::gpw::GpwDft` on high-spin H₂ (DZV-GTH-PADE /
GTH-PADE-q1), Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry, `n_alpha=2`,
`n_beta=0`, Padé LSD, DIIS. Oracle: **CPU twin** (not CP2K; the CPU
`reg_gpw-uks-o2-gamma` MT O₂ twin is a different surface class — CUDA rejects
`nonperiodic`).

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Both SCF converged | required |
| CUDA `<S2>` | > 1.5 (high-spin identity) |
| Force dual-gate | **not** this slug (`cuda-gpw-uks-forces`) |

## Perf

**perf_exempt** — small cell correctness-only (same class as other CUDA ES
dual-gates).

## Non-goals

UKS forces/stress, UKS-PBE, multi-k UKS, multigrid UKS, OT UKS, MT, device
DIIS/eigen residual.

## GH200 evidence (re-verify 2026-08-10)

Job **1126430** (after rebase onto wave-7 Assign): unit
`test_cuda_gpw_full_product` PASS; dual-gate
CPU E=−0.508766760868 Ha, CUDA E=−0.508771520730 Ha,
\|ΔE\|=**4.760e-06** Ha (tol 5e-6), ⟨S²⟩=2.0 both. Prior **1125905** same
numbers on pre-rebase tip.
