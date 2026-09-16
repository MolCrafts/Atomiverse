# `cuda-gpw-full-product`

## What

First public CUDA GPW method product: `ATV::cuda::gpw::GpwDft` matching the
CPU `GpwDft` public shape (members + `build` / `run` / `last_result` + forces
family). Composes landed `cuda/gpw/internal/*` stage units (device fine-grid
KS via `KsChainDevice::run_from_rho`) with host STVnl / cbg collocate /
integrate / DIIS / diagonalize.

## Twin

Same-settings CPU `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / GTH-PADE-q1),
Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry, closed-shell RKS, Padé LDA,
DIIS. Oracle: **CPU twin** (not CP2K for this correctness gate).

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |

## Perf

**perf_exempt** — small cell correctness-only first product (same class as
other CUDA ES footholds / PW dual-gates).

## Non-goals (this pin)

Multi-k, UKS, OT, multigrid product energy path, nonperiodic MT, vdW, XC
beyond PADE/PBE.
