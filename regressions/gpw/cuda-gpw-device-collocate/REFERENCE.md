# `cuda-gpw-device-collocate`

## What

Depth on public `ATV::cuda::gpw::GpwDft`: mid-SCF density collocate and V_eff
integrate use landed device `product_collocate` / `product_integrate` (via
`KsChainDevice::run`), not host `collocate_density` / `integrate_potential`.
Geometry-fixed products from `build_products` (ngrids=1); P enters as
`DeviceProduct.weight`. Host STVnl / DIIS / diagonalize / acc→V scatter remain.

## Twin

Same-settings CPU `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / GTH-PADE-q1),
Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry, closed-shell RKS, Padé LDA,
DIIS. Oracle: **CPU twin**.

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |

## Perf

**perf_exempt** — small cell correctness-only product depth (compose stage
units into the public method).

## Non-goals

Multigrid product energy path, UKS, multi-k, OT, device forces, non-default
stream.

## GH200 evidence

Job **1122198** (partition gpu, module `GPU/buildenv-gcccuda/2026.03-cu13.0`):

| Quantity | Value |
| --- | ---: |
| \|E_cuda − E_cpu\| | **1.617×10⁻⁷ Ha** |
| max\|ΔF\| | **2.360×10⁻⁶ Ha/Bohr** |
| CPU iters / CUDA iters | 5 / 9 |
| Units | test_cuda_gpw_full_product + product collocate/integrate **PASS** |

Tols: energy 5×10⁻⁶ Ha, forces 5×10⁻⁵ Ha/Bohr. **perf_exempt** small cell.
