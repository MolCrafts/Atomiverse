# `cuda-gpw-multigrid-device-ks`

## What

Staff iron-law restore of **device monom multi-level KS** mid-SCF for Si-class
Γ RKS PADE ngrids=4. Lands group-max mag + per-level sphere eps; dual-gate
energy residual remains above tol.

## Twin

Same as `cuda-gpw-multigrid-si-energy-path` (CPU diamond Si Γ PADE ngrids=4).

## Gates

| Gate | Tol | Status |
| --- | ---: | --- |
| Energy \|dE\| | **5×10⁻⁵ Ha** | **open** — best **7.6e-5** (GH200 **1184281**) |
| Soft-tol / pin-unsafe eps fit | **forbidden** | held |

## Lever tried

1. Group max\|w\| → `DeviceProduct.mag` (collint group window mag).
2. Per-level sphere eps: fine mass-margin (product-si hold); coarse collint-class.
3. Coarse eps factor scan 0.005–0.02: residual discontinuous; no pin-safe ≤5e-5.
4. Production default remains **host collint**. Device monom: `ATV_GPW_CUDA_DEVICE_MG=1`.

## Evidence (GH200)

| Job | Coarse eps | \|dE\| | Note |
| --- | ---: | ---: | --- |
| 1183276 | 0.02 all levels | 7.67e-5 | product-si broken |
| 1183591 | mass-margin + group-max | 4.29e-4 | product-si hold; ≈ monom class |
| **1184281** | fine mass-margin, coarse 0.02 | **7.60e-5** | product-si hold; **best** |
| 1184590 | coarse 0.005 | 4.58e-4 | under-correct |
| 1185043 | coarse 0.015 | 4.00e-4 | under-correct |

## Next lever

Device collint (cxyz / cmax radius) — not pin-unsafe eps fitting.

## Perf

**perf_exempt**.

## Status

**acceptance_met=false** — residual \|dE\| > 5e-5; infrastructure landed.
