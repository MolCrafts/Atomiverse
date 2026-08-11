# `cuda-gpw-multigrid-device-ks`

## What

Iron-law restore of **device monom multi-level KS** mid-SCF for Si-class Γ RKS
PADE ngrids=4: pack→`KsMultiGridChainDevice`→scatter with collint **group
max\|w\| + group Ltot** sphere windows. Production default (no host collint
mid-path DH).

## Twin

Same as `cuda-gpw-multigrid-si-energy-path` (CPU diamond Si Γ PADE ngrids=4).

## Gates

| Gate | Tol | Status |
| --- | ---: | --- |
| Energy \|dE\| | **5×10⁻⁵ Ha** | pending GH200 |
| Soft-tol / pin-unsafe eps fit | **forbidden** | held |
| Mid-path scientific matrix DH | **zero** | device default |

## Lever

1. Group max\|w\| → `DeviceProduct.mag` (prior wave).
2. **Group max Ltot → `DeviceProduct.L_win`** (this wave) — collint
   `group_angular_max` twin; monom continuous spheres no longer under-size
   s-monoms in mixed p/d geometry groups.
3. Device MG production default; `ATV_GPW_CUDA_HOST_MG=1` host collint debug.

Prior pin-unsafe coarse eps brackets (0.005–0.02, best **7.6e-5** on **1184281**)
abandoned.

## Skills

choose-launch-configuration, optimize-global-memory-access,
handle-boundary-conditions, av-perf (group-task launch; sphere window; residency).

## Perf

**perf_exempt**.

## Evidence

GH200 job id filled after dual-gate run.
