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
| Energy \|dE\| | **5×10⁻⁵ Ha** | **PASS** — GH200 **1186912** \|dE\|=**8.652e-06** |
| Soft-tol / pin-unsafe eps fit | **forbidden** | held (coarse uses host collint `kProductWindowEps=0.02`) |
| Mid-path scientific matrix DH | **zero** | device default |

## Lever

1. Group max\|w\| → `DeviceProduct.mag`.
2. **Group max Ltot → `DeviceProduct.L_win`** — collint `group_angular_max` twin.
3. Fine mass-margin (product-si hold); coarse collint-class `eps·0.02`.
4. Device MG production default; `ATV_GPW_CUDA_HOST_MG=1` host collint debug.

## Skills

choose-launch-configuration, optimize-global-memory-access,
handle-boundary-conditions, av-perf (group-task launch; sphere window; residency).

## Evidence (GH200)

| Job | Config | \|dE\| | Note |
| --- | --- | ---: | --- |
| 1180025 | monom default (pre-slug) | 4.3e-4 | SG-like |
| 1184281 | coarse 0.02, no group L | 7.6e-5 | best pre-L |
| 1186579 | group L + mass-margin all | 4.29e-4 | SG-like |
| **1186912** | **group L + fine MM / coarse 0.02** | **8.65e-06** | **PASS** |

Holds on **1186912**: `reg_cuda-gpw-multigrid-run` \|dE\|=9.8e-09;
`reg_cuda-gpw-product-si-gate` \|dE\|=3.53e-05; unit collocate/integrate/ks_multigrid 3/3.

## Perf

**perf_exempt**.

## Status

**acceptance_met=true**
