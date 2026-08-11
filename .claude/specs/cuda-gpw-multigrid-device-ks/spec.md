# cuda-gpw-multigrid-device-ks

## Goal

Restore **device monom multi-level KS** mid-SCF for Γ RKS multigrid
(`pack → KsMultiGridChainDevice → scatter`) under **Si-class PBC**, killing the
host collint mid-path P D2H + V H2D introduced by `multigrid-si-energy-path`.

## Method

`method=cuda` (CUDA track).

## Twin / oracle

Same-settings CPU diamond Si bct 4-atom Γ RKS PADE **ngrids=4** rel_cutoff=20
(same class as `reg_cuda-gpw-multigrid-si-energy-path`). Oracle: **CPU twin**,
not CP2K.

## Lever (file + function + metric)

| Piece | Location |
| --- | --- |
| Default MG mid-SCF path | `src/cuda/gpw/gpw_solver.cu` `build_energy_device` |
| Sphere window radius | `product_collocate.cu` / `product_integrate.cu` |
| Group-max window mag | `product_weight_pack` + `DeviceProduct.mag` |
| Metric | GH200 \|E_cuda − E_cpu\| (Ha); mid-path scientific matrix DH count = 0 |

**Root cause:** host collint multi-level was a dual-gate escape after monom MG
under-corrected Si (GH200 **1180025** \|dE\|=**4.3e-4**; CUDA MG E ≈ SG). Monom
sphere windows used per-monom \|weight\| + mass-margin eps, not collint **group
max\|w\|** + `eps_win = eps·0.02` — multi-level residual collapsed toward SG.
Fix continuous monom multi-level windows to collint-faithful group mag + eps_win;
restore device pack→`KsMultiGridChainDevice`→scatter as default (no mid-path P/V
matrix DH). Optional `ATV_GPW_CUDA_HOST_MG=1` keeps host collint for debug only.

## Acceptance

| Gate | Tol |
| --- | ---: |
| Si MG dual-gate \|dE\| | ≤ **5×10⁻⁵** Ha (hold energy-path) |
| Both SCF converged | required |
| Mid-path scientific matrix DH | **zero** (scalar residual OK) |
| Skills named | choose-launch-configuration, optimize-global-memory-access, handle-boundary-conditions, av-perf |
| Soft-tol | **forbidden** |

**perf_exempt** — iron-law residency / correctness dual-gate (not fair-wall vs CP2K).

## Non-goals

MG forces/stress, multi-k MG, UKS×MG, MG×PBE, soft-tol, dual-gate invent around
host collint.

## Regression

`regressions/gpw/cuda-gpw-multigrid-device-ks/` — dual-gate Si ngrids=4; asserts
device path (no host collint env required).
