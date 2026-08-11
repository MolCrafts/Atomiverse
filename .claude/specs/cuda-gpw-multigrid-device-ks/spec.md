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
| Group window mag + **group Ltot** | `product_weight_pack.cu` `group_max_mag_kernel` → `DeviceProduct.mag` / `L_win` |
| Sphere radius uses group L | `product_collocate.cu` / `product_integrate.cu` |
| Metric | GH200 \|E_cuda − E_cpu\| (Ha); mid-path scientific matrix DH count = 0 |

**Root cause (wave-6):** monom multi-level used **per-monom Ltot** for sphere
radius while collint uses `group_angular_max` Ltot. s-character monoms in mixed
p/d geometry groups under-sized vs collint → multi-level residual collapsed
toward SG (GH200 **1180025** \|dE\|=**4.3e-4**; pin-unsafe coarse eps brackets
best **7.6e-5** on **1184281**). Fix: lift monom `L_win` to collint group max
Ltot with group max\|w\|; restore device pack→`KsMultiGridChainDevice`→scatter
as production default. Host collint: `ATV_GPW_CUDA_HOST_MG=1` debug only.

## Acceptance

| Gate | Tol |
| --- | ---: |
| Si MG dual-gate \|dE\| | ≤ **5×10⁻⁵** Ha |
| Both SCF converged | required |
| Mid-path scientific matrix DH | **zero** on claimed device path |
| Skills named | choose-launch-configuration, optimize-global-memory-access, handle-boundary-conditions, av-perf |
| Soft-tol / pin-unsafe eps | **forbidden** |

**perf_exempt** — iron-law residency / correctness dual-gate (not fair-wall vs CP2K).

## Non-goals

MG forces/stress, multi-k MG, UKS×MG, MG×PBE, soft-tol, dual-gate invent around
host collint, full device cxyz collint port (follow-on if group L residual).

## Regression

`regressions/gpw/cuda-gpw-multigrid-device-ks/` — dual-gate Si ngrids=4; asserts
device path (HOST_MG unset).
