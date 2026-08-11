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
| Fine/coarse eps | fine `kProductMassMargin`; coarse collint `kProductWindowEps=0.02` |
| Metric | GH200 \|E_cuda − E_cpu\| (Ha); mid-path scientific matrix DH count = 0 |

**Root cause:** monom multi-level used **per-monom Ltot** for sphere radius while
collint uses `group_angular_max` Ltot; coarse monom also needs collint-class
window eps. Fix: group `L_win` + group max\|w\|; fine mass-margin / coarse 0.02;
device pack→`KsMultiGridChainDevice`→scatter as production default.

## Acceptance

| Gate | Tol | Result |
| --- | ---: | --- |
| Si MG dual-gate \|dE\| | ≤ **5×10⁻⁵** Ha | **PASS** GH200 **1186912** \|dE\|=**8.652e-06** |
| Both SCF converged | required | met |
| Mid-path scientific matrix DH | **zero** on claimed device path | met (device default) |
| Skills named | choose-launch-configuration, optimize-global-memory-access, handle-boundary-conditions, av-perf | met |
| Soft-tol / pin-unsafe eps | **forbidden** | held |

**acceptance_met=true**. **perf_exempt**.

## Non-goals

MG forces/stress, multi-k MG, UKS×MG, MG×PBE, soft-tol, dual-gate invent around
host collint.

## Regression

`regressions/gpw/cuda-gpw-multigrid-device-ks/` — dual-gate Si ngrids=4; asserts
device path (HOST_MG unset).
