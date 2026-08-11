# `cuda-gpw-multigrid-device-ks`

## What

Restore **device monom multi-level KS** mid-SCF on public `cuda::gpw::GpwDft`
for Si-class Γ RKS PADE **ngrids=4** (pack→`KsMultiGridChainDevice`→scatter).
Kills host collint mid-path P D2H + V H2D left by `multigrid-si-energy-path`.

## Twin

| | |
| --- | --- |
| System | diamond Si bct 4-atom, a = 5.43 Å |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Settings | Γ, CUTOFF 40 Ry, **ngrids=4**, **rel_cutoff=20**, n_occ=8, PADE, DIIS |
| Oracle | **same-settings CPU** (not CP2K) |

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |
| Mid-path scientific matrix DH | **zero** (scalar residual OK) |

## Lever

1. Collint-faithful monom windows: group max\|w\| → `DeviceProduct.mag` +
   `eps_win = eps·0.02` (not mass-margin) in sphere collocate/integrate.
2. Default MG mid-SCF: device pack→`KsMultiGridChainDevice`→scatter
   (`ATV_GPW_CUDA_HOST_MG=1` only for host-collint debug).

Skills: choose-launch-configuration, optimize-global-memory-access,
handle-boundary-conditions, av-perf.

## Perf

**perf_exempt** — iron-law residency / correctness dual-gate (not fair-wall).

## Status

Pending GH200 dual-gate after feat land.
