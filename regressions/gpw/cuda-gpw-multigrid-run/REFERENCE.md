# `cuda-gpw-multigrid-run`

## What

Lift `ngrids!=1` reject on public `cuda::gpw::GpwDft` and wire landed
`KsMultiGridChainDevice` (spectral prolong/restrict + fine KS + per-level
product collocate/integrate) into product **energy**.

## Twin

Same-settings CPU `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / GTH-PADE-q1),
Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry, **ngrids=4**, **rel_cutoff=40**,
closed-shell RKS, Padé LDA, DIIS. Oracle: **CPU twin** (not CP2K).

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Both SCF converged | required |

## Perf

**perf_exempt** — small cell correctness-only multigrid product energy (not a
fair-wall twin vs CP2K).

## Non-goals

Multigrid forces, multi-k × MG, UKS × MG, PBE product dual-gate, device
DIIS/eigen.

## Evidence

| Job | Host | Result |
| --- | --- | --- |
| GH200 **1125894** | tip pre-rebase | unit + reg PASS; \|dE\|=**1.587e-07** Ha |
| GH200 **1126433** | tip rebased onto `dev` @ Assign wave 7 | `test_cuda_gpw_full_product` + `test_cuda_gpw_ks_multigrid` PASS; reg \|dE\|=**1.587e-07** Ha (tol 5e-6); `GPU_JOB_OK`; builds wiped |

Oracle: same-settings CPU twin (not CP2K). **perf_exempt** small cell.
