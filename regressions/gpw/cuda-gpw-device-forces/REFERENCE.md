# `cuda-gpw-device-forces`

## What

Device grid force product on public `ATV::cuda::gpw::GpwDft` free-function
forces: host AO Pulay (W·dS, P·dT, P·dVnl) + device V_eff grid term, ionic
HF, and local-PP HF. Replaces the previous full CPU `gpw_forces_from_scf`
call. Mid-SCF collocate/integrate remains device product (prior slug).

## Twin

Same-settings CPU `cpu::gpw::GpwDft` + `gpw_forces_analytic` on H₂
(DZV-GTH-PADE / GTH-PADE-q1), Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry,
closed-shell RKS, Padé LDA, DIIS. Oracle: **CPU twin**.

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |

## Perf

**perf_exempt** — small cell correctness-only product depth (compose device
grid HF into free-fn forces).

## Non-goals

UKS / multi-k / multigrid / stress / PBE force dual-gate / dispersion.

## GH200 evidence

Post-rebase job **1126434** (partition gpu, n74 GH200, module
`GPU/buildenv-gcccuda/2026.03-cu13.0`, sacct **COMPLETED 0:0** ~6m13s;
prior tip gate **1125926** ~7m47s same class):

| Quantity | Value |
| --- | ---: |
| \|E_cuda − E_cpu\| | **1.617×10⁻⁷ Ha** |
| max\|ΔF\| | **2.321×10⁻⁶ Ha/Bohr** |
| CPU iters / CUDA iters | 6 / 9 |
| Units | `test_cuda_gpw_forces` + `test_cuda_gpw_full_product` **2/2 PASS** |
| Exit | **GPU_JOB_OK** |

Tols: energy 5×10⁻⁶ Ha, forces 5×10⁻⁵ Ha/Bohr. **perf_exempt** small cell.
