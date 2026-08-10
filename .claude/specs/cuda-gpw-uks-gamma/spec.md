# cuda-gpw-uks-gamma

## Lever

| | |
| --- | --- |
| **Track** | cuda |
| **Files** | `include/atomiverse/cuda/gpw/gpw_solver.cuh`, `src/cuda/gpw/gpw_solver.cu`, `include/atomiverse/cuda/gpw/internal/ks_chain.cuh`, `src/cuda/gpw/internal/ks_chain.cu`, unit + `regressions/gpw/cuda-gpw-uks-gamma/` |
| **Function** | Lift collinear UKS reject on public `cuda::gpw::GpwDft`; dual-spin mid-SCF product path via `KsChainDevice::run_uks` (collocate ρ_α/ρ_β → Poisson(total) → Padé LSD → dual V_eff → dual integrate) + host dual DIIS/eigen |
| **Metric** | \|E_cuda − E_cpu\| dual-gate (same-settings CPU twin); both SCF converged |

## Acceptance

1. `n_alpha`/`n_beta` accepted on product (Γ, DIIS, `ngrids==1`, periodic, fixed occ).
2. First-cut XC for UKS: **PADE only** (PBE-UKS → `cuda-gpw-uks-pbe-gate`).
3. Device mid-SCF remains product collocate/KS/integrate (no host cbg for H).
4. Unit + regression dual-gate vs CPU `GpwDft` high-spin H₂ (Γ UKS PADE class; O₂-class twin shape).
5. GH200 log evidence; RKS path unchanged; OT/MG/multi-k/MT still fail-closed.
6. Forces free-fn not owned (energy slug); may stay CPU-backed if grids present.
7. **perf_exempt** small cell.

## Non-goals

UKS forces, UKS-PBE, multi-k UKS, multigrid UKS, OT UKS, device DIIS/eigen.
