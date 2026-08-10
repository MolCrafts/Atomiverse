# cuda-gpw-kpoints

**Track:** method=cuda  
**Branch:** `wf/cuda-gpw-kpoints`  
**Oracle:** same-settings CPU `GpwDft` multi-k RKS DIIS (class of `reg_gpw-kpoints-si`)

## Lever

| | |
| --- | --- |
| **File** | `src/cuda/gpw/gpw_solver.cu` (`GpwDft::run` multi-k branch) + `include/atomiverse/cuda/gpw/gpw_solver.cuh` |
| **Function** | Multi-k closed-shell RKS DIIS energy: host phased \(S_k/T_k/V_{\mathrm{nl}}^k\) + \(\chi^k\) collocate/integrate (`cpu::gpw::internal` kpoints); device Poisson+XC via `KsChainDevice::run_from_rho` |
| **Metric** | CUDA↔CPU \(\lvert\Delta E\rvert\) dual-gate [Ha]; GH200 log evidence |

## Scope

- Lift `k_mesh=={1,1,1}` / empty `k_points` reject for **multi-k RKS DIIS** (PADE or PBE, single-grid, fixed occupations, periodic, `local_pp_on_grid=true`).
- Explicit `k_points` + weights + `k_time_reversal` members mirror CPU product surface.
- **Not** this slug: multi-k forces (`cuda-gpw-kpoint-forces`), multi-k stress, multi-k UKS, multi-k multigrid, OT, metallic smear, device \(\chi^k\)/STVnl.

## Acceptance

1. `k_mesh` ≠ Γ or non-empty `k_points` runs multi-k RKS DIIS without throw (supported slice).
2. Fail-closed: multi-k + UKS; multi-k + `ngrids>1`; multi-k + OT; MT / non-PADE-PBE still fail.
3. Unit: fail-closed joints + small H₂ mesh dual-gate energy (device required).
4. Regression `reg_cuda-gpw-kpoints`: H₂ `k_mesh={2,1,1}` PADE energy dual-gate vs CPU; stated tol.
5. GH200 green log; **perf_exempt** (small-molecule multi-k foothold; not a fair twin wall cut).

## Dual-gate tol

| Quantity | Tol |
| --- | ---: |
| \(\lvert E_{\mathrm{cuda}}-E_{\mathrm{cpu}}\rvert\) | \(5\times10^{-6}\) Ha (H₂ pure-s multi-k) |
