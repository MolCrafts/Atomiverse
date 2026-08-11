# cuda-gpw-kpoint-h-assemble

## Lever

| | |
| --- | --- |
| **Track** | method=cuda |
| **Hot path** | `src/cuda/gpw/internal/kpoint_integrate.{cu,cuh}` + multi-k branch of `src/cuda/gpw/gpw_solver.cu` (`run_cuda_kpoints_rks`) |
| **What today** | After device `integrate` (V→Vmat), mid-SCF still `download_Vmat` + host `H_k = T_k + Vnl_k + Vg` every k every iter |
| **Change** | Upload `H0_k = T_k + Vnl_k` once per k; after integrate, fuse device `H_k = H0_k + Vmat`; mid-SCF D2Hs `H_k` (not Vmat + host add) |
| **Metric** | Dual-gate `\|E_cuda − E_cpu\|` (Ha); unit max`\|ΔH\|` vs host assemble |
| **Does not own** | KDiis (`kpoint-device-diis`), STVnl (`kpoint-device-stvnl`), \(P_k\) residency |

## Acceptance

- Unit: device `assemble_H` matches host `T+Vnl+V` to stated abs/rel tol on H₂-class χ^k + synthetic V
- Regression: public multi-k RKS DIIS energy dual-gate H₂ `k_mesh={2,1,1}` TR-off PADE; `\|dE\| ≤ 5e-6` Ha; both converged; `n_kpts ≥ 2`
- GH200 log evidence required for green claim
- **perf_exempt** — multi-k device-spine foothold (not fair-wall)

## Oracle

Same-settings CPU `cpu::gpw::GpwDft` multi-k energy (not CP2K; not self-golden).
