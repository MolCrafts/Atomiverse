# cuda-gpw-kpoint-h-assemble

## What

Multi-k RKS DIIS **energy** dual-gate after device mid-SCF \(H_k\) assemble
(`KpointIntegrateDevice::assemble_H` with resident \(H0_k = T_k+V_{\mathrm{nl}}^k\))
on the public `cuda::gpw::GpwDft` multi-k path. Mid-SCF no longer
`download_Vmat` + host `T+Vnl+Vg` every iter; D2H is assembled \(H_k\) for
residual host KDiis/diag.

## System

- H₂, cell 12 Bohr cubic, DZV-GTH-PADE, GTH-PADE-q1
- `k_mesh = {2,1,1}`, `k_time_reversal = false` (nk=2)
- PADE LDA, single-grid, cutoff 80 Ry, DIIS depth 6, tol 1e-8

## Oracle

Same-settings CPU `cpu::gpw::GpwDft` multi-k energy (not CP2K; not self-golden).

## Gate

- `|E_cuda − E_cpu| ≤ 5e-6` Ha
- both SCF converged; `n_kpts ≥ 2`
- **perf_exempt** — multi-k device-spine foothold (not fair-wall claim)

## GH200 evidence

| | |
| --- | --- |
| Job | **1132528** (n98 aarch64 GH200) |
| HEAD | feat tip `wf/cuda-gpw-kpoint-h-assemble` (GH200 ran pre-commit; log 1132528) |
| Unit | `test_cuda_gpw_kpoint_h_assemble` + hold `test_cuda_gpw_kpoint_integrate` / `test_cuda_gpw_kpoints` **PASS** |
| Dual-gate | `E_cpu = E_cuda = -1.040581477089` Ha; `\|dE\| = 3.397e-14` Ha; nk=2; both conv 5 iters; parent `reg_cuda-gpw-kpoints` hold `\|dE\|=3.331e-14` |
| Log | `.claude/cuda-gpw-kpoint-h-assemble-1132528.out` → `GPU_JOB_OK` |

## Lever

`src/cuda/gpw/internal/kpoint_integrate.cu` (`upload_H0` / `assemble_H` /
`download_H`) + multi-k branch of `src/cuda/gpw/gpw_solver.cu`
(`run_cuda_kpoints_rks`). Does **not** own KDiis, STVnl, or \(P_k\) residency.
