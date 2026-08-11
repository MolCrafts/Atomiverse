# cuda-gpw-kpoint-device-stvnl

## What

Multi-k RKS DIIS **energy** dual-gate after device phased STVnl setup
(`KpointStvnlDevice` for \(S_k/T_k\); \(V_{\mathrm{nl}}^k=0\) on H GTH-PADE-q1)
on the public `cuda::gpw::GpwDft` multi-k path. Product no longer calls host
`build_overlap_kinetic_k` for multi-k setup.

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

## Lever

`src/cuda/gpw/internal/kpoint_stvnl.cu` (`KpointStvnlDevice::compute`) +
multi-k branch of `src/cuda/gpw/gpw_solver.cu` (`run_cuda_kpoints_rks`).
Does **not** own KDiis, mid-SCF H assemble, \(P_k\) residency, or payload.

## GH200 evidence

| | |
| --- | --- |
| Job | **1133851** (n40 aarch64 GH200) |
| HEAD | feat tip `wf/cuda-gpw-kpoint-device-stvnl` |
| Unit | `test_cuda_gpw_kpoint_stvnl` + hold `test_cuda_gpw_kpoint_h_assemble` / `test_cuda_gpw_kpoints` **PASS** |
| Dual-gate | `E_cpu = E_cuda = -1.040581477089` Ha; `\|dE\| = 3.331e-14` Ha; nk=2; both conv 5 iters; parent `reg_cuda-gpw-kpoints` hold `\|dE\|=3.331e-14` |
| Log | `cuda-gpw-kpoint-device-stvnl-1133851.out` → `GPU_JOB_OK` |
