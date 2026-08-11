# cuda-gpw-kpoint-h-residency

## What

Multi-k RKS DIIS **energy** dual-gate after mid-SCF **device H residual
residency**: assembled \(H_k\) from `KpointIntegrateDevice` is ingested into
`KpointDiisEigenDevice` via device RM→CM transpose and residual is formed
without full mid-SCF \(H_k\) D2H → re-H2D. Parents
`kpoint-h-assemble` + `kpoint-device-diis` remain on base.

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

`KpointDiisEigenDevice::ingest_H_k_device_rm` + `residual_from_device` in
`kpoint_diis_eigen.{cu,cuh}` wired in `run_cuda_kpoints_rks`
(`gpw_solver.cu`). Does **not** own P residency, STVnl, energy-trace, or
payload re-diag.

## GH200 evidence

| | |
| --- | --- |
| Job | **1136613** (n30 aarch64 GH200) |
| HEAD | `wf/cuda-gpw-kpoint-h-residency` rebased on GapCensus wave-17 `dev` @ `0f56cbca` (STVnl on base); dual-gate binary at feat tip (REFERENCE evidence amend only after green) |
| Unit | `test_cuda_gpw_kpoint_h_residency` + hold `test_cuda_gpw_kpoint_diis_eigen` / `test_cuda_gpw_kpoint_h_assemble` / `test_cuda_gpw_kpoints` **PASS** |
| Dual-gate | `E_cpu = E_cuda = -1.040581477089` Ha; `\|dE\| = 3.264e-14` Ha; nk=2; both conv 5 iters; parent `reg_cuda-gpw-kpoints` hold `\|dE\|=3.153e-14` |
| Log | `.claude/cuda-gpw-kpoint-h-residency-1136613.out` → `GPU_JOB_OK` |
| Prior tip | **1133843** \|dE\|=**3.442e-14** on pre-STVnl base (superseded by re-verify) |
