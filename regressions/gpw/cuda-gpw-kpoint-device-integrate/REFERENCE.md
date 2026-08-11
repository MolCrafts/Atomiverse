# cuda-gpw-kpoint-device-integrate

## What

Multi-k RKS DIIS **energy** dual-gate after device χ^k V→H_k integrate
(`KpointIntegrateDevice`) on the public `cuda::gpw::GpwDft` multi-k path.
Mid-SCF keeps V_eff device-resident (no full-grid D2H + host
`integrate_potential_k` for H_k assembly).

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

`src/cuda/gpw/internal/kpoint_integrate.cu` + multi-k branch of
`src/cuda/gpw/gpw_solver.cu` (`run_cuda_kpoints_rks`).
