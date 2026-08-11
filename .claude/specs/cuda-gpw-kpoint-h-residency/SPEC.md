# cuda-gpw-kpoint-h-residency

## Lever

| | |
| --- | --- |
| Track | **cuda** |
| Files | `include/.../kpoint_diis_eigen.cuh`, `src/.../kpoint_diis_eigen.cu`, `src/cuda/gpw/gpw_solver.cu` |
| Functions | `KpointDiisEigenDevice::ingest_H_k_device_rm` + `residual_from_device`; `run_cuda_kpoints_rks` mid-SCF assemble→DIIS |
| Metric | Drop mid-SCF full \(H_k\) D2H → host residual H2D; energy dual-gate \|dE\| |

## Problem

After landed `kpoint-h-assemble` + `kpoint-device-diis`, mid-SCF still:

1. `KpointIntegrateDevice::assemble_H` (single-slot `d_H_`)
2. `download_H` → host `H_k`
3. `KpointDiisEigenDevice::residual(H_ptrs)` re-H2Ds RM→CM into per-k `d_H_`

Fuse/DIIS ≠ device-resident H product.

## Design

1. `ingest_H_k_device_rm(ik, d_H_rm)` — device transpose row-major integrate layout → column-major DIIS `d_H_[ik]` (D2D, no host).
2. `residual_from_device()` — `form_error_k_` over all k without host H upload.
3. Product loop: for each k after `integrate`+`assemble_H`, ingest into DIIS; residual uses device path only.
4. Keep host `residual(H_hosts)` for unit oracles; `download_H` remains for payload / unit assemble tests (payload-device is a child slug).

## Acceptance

- Unit: device residual after ingest matches host residual(H) within 1e-10 on toy + H₂-class assemble chain.
- Regression dual-gate H₂ k=2×1×1 TR-off RKS PADE: \|E_cuda−E_cpu\| ≤ 5e-6 Ha; both converged; nk≥2.
- GH200 log evidence required; **perf_exempt** (small-cell foothold).
- Does **not** own P residency / STVnl / energy-trace / payload.

## Oracle

Same-settings CPU `cpu::gpw::GpwDft` multi-k energy.
