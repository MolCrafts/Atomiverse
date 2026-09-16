# cuda-gpw-kpoint-device-diis — multi-k device Pulay DIIS dual-gate

## What

Public `cuda::gpw::GpwDft` multi-k closed-shell RKS DIIS mid-SCF with
**device** `KpointDiisEigenDevice` (complex residual, k-summed Pulay,
cuSOLVER `zheevd`, density) vs same-settings CPU `GpwDft` (host `KDiis` +
`diagonalize_k`). H₂, DZV-GTH-PADE / GTH-PADE-q1, cell 12 Bohr, CUTOFF 80 Ry,
`k_mesh={2,1,1}` with **`k_time_reversal=false`** (nk=2), single-grid,
`local_pp_on_grid=true`.

Oracle is **same-settings CPU**, not CP2K. Absolute energy is not pinned —
only CUDA↔CPU parity.

## Dual-gate

| Quantity | Tol |
| --- | ---: |
| \(\lvert E_{\mathrm{cuda}}-E_{\mathrm{cpu}}\rvert\) | \(5\times10^{-6}\) Ha |

## Perf

**perf_exempt** — multi-k device spine foothold (host KDiis/eigen → device);
not a fair twin wall cut vs CP2K.

## Lever

`run_cuda_kpoints_rks` mid-SCF: replace host `KDiis::step` + `diagonalize_k` +
`density_from_fixed_k` with `internal::KpointDiisEigenDevice`. Residual host
H_k assemble after device integrate and host \(P_k\) H2D into collocate are
deeper slugs (`kpoint-h-assemble`, `kpoint-p-residency`).
