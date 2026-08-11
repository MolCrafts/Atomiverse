# cuda-gpw-kpoint-device-diis

## Track
method=cuda | loop=es-cpu-cuda | parent=dev@eec0b266

## Lever
**File / function:** `src/cuda/gpw/gpw_solver.cu` → `run_cuda_kpoints_rks` mid-SCF
Pulay + eigen; new `internal::KpointDiisEigenDevice`
(`kpoint_diis_eigen.{cuh,cu}`).

**Metric:** replace host `KDiis::step` + `diagonalize_k` +
`density_from_fixed_k` with device complex residual
`e^k = X†(HPS−SPH)X`, k-summed Pulay (shared B), cuSOLVER `zheevd`, and
`P = 2 C_occ C_occ†` per k.

**Does not own:** device H_k assemble (`kpoint-h-assemble`), \(P_k\) residency
for collocate (`kpoint-p-residency`), phased STVnl (`kpoint-device-stvnl`),
forces/stress/UKS×k/MG×k.

## Acceptance
1. Unit: device residual / eigen+density ≡ host `k_diis_error` /
   `diagonalize_k` / density on a small complex toy (tol ~1e-10).
2. Product dual-gate: H₂ `k_mesh={2,1,1}` TR-off RKS PADE energy
   `|E_cuda−E_cpu| ≤ 5e-6` Ha (`reg_cuda-gpw-kpoint-device-diis` + hold
   `test_cuda_gpw_kpoints`).
3. GH200 log evidence required; login has no toolkit.
4. **perf_exempt** — multi-k device spine foothold, not fair-twin wall.

## Oracle
`oracle: cpu-twin` — same-settings `cpu::gpw::GpwDft` (not CP2K live).
