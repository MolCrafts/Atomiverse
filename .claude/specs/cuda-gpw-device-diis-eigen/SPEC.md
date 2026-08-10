# cuda-gpw-device-diis-eigen

## Lever

**File / function:** `src/cuda/gpw/gpw_solver.cu` RKS mid-SCF loop
(`GpwDft::run` closed-shell branch) + new
`cuda/gpw/internal/ao_diis_eigen.{cuh,cu}` (`AoDiisEigenDevice`).

**Metric:** every RKS DIIS SCF iteration moves Pulay DIIS + Löwdin
orthogonal AO eigen + density rebuild off host
(`cpu::gpw::internal::GpwDiis::step`, `diagonalize_fock_shifted`,
`build_density_matrix`) onto device (cuBLAS commutator/history + host-tiny
Pulay `(m+1)` solve + cuSOLVER `dsyevd`). Host residual that remains:
product weight pack + integrate-acc scatter → H (deeper slug), STVnl build.

## Acceptance

| Gate | Criterion |
| --- | --- |
| Unit | `test_cuda_gpw_diis_eigen` — device DIIS error / eigen / density vs host on fixed H,P,S,X; max abs tol ≤ 1e-10 (error/P) / 1e-9 (eig) |
| Dual-gate | `reg_cuda-gpw-device-diis-eigen` — H₂ Γ RKS PADE energy CUDA↔CPU ≤ 5e-6 Ha (same as full-product class) |
| Fail-closed | OT / multi-k / MT / non-PADE-PBE paths unchanged |
| Evidence | GH200 sbatch log (ctest unit + run_reg) |
| Perf | **perf_exempt** — small-cell foothold; architectural residual removal, not fair-twin wall |

## Out of scope

- Device P→product-weight pack / device scatter-acc (next residual after this tip)
- UKS device DIIS (host dual GpwDiis retained; twin is RKS)
- Device AO Pulay forces / STVnl
