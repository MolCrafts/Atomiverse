# cuda-gpw-uks-device-diis

method=cuda | twin: same-settings CPU Γ UKS PADE energy (H₂/O₂ class)

## Lever
`src/cuda/gpw/gpw_solver.cu` UKS branch still runs **host dual**
`cpu::gpw::internal::GpwDiis` + host eigen after RKS `AoDiisEigenDevice`
landed. Compose dual-spin device Pulay DIIS + cuSOLVER eigen on product UKS
path (pattern from `ao_diis_eigen.{cu,cuh}`).

## Acceptance
- GH200 dual-gate energy vs CPU Γ UKS PADE; stated tol; **perf_exempt** small cell
- UKS free-fn forces/stress/PBE not owned by this slug
- device-only kernels in new DIIS path; no soft tols
