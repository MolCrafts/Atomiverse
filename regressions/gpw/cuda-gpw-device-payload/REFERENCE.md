# cuda-gpw-device-payload

## Twin

Same-settings CPU `GpwDft` H₂ Γ RKS PADE energy after final payload C/ε move
onto device (`AoDiisEigenDevice::eigen_from_device` of assembled H — no host
`diagonalize_fock` re-diag). Geometry/settings match `cuda-gpw-device-diis-eigen`.

## Oracle

`oracle: cpu-twin` — same-settings `cpu::gpw::GpwDft` on the dual-gate binary.

## Tolerances

| Quantity | Tol |
| --- | ---: |
| \|E_cuda − E_cpu\| | ≤ 5×10⁻⁶ Ha |
| max\|ε_cuda − ε_cpu\| | ≤ 5×10⁻⁵ Ha |

**perf_exempt** — small-cell architectural residency (payload boundary), not a
fair-twin wall measurement.

## Lever

`gpw_solver.cu` final RKS block + `AoDiisEigenDevice::eigen_from_device` —
D2D H → device dsyevd → boundary D2H C/ε/H_ks; DIIS P kept for energy density.

## Evidence

| Job | Node | Result |
| --- | --- | --- |
| GH200 **1178026** | n34 | tip green (pre-rebase): unit PASS; dual-gate \|dE\|=**1.252e-08** max\|deps\|=**1.722e-06** Ha |
| GH200 **1180932** | n552 | rebased on wave-4 `dev` `213d794d`: unit `test_cuda_gpw_diis_eigen` PASS; dual-gate \|dE\|=**1.252e-08** Ha max\|deps\|=**1.899e-06** Ha (E_cpu −1.040591553127; E_cuda −1.040591565649; iters 5/8); `BUILDS_WIPED_OK` |
| GH200 **1182615** | n555 | rebased on wave-5 `dev` `d3c6c5dc` (si-energy-path on base): unit `test_cuda_gpw_diis_eigen` PASS; dual-gate \|dE\|=**1.252e-08** Ha max\|deps\|=**1.888e-06** Ha (E_cpu −1.040591553127; E_cuda −1.040591565649; iters 5/8); `BUILDS_WIPED_OK` |

Skills applied: **av-perf** (residency — boundary-only D2H of C/ε/H_ks; mid-path D2D H); **write-numerically-stable-kernel** + **optimize-global-memory-access** (reuse existing cuSOLVER `dsyevd` / cublas gemm path in `diagonalize_and_density_`; no new ad-hoc kernels).

Mid-path scientific matrix DH in diff: **zero** (payload D2H only after convergence).
