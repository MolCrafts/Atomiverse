# cuda-gpw-uks-payload-device

## Twin

Same-settings CPU `GpwDft` high-spin H₂ Γ UKS PADE energy after final dual
payload C/ε move onto device (dual `AoDiisEigenDevice::eigen_from_device` of
assembled \(H_\alpha/H_\beta\) — no host dual `diagonalize_fock` re-diag).
Geometry/settings match `cuda-gpw-uks-device-diis` / `cuda-gpw-uks-gamma`.

## Oracle

`oracle: cpu-twin` — same-settings `cpu::gpw::GpwDft` on the dual-gate binary.

## Tolerances

| Quantity | Tol |
| --- | ---: |
| \|E_cuda − E_cpu\| | ≤ 5×10⁻⁶ Ha |
| max\|ε_α/β_cuda − ε_α/β_cpu\| | ≤ 5×10⁻⁵ Ha |

**perf_exempt** — small-cell architectural residency (payload boundary), not a
fair-twin wall measurement.

## Lever

`gpw_solver.cu` final UKS block + dual `AoDiisEigenDevice::eigen_from_device` —
D2D H → device dsyevd per spin → boundary D2H C/ε/H_ks α/β; DIIS P kept for
energy density; ⟨S²⟩ from boundary C only.

## Skills applied

- **av-perf** — residency: boundary-only D2H of dual C/ε/H_ks; mid-path D2D H
- **write-numerically-stable-kernel** + **optimize-global-memory-access** —
  reuse existing cuSOLVER `dsyevd` / cublas gemm path in
  `diagonalize_and_density_`; no new ad-hoc kernels

## Not owned

UKS free-fn F/σ, UKS-PBE, multi-k UKS payload (`kpoint-payload-device`),
cold-start densify (`device-cold-start`).

## Evidence

| Job | Node | Result |
| --- | --- | --- |
| GH200 **1186414** | n514 | unit `test_cuda_gpw_diis_eigen` PASS; dual-gate \|dE\|=**4.163e-09** Ha max\|deps\|=**1.481e-06** Ha (E_cpu −0.508766760868; E_cuda −0.508766756705; iters 8/6; ⟨S²⟩=2); smoke `uks-device-diis` same \|dE\|; `GPU_JOB_OK`; `BUILDS_WIPED_OK` |

Skills applied: **av-perf** (residency — boundary-only D2H of dual C/ε/H_ks; mid-path D2D H); **write-numerically-stable-kernel** + **optimize-global-memory-access** (reuse existing cuSOLVER `dsyevd` / cublas gemm path in `diagonalize_and_density_`; no new ad-hoc kernels).

Mid-path scientific matrix DH in diff: **zero** (payload D2H only after
convergence).
