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
| GH200 *(pending)* | | dual-gate after impl |

Mid-path scientific matrix DH in diff: **zero** (payload D2H only after
convergence).
