# cuda-gpw-uks-device-diis

method=cuda | twin: same-settings CPU Γ UKS PADE energy (H₂/O₂ class)

## Lever

`src/cuda/gpw/gpw_solver.cu` UKS branch still runs **host dual**
`cpu::gpw::internal::GpwDiis` + host `diagonalize_fock_shifted` after RKS
`AoDiisEigenDevice` landed. Compose **dual** `internal::AoDiisEigenDevice`
(α + β) on the product UKS mid-SCF path:

| Channel | `n_occ` | `occ_factor` |
| --- | ---: | ---: |
| α | `n_alpha` | 1.0 |
| β | `n_beta` | 1.0 |

Shared geometry `S`/`X` upload once; residual = max(|e_α|, |e_β|); **no**
ascent DIIS reset (matches host UKS / O₂ thrash policy). Not owned: free-fn
forces/stress, UKS-PBE, multi-k UKS, weight-pack.

## Acceptance

- GH200 dual-gate energy vs same-settings CPU Γ UKS PADE (H₂ high-spin class)
- stated tol \|E_cuda−E_cpu\| ≤ 5×10⁻⁶ Ha (H₂ pure-s; same class as uks-gamma)
- unit: UKS channel `occ_factor=1` residual/eigen/density vs host oracle
- **perf_exempt** small-cell architectural lift (not fair-twin wall)
- device-only DIIS/eigen kernels on new path; no soft tols; no host GpwDiis
  in Γ UKS product loop
