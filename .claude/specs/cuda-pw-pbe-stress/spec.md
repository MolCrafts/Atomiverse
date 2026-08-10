# cuda-pw-pbe-stress

## Goal

Dual-gate closed-shell **PBE** Nielsen–Martin cell stress on CUDA
(`cuda::pw::pw_stress` with `XCType::PBE`), twin of landed LSDA
`reg_cuda-pw-stress`.

## Lever

| Item | Value |
| --- | --- |
| file | `src/cuda/planewave/internal/pw_stress.cu` (`pw_stress_analytic_impl`, already takes `XCType`) |
| public | `cuda::pw::pw_stress` → `method.xc` |
| metric | max_v \|σ_cuda − σ_cpu\| [eV/Å³] |
| acceptance | dual-gate Si FCC ≤ 5e-2 eV/Å³; cubic symmetry; GH200 log |

**No energy-path change** if stress-only: device SCF + NSCF strain-FD already
route through `DeviceDensity` PBE (`xc_pbe.cu` + spectral ∇ρ). Gap is the
product dual-gate pin under PBE (queue claim was LSDA-only).

## System / oracle

- 2-atom Si FCC primitive, a = 5.43 Å, GTH-PBE-q4, ecut 8 Ry, Γ, tol 1e-6
- Oracle: same-settings `cpu::pw::pw_stress` with `XCType::PBE` (not CP2K)
- Fail-closed: spin / smear unchanged

## Non-goals

Spin stress, smear stress, multi-SCF FD, energy rewrites, GPW stress.
