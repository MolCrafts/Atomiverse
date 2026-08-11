# cuda-pw-smear-forces

method=cuda | twin: CPU forces under metallic occ + dual-gate

## Lever

| Site | Change |
| --- | --- |
| `src/cuda/planewave/pw_solver.cu` `pw_forces_analytic` | Lift restricted `smearing != none` reject; force-path SCF with FD/MP/Gaussian fractional occupations (same shape as `PW::run` smear branch) |
| `src/cuda/planewave/internal/pw_forces.cu` + `device_solver.{cu,cuh}` | NL Hellmann–Feynman with per-band `occ[b] ∈ [0,2]` |
| `src/cpu/planewave/internal/pw_forces.cpp` | Wire oracle: `run_scf_smear_restricted` + fractional `accumulate_nl_forces_k` (CPU fixed-occ path ignored smear — dual-gate would be dishonest) |

Metric: dual-gate max|F_cuda − F_cpu| [eV/Å] on stretched Al₂ dimer under
restricted FD smear (monatomic 1-atom F≡0 by gauge — relative bond required).

## Acceptance

- GH200: `reg_cuda-pw-smear-forces --device=both` green; max|ΔF| **stated**, design-time tol **5e-3 eV/Å** (same class as `cuda-pw-forces`)
- Measured **GH200 1130401**: max|ΔF|=**1.358e-11** eV/Å; max|F_cpu|=1.481 eV/Å; units 2/2; (prior 1130359 same class)
- Restricted metallic-smear **energy** path unchanged (`reg_cuda-pw-metallic-smear`)
- spin+smear forces stay reject (child `cuda-pw-spin-smear-forces`)
- `pw_stress` smear reject unchanged (child `cuda-pw-smear-stress`)
- Same-settings CPU oracle; no self-golden pins; no soft tols
- **perf_exempt** small cell (correctness dual-gate only)

## Out of scope

spin∧smear F/σ; restricted smear stress; smear×PBE force depth; multi-k product claims beyond the reg k-mesh.
