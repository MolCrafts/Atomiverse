# Spec — `cuda-pw-pbe-device`

**method:** cuda · **oracle:** same-settings CPU twin · **branch:** `wf/cuda-pw-pbe-device`

## Lever

| Item | Value |
| ---- | ----- |
| Files | `src/cuda/planewave/internal/xc_pbe.cu` (+ `.cuh`), `device_density.cu` GGA branch, `pw_solver.cu` whitelist |
| Function | `xc_pbe_grid` (pointwise closed-shell PBE X+C + flux); `DeviceDensity` spectral ∇ρ → PBE → −∇·flux |
| Metric | Pointwise Vxc/ε/flux vs CPU `evaluate_xc(PBE,…)` at **1e-12·max(1,\|ref\|)**; DeviceDensity stage parity vs `XCBuilder::step(ws, PBE)` at LDA-stage bounds; e2e \|E_cuda−E_cpu\| **< 1e-3 Ry** |
| What lifts | CUDA `PW` was LSDA-only; default `xc=PBE` threw. Device PBE-GGA closes the XC functional gap vs CPU for restricted γ / k PW. |

## Acceptance

1. Unit `test_cuda_pw_xc_pbe` green on GH200 (pointwise + PW92-not-VWN5 pin).
2. Unit `test_cuda_pw_density` PBE stage parity green (reuses LDA-stage tols).
3. Guards: default PBE runnable; PBEsol/revPBE/RPBE/BLYP throw (no silent wrong-constants).
4. E2E `reg_cuda-pw-pbe-device` dual-gate: both legs converge; \|E_cuda−E_cpu\| < 1e-3 Ry on Si diamond + GTH-PBE-q4.
5. No host loops over grid points in new `.cu` kernels; device-only XC path.
6. **perf_exempt** — 2-atom correctness foothold; small cells never claim CUDA wall wins (ES order step 4).

## Out of scope

- Device forces/stress (`cuda-pw-forces` / `cuda-pw-stress`).
- PBE-family constant variants (PBEsol/revPBE/RPBE).
- Spin-polarized GGA on CUDA.
- Large-cell wall (`cuda-pw-large-cell-measure`).

## Gate evidence

GH200 sbatch log with `CUDA_PW_PBE_JOB_OK` + ctest 100% on the scoped suite + `run_reg_cuda-pw-pbe-device`.
