# cuda-gpw-product-h2o-gate

method=cuda | track=cuda | parent `dev`

## Lever

- **What:** Production-class dual-gate for public `cuda::gpw::GpwDft` beyond the
  H₂ foothold — water Γ RKS PADE energy + free-fn forces.
- **Where (files):**
  - `src/cuda/gpw/gpw_solver.cu` — RKS mid-SCF: on energy ascent
    `dE > kDiisAscentResetHa` call `AoDiisEigenDevice::reset()` (post
    `cuda-gpw-device-diis-eigen`; host `GpwDiis::clear` remains for host DIIS
    API / UKS residual — UKS path intentionally does **not** ascent-reset).
  - `regressions/gpw/cuda-gpw-product-h2o-gate/` + CMake
    `reg_cuda-gpw-product-h2o-gate` / `run_reg_cuda-gpw-product-h2o-gate`
  - unit `tests/cuda/gpw/test_gpw_full_product.cu::WaterGammaPadeEnergyDualGate`
  - **No new kernel.**
- **Metric:** both SCF converge; `|E_cuda−E_cpu| ≤ 5e-5` Ha;
  max`|ΔF| ≤ 5e-4` Ha/Bohr; GH200 log evidence.
- **SCF residual (prior no-land):** wave-9 CUDA water RKS non-stationarity
  (jobs 1128194/1128217/1128269/1128295). Hardening: DIIS subspace reset on
  energy ascent — not soft pins / not level-shift defaults in the reg.

## Twin / oracle

Same-settings CPU `cpu::gpw::GpwDft` — water Γ RKS **PADE**, periodic,
O:SZV-GTH-PADE / GTH-PADE-q6 + H:DZV-GTH-PADE / GTH-PADE-q1, `n_occ=4`.
Oracle: **CPU twin** (not CP2K for this correctness gate).

## Acceptance

| Gate | Criterion |
| --- | --- |
| CPU SCF | converged |
| CUDA SCF | converged (GH200) |
| Energy dual-gate | `\|dE\| ≤ 5e-5` Ha (p-shell multi-center class, O₂ gate) |
| Forces dual-gate | max`\|dF\| ≤ 5e-4` Ha/Bohr |
| Perf | **perf_exempt** small-molecule correctness |
| Scope | no hybrid/GAPW/PAW/meta-GGA; no multi-k; no MT |

`acceptance_met=true` only with GH200 dual-gate log after rebase onto
device-diis-eigen (`dev` ≥ `e164b767`).

## GH200 evidence

### Re-verify after rebase onto `dev` @ `7f8da1f0` (Assign wave 5 base) — job **1130195** (COMPLETED)

- Node GH200 `n46`; `test_cuda_gpw_full_product` — **100%** (incl.
  `WaterGammaPadeEnergyDualGate`)
- `run_reg_cuda-gpw-product-h2o-gate`:
  - CPU E=−16.902354386489 Ha (7 iters); CUDA E=−16.902355139062 Ha (8 iters)
  - **|dE|=7.526e−07 Ha** (tol 5e−5); **max|dF|=3.213e−06 Ha/Bohr** (tol 5e−4)
  - PASS; **perf_exempt** small molecule
- Lever: `AoDiisEigenDevice::reset()` on ascent (`kDiisAscentResetHa`); no new
  kernel. Log: `.claude/cuda-gpw-product-h2o-gate-1130195.out`.
  **acceptance_met=true**.

### Prior re-verify on `dev` @ `e13cdb91` (Assign wave 4 base) — job **1130140** (COMPLETED)

Same dual-gate numbers (n193). Superseded by **1130195** after rebase onto
Assign wave 5 base.

### Prior re-verify on `dev` @ `46d65433` — job **1130066** (COMPLETED)

Same dual-gate numbers (n210). Superseded by **1130140**.

### Prior (pre device-diis land) — jobs **1129720** / **1129294**

Same dual-gate numbers on host-GpwDiis RKS tip (`bcb873e0` base).
