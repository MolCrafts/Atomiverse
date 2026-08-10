# cuda-gpw-product-h2o-gate

method=cuda | track=cuda | parent `dev`

## Lever

- **What:** Production-class dual-gate for public `cuda::gpw::GpwDft` beyond the
  H₂ foothold — water Γ RKS PADE energy (+ free-fn forces when SCF green).
- **Where:** new `regressions/gpw/cuda-gpw-product-h2o-gate/` + CMake
  `reg_cuda-gpw-product-h2o-gate` / `run_reg_cuda-gpw-product-h2o-gate`; optional
  unit in `tests/cuda/gpw/test_gpw_full_product.cu`. **No new kernel.**
- **Metric:** both SCF converge; `|E_cuda−E_cpu|` and (if forces) max`|ΔF|`
  under stated tols; GH200 log evidence.
- **SCF residual (prior no-land):** wave-9 failed on CUDA water RKS non-
  stationarity (jobs 1128194/1128217/1128269/1128295). If pure DIIS stalls,
  harden the **existing** product SCF loop (DIIS subspace reset on energy
  ascent / modest level-shift defaults in the reg only — not soft pins).

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
| Forces dual-gate | max`\|dF\| ≤ 5e-4` Ha/Bohr (optional if free-fn green) |
| Perf | **perf_exempt** small-molecule correctness |
| Scope | no hybrid/GAPW/PAW/meta-GGA; no multi-k; no MT |

`acceptance_met=true` only with GH200 dual-gate log.

## GH200 evidence

### Re-verify after rebase onto `dev` @ `bcb873e0` — job **1129720** (COMPLETED)

- Node GH200 `n211`; `test_cuda_gpw_full_product` — **100%** (incl.
  `WaterGammaPadeEnergyDualGate`)
- `run_reg_cuda-gpw-product-h2o-gate`:
  - CPU E=−16.902354386489 Ha (7 iters); CUDA E=−16.902355139062 Ha (8 iters)
  - **|dE|=7.526e−07 Ha** (tol 5e−5); **max|dF|=3.213e−06 Ha/Bohr** (tol 5e−4)
  - PASS; **perf_exempt** small molecule

### Prior tip evidence (job **1129294**, COMPLETED; same numbers)

Same dual-gate result on pre-rebase tip `17b955ed`.
