# cuda-gpw-pbe-product-gate

## Lever

| Field | Value |
| --- | --- |
| File | `tests/cuda/gpw/test_gpw_full_product.cu` + `regressions/gpw/cuda-gpw-pbe-product-gate/` |
| Function | Public `cuda::gpw::GpwDft::run` with `xc = XcKind::PBE` (already routes via `to_ks_xc` → `KsChainDevice` PBE/`SpectralFftDevice` from `cuda-gpw-ks-gga`) |
| Metric | \|E_cuda − E_cpu\| [Ha] on H₂ Γ RKS **PBE** dual-gate; optional free-fn max\|ΔF\| |

## Research (before code)

- Product **accepts** `XcKind::PBE` on base (`gpw_solver.cu` validate + `to_ks_xc`).
- Stage unit `cuda-gpw-ks-gga` dual-gates ρ→V_eff PBE vs CPU `ks_grid_build_rks`.
- e2e dual-gates on base are **PADE-only** H₂ (unit + `reg_cuda-gpw-full-product`) — PBE is claimed-but-ungated.
- Compose only: extend unit/reg; **no** new kernel foothold.

## Acceptance

1. Unit `H2GammaPbeEnergyDualGate` (+ forces if free-fn path exercised) green on GH200.
2. `reg_cuda-gpw-pbe-product-gate` dual-gate vs same-settings CPU Γ RKS PBE (H₂, GTH-PBE-q1, DZV-GTH-PADE compact basis).
3. Stated tols: \|ΔE\| ≤ **5e-6** Ha; max\|ΔF\| ≤ **5e-5** Ha/Bohr (same class as PADE product).
4. Both SCF converged; finite energies.
5. Fail-closed paths unchanged (UKS+PBE, BLYP, OT, multi-k, MT, …).
6. **perf_exempt** small-cell correctness (not fair-wall vs CP2K).
7. No soft tols / no self-golden pins.

## Non-goals

UKS-PBE (`cuda-gpw-uks-pbe-gate`), H₂O×PBE (`product-h2o-pbe`), MG×PBE, multi-k×PBE, PBE forces-as-new-kernel, stress, stream, DIIS/eigen device residual.
