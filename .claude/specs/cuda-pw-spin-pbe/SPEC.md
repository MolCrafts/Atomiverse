# cuda-pw-spin-pbe

## Lever

| Item | Value |
| --- | --- |
| Track | **cuda** |
| Lever | Lift collinear `spin_polarized` ∧ `XCType::PBE` reject on `cuda::pw::PW::run`; implement device true spin-GGA energy |
| Files | `src/cuda/planewave/internal/xc_pbe.cu` (+ header), `device_density.cu/.cuh`, `pw_solver.cu/.cuh` |
| Metric | same-settings CPU↔CUDA \|E\| [Ha] on O₂ triplet spin-PBE; mag = 2 |
| Not in scope | forces, stress, spin+smear |

## Acceptance

1. `DeviceDensity::exchange_correlation_spin` accepts `XCType::PBE` and runs spectral ∇ρ_σ + `xc_pbe_spin_grid` + div(flux_σ) matching `XCBuilder::step_spin` GGA branch.
2. Unit: `test_cuda_pw_xc_pbe_spin` — device kernel vs `evaluate_xc(PBE, …)` within `1e-12·max(1,|ref|)` (and flux formulas).
3. Regression `reg_cuda-pw-spin-pbe --device=both`: mag = 2, both converge, \|E_cuda − E_cpu\| ≤ **2e-3 Ha**, \|E_PBE − E_LSDA\| > 1e-4 Ha (GGA activation on CUDA or via CPU twin).
4. Fail-closed: spin forces/stress/smear still throw; PBEsol/revPBE/RPBE still throw.
5. GH200 log evidence required for green; `perf_exempt` (correctness dual-gate foothold).

## Twin

- CPU: `reg_pw-spin-gga` class (O₂, ecut 12 Ry, k=1, mult=3, mix_beta=0.3, PBE).
- Oracle: same-settings `cpu::pw::PW` (not self-golden CUDA).
