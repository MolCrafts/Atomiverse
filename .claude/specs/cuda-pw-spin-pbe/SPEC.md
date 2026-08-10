# cuda-pw-spin-pbe

**acceptance_met=true** (GH200 **1124645** reconfirm after rebase onto
`dev`@`44bd64bb`, exit 0; prior **1123961** / **1122874**).

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

## Gate evidence (GH200 **1124645**, n40 — wave-4 reconfirm)

| Check | Result |
| --- | --- |
| `test_cuda_pw_xc_pbe_spin` | Passed |
| `test_cuda_pw_guards` | Passed |
| CPU PBE E | **−28.7585210635 Ha**, mag=2, conv |
| CUDA PBE E | **−28.7585210635 Ha**, mag=2, conv |
| \|E_cuda − E_cpu\| | **1.14e-13 Ha** (tol 2e-3) |
| \|E_PBE − E_LSDA\|_cpu | **0.129 Ha** (floor 1e-4) |
| RCs | BUILD=0 UNIT_SPIN=0 UNIT_GUARD=0 REG=0 |

Prior gates: **1123961** (n454, after rebase onto `4913b33b`); **1122874**
(n416) same dual-gate numbers on pre-rebase tip.
