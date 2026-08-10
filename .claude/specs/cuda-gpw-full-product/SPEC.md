# Spec: `cuda-gpw-full-product`

**Track:** cuda · **P0** · **Branch:** `wf/cuda-gpw-full-product`  
**Parent:** `dev` · **Loop:** `es-cpu-cuda-opt`  
**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../notes/cuda-gpw-public-shape.md)

## Lever (research → impl)

| Item | Value |
| --- | --- |
| Public method | `include/atomiverse/cuda/gpw/gpw_solver.cuh` + `src/cuda/gpw/gpw_solver.cu` |
| Device stage | `KsChainDevice::run_from_rho` (Poisson + Padé/PBE + V_eff) |
| Host twin path | STVnl / cbg collocate / integrate / DIIS / diagonalize |
| Metric | CUDA↔CPU \|ΔE\| [Ha], max\|ΔF\| [Ha/Bohr] on H₂ Γ RKS PADE |
| Forces | free `cuda::gpw::gpw_forces_analytic` / `gpw_forces_from_scf` |

Stage footholds stay under `cuda/gpw/internal/` — **not** the user API.

## First slice (acceptance minimum)

1. **Γ closed-shell DIIS RKS** — LDA (PADE) and/or PBE.
2. **Energy** via `build` → `run` → `last_result` (no `*Config`, no factories).
3. **Forces** free-function family matching CPU.
4. Multi-k / UKS / OT / vdW / multigrid energy **fail-closed** (how-to throw).

## Gates

| Gate | Criterion |
| --- | --- |
| Shape | Public surface equals freeze note; stages stay `internal/` |
| Correctness | \|ΔE\| ≤ **5e-6 Ha**, max\|ΔF\| ≤ **5e-5 Ha/Bohr** (H₂ dual-gate) |
| Device-only | No host science loops in device KS kernels (`.cu` hot path) |
| Evidence | **GH200** log — ctest/reg PASS |
| Fail-closed | Unsupported members throw with how-to (no silent CPU-SCF fallback) |
| Perf | **perf_exempt** small-cell first product |

## Tests / regressions

- Unit: `test_cuda_gpw_full_product` — shape, fail-closed, H₂ dual-gate
- `reg_cuda-gpw-full-product` — dual-gate energy + forces vs CPU
- Existing stage units remain green

## acceptance_met

`false` until GH200 evidence + dual-gate green + shape freeze still holds.
