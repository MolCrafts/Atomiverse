# Spec: `cuda-gpw-full-product`

**Track:** cuda · **P0** · **Branch:** `wf/escuda-cuda-gpw-full-product`  
**Parent:** `dev` · **Loop:** `es-cpu-cuda-opt`  
**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../notes/cuda-gpw-public-shape.md)

## Lever

Compose landed `cuda/gpw/internal/*` stage units into a **public method**
`ATV::cuda::gpw::GpwDft` that mirrors CPU `ATV::cpu::gpw::GpwDft` public shape
(members + `build` / `run` / `last_result` + forces family). Stage footholds
must **not** become the user API.

| Layer | Path |
| --- | --- |
| Public method (new) | `include/atomiverse/cuda/gpw/gpw_solver.cuh` + `src/cuda/gpw/gpw_solver.cu` (names may match CPU) |
| Stages (compose, do not promote) | `ks_chain`, `multigrid_transfer`, product collocate/integrate, Poisson, XC |
| Oracle | same-settings CPU `GpwDft` |

## First slice (acceptance minimum)

1. **Γ closed-shell DIIS RKS** — LDA (PADE) and/or PBE.
2. **Energy** via `build` → `run` → `last_result` (no `*Config`, no factories).
3. **Forces** path for that slice (free function and/or Frame evaluator family
   matching CPU).
4. Multi-k / UKS / OT / vdW **not** required for first land.

## Gates

| Gate | Criterion |
| --- | --- |
| Shape | Public surface equals freeze note; stages stay `internal/` |
| Correctness | CUDA↔CPU energy (and forces) within **stated** tols |
| Device-only | No host science loops in `.cu` hot path |
| Evidence | **GH200** log — ctest/reg PASS; no green claim from login-only |
| Fail-closed | Unsupported members throw with how-to (no silent CPU fallback) |
| FD | If energy path shared with CPU, re-run FD force oracle where required |

## Non-goals

- Hybrid / GAPW / PAW / meta-GGA product
- Softening tols / self-golden pins
- Promoting `KsChainDevice` as public method
- Full CUDA `Driver<gpw::Phase>` nested node (optional later; public shape first)

## Tests / regressions (planned)

- Unit: public shape / fail-closed guards (login-buildable where possible)
- `reg_cuda-gpw-full-product` (or slice name): dual-gate energy (+ forces) vs CPU
  Γ RKS small molecule; **perf_exempt** small cell OK for first slice
- Existing stage units must remain green (`test_cuda_gpw_*`, `reg_cuda-gpw-ks-*`)

## acceptance_met

`false` until GH200 evidence + dual-gate green + shape freeze still holds.
