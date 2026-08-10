# cuda-gpw-device-collocate

method=cuda | track: es-cpu-cuda | parent: `dev`

## Lever

| | |
| --- | --- |
| File | `src/cuda/gpw/gpw_solver.cu` |
| Function | `GpwDft::run` → `build_H_and_energy` |
| Metric | Drop host `collocate_density` / `integrate_potential` from the mid-SCF path; no H2D ρ for KS; no D2H V_eff for integrate |

Compose landed stage units:

- geometry-fixed `cpu::gpw::build_products` (ngrids=1 MultiGrid) → host product list
- per-iter density weight → `DeviceProduct.weight`
- device `KsChainDevice::run` (collocate → Poisson → XC → V_eff → integrate shapes)
- host scatter of device acc → KS V_grid (same `k · w · coeff` map as CPU `integrate_products`)

Still host: STVnl / DIIS / diagonalize. Still D2H V_eff / V_es for `result_` force reuse (final KS only preferred). Fail-closed non-Γ paths unchanged.

## Twin / oracle

Same-settings CPU `cpu::gpw::GpwDft` (H₂ Γ RKS PADE class — existing full-product dual-gate). Oracle: **CPU twin**, not CP2K.

## Acceptance

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |
| GH200 evidence | required |
| Mid-SCF path | device collocate + device integrate (not host cbg coll/int for H) |

**perf_exempt** — small-cell product depth (compose footholds; not fair-wall cut).

## Non-goals

New kernels; multigrid product energy; UKS; multi-k; OT; non-default stream (separate slug); device forces (separate slug).
