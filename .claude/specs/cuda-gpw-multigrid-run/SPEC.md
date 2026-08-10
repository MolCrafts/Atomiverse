# cuda-gpw-multigrid-run

method=cuda | track es-cpu-cuda | P1 product depth

## Lever

| | |
| --- | --- |
| File | `src/cuda/gpw/gpw_solver.cu` (+ public docs in `gpw_solver.cuh`) |
| Function | `GpwDft::validate_slice_`, `GpwDft::run` → mid-SCF `build_H_and_energy` |
| Metric | CUDA↔CPU energy dual-gate on Γ RKS **multigrid** (`ngrids>1`, `rel_cutoff>0`) |

Wire landed stage unit `KsMultiGridChainDevice` (spectral prolong/restrict +
fine KS + per-level product collocate/integrate) into public product energy.
Lift `ngrids!=1` reject. Single-grid (`ngrids==1`) path stays on
`KsChainDevice` (unchanged dual-gate class).

## Acceptance

1. `ngrids>1 && rel_cutoff>0` accepted by `build`/`run` (fail-closed if
   `ngrids>1 && rel_cutoff<=0` with how-to).
2. Unit `test_cuda_gpw_full_product` (or MG-specific unit) +
   `reg_cuda-gpw-multigrid-run`: H₂ Γ RKS PADE, ngrids=4, rel_cutoff class,
   both SCF converge; \|E_cuda − E_cpu\| within stated tol (same order as
   full-product **5e-6 Ha**, or stated looser if cuFFT/MG residual requires).
3. GH200 log evidence required for green.
4. **perf_exempt** — small-cell correctness foothold (not fair-wall vs CP2K).

## Non-goals

- Multigrid **forces** (`cuda-gpw-multigrid-forces`)
- Multi-k × MG (`cuda-gpw-multigrid-kpoints`)
- UKS × MG, PBE product gate, device DIIS/eigen
- Soften tols / self-golden pins

## Twin / oracle

Same-settings CPU `cpu::gpw::GpwDft` (not CP2K live). Offline pins only if
cited; primary gate is CPU twin energy.
