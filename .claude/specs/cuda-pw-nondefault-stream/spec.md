# cuda-pw-nondefault-stream

## Lever

| Item | Value |
| --- | --- |
| **Metric** | Correctness only — no wall claim on 2-atom Si (ES order step 3 polish) |
| **Files** | `device_density.{cu,cuh}`, `device_solver.{cu,cuh}`, `pw_solver.cu` |
| **Change** | One non-default `cudaStream_t` owned by `DeviceDensity`; borrowed by `DeviceBandSolver` so density + band sides share ordering. All kernels, cuFFT, cuBLAS, cuSOLVER bound to that stream. Setup/H2D may stay blocking; mid-SCF host decisions use stream-ordered D2H. |
| **Not** | Physics change, default-stream dual path, multi-stream overlap, bit-identity vs prior default-stream run |

## Acceptance

1. `DeviceDensity::stream()` is non-null (not the default stream).
2. `PW::run` shares that stream into `DeviceBandSolver`.
3. GH200: `test_cuda_pw_*` + `test_cuda_scf_rhf` 100% PASS; `diag_band_parity` MAX_EIG_DIFF at roundoff; `reg_pw-cuda-lda-si --device=both` and `reg_cuda-pw-nondefault-stream` PASS at existing tols (parity 1e-3 Ry, ABINIT window).
4. No host loops in `.cu` hot paths; device-only kernels.

## Twin / oracle

- Unit: `tests/cuda/planewave/test_density.cu` + new stream assertion; foothold unit suite.
- E2E: same-settings CPU twin (`pw-cuda-lda-si` / `cuda-pw-nondefault-stream`) + literature ABINIT window.

## Perf

`perf_exempt` — residency/stream polish; 2-atom cell cannot resolve stream overhead (same as kpoint-hoist note).
