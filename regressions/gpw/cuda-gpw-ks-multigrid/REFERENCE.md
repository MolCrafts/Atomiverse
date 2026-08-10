# cuda-gpw-ks-multigrid

## What

Composed CUDA GPW **multi-grid KS stage unit** (not a full method):

```
per-level product collocate → spectral prolong to fine
  → n = ρ − n_ion → Poisson/FFT → Padé LDA XC → V_eff
  → spectral restrict to levels → per-level product integrate
```

Dual-gate vs same-settings CPU Γ multigrid fragment:
`prolong_fft` / `restrict_fft` + `ks_grid_build_rks` + continuous product
shapes on fine 16³ + coarse 8³ / L=8 Bohr.

## Oracle

| Item | Value |
| --- | --- |
| Oracle | `cpu-twin` (CPU `prolong_fft`/`restrict_fft` + `ks_grid_build_rks` + continuous products) |
| Literature | n/a (internal stage parity) |
| Units | Hartree |

## Tols (stated)

| Quantity | Tol |
| --- | --- |
| e_hartree, e_xc | `1e-9 · max(1, \|ref\|)` |
| max\|ΔV_eff\| | `1e-8 · max(1, peak) + 1e-11` |
| max\|Δacc\| (per level) | `1e-8 · max(1, \|ref\|) + 1e-11` |

Not bit-exact (device reduction order / cuFFT vs FFTW).

## Perf

`perf_exempt` — tiny-grid correctness foothold; not a fair CP2K twin wall.

## Run

```bash
# On GH200 only (login has no CUDA toolkit)
cmake --build build -j$(nproc) --target atomiverse test_cuda_gpw_ks_multigrid reg_cuda-gpw-ks-multigrid
ctest --test-dir build -R 'test_cuda_gpw_ks_multigrid' --output-on-failure
./build/regressions/reg_cuda-gpw-ks-multigrid --device=both
```
