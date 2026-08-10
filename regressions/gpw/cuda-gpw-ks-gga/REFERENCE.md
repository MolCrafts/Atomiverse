# cuda-gpw-ks-gga

## What

Composed CUDA GPW **KS stage unit** with closed-shell **PBE-GGA** (not a full
method):

```
product collocate → n = ρ − n_ion → Poisson/FFT
  → spectral ∇ρ → device PBE (local v_ρ + flux) → spectral −∇·flux
  → V_eff = V_es + V_xc + V_loc → product integrate
```

Dual-gate vs same-settings CPU `ks_grid_build_rks(..., XcKind::PBE)` +
continuous product collocate/integrate on a Γ 16³ / L=8 Bohr multi-s density.

## Oracle

| Item | Value |
| --- | --- |
| Oracle | `cpu-twin` (`ks_grid_build_rks` PBE + continuous product shapes) |
| Literature | n/a (internal stage parity) |
| Units | Hartree |

## Tols (stated)

| Quantity | Tol |
| --- | --- |
| e_hartree, e_xc, e_locgrid | `1e-10 · max(1, \|ref\|)` |
| max\|ΔV_eff\| | `1e-9 · max(1, peak) + 1e-12` |
| max\|Δacc\| | `1e-9 · max(1, peak) + 1e-12` |

Not bit-exact (cuFFT vs FFTW, atomic E_xc reduction). Observed on GH200
~1e-15 energy / ~1e-16 V_eff (far under tol).

## Perf

`perf_exempt` — tiny-grid correctness foothold; not a fair CP2K twin wall.

## Run

```bash
# On GH200 only (login has no CUDA toolkit)
sbatch .claude/job-cuda-gpw-ks-gga.sh
# or manual:
cmake --build build -j$(nproc) --target atomiverse test_cuda_gpw_ks_gga reg_cuda-gpw-ks-gga
ctest --test-dir build -R 'test_cuda_gpw_ks_gga|test_cuda_gpw_ks_chain' --output-on-failure
cmake --build build --target run_reg_cuda-gpw-ks-gga
```

## Evidence

| Item | Value |
| --- | --- |
| GH200 job | **1050680** (`GPU_JOB_OK`, exit 0; rebased on `dev` `9f05a4fc`) |
| HEAD | `93c26d40` on `wf/escuda-cuda-gpw-ks-gga` |
| Unit | 1/1 PASS (`test_cuda_gpw_ks_gga`) |
| Dual-gate | `PASS cuda-gpw-ks-gga dual-gate` — e_H/e_xc ~1e-15; max\|ΔV_eff\|=8.0e-16; max\|Δacc\|=3.9e-14 |
| Log | `.claude/worktrees/wf_escuda_cuda_gpw_ks_gga/.claude/cuda-gpw-ks-gga-1050680.out` |
| Prior | job **1050277** (pre-rebase) same tols |
