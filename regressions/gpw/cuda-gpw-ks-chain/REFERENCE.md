# cuda-gpw-ks-chain

## What

Composed CUDA GPW **KS stage unit** (not a full method):

```
product collocate → n = ρ − n_ion → Poisson/FFT → Padé LDA XC
  → V_eff = V_es + V_xc + V_loc → product integrate
```

Dual-gate vs same-settings CPU `ks_grid_build_rks` + continuous product
collocate/integrate on a Γ 16³ / L=8 Bohr multi-s water-like density.

## Oracle

| Item | Value |
| --- | --- |
| Oracle | `cpu-twin` (`ks_grid_build_rks` + continuous product shapes) |
| Literature | n/a (internal stage parity) |
| Units | Hartree |

## Tols (stated)

| Quantity | Tol |
| --- | --- |
| e_hartree, e_xc, e_locgrid | `1e-10 · max(1, \|ref\|)` |
| max\|ΔV_eff\| | `1e-9 · max(1, peak) + 1e-12` |
| max\|Δacc\| | `1e-9 · max(1, peak) + 1e-12` |

## Perf

`perf_exempt` — tiny-grid correctness foothold; not a fair CP2K twin wall.

## Run

```bash
# On GH200 only (login has no CUDA toolkit)
cmake --build build -j$(nproc) --target atomiverse test_cuda_gpw_ks_chain reg_cuda-gpw-ks-chain
ctest --test-dir build -R '^test_cuda_gpw_ks_chain$' --output-on-failure
cmake --build build --target run_reg_cuda-gpw-ks-chain
```

## Evidence

| Item | Value |
| --- | --- |
| GH200 job | **1049858** (`GPU_JOB_OK`, exit 0) |
| HEAD | `ce4e2944` on `wf/escuda-cuda-gpw-ks-chain` |
| Unit | 5/5 PASS (`test_cuda_gpw_ks_chain` + 4 footholds) |
| Dual-gate | `PASS cuda-gpw-ks-chain dual-gate` — e_H/e_xc/e_loc ~1e-15; max\|ΔV_eff\|=4.4e-16; max\|Δacc\|=6.9e-14 |
| Log | `.claude/worktrees/wf_escuda_cuda_gpw_ks_chain/.claude/cuda-gpw-ks-chain-1049858.out` |
