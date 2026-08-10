# cuda-gpw-ks-chain

**method:** cuda  
**branch:** `wf/cuda-gpw-ks-chain`  
**oracle:** same-settings CPU (`ks_grid_build_rks` + continuous product collocate/integrate)

## Lever

Compose four landed CUDA GPW unit footholds into one **device stage** under
`src/cuda/gpw/internal/`:

```
product collocate → n = ρ − n_ion → Poisson/FFT → Padé LDA XC
  → V_eff = V_es + V_xc [+ V_loc] → product integrate
```

| Piece | File | Metric |
| --- | --- | --- |
| Stage glue | `ks_chain.{cuh,cu}` | device-only chain; no host grid loops |
| Collocate | reuses `product_collocate` | ρ from products |
| Poisson | reuses `PoissonFftDevice` | V_es, e_H |
| XC | reuses `xc_lda_grid` + `xc_energy` | V_xc, e_xc |
| Integrate | reuses `product_integrate` | V_eff · shape → acc |

**Not in scope:** `GpwDft::run` product, multigrid, k-points, GGA/vdW, public method API.

## Acceptance

1. Unit `test_cuda_gpw_ks_chain` on GH200: composed stage vs CPU twin on Γ small grid.
2. Regression `reg_cuda-gpw-ks-chain` dual-gate (CPU path + CUDA path parity).
3. Stated tols (Hartree unless noted):
   - max\|V_eff\|: `1e-9 · max(1, peak) + 1e-12`
   - \|e_H\|, \|e_xc\|: `1e-10 · max(1, \|ref\|)`
   - integrate acc: `1e-9 · max(1, peak) + 1e-12`
4. Device-only kernels in `ks_chain.cu` (no host loops over grid/products).
5. **No** public full GPW method; stage stays under `cuda::gpw::internal`.
6. GH200 log evidence required for green.

## Perf

`perf_exempt` — composed foothold stage on tiny Γ grid; correctness only (not a fair twin wall).
