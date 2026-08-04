<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture (binding conclusions)

Physics pipeline is one algorithm; SCF control flow is still forked (Γ/k ×
RKS/UKS) but **workspace + shared fine-grid KS build are landed**. Product
collocate/integrate now uses a **CP2K-style collint engine** with geometry
batching; the remaining wall vs CP2K is still large (~30×) but no longer the
old separable-per-product loop.

### One pipeline

```
P / {P_k} → collocate ρ → Poisson → XC → integrate V → H → DIIS|OT → P'
```

Geometry-fixed once per `run()`: S/T/Vpp or S_k/T_k/Vnl_k, n_ion, v_locsr,
cbg / ProductSet / KpointBasisGrid. Iteration bags live on
`internal::GpwWorkspace`; fine-grid ρ→V_eff is `ks_grid_build_rks/uks`.

### Fair wall vs CP2K (2026-08-04, arrhenius1, OMP=4 bind)

Twin `gpw-multigrid-kpoints` MAIN (MP 2×2×2, 150 Ry, NGRIDS 4):

| Side | Wall [s] | Notes |
| --- | ---: | --- |
| ATV best MAIN | **~2.1** | best-of-4 OMP=4 bind |
| CP2K 2026.1 same host | **~1.28** quiet | re-run 2026-08-04 |
| **ratio ATV/CP2K** | **~1.6×** | ≤2× gate closed |

Landed: collint stack; STVnl gen-stamp Hermite + s–s + fused `prim_ST`
(~1500→520 ms); integrate group-idx + segment SIMD accumulate. Per-iter
coll/int ~80–90 ms. Residual is optional polish toward ~1×.

### Product collint (landed)

Files: `internal/product_collint.{h,cpp}` + wire in `multigrid.cpp`.

| Piece | Status |
| --- | --- |
| cab→cxyz expand (per-axis binomial about pc) | landed |
| pol[dir][l][i] + exp recurrence | landed |
| z-fast contract + portable xsimd | landed |
| grow-only thread_local scratch (no per-product heap) | landed |
| geometry sort + group batch (shared pol / monom I) | landed |
| sphere-only Fz contract (skip cube corners) | landed |
| reflection pairs `(k,1−k)` | landed |
| CP2K sphere-bounds cache | landed |
| integrate group-idx + segment SIMD | landed |
| shell-level task list beyond cartesian-prim groups | **open** |

Algorithm after CP2K `src/grid/cpu/grid_cpu_collint.h` (BSD-3, Ole Schuett).
Public API / Frame laws unchanged. SIMD: portable **xsimd only**.

### SIMD policy

Portable **xsimd only** (no arch intrinsics, no global `XTENSOR_USE_XSIMD`).
Landed: z-run axpy/dot; 1D poly/E2 (legacy collocate_one); collint z-contract;
`reduce_add` on dots. Exp recurrence stays scalar.

### Critical bottlenecks (re-ranked post-collint)

| Rank | Issue | Where |
| --- | --- | --- |
| **1** | Residual setup (STVnl ~0.52 s, products_mg ~0.3 s, kbg ~0.2 s) | k-path |
| 2 | Collint residual (~4× CP2K per-iter; ~1.6× overall) | `product_collint.cpp` |
| 3 | Multi-k dense χ^k until products default | `kpoints.cpp` |
| 4 | Force path rebuilds grids | `gpw_forces.cpp` |

**Done (do not re-open as primary):** GpwWorkspace; `ks_grid_build_*`; OT
economics; |V(pc)| windows; 1D poly SIMD; **first collint + geometry batch**.

### Extract / rewrite order

1. ~~`GpwWorkspace`~~ landed  
2. ~~`KsGridBuild`~~ landed  
3. ~~CP2K collint product path (cxyz + pol + geometry batch)~~ **landed**  
4. **Active:** reflection pairs + sphere cache + tighter task locality  
5. Force grid reuse; MPI product/grid distribute; products-default-k after (4)

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`; arch
intrinsics; weakening SCF tol; 5th collocation layout without deleting dense
χ^k; file-split without shared KS build.

### Disk
