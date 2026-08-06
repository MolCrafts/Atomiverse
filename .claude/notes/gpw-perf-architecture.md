<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture (binding conclusions)

Physics pipeline is one algorithm; SCF control flow is still forked (Γ/k ×
RKS/UKS) but **workspace + shared fine-grid KS build are landed**. Product
collocate/integrate now uses a **CP2K-style collint engine** with geometry
batching + shell-level tasks; fair wall vs CP2K is **~1.25×** MAIN (was
~30–47× pre-collint; ~1.5× post products_mg).

### One pipeline

```
P / {P_k} → collocate ρ → Poisson → XC → integrate V → H → DIIS|OT → P'
```

Geometry-fixed once per `run()`: S/T/Vpp or S_k/T_k/Vnl_k, n_ion, v_locsr,
cbg / ProductSet / KpointBasisGrid. Iteration bags live on
`internal::GpwWorkspace`; fine-grid ρ→V_eff is `ks_grid_build_rks/uks`.

### Fair wall vs CP2K (2026-08-06 wave-6, arrhenius1, OMP=4 bind)

Twin `gpw-multigrid-kpoints` MAIN (MP 2×2×2, 150 Ry, NGRIDS 4):

| Side | Wall [s] | Notes |
| --- | ---: | --- |
| ATV MAIN | **~1.6** | post shell-level collint tasks (coll ~30 / int ~58 ms) |
| CP2K 2026.1 same host | **~1.28** quiet | re-run 2026-08-04 |
| **ratio ATV/CP2K** | **~1.25×** | ≤2× closed; chase ~1× via integrate residual |

Landed: collint residual (hybrid xsimd FMA, cab expand, TLS sphere); STVnl
gen-stamp Hermite + s–s + fused `prim_ST` (~1500→520→**435** ms); products_mg
geometry-batch (**~285→~18 ms**); shell-level `CollintTask` list (MAIN
**~1.9→~1.6 s**; coll **~63→~30**, int **~76→~58**); OT g·e DIIS + OUTER_SCF
polish. Remaining residual: **integrate kernel** toward CP2K ~20 ms (coll
secondary ~1.5×).

### Product collint (landed)

Files: `internal/product_collint.{h,cpp}` + wire in `multigrid.cpp`.

| Piece | Status |
| --- | --- |
| cab→cxyz expand (per-axis binomial about pc) | landed |
| pol[dir][l][i] + exp recurrence | landed |
| z-fast contract + portable xsimd (+ FMA / 4-wide residual) | landed |
| grow-only thread_local scratch (no per-product heap) | landed |
| geometry sort + group batch (shared pol / monom I) | landed |
| sphere-only Fz contract (skip cube corners) | landed |
| reflection pairs `(k,1−k)` | landed |
| CP2K sphere-bounds cache (+ TLS multi-imr) | landed |
| integrate group-idx + segment SIMD | landed |
| cab-style alpha table for large geometry groups | landed |
| lp_z specialize through 9 (GRID_MAX_LP_OPTIMIZED) | landed |
| products_mg geometry-batch build_products | **landed** (wave 5) |
| shell-level task list (`CollintTask` once in build_products) | **landed** (wave 6) |
| integrate residual toward CP2K ~20 ms | **open** |

Algorithm after CP2K `src/grid/cpu/grid_cpu_collint.h` (BSD-3, Ole Schuett).
Public API / Frame laws unchanged. SIMD: portable **xsimd only**.

### SIMD policy

Portable **xsimd only** (no arch intrinsics, no global `XTENSOR_USE_XSIMD`).
Landed: z-run axpy/dot; 1D poly/E2 (legacy collocate_one); collint z-contract
with FMA + 4-wide residual on wide hosts (gated so NEON/aarch64 still builds);
`reduce_add` on dots. Exp recurrence stays scalar.

### Critical bottlenecks (re-ranked post wave-6)

| Rank | Issue | Where |
| --- | ---: | --- |
| **1** | integrate still ~3× CP2K per-iter (~58 vs ~19 ms) | `product_collint.cpp` |
| 2 | collocate ~1.5× CP2K (~30 vs ~18 ms) | `product_collint.cpp` |
| 3 | MG MAIN overall **~1.25×** (chase ~1×) | MAIN twin |
| 4 | OT outer residual **~1.4×** UKS O₂ (≤3× aim met) | `ot.cpp` / `gpw_solver.cpp` polish |
| 5 | Multi-k dense χ^k until products default | `kpoints.cpp` |

**Done (do not re-open as primary):** GpwWorkspace; `ks_grid_build_*`; OT
economics + **g·e DIIS outer** + **OUTER_SCF polish**; |V(pc)| windows; 1D poly
SIMD; **first collint + geometry batch + residual polish**; spectral FFT;
STVnl Hermite key reuse; **products_mg geometry-batch setup**; **shell-level
collint tasks**.

### Extract / rewrite order

1. ~~`GpwWorkspace`~~ landed  
2. ~~`KsGridBuild`~~ landed  
3. ~~CP2K collint product path (cxyz + pol + geometry batch + residual)~~ **landed**  
4. ~~OT g·e DIIS outer + OUTER_SCF polish~~ **landed**  
5. ~~products_mg setup geometry-batch~~ **landed**  
6. ~~shell-level collint tasks~~ **landed**  
7. **Active:** integrate residual toward ~1× MG  
8. products-default-k after (7) closer to parity; MPI product/grid distribute

### CUDA GPW foothold (landed, not product)

Thin Γ device slices under `src/cuda/gpw/internal/`:

| Slice | Files | Oracle |
| --- | --- | --- |
| Poisson/FFT | `poisson_fft.{cu,cuh}` | `cpu::gpw::solve_poisson` |
| Padé LDA XC | `xc_lda.{cu,cuh}` | `cpu::xc::pade_lsd` / `xc_grid_pade` |
| product ρ collocate | `product_collocate.{cu,cuh}` | `collint_collocate_product` continuous + collint |
| product V·shape integrate | `product_integrate.{cu,cuh}` | `collint_integrate_product` continuous + collint |

Unit oracle tols stated; **not** a full GPW method product. Full CUDA GPW
remains blocked pending architecture freeze.

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`; arch
intrinsics; weakening SCF tol; 5th collocation layout without deleting dense
χ^k; file-split without shared KS build.
