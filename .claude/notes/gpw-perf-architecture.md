<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture (binding conclusions)

Physics pipeline is one algorithm; SCF control flow is still forked (Γ/k ×
RKS/UKS) but **workspace + shared fine-grid KS build are landed**. Product
collocate/integrate now uses a **CP2K-style collint engine** with geometry
batching; fair wall vs CP2K is **~1.8×** MAIN (was ~30–47× pre-collint).

### One pipeline

```
P / {P_k} → collocate ρ → Poisson → XC → integrate V → H → DIIS|OT → P'
```

Geometry-fixed once per `run()`: S/T/Vpp or S_k/T_k/Vnl_k, n_ion, v_locsr,
cbg / ProductSet / KpointBasisGrid. Iteration bags live on
`internal::GpwWorkspace`; fine-grid ρ→V_eff is `ks_grid_build_rks/uks`.

### Fair wall vs CP2K (2026-08-06 wave-4, arrhenius1, OMP=4 bind)

Twin `gpw-multigrid-kpoints` MAIN (MP 2×2×2, 150 Ry, NGRIDS 4):

| Side | Wall [s] | Notes |
| --- | ---: | --- |
| ATV MAIN remeasure | **~2.4** | post collint residual (coll ~63 / int ~76 ms) |
| CP2K 2026.1 same host | **~1.28** quiet | re-run 2026-08-04 |
| **ratio ATV/CP2K** | **~1.9×** | ≤2× gate closed; chase ~1× via setup |

Landed: collint residual (hybrid xsimd FMA, cab expand, TLS sphere); STVnl
gen-stamp Hermite + s–s + fused `prim_ST` (~1500→520→**435** ms); OT g·e DIIS
+ OUTER_SCF polish. Remaining residual: products_mg setup + shell-level tasks
(coll/int still ~3–4× CP2K per-iter class).

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
| shell-level task list beyond cartesian-prim groups | **open** |

Algorithm after CP2K `src/grid/cpu/grid_cpu_collint.h` (BSD-3, Ole Schuett).
Public API / Frame laws unchanged. SIMD: portable **xsimd only**.

### SIMD policy

Portable **xsimd only** (no arch intrinsics, no global `XTENSOR_USE_XSIMD`).
Landed: z-run axpy/dot; 1D poly/E2 (legacy collocate_one); collint z-contract
with FMA + 4-wide residual on wide hosts; `reduce_add` on dots. Exp recurrence
stays scalar.

### Critical bottlenecks (re-ranked post wave-4)

| Rank | Issue | Where |
| --- | ---: | --- |
| **1** | Residual setup products_mg ~0.3 s (+ STVnl ~0.44 s) + collint still ~3–4× CP2K per-iter | `multigrid.cpp` + `product_collint.cpp` |
| 2 | MG MAIN overall **~1.9×** (≤2× closed; chase ~1×) | MAIN twin |
| 3 | OT outer residual **~1.4×** UKS O₂ (≤3× aim met) | `ot.cpp` / `gpw_solver.cpp` polish |
| 4 | Multi-k dense χ^k until products default | `kpoints.cpp` |
| 5 | Force path rebuilds grids | `gpw_forces.cpp` |

**Done (do not re-open as primary):** GpwWorkspace; `ks_grid_build_*`; OT
economics + **g·e DIIS outer** + **OUTER_SCF polish**; |V(pc)| windows; 1D poly
SIMD; **first collint + geometry batch + residual polish**; spectral FFT;
STVnl Hermite key reuse.

### Extract / rewrite order

1. ~~`GpwWorkspace`~~ landed  
2. ~~`KsGridBuild`~~ landed  
3. ~~CP2K collint product path (cxyz + pol + geometry batch + residual)~~ **landed**  
4. ~~OT g·e DIIS outer + OUTER_SCF polish~~ **landed**  
5. **Active:** products_mg setup + shell-level collint tasks toward ~1× MG  
6. Force grid reuse; MPI product/grid distribute; products-default-k after (5)

### CUDA GPW foothold (landed, not product)

Thin Γ device Poisson/FFT under `src/cuda/gpw/internal/poisson_fft.{cu,cuh}`
mirrors `cpu::gpw::solve_poisson` (cuFFT r2c → 4π/|G|² → c2r + Hartree). Padé
LDA XC grid under `src/cuda/gpw/internal/xc_lda.{cu,cuh}` mirrors
`cpu::xc::pade_lsd` / `xc_grid_pade`. Unit oracle tols stated; **not** a full
GPW method product. Full CUDA GPW remains blocked pending architecture freeze.

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`; arch
intrinsics; weakening SCF tol; 5th collocation layout without deleting dense
χ^k; file-split without shared KS build.

### Disk

Wipe `build` / `build-*` after every cmake/ctest. No multi-GB leftovers.
