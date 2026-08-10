<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture (binding conclusions)

Physics pipeline is one algorithm; SCF control flow is still forked (Γ/k ×
RKS/UKS) but **workspace + shared fine-grid KS build are landed**. Product
collocate/integrate now uses a **CP2K-style collint engine** with geometry
batching + shell-level tasks; fair wall vs CP2K is **2.33×** MAIN per
[`gpw-perf-baseline.md`](gpw-perf-baseline.md) (the ~1.25× figure was
~30–47× pre-collint; ~1.5× post products_mg). Integrate residual wave-7 cut
int **~58→~45** ms (still ~2.4× CP2K ~19 ms class).

### One pipeline

```
P / {P_k} → collocate ρ → Poisson → XC → integrate V → H → DIIS|OT → P'
```

Geometry-fixed once per `run()`: S/T/Vpp or S_k/T_k/Vnl_k, n_ion, v_locsr,
cbg / ProductSet / KpointBasisGrid. Iteration bags live on
`internal::GpwWorkspace`; fine-grid ρ→V_eff is `ks_grid_build_rks/uks`.

### Fair wall vs CP2K (2026-08-06 wave-7, arrhenius1, OMP=4 bind)

Twin `gpw-multigrid-kpoints` MAIN (MP 2×2×2, 150 Ry, NGRIDS 4):

| Side | Wall [s] | Notes |
| --- | ---: | --- |
| ATV MAIN | **~1.6** | post integrate residual (coll ~29 / int **~45** ms) |
| CP2K 2026.1 same host | **~1.28** quiet | re-run 2026-08-04 |
| **ratio ATV/CP2K** | **2.33×** | per gpw-perf-baseline.md re-measure; chase ~1× |

Landed: collint residual (hybrid xsimd FMA, cab expand, TLS sphere); STVnl
gen-stamp Hermite + s–s + fused `prim_ST` (~1500→520→**435** ms); products_mg
geometry-batch (**~285→~18 ms**); shell-level `CollintTask` list (MAIN
**~1.9→~1.6 s**; coll **~63→~30**, int **~76→~58**); integrate residual
(width-4 accumulate, Ltot skip, short segments; int **~58→~45**); OT g·e DIIS
+ OUTER_SCF polish. Remaining residual: **integrate still ~2.4× CP2K** (~45 vs
~19 ms; coll secondary ~1.5×).

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
| integrate residual (width-4 / Ltot / short seg) | **landed** (wave 7) |
| further int residual toward CP2K ~20 ms | open after MAIN path single-owner (`arch-mg-main-structural`); not pin-safe micro-opt |
| multi-level MAIN product composition | **landed** (`arch-mg-main-structural`) — `multigrid_collocate_rho*` / `multigrid_integrate_veff*` single owner for Γ+multi-k; wall residual still open |
| pure-s + mixed-angular 2-center S/T xsimd | **landed** (slab-class; not fair-wall lever) |

Algorithm after CP2K `src/grid/cpu/grid_cpu_collint.h` (BSD-3, Ole Schuett).
Public API / Frame laws unchanged. SIMD: portable **xsimd only**.

### SIMD policy

Portable **xsimd only** (no arch intrinsics, no global `XTENSOR_USE_XSIMD`).
Landed: z-run axpy/dot; 1D poly/E2 (legacy collocate_one); collint z-contract
with FMA + 4-wide residual on wide hosts (gated so NEON/aarch64 still builds);
**integrate prefers width-4 segment accumulate on AVX-512** (avoids ZMM spill);
`reduce_add` on dots. Exp recurrence stays scalar.

### Critical bottlenecks (re-ranked post wave-7)

| Rank | Issue | Where |
| --- | ---: | --- |
| **1** | integrate still ~2.4× CP2K per-iter (~45 vs ~19 ms) | `product_collint.cpp` |
| 2 | collocate ~1.5× CP2K (~29 vs ~18 ms) | `product_collint.cpp` |
| 3 | MG MAIN overall **2.33×** (chase ~1×) | MAIN twin |
| 4 | OT outer residual **~1.4×** UKS O₂ (≤3× aim met) | `ot.cpp` / `gpw_solver.cpp` polish |
| 5 | Multi-k dense χ^k until products default | `kpoints.cpp` |

**Done (do not re-open as primary):** GpwWorkspace; `ks_grid_build_*`; OT
economics + **g·e DIIS outer** + **OUTER_SCF polish**; |V(pc)| windows; 1D poly
SIMD; **first collint + geometry batch + residual polish**; spectral FFT;
STVnl Hermite key reuse; **products_mg geometry-batch setup**; **shell-level
collint tasks**; **integrate residual (wave 7)**.

### Extract / rewrite order

1. ~~`GpwWorkspace`~~ landed  
2. ~~`KsGridBuild`~~ landed  
3. ~~CP2K collint product path (cxyz + pol + geometry batch + residual)~~ **landed**  
4. ~~OT g·e DIIS outer + OUTER_SCF polish~~ **landed**  
5. ~~products_mg setup geometry-batch~~ **landed**  
6. ~~shell-level collint tasks~~ **landed**  
7. ~~integrate residual~~ **landed** (wave 7; thin kernel residual-2 **exhausted** wave-8)  
8. ~~MG MAIN multi-level product composition single-owner~~ **landed**
   (`arch-mg-main-structural`: `multigrid_collocate_rho*` /
   `multigrid_integrate_veff*` — forked solver lambdas deleted; wall still
   ~2.33× until non-pin-safe sphere/window or deeper task rewrite)  
9. products-default-k fresh A/B only if MAIN near parity; then MPI product/grid  
10. **now (wall residual):** integrate/collocate work reduction on the shared
    MAIN path only — no second forked loop.

### CUDA GPW foothold + LDA stage (landed, not product)

Thin Γ device slices under `src/cuda/gpw/internal/`:

| Slice | Files | Oracle |
| --- | --- | --- |
| Poisson/FFT | `poisson_fft.{cu,cuh}` | `cpu::gpw::solve_poisson` |
| Padé LDA XC | `xc_lda.{cu,cuh}` | `cpu::xc::pade_lsd` / `xc_grid_pade` |
| product ρ collocate | `product_collocate.{cu,cuh}` | `collint_collocate_product` continuous + collint |
| product V·shape integrate | `product_integrate.{cu,cuh}` | `collint_integrate_product` continuous + collint |
| composed KS chain (LDA + closed-shell PBE) | `ks_chain.{cu,cuh}` | `ks_grid_build_rks` + continuous collint (LDA GH200 **1049858**; GGA **1050680**) |
| multi-grid product KS stage | `multigrid_transfer.{cu,cuh}` + `KsMultiGridChainDevice` | CPU Γ MG fragment dual-gate (GH200 **1051042** max\|ΔV_eff\|=8.1e-10) |

Unit/stage oracle tols stated — **no** public `GpwDft`. Public shape for the
full method product is **frozen** to equal CPU `GpwDft` — see
[cuda-gpw-public-shape.md](cuda-gpw-public-shape.md). Stage units stay
`internal/` and are composed under that product; OT residual does not block.
`cuda-gpw-full-product` is staffable under that contract (freeze green on
`dev` @ `0147fe02`; still a product implement, not auto-landed).

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`; arch
intrinsics; weakening SCF tol; 5th collocation layout without deleting dense
χ^k; file-split without shared KS build.
