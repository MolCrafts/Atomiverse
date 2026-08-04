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
| ATV best MAIN | **~2.6** | two-shot OMP=4 bind |
| CP2K 2026.1 same host | **~1.28** quiet | re-run 2026-08-04 |
| **ratio ATV/CP2K** | **~2.0×** | ≤2× gate met on quiet runs |

Landed: geometry pre-sort at `build_products`; process-wide sphere-bounds
cache; `lp_z` 0..6 specialized collocate; zero-skip monoms; max\|w\| window;
reflection + z xsimd. Per-iter collocate/integrate ~70–100 ms (was ~550/600).
Remaining headroom: STVnl_k setup ~1.3 s (once), further kernel polish.

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
| reflection pairs `(k,1−k)` | **open** |
| CP2K sphere-bounds cache | **open** |
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
| **1** | STVnl_k setup ~1.3 s (dominates remaining MAIN gap vs CP2K) | k-path setup |
| 2 | Product collint still ~2× overall (kernel polish / cab-level) | `product_collint.cpp` |
| 3 | Multi-k dense χ^k O(npts·nb²·nk) until products default | `kpoints.cpp` |
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
