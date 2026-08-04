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
| ATV best MAIN | **~9.8** | two-shot stable OMP=4 bind |
| CP2K 2026.1 same host | **0.885** | `tools/cp2k_oracle/inputs/gpw-multigrid-kpoints/si_kpoints_mg.inp` |
| **ratio ATV/CP2K** | **~11×** | was ~47× pre-collint, ~30× mid-batch |

Collint + geometry monom batch + sphere-only Fz contract. CP2K grid_collocate /
grid_integrate ~0.018–0.019 s/iter. Stale: pre-collint ~42 s / 47×.

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
| **1** | Product collint still ~11× vs CP2K (reflection / sphere cache / tasks) | `product_collint.cpp` |
| 2 | Multi-k dense χ^k O(npts·nb²·nk) until product engine wins fully | `kpoints.cpp` |
| 3 | Force path rebuilds grids (geometry-fixed reuse incomplete) | `gpw_forces.cpp` |
| 4 | Forked SCF control flow (structure only; KS build shared) | `gpw_solver.cpp` |

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
