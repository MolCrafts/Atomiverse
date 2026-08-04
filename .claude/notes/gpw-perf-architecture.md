<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture (binding conclusions)

Physics pipeline is one algorithm; SCF control flow is still forked (Γ/k ×
RKS/UKS) but **workspace + shared fine-grid KS build are landed**. The wall
that remains is the **product collocate/integrate constant** vs CP2K collint.

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
| ATV best MAIN | **~42** | load-sensitive (contended node 100+ s) |
| CP2K 2026.1 same host | **0.885** | `tools/cp2k_oracle/inputs/gpw-multigrid-kpoints/si_kpoints_mg.inp` |
| **ratio ATV/CP2K** | **~47×** | use this, not the stale 1.9× table |

Per-iter (ATV `ATV_GPW_PROF` MAIN): collocate ~0.9 s, **integrate ~4.2–4.5 s**,
Poisson+XC ~10 ms. CP2K `grid_integrate_task_list` ~0.019 s/iter. **Gap is
collint-class product engine, not 1D poly fill.**

Stale: wave2 CP2K wall 23.6 s / ratio 1.9 — superseded by re-measurement above.

### SIMD policy

Portable **xsimd only** (no arch intrinsics, no global `XTENSOR_USE_XSIMD`).
Landed: z-run `axis_dot`/`axis_axpy`; 1D poly/E2 fill; `reduce_add` on dots.
Exp recurrence stays scalar (sequential). Multi-monomial product batching
still open.

### Critical bottlenecks (re-ranked 2026-08-04)

| Rank | Issue | Where |
| --- | --- | --- |
| **1** | Product collocate/integrate vs CP2K collint (~47×) | `multigrid.cpp` |
| 2 | Multi-k dense χ^k O(npts·nb²·nk) until product engine wins | `kpoints.cpp` |
| 3 | Force path rebuilds grids (geometry-fixed reuse incomplete) | `gpw_forces.cpp` |
| 4 | Forked SCF control flow (structure only; KS build shared) | `gpw_solver.cpp` |

**Done (do not re-open as primary):** GpwWorkspace; `ks_grid_build_*`; OT
economics Γ RKS/UKS + multi-k integer; |V(pc)| integrate windows; 1D poly SIMD.

### Extract / rewrite order

1. ~~`GpwWorkspace`~~ landed  
2. ~~`KsGridBuild`~~ landed  
3. **CP2K collint product path** (cxyz + reflection + Horner) — **active**  
4. Force grid reuse; MPI product/grid distribute; products-default-k after (3)

### Collint strategy (user: may copy CP2K algorithm)

Reference: CP2K `src/grid/cpu/grid_cpu_collint.h` (BSD-3, Ole Schuett). Ortho path:

1. Expand `(r−ra)^{la}(r−rb)^{lb}` → coefficients of `(r−rp)^{l}` (per axis binomial).  
2. Form `cxyz[lz][ly][lx]`.  
3. Precompute `pol[dir][l][i] = (r−rp)^{l} exp(−p(r−rp)²)` with exp recurrence.  
4. Contract `cxyz → cxy → cx → grid` (Horner); process reflection pairs
   `(k,1−k)` / `(j,1−j)`.  
5. Inner x-loop: xsimd (not AVX2 intrinsics) for portable SIMD.

Integrate is the transpose of collocate. Keep public API and Frame laws.

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`; arch
intrinsics; weakening SCF tol; 5th collocation layout without deleting dense
χ^k; file-split without shared KS build.

### Disk

Perf loops **always** wipe builds after worker/merge/clearup.

Queue: `docs/developer/gpw-perf-queue.md`.
