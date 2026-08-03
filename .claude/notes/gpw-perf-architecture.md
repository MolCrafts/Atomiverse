<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture (binding conclusions)

Capability-first waves left **four nearly complete SCF copies** (Γ/k × RKS/UKS)
in `src/cpu/gpw/gpw_solver.cpp` (~5.9k). Physics pipeline is one algorithm;
control flow is forked. Correctness is good; the remaining wall is the grid
kernel constant + OT economics on the not-yet-converted paths + structure.

### One pipeline

```
P / {P_k} → collocate ρ → Poisson → XC → integrate V → H → DIIS|OT → P'
```

Geometry-fixed once per `run()`: S/T/Vpp or S_k/T_k/Vnl_k, n_ion, v_locsr,
cbg / ProductSet / KpointBasisGrid.

### SIMD policy (2026-08-03, user decision: all levers on)

SIMD is a **first-class lane**, implemented with **portable xsimd only** —
one source, per-arch codegen. Arch intrinsics stay banned. Evidence for the
choice: CP2K's `grid_cpu_collint.h` hand-writes an **AVX2+FMA-only** kernel
with a `#pragma omp simd` scalar fallback — on aarch64 CP2K itself runs the
scalar path, so single-source xsimd matches its x86 weapon and beats its
aarch64 story. First cash-in: z-run `axis_dot`/`axis_axpy` point-fill
(−30.9% on the MG twin, `327a7ca8`). Every vectorized reduction lands under
the tolerance + FD-force laws (not bit-exact, deterministic).

### Critical bottlenecks (re-ranked post 2026-08-02 waves)

| Rank | Issue | Where |
| --- | --- | --- |
| 1 | Remaining point-fill constant vs CP2K (1D pol/exp fills scalar; one run per monomial; sphere bounds sqrt per column) | `multigrid.cpp` |
| 2 | Integrate windows sized density-independent worst-case (`A.maxc·B.maxc·prod.k`, `kProductMassMargin=1e-4` ⇒ eps ~1e-11 whole-cell sweeps for diffuse pairs) | `multigrid.cpp` |
| 3 | OT line-search economics only fixed on Γ RKS; UKS Γ + multi-k OT still rebuild per Armijo step | `gpw_solver.cpp` |
| 4 | Multi-k dense χ^k O(npts·nb²·nk) — but products flip is **refuted** until the product engine is fixed (see `wf/perf-p1-products-default-k` @ `3ce71310` A/B) | `kpoints.cpp` |
| 5 | Four forked SCF copies (workspace / KsGridBuild extraction) | `gpw_solver.cpp` |

### Extract first

1. `GpwWorkspace` — plans, grids, DIIS, scratch (like `PWWorkspace`)
2. `KsGridBuild` — one ρ→Poisson→XC→V_eff
3. Unified product collocate/integrate
4. Parameterized OT step

### Phases (SIMD promoted; all levers authorized)

| Phase | Work |
| --- | --- |
| **done** | Profile; FFT plan cache; buffer reuse; zgemm; k-OT double energy; z-run point-fill SIMD; Γ RKS OT economics |
| **P1** | Deeper point-fill SIMD (1D fills, multi-monomial register blocking); |V|/P-aware integrate windows; OT economics → UKS Γ + multi-k; Workspace + KsGridBuild |
| **P2** | cxyz+reflection+Horner collocation rewrite (CP2K collint structure, dedicated wave); 2-center S/T setup SIMD; force grid reuse; MPI product/grid distribute; products-default-k (re-gated on engine fixes) |

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`; arch
intrinsics / per-arch dispatch; weakening SCF tol; 5th collocation layout
without deleting dense χ^k; file-split without shared KS build.

### Disk

The perf loops **always** wipe builds after worker/merge/clearup.

Queue: `docs/developer/gpw-perf-queue.md`. Workflows:
`.grok/workflows/cp2k-perf-loop.rhai` (CP2K-anchored rewrite, primary) and
`.grok/workflows/gpw-perf-loop.rhai` (superseded in spirit; same queue).
