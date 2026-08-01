<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture (binding conclusions)

Capability-first waves left **four nearly complete SCF copies** (Γ/k × RKS/UKS)
in `src/cpu/gpw/gpw_solver.cpp` (~5.8k). Physics pipeline is one algorithm;
control flow is forked. Correctness is good; wall loses to structure +
allocation.

### One pipeline

```
P / {P_k} → collocate ρ → Poisson → XC → integrate V → H → DIIS|OT → P'
```

Geometry-fixed once per `run()`: S/T/Vpp or S_k/T_k/Vnl_k, n_ion, v_locsr,
cbg / ProductSet / KpointBasisGrid.

### Critical bottlenecks

| Rank | Issue | Where |
| --- | --- | --- |
| 1 | Multi-k dense χ^k O(npts·nb²·nk) | `kpoints.cpp` |
| 2 | MG product scatter + full-grid chunk temps + per-product heap | `multigrid.cpp` |
| 3 | FFTW plan thrash every Poisson/GGA/prolong | `poisson_fft.cpp`, `grid_fft.cpp` |
| 4 | OT multiplies full KS rebuilds; k-OT double energy | `gpw_solver.cpp` |
| 5 | Hand complex matmul (no zgemm) | `kpoints.cpp` |

### Extract first

1. `GpwWorkspace` — plans, grids, DIIS, scratch (like `PWWorkspace`)
2. `KsGridBuild` — one ρ→Poisson→XC→V_eff
3. Unified product collocate/integrate
4. Parameterized OT step

### Phases

| Phase | Work |
| --- | --- |
| **P0** | Profile; FFT plan cache; buffer reuse; zgemm; fix k-OT double energy |
| **P1** | Workspace + KsGridBuild; products-default multi-k; OT economics |
| **P2** | SIMD products; force reuse; MPI product/grid distribute |

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`;
weakening SCF tol; 5th collocation layout without deleting dense χ^k;
file-split without shared KS build.

### Disk

`gpw-perf-loop` **always** wipes builds after worker/merge/clearup.

Queue: `docs/developer/gpw-perf-queue.md`. Workflow: `.grok/workflows/gpw-perf-loop.rhai`.
