<!-- mol:note:topic:gpw-perf-architecture -->
## GPW performance architecture

Walls and provenance: [gpw-perf-baseline.md](gpw-perf-baseline.md).

### One pipeline

```
P / {P_k} → collocate ρ → Poisson → XC → integrate V → H → DIIS|OT → P'
```

- Geometry-fixed once per `run()`: S/T/Vpp or S_k/T_k/Vnl_k, n_ion, v_locsr,
  cbg / ProductSet / KpointBasisGrid.
- Iteration state lives on `internal::GpwWorkspace`; fine-grid ρ→V_eff is
  `ks_grid_build_rks/uks`; multi-level MAIN collocate/integrate is
  `multigrid_collocate_rho*` / `multigrid_integrate_veff*` (single owner for
  Γ and multi-k — no forked solver lambdas).

### Product collint

`internal/product_collint.{h,cpp}` wired from `multigrid.cpp`, after CP2K
`src/grid/cpu/grid_cpu_collint.h` (BSD-3, Ole Schuett): cab→cxyz expansion
about pc, pol[dir][l][i] with exp recurrence, z-fast contract, grow-only
thread-local scratch, geometry sort + group batch, sphere-only Fz contract,
reflection pairs, CP2K sphere-bounds cache, group-idx integrate with segment
SIMD, cab-style alpha table, lp_z specialized through 9, geometry-batched
`build_products`, shell-level `CollintTask` list.

### SIMD policy

Portable xsimd only. z-run axpy/dot; collint z-contract with FMA + 4-wide
residual on wide hosts (NEON still builds); integrate prefers width-4 segment
accumulation on AVX-512 (avoids ZMM spill); `reduce_add` on dots; the exp
recurrence stays scalar.

### Bottlenecks (MG MAIN class)

| Rank | Issue | Where |
| --- | --- | --- |
| 1 | integrate ~2.4–2.8× CP2K per call | `product_collint.cpp` |
| 2 | collocate ~1.5–2× CP2K per call | `product_collint.cpp` |
| 3 | OT outer residual (UKS O₂) | `ot.cpp` / `gpw_solver.cpp` |
| 4 | multi-k dense χ^k until products are default | `kpoints.cpp` |

Pin-safe kernel levers are exhausted; the next cut is work reduction on the
shared MAIN path (`mg-main-structural`). Products-default-k A/B only once MAIN
is near parity; then MPI product/grid.

### Forbidden

Public `*Config`; e2e/timing in `tests/`; global `XTENSOR_USE_XSIMD`; arch
intrinsics; weakening SCF tol; a fifth collocation layout without deleting
dense χ^k; a second forked loop beside the shared MAIN path.
