# GPW performance optimization queue

Worklist for workflow `gpw-perf-loop`. Optimize **existing** CPU GPW code —
correctness pins must stay green; fair wall vs CP2K (matched algorithm) is the
speed gate. Do not invent capability gaps here (use `cp2k-method-loop`).

**Disk law (binding):** every worker, merge, and clearup **wipes** `build/`,
`build-*`, `CMakeCache.txt`, `CMakeFiles/`, `Testing/`, `.cache/`, and
`/tmp/atv-*` temps on the tree it touched. Never leave multi‑GB builds on
shared nobackup. Never delete source / regressions / specs / `.git`.

Architecture conclusions: [`.claude/notes/gpw-perf-architecture.md`](../../.claude/notes/gpw-perf-architecture.md).

## Fair baseline twins (1 MPI × OMP=4 unless noted)

| Twin | Role |
| --- | --- |
| `gpw-uks-o2-gamma` | Γ UKS single-grid SCF (~1.5× CP2K today) |
| `gpw-kpoints-si` | multi-k RKS (dense χ^k path) |
| `gpw-multigrid-kpoints` | multi-k MG products (worst wall class) |
| `ot-minimizer-water` | OT path (iter economics) |

Wall = **same single twin SCF job**, not multi-gate reg wall.

## pending

Seeded P0 → P1 → P2. Census may re-order; do not erase done/blocked.

- [ ] `p0-fft-plan-cache` | Cache FFTW plans (Poisson, prolong/restrict, GGA) keyed by grid shape; init once per run | phase P0 | files: `poisson_fft.cpp`, `grid_fft.cpp`, multigrid prolong | gate: twins green; FFT-heavy step wall ↓ |
- [ ] `p0-collocate-buffers` | Preallocate collocate/integrate/product scratch; kill per-product heap in `collocate_one`; reuse chunk partials | phase P0 | files: `collocate.cpp`, `multigrid.cpp` | gate: twins green; product path wall ↓ |
- [ ] `p0-k-zgemm` | Replace hand complex `mul`/`dagger_mul` with BLAS zgemm (or `cpu::linalg`) | phase P0 | files: `kpoints.cpp` | gate: multi-k regs green |
- [ ] `p0-ot-double-energy` | k-OT: one `build_k_energy` per outer iter (remove double rebuild) | phase P0 | files: `gpw_solver.cpp` OT loops | gate: OT regs green; fewer energy builds |
- [ ] `p0-profile-baseline` | Instrument / document ATV_GPW_PROF breakdown on the four twins; write numbers into queue note or handoff | phase P0 | files: notes / handoff | gate: profile table committed |
- [ ] `p1-gpw-workspace` | Extract `GpwWorkspace` (grid, cbg/kbg/products, plans, DIIS history, scratch) — mirror PW workspace | phase P1 | files: `gpw_solver.*`, new internal header | gate: no public API change; regs green |
- [ ] `p1-ks-grid-build` | Single `KsGridBuild` (ρ→Poisson→XC→V_eff) shared by Γ/k × RKS/UKS | phase P1 | files: `gpw_solver.cpp` | gate: energy pins bit-stable within tol |
- [ ] `p1-products-default-k` | Multi-k production collocation via products/MG; dense χ^k transitional only | phase P1 | files: `kpoints.cpp`, solver k-path | gate: k twin wall large ↓; pins hold |
- [ ] `p1-ot-convergence` | OT line-search / precond economics (fewer full rebuilds; align with CP2K OUTER_SCF spirit) | phase P1 | files: `ot.cpp`, solver OT | gate: OT twin wall ↓; energy tol hold |
- [ ] `p2-product-simd` | SIMD / better screening on product collocate_one | phase P2 | files: `multigrid.cpp` | gate: MG twin wall ↓ |
- [ ] `p2-force-grid-reuse` | Forces reuse SCF cbg/products/V_eff; no second kbg build | phase P2 | files: `gpw_forces.cpp` | gate: force regs + wall |

## in_progress

(empty)

## blocked

- [!] `global-xtensor-xsimd` | blocked — never enable global `XTENSOR_USE_XSIMD` (breaks reshape elsewhere)

## done

(empty — harness moves rows here)
