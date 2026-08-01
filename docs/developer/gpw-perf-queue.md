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

Wave-1 PERF-CENSUS (2026-08-01 on `dev`): re-diffed `cpu/gpw` hotspots.
Architecture ranks still hold. Baseline table landed (`p0-profile-baseline`).
Order: P0 (measure + cheap wins) → P1 (structure) → P2 (SIMD / reuse / MPI).

### P0 — profile + thrash / double work

- [ ] `p0-fft-plan-cache` | Cache FFTW plans + r/c buffers by grid shape; create once per geometry (mirror `cpu::pw::internal::FFTPlan3D` / `FFTWBuffer`). Today every call `fftw_plan_*` + `fftw_destroy_plan` + malloc: `solve_poisson` / `solve_poisson_q`, GGA `gradient`/`divergence`, MG `fwd_dft`/`inv_dft_real` (prolong/restrict), also `mt_poisson` / `vdw_df` / `rvv10` thrash | phase P0 | files: `poisson_fft.cpp`, `grid_fft.cpp`, `multigrid.cpp`, optionally `mt_poisson.cpp` | gate: twins green; FFT-heavy step wall ↓; no public API |
- [ ] `p0-collocate-buffers` | Preallocate product/collocate/integrate scratch: kill per-call `std::vector` in `collocate_one` (fbuf/ebuf/wbuf); reuse fixed-chunk full-grid scatter partials across SCF iters (today `collocate_products` reallocs `tl[nc]` every call); same for integrate chunk nb² accums | phase P0 | files: `multigrid.cpp` (`collocate_one`, `collocate_products*`, `integrate_products*`), `collocate.cpp` | gate: twins green; MG/product path wall ↓ |
- [ ] `p0-k-zgemm` | Replace hand complex matmul with BLAS zgemm (or thin OpenBLAS wrappers). Sites: `kpoints.cpp` anonymous `mul`/`dagger_mul` (diag + DIIS error path) **and** `linalg::c_mul` / `c_dagger_mul` in `orbital_tangent.cpp` (OT band_gradient / s_tangent / param — every OT step) | phase P0 | files: `kpoints.cpp`, `orbital_tangent.cpp` (+ header if needed) | gate: multi-k + OT regs green; no tol soften |
- [ ] `p0-ot-double-energy` | Multi-k OT outer loop rebuilds KS **twice** per iter: RKS `(void)build_k_energy` then `build_k_energy` after `refresh_occupations` (~L1883/1888); UKS twin `build_uks_k_energy` ×2 (~L3364/3374). Collapse to one ρ→H rebuild when dens/occ update is fused (or skip first when occupations frozen / integer) | phase P0 | files: `gpw_solver.cpp` multi-k OT RKS + UKS | gate: OT regs green (`ot-kpoints`, `ot-uks-kpoints`, `ot-minimizer-water`); fewer energy builds / wall ↓ |

### P1 — one pipeline + products path + OT economics

- [ ] `p1-gpw-workspace` | Extract private `GpwWorkspace` (grid, cbg/kbg/products, FFT plans, DIIS/KDiis history, collocate scratch, last_veff) — mirror `PWWorkspace`; kill four nearly complete SCF copies' local bags without public API | phase P1 | files: `gpw_solver.*`, new `internal/gpw_workspace.h` | gate: no public API / no *Config; regs green; no wall regression on four twins |
- [ ] `p1-ks-grid-build` | Single internal `KsGridBuild` (ρ→Poisson→XC→V_eff [+ nonlocal vdW]) shared by Γ/k × RKS/UKS DIIS+OT; today `build_k_energy` / `build_uks_k_energy` / Γ lambdas are forked copies in ~5.8k `gpw_solver.cpp` | phase P1 | files: `gpw_solver.cpp` (+ workspace) | gate: energy pins within existing tol |
- [ ] `p1-products-default-k` | Production multi-k collocation via products/MG; dense χ^k (`KpointBasisGrid`, O(npts·nb²·nk)) transitional / opt-out only. Twin `gpw-kpoints-si` still single-grid dense (no `rel_cutoff`); MG only when `ngrids>1 && rel_cutoff>0`. Do **not** add a 5th collocation layout without deleting dense χ^k | phase P1 | files: `kpoints.cpp`, solver k-path, public multigrid defaults if needed (ctor/members only — no *Config) | gate: k twin wall large ↓; pins hold |
- [ ] `p1-ot-convergence` | After double-energy fix: OT line-search / precond economics — fewer full `build_*_energy` in Armijo; fixed-H inner steps where safe; align with CP2K OUTER_SCF spirit already sketched for OT+U | phase P1 | files: `ot.cpp`, `gpw_solver.cpp` OT | gate: `ot-minimizer-water` wall ↓; energy tol hold |

### P2 — SIMD, force reuse, MPI grids

- [ ] `p2-product-simd` | SIMD / better screening on separable `collocate_one` outer-product fill (comment already says vectorizable; no xsimd yet). Respect iron law: no global `XTENSOR_USE_XSIMD` | phase P2 | files: `multigrid.cpp` | gate: MG twin wall ↓ |
- [ ] `p2-force-grid-reuse` | Forces rebuild full `kbg` + `kbgd` (and Γ `cbg`/`cbgd`) + Poisson/XC every call (`gpw_forces.cpp` ~L1129–1244); reuse SCF products/cbg/V_eff / last density path where geometry-fixed | phase P2 | files: `gpw_forces.cpp`, solver result/workspace handoff | gate: force regs green + force wall ↓ |
- [ ] `p2-mpi-product-grid` | Architecture P2: product/grid distribute. Today only k RR + Allreduce density/traces; setup STVpp, products, grids, Poisson replicated on every rank (commented foothold). Split product scatter / grid FFT across ranks without changing Frame API | phase P2 | files: `gpw_solver.cpp`, `multigrid.cpp`, MPI layout | gate: `mpi-gpw-scaling-4rank` + multi-k regs; scaling wall ↓ |

## in_progress

(empty)

## blocked

- [!] `global-xtensor-xsimd` | blocked — never enable global `XTENSOR_USE_XSIMD` (breaks reshape elsewhere)

## done

- [x] `p0-profile-baseline` | 2026-08-01 | fair-wall + ATV_GPW_PROF table in `.claude/notes/gpw-perf-baseline.md`; thin k/uks prof marks; four twins PASS; unit `test_cpu_(gpw|xc)_` 12/12 |
