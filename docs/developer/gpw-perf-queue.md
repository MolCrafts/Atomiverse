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

Wave-1 PERF-CENSUS re-diff (2026-08-01 on `dev` @ `3de007ef`): re-read
`cpu/gpw` hotspots + baseline table. Architecture ranks still hold; measured
wall order is **MG product integrate ≫ Γ UKS integrate/collocate ≫ dense χ^k
≪ OT rebuild multiplicity ≫ Poisson/FFT on current twins**. `p0-profile-baseline`
done; `p0-fft-plan-cache` stays `[~]` (unmerged tip). No new capability slugs.
Order: P0 cheap thrash/double-work (by measured impact) → P1 structure → P2.

### P0 — thrash / double work (measured first)

- [ ] `p0-collocate-buffers` | **#1 measured wall** (`gpw-multigrid-kpoints` integrate ~7.1 s/KS = 87% of rebuild). Preallocate product/collocate/integrate scratch: kill per-call `std::vector` in `collocate_one` (~L653–662 `fbuf`/`ebuf`/`wbuf`); reuse fixed-chunk full-grid scatter partials across SCF iters (`collocate_products` reallocs `tl[nc]` every call ~L794–798; `integrate_products` same ~L853–856; `*_cells` twins); Γ CSR path `integrate_potential` allocates `part[nc*nn]` every call (`collocate.cpp` ~L283) | phase P0 | files: `multigrid.cpp` (`collocate_one`, `collocate_products*`, `integrate_products*`), `collocate.cpp` | gate: twins green; MG twin MAIN wall ↓ vs 65.6 s baseline |
- [ ] `p0-ot-double-energy` | Multi-k OT outer loop rebuilds KS **twice** per iter (still present): RKS `(void)build_k_energy` then `build_k_energy` after `refresh_occupations` (~L1926/1931); UKS twin `build_uks_k_energy` ×2 (~L3407/3417). Baseline OT water ~2.3 full KS builds/outer (59 rebuilds / 25 iters). Collapse to one ρ→H when dens/occ fused (or skip first when occupations frozen / integer) | phase P0 | files: `gpw_solver.cpp` multi-k OT RKS + UKS | gate: OT regs green (`ot-kpoints`, `ot-uks-kpoints`, `ot-minimizer-water`); fewer energy builds / wall ↓ |
- [ ] `p0-k-zgemm` | Hand complex matmul still O(n³) triple loops (no zgemm): `kpoints.cpp` anonymous `mul`/`dagger_mul` (~L638–672; used by `diagonalize_k` + `k_diis_error`) **and** `linalg::c_mul` / `c_dagger_mul` in `orbital_tangent.cpp` (~L51–82; OT `band_gradient_k` / `s_tangent_k` / param — every OT step). Replace with BLAS zgemm (or thin OpenBLAS wrappers; row-major care) | phase P0 | files: `kpoints.cpp`, `orbital_tangent.cpp` (+ header if needed) | gate: multi-k + OT regs green; no tol soften |

### P1 — one pipeline + products path + OT economics

- [ ] `p1-products-default-k` | Production multi-k collocation via products/MG; dense χ^k (`KpointBasisGrid`, O(npts·nb²·nk)) transitional / opt-out only. Twin `gpw-kpoints-si` MAIN still single-grid dense (`use_mg = ngrids>1 && rel_cutoff>0` only; kbg_build ~1.3 s setup + ~100 ms/step collocate). Do **not** add a 5th collocation layout without deleting dense χ^k | phase P1 | files: `kpoints.cpp`, solver k-path (`gpw_solver.cpp` ~L1091/1222), public multigrid defaults if needed (ctor/members only — no *Config) | gate: k twin wall large ↓; pins hold |
- [ ] `p1-ot-convergence` | After double-energy fix: OT line-search / precond economics — fewer full `build_*_energy` in Armijo (baseline sum `ls_rebuilds`=33 / 25 iters); fixed-H inner steps where safe; align with CP2K OUTER_SCF spirit already sketched for OT+U | phase P1 | files: `ot.cpp`, `gpw_solver.cpp` OT | gate: `ot-minimizer-water` wall ↓ vs ~10.8 s OT-only baseline; energy tol hold |
- [ ] `p1-gpw-workspace` | Extract private `GpwWorkspace` (grid, cbg/kbg/products, FFT plans, DIIS/KDiis history, collocate scratch, last_veff) — mirror `PWWorkspace`; kill four nearly complete SCF copies' local bags without public API. `gpw_solver.cpp` still ~5.9k forked control flow | phase P1 | files: `gpw_solver.*`, new `internal/gpw_workspace.h` | gate: no public API / no *Config; regs green; no wall regression on four twins |
- [ ] `p1-ks-grid-build` | Single internal `KsGridBuild` (ρ→Poisson→XC→V_eff [+ nonlocal vdW]) shared by Γ/k × RKS/UKS DIIS+OT; today `build_k_energy` (~L1563) / `build_uks_k_energy` (~L2906) / Γ lambdas are forked copies | phase P1 | files: `gpw_solver.cpp` (+ workspace) | gate: energy pins within existing tol |

### P2 — SIMD, force reuse, MPI grids

- [ ] `p2-product-simd` | SIMD / better screening on separable `collocate_one` outer-product fill (~L715–754; comment ~L623 says vectorizable; no xsimd yet). Respect iron law: no global `XTENSOR_USE_XSIMD` | phase P2 | files: `multigrid.cpp` | gate: MG twin wall ↓ |
- [ ] `p2-force-grid-reuse` | Forces rebuild full `kbg` + `kbgd` (and Γ `cbg`/`cbgd`) + Poisson/XC every call (`gpw_forces.cpp` ~L1129–1244); reuse SCF products/cbg/V_eff / last density path where geometry-fixed | phase P2 | files: `gpw_forces.cpp`, solver result/workspace handoff | gate: force regs green + force wall ↓ |
- [ ] `p2-mpi-product-grid` | Architecture P2: product/grid distribute. Today only k RR + Allreduce density/traces; setup STVpp, products, grids, Poisson replicated on every rank (foothold comments ~L969–1010). Split product scatter / grid FFT across ranks without changing Frame API | phase P2 | files: `gpw_solver.cpp`, `multigrid.cpp`, MPI layout | gate: `mpi-gpw-scaling-4rank` + multi-k regs; scaling wall ↓ |

## in_progress

- [~] `p0-fft-plan-cache` | Cache FFTW plans + r/c buffers by grid shape; create once per geometry (mirror `cpu::pw::internal::FFTPlan3D` / `FFTWBuffer`). Sites: `solve_poisson` / `solve_poisson_q`, GGA `gradient`/`divergence`, MG `fwd_dft`/`inv_dft_real`, `mt_poisson` / `vdw_df` / `rvv10` thrash | phase P0 | files: `poisson_fft.cpp`, `grid_fft.cpp`, `multigrid.cpp`, `mt_poisson.cpp`, new `internal/fft_cache.{h,cpp}` | gate: twins green; FFT-heavy step wall ↓; no public API | resume `wf/perf-p0-fft-plan-cache` @ `af9e8c82` (feat tip; not merged — do not demote) |

## blocked

- [!] `global-xtensor-xsimd` | blocked — never enable global `XTENSOR_USE_XSIMD` (breaks reshape elsewhere)

## done

- [x] `p0-profile-baseline` | 2026-08-01 | fair-wall + ATV_GPW_PROF table in `.claude/notes/gpw-perf-baseline.md`; thin k/uks prof marks; four twins PASS; unit `test_cpu_(gpw|xc)_` 12/12 |
