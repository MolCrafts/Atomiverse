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

Post wave-2 land (2026-08-02): FFT cache + collocate buffers + multi-k OT
double-energy on `dev`. Remaining P0: zgemm. Then P1 structure / products-k /
OT Armijo economics.

### P0 — thrash / double work (measured first)

- [ ] `p0-k-zgemm` | Hand complex matmul still O(n³) triple loops (no zgemm on dev): `kpoints.cpp` anonymous `mul`/`dagger_mul` (~L638–672; `diagonalize_k` + `k_diis_error`) **and** `linalg::c_mul` / `c_dagger_mul` in `src/cpu/linalg/orbital_tangent.cpp` (~L51–82; OT `band_gradient_k` / `s_tangent_k` / param — every OT step). Replace with BLAS zgemm (or thin OpenBLAS wrappers; row-major care) | phase P0 | files: `kpoints.cpp`, `orbital_tangent.cpp` (+ header if needed) | gate: multi-k + OT regs green; no tol soften |

### P1 — one pipeline + products path + OT economics

- [ ] `p1-products-default-k` | Production multi-k collocation via products/MG; dense χ^k (`KpointBasisGrid`, O(npts·nb²·nk)) transitional / opt-out only. Twin `gpw-kpoints-si` MAIN still single-grid dense (`use_mg = ngrids>1 && rel_cutoff>0` only at ~L1091/2436/4485; kbg_build ~1.3 s setup + ~100 ms/step collocate). Do **not** add a 5th collocation layout without deleting dense χ^k | phase P1 | files: `kpoints.cpp`, solver k-path (`gpw_solver.cpp` ~L1091/1222), public multigrid defaults if needed (ctor/members only — no *Config) | gate: k twin wall large ↓; pins hold |
- [ ] `p1-ot-convergence` | After double-energy fix: OT line-search / precond economics — fewer full `build_*_energy` in Armijo (baseline sum `ls_rebuilds`=33 / 25 iters); fixed-H inner steps where safe; align with CP2K OUTER_SCF spirit already sketched for OT+U | phase P1 | files: `ot.cpp`, `gpw_solver.cpp` OT | gate: `ot-minimizer-water` wall ↓ vs ~10.8 s OT-only baseline; energy tol hold |
- [ ] `p1-gpw-workspace` | Extract private `GpwWorkspace` (grid, cbg/kbg/products, FFT plans, DIIS/KDiis history, collocate scratch, last_veff) — mirror `PWWorkspace`; kill four nearly complete SCF copies' local bags without public API. `gpw_solver.cpp` still ~5.9k forked control flow; no `gpw_workspace` / `fft_cache` on `dev` yet | phase P1 | files: `gpw_solver.*`, new `internal/gpw_workspace.h` | gate: no public API / no *Config; regs green; no wall regression on four twins |
- [ ] `p1-ks-grid-build` | Single internal `KsGridBuild` (ρ→Poisson→XC→V_eff [+ nonlocal vdW]) shared by Γ/k × RKS/UKS DIIS+OT; today `build_k_energy` (~L1563) / `build_uks_k_energy` (~L2906) / Γ lambdas remain forked copies | phase P1 | files: `gpw_solver.cpp` (+ workspace) | gate: energy pins within existing tol |

### P2 — SIMD, force reuse, MPI grids

- [ ] `p2-product-simd` | SIMD / better screening on separable `collocate_one` outer-product fill (~L715–754; comment ~L623 says vectorizable; no xsimd yet). Respect iron law: no global `XTENSOR_USE_XSIMD` | phase P2 | files: `multigrid.cpp` | gate: MG twin wall ↓ |
- [ ] `p2-force-grid-reuse` | Forces rebuild full `kbg` + `kbgd` (and Γ `cbg`/`cbgd`) + Poisson/XC every call (`gpw_forces.cpp` ~L1129–1244); reuse SCF products/cbg/V_eff / last density path where geometry-fixed | phase P2 | files: `gpw_forces.cpp`, solver result/workspace handoff | gate: force regs green + force wall ↓ |
- [ ] `p2-mpi-product-grid` | Architecture P2: product/grid distribute. Today only k RR + Allreduce density/traces; setup STVpp, products, grids, Poisson replicated on every rank (foothold comments ~L969–1010). Split product scatter / grid FFT across ranks without changing Frame API | phase P2 | files: `gpw_solver.cpp`, `multigrid.cpp`, MPI layout | gate: `mpi-gpw-scaling-4rank` + multi-k regs; scaling wall ↓ |

## in_progress

(empty)

## blocked

- [!] `global-xtensor-xsimd` | blocked — never enable global `XTENSOR_USE_XSIMD` (breaks reshape elsewhere)

## done

- [x] `p0-profile-baseline` | 2026-08-01 | fair-wall + ATV_GPW_PROF table in `.claude/notes/gpw-perf-baseline.md`; thin k/uks prof marks; four twins PASS; unit `test_cpu_(gpw|xc)_` 12/12 |
- [x] `p0-fft-plan-cache` | 2026-08-02 | FFTW plan+buffer cache by grid shape; UKS fair wall ~18.2→18.0 s (noise); micro/Poisson ~5–10% | merged `dc885a96` |
- [x] `p0-collocate-buffers` | 2026-08-02 | prealloc collocate_one + product scatter/integrate pools; **gpw-multigrid-kpoints MAIN 91.8→65.1 s** (OMP=4) | merged `2c476da2` |
- [x] `p0-ot-double-energy` | 2026-08-02 | multi-k OT single KS rebuild/outer; ATV_GPW_PROF builds **115→79** on ot-kpoints class | merged `419ad172` |
