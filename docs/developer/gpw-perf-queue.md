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
| `gpw-uks-o2-gamma` | Γ UKS single-grid SCF (0.90× CP2K, 2026-08-02) |
| `gpw-kpoints-si` | multi-k RKS (dense χ^k path) |
| `gpw-multigrid-kpoints` | multi-k MG products (worst wall class) |
| `ot-minimizer-water` | OT path (iter economics) |

Wall = **same single twin SCF job**, not multi-gate reg wall.

## pending

Post wave-2 (2026-08-02): P0 exhausted; `p1-ot-convergence` landed
(`97de6c19`). Next: P1 structure (workspace / ks-grid-build) + products-k,
then P2.

### P0 — thrash / double work (measured first)

(empty — all P0 items done)

### P1 — one pipeline + products path + OT economics

- [ ] `p1-products-default-k` | Production multi-k collocation via products/MG; dense χ^k (`KpointBasisGrid`, O(npts·nb²·nk)) transitional / opt-out only. Twin `gpw-kpoints-si` MAIN still single-grid dense (`use_mg = ngrids>1 && rel_cutoff>0` only at ~L1091/2436/4485; kbg_build ~1.3 s setup + ~100 ms/step collocate). Do **not** add a 5th collocation layout without deleting dense χ^k | phase P1 | files: `kpoints.cpp`, solver k-path (`gpw_solver.cpp` ~L1091/1222), public multigrid defaults if needed (ctor/members only — no *Config) | gate: k twin wall large ↓; pins hold |
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
- [x] `p0-k-zgemm` | 2026-08-02 | BLAS zgemm (Fortran `zgemm_` binding in `linalg::c_mul`/`c_dagger_mul`; `kpoints.cpp` duplicate mul/dagger_mul deleted, forwards to linalg pair) replaces O(n³) scalar complex matmul on multi-k + OT. Walls (OMP=4, min of 2): `gpw-kpoints-si` reg 19.17→19.00 s (ratio vs CP2K MAIN pin 3.3 s: 5.81→5.76; reg wall inflated vs MAIN-only), `ot-minimizer-water` 14.00→13.97 s, `gpw-multigrid-kpoints` MAIN 65.1→65.1 s — parity within noise on small-nb twins (nb 45–173); removes the scalar scaling ceiling. Regs green, no tol soften; not bit-exact (zgemm FP reorder) | merged `c2fb3324` |
- [x] `p1-ot-convergence` | 2026-08-02 | Γ RKS OT line-search economics (`gpw_solver.cpp`): accepted Armijo trial's KS build carried into the next iteration as E(0); quadratic-model retry α safeguarded to [0.1, 0.5]·α; budget-exhaust exit reuses the carried H. KS builds per OT iter **2.32 → 1.11** (58/25 → 30/27). Official post-merge walls (OMP=4, min of 3, loadavg ~10–16): `ot-minimizer-water` reg **13.97 → 8.87 s** (−36.5%); other twins flat — kpoints-si reg 19.07 s, MG MAIN 65.1 s, uks `wall=` 19.1 s (17.6 on quieter node; load drift, path untouched). Not bit-exact (trajectory 25→27 OT iters); tolerances UNCHANGED (CP2K pin 1.5e-3 Ha, OT/DIIS parity 1e-4 Ha) | merged `97de6c19` (worker `a2facc8e`) |
