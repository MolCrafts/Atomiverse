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

Post wave-2 simd-retry (2026-08-02): P0 exhausted. **SIMD policy change
(2026-08-03): all optimization levers authorized; SIMD (portable xsimd only)
is first-class at any priority** — see the architecture note. Wave 3 staffs
the top MG-kernel + OT-extension rows (in_progress below).

### P0 — thrash / double work (measured first)

(empty — all P0 items done)

### P1 — kernel constant + OT + structure

- [ ] `p2-cxyz-reflection` | The big coupled collocation rewrite (CP2K collint structure): fold the polynomial `(r-ra)^la·(r-rb)^lb` into cxyz coefficients, exploit Gaussian reflection symmetry about pc (4/8-fold), Horner cxyz→cxy→cx contraction. Substantial + parity-risk — **dedicated wave, single worker, full FD-force battery** | phase P2 | files: `multigrid.cpp` (major) | gate: MG MAIN large ↓; all energy+force pins unchanged tols |
- [ ] `p2-simd-2center-setup` | xsimd the analytic 2-center S/T primitive loops (setup STVpp ~99 s at slabw scale after KB/grid-PP fixes; invisible on small twins) | phase P2 | files: overlap/kinetic internals | gate: setup microbench ↓ on a slab-class fragment; four twins no regression |
- [ ] `p1-products-default-k` | **RE-GATED (2026-08-02 A/B, branch `wf/perf-p1-products-default-k` @ `3ce71310`): flip is a ~70–100× regression until the product engine is fixed** (per-iter primitive re-eval, worst-case integrate windows, `kProductMassMargin`). Revisit only after `p1-integrate-windows` + deeper SIMD land, with a fresh same-binary A/B. Original goal: production multi-k collocation via products/MG; dense χ^k (`KpointBasisGrid`, O(npts·nb²·nk)) transitional / opt-out only. Do **not** add a 5th collocation layout without deleting dense χ^k | phase P2 | files: `kpoints.cpp`, solver k-path | gate: fresh A/B shows products ≤ dense on the k twin; pins hold |


### P2 — SIMD, force reuse, MPI grids

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
- [x] `p2-product-simd` | 2026-08-02 | xsimd z-axis (stride-1) runs for the product collocate/integrate point-fill in `multigrid.cpp` only: `collocate_one` emits contiguous z-runs (split only at the periodic wrap) instead of per-point lambda callbacks; the four product kernels (collocate/integrate × Γ/`_cells` k) consume runs via portable `axis_dot` / `axis_axpy` `xsimd::batch<double>` helpers (native_integrals idiom: raw pointers, `load_unaligned`, scalar tail; no per-arch intrinsics, no global `XTENSOR_USE_XSIMD`). **gpw-multigrid-kpoints MAIN 65.1 → 45.0 s (−30.9%)** (OMP=4, min of 3); uks-o2-gamma 19.25→19.28 s (no regression). Energy pin exact (−15.8024168854 Ha, iters=7); all force regs PASS, tolerances UNCHANGED; not bit-exact (deterministic batch-width sum reorder) | merged `327a7ca8` (worker `cc6068ad`) |
- [x] `p1-ot-convergence` | 2026-08-02 | Γ RKS OT line-search economics (`gpw_solver.cpp`): accepted Armijo trial's KS build carried into the next iteration as E(0); quadratic-model retry α safeguarded to [0.1, 0.5]·α; budget-exhaust exit reuses the carried H. KS builds per OT iter **2.32 → 1.11** (58/25 → 30/27). Official post-merge walls (OMP=4, min of 3, loadavg ~10–16): `ot-minimizer-water` reg **13.97 → 8.87 s** (−36.5%); other twins flat — kpoints-si reg 19.07 s, MG MAIN 65.1 s, uks `wall=` 19.1 s (17.6 on quieter node; load drift, path untouched). Not bit-exact (trajectory 25→27 OT iters); tolerances UNCHANGED (CP2K pin 1.5e-3 Ha, OT/DIIS parity 1e-4 Ha) | merged `97de6c19` (worker `a2facc8e`) |
- [x] `p1-gpw-workspace` | 2026-08-04 | private `internal::GpwWorkspace` owns grid/mg/products/cbg/kbg/last_veff on Γ + multi-k RKS/UKS | `19576a1d` |
- [x] `p1-ks-grid-build` | 2026-08-04 | shared `ks_grid_build_rks/uks` + `evaluate_xc_*` for ρ→Poisson→XC→V_eff [+vdW]; all SCF energy builders wired | `e47756a9` |
- [x] `p1-ot-uks-k` | 2026-08-04 | Γ UKS OT carried (Ha,Hb,E) + quadratic Armijo retry + budget-exhaust reuse; multi-k RKS/UKS integer path skips KS rebuild after accept + quadratic retry | four twins + `ot-uks-o2` PASS |
- [x] `p1-integrate-windows` | 2026-08-04 | integrate windows sized by local \|V(pc)\|·geom (floor 0.05), not global max\|V\| (that inflated every product) and not bare maxc·maxc·K; `integrate_products` + `_cells`. MG MAIN **45.0→42.9 s** (OMP=4, one shot); E pin within 5e-7; uks-o2-gamma + ot-water PASS; not bit-exact |
- [x] `p2-simd-pointfill-2` | 2026-08-04 | xsimd 1D poly/E2 fill (`fill_axis_poly_e2_w`); `axis_dot` via `reduce_add`; hoist sphere z-bound constants. Exp recurrence stays scalar (sequential). Multi-monomial blocking **not** done (no shared-window product batch). MG MAIN best **41.7 s** (OMP=4 bind); pins hold. CP2K same-host twin **0.885 s** → ratio ~**47×** (see note below) |

### Fair wall vs CP2K (2026-08-04, arrhenius1, OMP=4 bind)

`gpw-multigrid-kpoints` MAIN only (MP 2×2×2, 150 Ry, NGRIDS 4):

| Side | Wall [s] | Notes |
| --- | ---: | --- |
| ATV (best of 2, contended node) | **41.7** | MAIN line; run1 101.7 under load |
| CP2K 2026.1 `cp2k.psmp` np=1 | **0.885** | `si_kpoints_mg.inp`; energy −15.802416412503838 Ha |
| **ratio ATV/CP2K** | **~47** | integrate ~4.3 s/iter ATV vs CP2K `grid_integrate` ~0.019 s/iter |

Older baseline (wave2) listed CP2K 23.6 s / ATV 45.0 → 1.9×; re-measured CP2K on this host is ~25× faster than that table — use **today's** numbers for planning. Remaining gap is the product collocate/integrate constant (CP2K collint), not the 1D factor fill.
