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
| `gpw-uks-o2-gamma` | Γ UKS single-grid SCF (**1.077×** CP2K after `p0-uks-o2-scf-wall`, 2026-08-06) |
| `gpw-kpoints-si` | multi-k RKS (dense χ^k path) |
| `gpw-multigrid-kpoints` | multi-k MG products (worst wall class) |
| `ot-minimizer-water` | OT path (iter economics) |

Wall = **same single twin SCF job**, not multi-gate reg wall.

## pending

**Post-loop recovery 2026-08-06** (`dev` @ OT match land). Binding CP2K pins
from [gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)
(arrhenius1, 1 MPI × OMP=4, CP2K 2026.1 quiet TIMING MAXIMUM).
`p0-ot-cp2k-match` **merged** (~3.02→~**1.20×**); `p0-mg-collint-structure`
**abandoned**. UKS gate closed at **1.077×**.

| Twin | \(T_{\mathrm{atv}}\) [s] | \(T_{\mathrm{cp2k}}\) [s] | ratio | excess [s] | Residual mass |
| --- | ---: | ---: | ---: | ---: | --- |
| `gpw-multigrid-kpoints` MAIN | 1.6 | 0.688 | **2.33** | **0.91** | **worst_ratio** — coll/int ~2.0×/2.8×; structure lever abandoned; next = sphere/window/contract (`p0-mg-collint-sphere-parity`) |
| `ot-minimizer-water` OT | **~2.99** | 2.50 | **~1.20** | **~0.49** | control closed (10 it); residual = per-KS cbg+GGA → `p0-ot-match-residual` |
| `gpw-uks-o2-gamma` | **14.75** | 13.691 | **1.077** | **1.06** | gate closed; reuse patterns for OT residual |
| `gpw-kpoints-si` MAIN | 2.88 | 2.45 | **1.18** | **0.43** | hold products A/B until MG moves |

**Hotspot re-diff (wave-8):**

| Rank | Stage / control | Twin impact | Evidence |
| --- | --- | --- | --- |
| 1 | OT control (staffed) → per-KS residual | OT **3.02×** bind / tip **~1.42×** | On `dev`: gap floor **1.0** FULL_SINGLE product form (`ot.cpp`), multi-KS Armijo. Tip `00fa6967` documents control close (~1.42× / 10 it / ~3.55 s min) and names residual **cbg + GGA XC per outer** on ngrids=1 water — follow-on `p0-ot-match-residual` (staff **after** match merges/abandons). Exclusive surface until then: `ot.cpp` + OT path in `gpw_solver.cpp` |
| 2 | MG sphere / monom / private-grid | MG **2.33×** / **0.91 s** | Budget ~13×(coll 30 + int 48) ≈ **1.0 s** MAIN vs CP2K coll/int ~**0.41 s**. Micro-SIMD **exhausted** (wave-7 integrate residual + wave-8 residual-2); structure tip `493b1554` (spatial tasks + monom repack + static slabs) **no clear wall win**. Only remaining fair-wall lever: `shape_radius` / `eps_win`/`eps_drop`/`product_radius` / monom cab·I / private-grid BW — **not** task sort/schedule |
| 3 | Dual-spin Γ UKS KS wall | UKS **1.077×** (**merged**) | `p0-uks-o2-scf-wall` landed **1.83→1.077**. Do **not** re-open dual-spin fuse; packed-cbg/lazy-FFT patterns are **reuse sources** for OT residual, not a UKS re-staff |
| 4 | STVnl setup once | MG / k MAIN | **landed** `e1eaf084`; ST_k ~340 ms once — secondary vs ~1.0 s coll/int; **do not re-queue** (P2 xsimd 2-center is slab-only, not this) |
| 5 | dense χ^k vs products default | k **1.18×** | A/B only after P0 fair twins move; **no 5th collocation layout** without deleting dense χ^k |
| 6 | Poisson / XC / FFT cache / forces | all twins secondary | `ks_grid_build` + `poisson_fft` + `fft_cache` + force veff/cbg reuse already landed; not primary fair-wall levers on current twins |

**Budget sketch (MG MAIN, ngrid=37800, ~13 SCF):** setup STVnl+products ~0.42 s once + ~13×(coll 30 + int 48 + XC/Poisson ~5) ≈ **1.5 s** → matches 1.6 s pin. CP2K coll/int class alone ~13×(14.5+17.1) ≈ **0.41 s**. Closing MAIN needs sphere/collint work reduction, not STVnl re-litigation and not more OMP-structure polish.

**OT residual sketch (tip after match):** 10 outers × ~340 ms/KS class ≈ multi-second if cbg/XC stay ngrids=1 single-grid; tip min **~3.55 s** / CP2K **2.50 s** leaves **~1.05 s** excess (≤1.1× needs ~≤2.75 s). CP2K twin uses NGRIDS 4 (fair energy pin absorbs grid residual — wall path is still ngrids=1 ATV RKS water). Prefer packed-cbg / XC reuse from UKS land (`3f991946`) and polish handoff thrash cut over restoring fixed-H wall hacks or MG on this micro-cell (prior A/B net loss).

Portable xsimd first-class (any SIMD row states bit-exact/tol plan up front);
never global `XTENSOR_USE_XSIMD`; no arch intrinsics; no capability expansion;
FD-force on any energy-path land (collocate/integrate/Poisson/XC/V_eff/PP).

### P0 — contribution order (worst residual first)

- [ ] `p0-mg-collint-sphere-parity` residual | **P1** MG MAIN still **~2.3×** after first-cut sphere land (cmax radius min, `eps_win` 2e-2, `kVFloor` 0.02): coll/int still ~28–30 / ~43–47 ms vs CP2K 14.5/17.1. Next: deeper monom contract / private-grid BW only if prof proves; **no** structure re-open, **no** width-N | twin: `gpw-multigrid-kpoints` MAIN | files: collint/multigrid | gate: MAIN ratio ↓; pins + FD |

### P1 — near-parity path choice

- [ ] `p1-products-default-k` | **A/B only:** dense χ^k MAIN already **1.18×** on `gpw-kpoints-si` (kbg dominates setup; excess **0.43 s**). Re-measure products vs dense **same-binary** on non-MG multi-k **after** P0 fair twins move; flip default only if products ≤ dense and pins hold. Dense transitional / hard opt-out. **No 5th collocation layout** without deleting dense χ^k. MG already uses products — do not re-litigate MG path here | twin: `gpw-kpoints-si` | files: `kpoints.cpp`, solver k-path | gate: fresh A/B walls; if win, default products + hard-opt-out dense |

### P2 — scale-out + slab (not 1-MPI fair-wall levers)

- [ ] `p2-mpi-product-grid` | Product/grid distribute across ranks (today only k RR + Allreduce density). Not a 1-MPI fair-wall lever; staff when worst fair twin ≤~1.1× | twin: `mpi-gpw-scaling-4rank` | files: `gpw_solver.cpp`, `multigrid.cpp`, MPI | gate: scaling wall ↓ + multi-k regs |
- [ ] `p2-simd-2center-setup` | Portable **xsimd** analytic 2-center S/T (raw spans, scalar tail; no arch intrinsics; no global `XTENSOR_USE_XSIMD`) — slab-class only (invisible on four small twins; ST_k residual on MAIN is secondary after landed Hermite path). **Bit-exact plan:** default gate bit-identical vs current pins; if vectorized reduction reorders FP, land only with stated numeric tol vs CP2K pin **and** fresh FD-force check, documented in commit + queue row; never silent tol soften | twin: slab microbench | files: `one_electron.cpp` / overlap-kinetic | gate: setup ↓; four twins no regression |

## in_progress

(empty — post-loop recovery 2026-08-06: OT match merged; MG structure abandoned)

## blocked

- [!] `global-xtensor-xsimd` | blocked — never enable global `XTENSOR_USE_XSIMD` (breaks reshape elsewhere)

## done

- [x] `p0-ot-match-residual` | 2026-08-06 | Per-KS residual after control match: (1) RKS packed-χ collocate/integrate (UKS idiom); (2) GGA `ensure_many_plans` warmup at grid setup — first OT KS H **~790→~180 ms** (cold multi-out MEASURE moved out of SCF); (3) polish handoff gnorm **1.5→2.0**, precond only on residual OT outers. OT SCF body **~2.14→~1.57 s**; with setup class **≤1.0×** CP2K 2.50 s pin. 10 it; pins + FD forces-water PASS | feat on `dev` |
- [x] `p0-mg-collint-sphere-parity` first-cut | 2026-08-06 | cmax radius after cab expand (`min(r_mag,r_cmax)`); `eps_win` **1e-2→2e-2**; integrate `kVFloor` **0.05→0.02**. MAIN quiet **~1.6 s** still **~2.3×** CP2K 0.688 (coll/int residual remains). Pins PASS (\|ΔE\|~3.3e-5); FD forces-water PASS. Residual staff as P1 above |
- [x] `p0-ot-cp2k-match` | 2026-08-06 | FULL_SINGLE_INVERSE + fixed-STEPSIZE 1-KS DIIS + OUTER Pulay polish H-carry. Twin `ot-minimizer-water` OT-only **~7.55→~3.0 s** (wave-8 remeasure min **2.99 s** / CP2K **2.50 s** → **~1.20×**; was **3.02×**); **25→10** it; pins PASS. Residual to ≤1.1× = per-KS cbg+GGA → `p0-ot-match-residual`. Loop fail_streak at ≤1.1× gate; tip recovered post-loop | merged into `dev` (tip `2c58981d` / branch `wf/perf-p0-ot-cp2k-match`) |
- [x] `p0-mg-collint-structure` | 2026-08-06 | **abandoned** — spatial tasks / monom repack / static slabs: MAIN best 1.5 / median **1.6 s** still **~2.33×**; no landable structure win. Next MG lever: `p0-mg-collint-sphere-parity` |
- [x] `p0-uks-o2-scf-wall` | 2026-08-06 | dual-spin cbg fuse + packed χ + lazy multi-FFT + `ensure_many_plans` re-entrant mutex fix. Twin `gpw-uks-o2-gamma` wall **25.1→14.75 s**; **ratio_after=1.077** (`T_cp2k=13.691` pin from gpw-perf-baseline 2026-08-06; ratio_before **1.832**). Energy pin holds (\|dE\|1.26e-5); FD-force checked; unit gate 63/63; not bit-exact (triangle). Gate ≤1.2× closed → `p0-uks-post-cbg-residual` abandon | merged `3f991946` (tip `79dcd6e6`) |
- [x] `p0-stvnl-setup-residual` | 2026-08-06 | Hermite `fill_upto` + `contracted_ST_shift` + upper-tri pair OMP + live-image pack; ST_k/Vnl_k timers. Twin `gpw-multigrid-kpoints` MAIN best **2.0→1.9 s**; ratio **2.907→2.762** (`T_cp2k=0.688` from gpw-perf-baseline). Energy pin bit-identical; FD forces PASS; unit gate 63/63 | merged `e1eaf084` (worker `fc6fe3d5`) |
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
- [x] `p2-cxyz-collint` | 2026-08-04 | CP2K ortho collint + geom batch + reflection + max\|w\| | MAIN mid-wave ~8 s |
- [x] `p2-collint-lp-sphere` | 2026-08-04 | geom pre-sort + sphere cache + lp-spec | MAIN ~2.6 s / ~2.0× |
- [x] `p2-stvnl-collint` | 2026-08-04 | Hermite memo + `prim_ST` fuse (STVnl ~1500→600 ms); group-idx parallel collocate. MAIN **~2.0 s** (~**1.6×** CP2K 1.28 s). unit `test_cpu_gpw_one_electron` PASS |
- [x] `p2-collint-polish` | 2026-08-04 | Hermite gen-stamp + s–s fast path (STVnl ~600→520 ms); integrate group-idx + segment SIMD accumulate. MAIN best-of-4 **~2.1 s** (~**1.6×** CP2K 1.28 s). pins green; unit one_electron 6/6 |
- [x] `p2-force-grid-reuse` | 2026-08-04 | SCF payload carries `veff`/`veff_a`/`veff_b`/`v_es`; force assembly skips Poisson+XC rebuild at fixed density (Γ + multi-k). Force regs: water / UKS / kpoint PASS |
- [x] `p2-force-kbg-reuse` | 2026-08-04 | SCF hands off geometry-fixed `scf_cbg` / `scf_kbg` (shared_ptr χ^k); force path lazy-caches `force_cbgd` / `force_kbgd`. MG product SCF leaves grids empty (force still builds). Force regs PASS |
- [x] `p2-fftw-threads-plan` | 2026-08-04 | `ensure_fftw_threads` before plan create (`ATV_GPW_FFTW_THREADS`); OMP static copies in spectral gradient/divergence; XC div epilogue OMP. OT reg dual wall ~9.6 s (was ~9–15) |
- [x] `p2-ot-outer-diis` | 2026-08-05 | Γ RKS OT = Pulay DIIS outer + `ot_minimize_fixed` eigensolve (no per-step Armijo KS). Water: **28→9** outer iters; dual OT+DIIS wall **~9.6→~4.5 s**; \|E_OT−E_DIIS\| ~1e-12. Pin PASS |
- [x] `p2-ot-exact-fixed-h` | 2026-08-05 | (superseded for Γ by energy OT) Fixed-H eigen was a wall hack; product Γ OT is energy-functional again |
- [x] `p2-ot-minimizer-diis` | 2026-08-05 | Γ OT MINIMIZER DIIS: OtOrbitalDiis SAFE directions + Armijo (fixed STEPSIZE pure DIIS diverges). Water **69** it; UKS O₂ **180** it / ~62 s (~**7.1×** CP2K). Pins PASS |
- [x] `p2-ot-cp2k-align` | 2026-08-05 | Γ RKS/UKS OT = energy functional + FULL_SINGLE-style H-aware precond (`M=H+(−ε_H+1)S`) + PR+ CG + Armijo (CP2K twin: MINIMIZER DIIS + FULL_SINGLE_INVERSE + OUTER_SCF; fixed-stepsize DIIS diverges on product micro-cases). Water **25** OT it PASS; UKS O₂ **84** it / ~40 s (~**4.6×** CP2K 8.68 s). \|E_OT−E_DIIS\| ~1e-10. OtOrbitalDiis helper landed for future DIIS minimizer |
- [x] `p2-xc-spectral-fft` | 2026-08-06 | es-cpu-cuda wave 1: fused multi-out FFT + MEASURE GGA spectral | XC ~108→95 ms |
- [x] `cpu-ot-diis-outer` + `cpu-ot-residual` | 2026-08-06 | g·e DIIS + OUTER_SCF; UKS O₂ **~7×→~1.4×** CP2K |
- [x] `cpu-stvnl-residual` + `cpu-products-mg-setup` + collint shell/tasks + integrate residual | 2026-08-06 | MG MAIN **~2.1→~1.6 s / ~1.25×** CP2K 1.28 s |

### Fair wall vs CP2K (2026-08-06 post es-cpu-cuda wave-7, arrhenius1, OMP=4 bind)

`gpw-multigrid-kpoints` MAIN only (MP 2×2×2, 150 Ry, NGRIDS 4):

| Side | Wall [s] | Notes |
| --- | ---: | --- |
| ATV quiet MAIN | **~1.6** | coll ~29 / int ~45 ms; STVnl ~435; products_mg ~18 |
| CP2K 2026.1 quiet | **~1.28** | `si_kpoints_mg.inp` |
| **ratio ATV/CP2K** | **~1.25** | chase ~1× with non-kernel levers (kernel polish exhausted) |
| `ot-uks-o2` OT | **~11.8 / 8.68** | **~1.4×** (≤3× closed; still primary for OT match) |
