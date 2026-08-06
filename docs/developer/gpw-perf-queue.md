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

**Census 2026-08-06 cp2k-perf wave-4** (parent `dev` @ `f868bbe5`, binding walls
unchanged from [gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)
2026-08-06 re-pin — arrhenius1, 1 MPI × OMP=4 bind, CP2K 2026.1 quiet TIMING
`CP2K` TOTAL TIME MAXIMUM). Re-diff `cpu/gpw` (solver / OT / product_collint /
multigrid / collocate / one_electron / kpoints / ks_grid_build / XC) ranked by
expected contribution to \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) on the four fair
twins — not aesthetics. **No new electronic methods**; no arch-intrinsic work;
portable-xsimd rows state bit-exact/tol plan up front. Three wave-3 P0s stay in
`## in_progress` (do not re-queue same slugs); pending is **unstaffed residual**
only.

| Twin | \(T_{\mathrm{atv}}\) [s] | \(T_{\mathrm{cp2k}}\) [s] | ratio | excess [s] | Residual mass (wave-4 re-diff on `dev`) |
| --- | ---: | ---: | ---: | ---: | --- |
| `ot-minimizer-water` OT | 7.55 | 2.50 | **3.02** | **5.05** | **worst_ratio**; `OtPrecondSingleInverse` ENERGY_GAP **floor 1.0** + FULL_SINGLE form (no occupied projector); Γ OT-DIIS Armijo multi-KS on reject; **25** it vs CP2K **19**; **staffed** `p0-ot-cp2k-match` |
| `gpw-uks-o2-gamma` | 25.1 | 13.69 | **1.83** | **11.4** | **largest absolute excess**; ngrids=1 dual-spin **cbg** (`collocate.cpp` full-square integrate + dual ρ walks) + DIIS **26** it + Padé LSD; CP2K twin NGRIDS 4; **staffed** `p0-uks-o2-scf-wall` |
| `gpw-multigrid-kpoints` MAIN | 1.6 | 0.688 | **2.33** | **0.91** | steady coll/int **~2.0×/2.8×** (29–30 vs 14.5 / 47–49 vs 17.1 ms); STVnl **landed**; OMP/task structure **staffed** `p0-mg-collint-structure` — tip evidence **no wall win**, mass remains **sphere/collint kernel work class** |
| `gpw-kpoints-si` MAIN | 2.88 | 2.45 | **1.18** | **0.43** | near parity; dense χ^k / kbg setup dominates MAIN; not the active lever until P0 fair twins move |

**Hotspot re-diff (wave-4):**

| Rank | Stage / control | Twin impact | Evidence |
| --- | --- | --- | --- |
| 1 | OT control vs CP2K twin | OT **3.02×** / **5.05 s** | `ot.cpp` gap floor **1.0** (CP2K **0.08**); projector skipped; `gpw_solver.cpp` Γ OT-DIIS Armijo + polish; **in_progress** `p0-ot-cp2k-match` (tip claims ~3.02→~1.56 / 25→10 it — **unmerged**, not binding). Residual after land likely still **>1.1×** → follow-on below |
| 2 | Dual-spin Γ UKS KS wall | UKS **1.83×** / **11.4 s** | On `dev`: dual cbg collocate + full-nb² integrate (no triangle) + dual `ks_grid_build_uks` walks; DIIS 26. **in_progress** `p0-uks-o2-scf-wall` (tip: dual-spin CSR fuse + triangle integrate + MT kernel cache + Padé vacuum; claims ~1.29–1.48×; ngrids=4 products A/B **slower — do not adopt**). Absolute residual after land still multi-second → DIIS/prof follow-on below |
| 3 | MG collint sphere / monom work | MG **2.33×** / **0.91 s** | Budget: ~13×(coll 30 + int 48) ≈ **1.0 s** of MAIN vs CP2K coll/int ~**0.41 s**. Kernel micro-SIMD **exhausted** (width-4 / Ltot / short-seg; residual-2 no-land). Structure OMP/spatial/fuse-weights **staffed, tip no wall win**. Leftover is **sphere volume + monom contract + private-grid BW** inside `product_collint.cpp` / group path — **not** another schedule(dynamic) experiment |
| 4 | STVnl setup once | MG / k MAIN | **landed** `e1eaf084`; ST_k ~340 ms once — secondary; **do not re-queue** |
| 5 | dense χ^k vs products default | k **1.18×** | A/B only after P0 fair twins move; **no 5th collocation layout** without deleting dense χ^k |

**Budget sketch (MG MAIN, ngrid=37800, ~13 SCF):** setup STVnl+products ~0.42 s once + ~13×(coll 30 + int 48 + XC/Poisson ~5) ≈ **1.5 s** → matches 1.6 s pin. CP2K coll/int class alone ~13×(14.5+17.1) ≈ **0.41 s**. Closing MAIN needs sphere/collint work reduction, not STVnl re-litigation and not more OMP-structure polish.

Portable xsimd first-class (any SIMD row states bit-exact/tol plan up front);
never global `XTENSOR_USE_XSIMD`; no arch intrinsics; no capability expansion;
FD-force on any energy-path land (collocate/integrate/Poisson/XC/V_eff/PP).

### P0 — contribution order (worst residual after staffed work → stage mass)

Wave-3 staff (`p0-ot-cp2k-match`, `p0-uks-o2-scf-wall`, `p0-mg-collint-structure`)
remain in `## in_progress` — **do not re-queue those slugs**. Pending P0 =
follow-on residual distinct from staffed surfaces.

- [ ] `p0-mg-collint-sphere-parity` | **P0** Close MG MAIN coll/int residual (**~2.0×/2.8×** per-call; **~0.9 s** of the **0.91 s** MAIN gap; ratio **2.33×**) after structure path showed **no wall win**. Levers **inside sphere/group contract**, not OMP schedule/task-list restructure and not width-N/lp_z accumulate micro-SIMD (both exhausted): (1) sphere-volume / window parity with CP2K (`shape_radius`, `eps_win`/`eps_drop`, reflection cube waste) under pin-safe energy; (2) monom-contract / cab expand cost (group avg_g, shared pol already landed — cut remaining expand+I walks); (3) private-grid reduce bandwidth if still visible under `ATV_GPW_PROF` after (1–2). **Bit-exact plan:** prefer bit-identical vs current energy pin; window/radius changes that reorder or thin spheres land only with **stated** numeric tol vs CP2K pin **and** fresh FD-force, documented in commit + this row; never silent tol soften. Portable xsimd only if a second real hot loop needs it (raw spans, scalar tail; no arch intrinsics; no global `XTENSOR_USE_XSIMD`) | twin: `gpw-multigrid-kpoints` MAIN | files: `src/cpu/gpw/internal/product_collint.cpp`, `src/cpu/gpw/internal/multigrid.cpp` (sphere/group path only — **no** re-litigate task sort/schedule) | gate: MAIN ratio ↓ from **2.33**; coll/int ms ↓ toward CP2K 14.5/17.1; pins + FD force PASS |
- [ ] `p0-uks-post-cbg-residual` | **P0** After dual-spin cbg fuse lands, close remaining `gpw-uks-o2-gamma` absolute gap (binding **11.4 s** excess / **1.83×**; tip after fuse still **~1.3–1.5×**). Ordered levers: (1) `ATV_GPW_PROF` residual split on post-fuse path (XC / DIIS / leftover integrate); (2) **DIIS iter economics** vs CP2K (26 it class) — solver-loop edits **after** `p0-ot-cp2k-match` releases exclusive `gpw_solver.cpp`; (3) do **not** flip fair twin to ngrids=4 products (A/B slower on tip). Not a new method; not re-fuse cbg | twin: `gpw-uks-o2-gamma` | files: `src/cpu/gpw/gpw_solver.cpp` (DIIS path), residual only in `collocate.cpp` / `ks_grid_build.cpp` / `xc_grid.cpp` if prof proves | gate: twin ratio ≤**1.2×** (or best next step with measured ratio ↓); energy pin + UKS force regs PASS; FD if energy path touched; no tol soften |
- [ ] `p0-ot-match-residual` | **P0** Close fair twin `ot-minimizer-water` residual after `p0-ot-cp2k-match` merges (binding **3.02×** / **5.05 s**; tip claims ~**1.56×** / **3.89 s** still **>1.1×**). Levers: outer-iter vs CP2K **19** OUTER_SCF, KS rebuilds/outer → 1, precond/PD without restoring fixed-H wall hack, polish handoff without thrash. **Staff only after** `p0-ot-cp2k-match` merges or is abandoned — exclusive `ot.cpp` + OT path in `gpw_solver.cpp` | twin: `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `src/cpu/gpw/gpw_solver.cpp` (OT path) | gate: OT ratio ≤**1.1×** vs real CP2K pin; energy pin + OT/DIIS parity + FD if energy path; no tol soften |

### P1 — near-parity path choice

- [ ] `p1-products-default-k` | **A/B only:** dense χ^k MAIN already **1.18×** on `gpw-kpoints-si` (kbg dominates setup; excess **0.43 s**). Re-measure products vs dense **same-binary** on non-MG multi-k **after** P0 fair twins move; flip default only if products ≤ dense and pins hold. Dense transitional / hard opt-out. **No 5th collocation layout** without deleting dense χ^k. MG already uses products — do not re-litigate MG path here | twin: `gpw-kpoints-si` | files: `kpoints.cpp`, solver k-path | gate: fresh A/B walls; if win, default products + hard-opt-out dense |

### P2 — scale-out + slab (not 1-MPI fair-wall levers)

- [ ] `p2-mpi-product-grid` | Product/grid distribute across ranks (today only k RR + Allreduce density). Not a 1-MPI fair-wall lever; staff when worst fair twin ≤~1.1× | twin: `mpi-gpw-scaling-4rank` | files: `gpw_solver.cpp`, `multigrid.cpp`, MPI | gate: scaling wall ↓ + multi-k regs |
- [ ] `p2-simd-2center-setup` | Portable **xsimd** analytic 2-center S/T (raw spans, scalar tail; no arch intrinsics; no global `XTENSOR_USE_XSIMD`) — slab-class only (invisible on four small twins; ST_k residual on MAIN is secondary after landed Hermite path). **Bit-exact plan:** default gate bit-identical vs current pins; if vectorized reduction reorders FP, land only with stated numeric tol vs CP2K pin **and** fresh FD-force check, documented in commit + queue row; never silent tol soften | twin: slab microbench | files: `one_electron.cpp` / overlap-kinetic | gate: setup ↓; four twins no regression |

## in_progress

**cp2k-perf wave-3 Assign** (parent `dev` @ `f868bbe5`; non-kernel levers only; no product_collint accumulate micro-polish; no capability; no CUDA). File surfaces kept **disjoint** (one owner of `gpw_solver.cpp`). `p0-stvnl-setup-residual` already done [x]. `p1-products-default-k` held until P0 fair twins move.

- [~] `p0-ot-cp2k-match` | **P0** Close fair twin `ot-minimizer-water` **3.02× → ≤1.1×** (ATV OT **7.55 s** / CP2K **2.50 s**; excess **5.05 s** — binding worst_ratio). Levers (control path only; ngrids=1 stays — MG was net loss on this micro-cell): (1) outer-iter **25 → ~19** matching CP2K OUTER_SCF MAX_SCF 20 + MINIMIZER DIIS; (2) dense-stable ENERGY_GAP toward CP2K 0.08 and/or FULL_SINGLE_INVERSE projector **when PD-safe** (today floor 1.0 FULL_SINGLE form in `ot.cpp`); (3) Armijo / DIIS history so KS rebuilds/outer → ~1 (cut multi-`build_H_and_energy` rejects); (4) OUTER_SCF polish thresholds without restoring Pulay-H + fixed-H wall hack. Per-outer ~302→~131 ms is fewer rebuilds × precond quality. Diagnostic `ot-uks-o2` only after fair twin moves | twin: `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `src/cpu/gpw/gpw_solver.cpp` (OT path only — exclusive owner of `gpw_solver.cpp` this wave) | gate: real CP2K twin wall ratio ↓; energy pin + OT/DIIS parity + FD force PASS; no tol soften |
- [~] `p0-uks-o2-scf-wall` | **P0** Close `gpw-uks-o2-gamma` **1.83× → ≤1.2×** (ATV **25.1 s** / CP2K **13.69 s**; excess **~11.4 s** — largest absolute gap). Ordered levers: (1) `ATV_GPW_PROF` split dual-spin collocate / integrate / XC / DIIS mass on the fair single-grid path; (2) cut redundant dual KS structure (shared total-ρ work, χ reuse, avoid double walks) in `collocate.cpp` + `build_uks_H` / `ks_grid_build_uks`; (3) **optional same-binary A/B** ngrids=4/rel_cutoff=60 product path (landed UKS MG / collint) vs cbg — adopt on fair twin **only if wall ↓ and pin holds** (historical pre-collint MG was **slower** 32.8 s — do not blind-switch); (4) DIIS vs CP2K Broyden iter economics if rebuild count dominates — **defer solver-loop edits** (OT owns `gpw_solver.cpp` this wave). Not a new method | twin: `gpw-uks-o2-gamma` | files: `src/cpu/gpw/internal/collocate.cpp`, `src/cpu/gpw/internal/ks_grid_build.cpp` (**no** `gpw_solver.cpp` this wave) | gate: twin wall ratio ↓; energy pin + UKS force regs PASS; FD force if energy path touched; no tol soften |
- [~] `p0-mg-collint-structure` | **P0** Non-kernel cut of MG coll+int residual (coll ~2.0×, int ~2.8×; ~0.9 s of the **0.91 s** MAIN gap; MAIN **2.33×**). Task/OMP over `CollintTask` (`dynamic,1`), private-grid reduce, sphere/cache locality, monom walk order, redundant work outside inner SIMD — **not** another width-N accumulate / lp_z / kernel micro-SIMD experiment (exhausted). STVnl setup residual **landed** — this row is **steady-state coll/int only**. Energy-path → FD-force gate | twin: `gpw-multigrid-kpoints` MAIN | files: `src/cpu/gpw/internal/multigrid.cpp`, `src/cpu/gpw/internal/product_collint.cpp` (**structure only** — no accumulate micro-polish) | gate: MAIN ratio ↓ from **2.33**; coll/int ms ↓ toward CP2K 14.5/17.1; pins + FD force PASS; no silent tol soften |

## blocked

- [!] `global-xtensor-xsimd` | blocked — never enable global `XTENSOR_USE_XSIMD` (breaks reshape elsewhere)

## done

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
