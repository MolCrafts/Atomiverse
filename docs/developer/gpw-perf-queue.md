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

**Census 2026-08-06 cp2k-perf wave-7** (parent `dev` @ `0b3758a3`, binding walls
unchanged from [gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)
2026-08-06 re-pin — arrhenius1, 1 MPI × OMP=4 bind, CP2K 2026.1 quiet TIMING
`CP2K` TOTAL TIME MAXIMUM). Re-diff `cpu/gpw` (solver / OT / product_collint /
multigrid / collocate / one_electron / kpoints / ks_grid_build / XC / fft /
poisson) ranked by expected contribution to \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\)
on the four fair twins — not aesthetics. **No new electronic methods**; no
arch-intrinsic work; portable-xsimd rows state bit-exact/tol plan up front.
Staffed `## in_progress` P0s (`p0-ot-cp2k-match` tip `00fa6967`,
`p0-mg-collint-structure` tip `493b1554`) stay staffed (do **not** re-queue
those slugs); pending is **unstaffed residual** only. Tip walls are **not**
binding until merge re-pin. UKS post residual stays pruned (gate closed at
1.077×). **No new slugs** this wave.

| Twin | \(T_{\mathrm{atv}}\) [s] | \(T_{\mathrm{cp2k}}\) [s] | ratio | excess [s] | Residual mass (wave-7 re-diff on `dev` + tip evidence) |
| --- | ---: | ---: | ---: | ---: | --- |
| `ot-minimizer-water` OT | 7.55 | 2.50 | **3.02** | **5.05** | **worst_ratio** on `dev` (`ot.cpp` gap floor **1.0** FULL_SINGLE form; 25 it vs CP2K 19 OUTER). **staffed** `p0-ot-cp2k-match` tip `00fa6967`/`52a2e00b`: **~3.02→~1.42×** / 7.55→**~3.55 s** min / 25→**10** it (FULL_SINGLE_INVERSE, no floor-1.0; fixed-STEPSIZE 1-KS DIIS; OUTER Pulay polish H-carry). Residual after land: **per-KS/setup** (cbg + GGA XC ~340 ms/KS, ngrids=1) — **not** more precond / control polish |
| `gpw-multigrid-kpoints` MAIN | 1.6 | 0.688 | **2.33** | **0.91** | steady coll/int **~2.0×/2.8×** (29–30 vs 14.5 / 47–49 vs 17.1 ms); STVnl **landed**. Structure tip `493b1554` **no wall win** (BEFORE ×3 all 1.6 s; AFTER best **1.5** / median **1.6** — still **~2.33×**) — residual is **sphere/window/monom contract** (`shape_radius` / `eps_win`/`eps_drop` in `multigrid.cpp` + cab·I in `product_collint.cpp`), **not** OMP/task restructure (**abandon** structure lever when tip clears/merges) |
| `gpw-uks-o2-gamma` | **14.75** | 13.691 | **1.077** | **1.06** | **merged** `p0-uks-o2-scf-wall` (`3f991946`): dual-spin cbg fuse + packed χ + lazy multi-FFT; ratio **1.83→1.077** (≤1.2× gate closed). No pending staff — residual demoted vs OT/MG excess |
| `gpw-kpoints-si` MAIN | 2.88 | 2.45 | **1.18** | **0.43** | near parity; dense χ^k / kbg setup dominates MAIN; hold products A/B until P0 fair twins move |

**Hotspot re-diff (wave-7):**

| Rank | Stage / control | Twin impact | Evidence |
| --- | --- | --- | --- |
| 1 | OT control (staffed) → per-KS residual | OT **3.02×** bind / tip **~1.42×** | On `dev`: gap floor **1.0**, multi-KS Armijo. Tip `00fa6967` documents control close (~1.42× / 10 it / ~3.55 s min) and names residual **cbg + GGA XC per outer** on ngrids=1 water — follow-on `p0-ot-match-residual` (staff **after** match merges/abandons) |
| 2 | MG sphere / monom / private-grid | MG **2.33×** / **0.91 s** | Budget ~13×(coll 30 + int 48) ≈ **1.0 s** MAIN vs CP2K coll/int ~**0.41 s**. Micro-SIMD **exhausted** (wave-7/8 integrate residual); structure tip `493b1554` (spatial tasks + monom repack + static slabs) **no clear wall win**. Only remaining fair-wall lever: `shape_radius` / `eps_win`/`eps_drop` / monom cab·I / private-grid BW |
| 3 | Dual-spin Γ UKS KS wall | UKS **1.077×** (**merged**) | `p0-uks-o2-scf-wall` landed **1.83→1.077**. Do **not** re-open dual-spin fuse |
| 4 | STVnl setup once | MG / k MAIN | **landed** `e1eaf084`; ST_k ~340 ms once — secondary; **do not re-queue** |
| 5 | dense χ^k vs products default | k **1.18×** | A/B only after P0 fair twins move; **no 5th collocation layout** without deleting dense χ^k |

**Budget sketch (MG MAIN, ngrid=37800, ~13 SCF):** setup STVnl+products ~0.42 s once + ~13×(coll 30 + int 48 + XC/Poisson ~5) ≈ **1.5 s** → matches 1.6 s pin. CP2K coll/int class alone ~13×(14.5+17.1) ≈ **0.41 s**. Closing MAIN needs sphere/collint work reduction, not STVnl re-litigation and not more OMP-structure polish.

**OT residual sketch (tip after match):** 10 outers × ~340 ms/KS class ≈ multi-second if cbg/XC stay ngrids=1 single-grid; tip min **~3.55 s** / CP2K **2.50 s** leaves **~1.05 s** excess (≤1.1× needs ~≤2.75 s). CP2K twin uses NGRIDS 4 (fair energy pin absorbs grid residual — wall path is still ngrids=1 ATV RKS water). Prefer packed-cbg / XC reuse from UKS land (`3f991946`) and polish handoff thrash cut over restoring fixed-H wall hacks or MG on this micro-cell (prior A/B net loss).

Portable xsimd first-class (any SIMD row states bit-exact/tol plan up front);
never global `XTENSOR_USE_XSIMD`; no arch intrinsics; no capability expansion;
FD-force on any energy-path land (collocate/integrate/Poisson/XC/V_eff/PP).

### P0 — contribution order (worst residual after staffed work → stage mass)

Wave-7: `p0-uks-o2-scf-wall` **done** (1.077×). Remaining in `## in_progress`:
`p0-ot-cp2k-match`, `p0-mg-collint-structure` — **do not re-queue those slugs**.
Pending P0 = follow-on residual distinct from staffed surfaces. No new slugs.

- [ ] `p0-ot-match-residual` | **P0** Close fair twin `ot-minimizer-water` residual **after** `p0-ot-cp2k-match` merges (binding **3.02×** / **5.05 s**; tip `00fa6967`/`52a2e00b` ~**1.42×** / **~3.55 s** min / 10 it — still **>1.1×**, excess **~1.05 s** vs CP2K 2.50). Control path (FULL_SINGLE_INVERSE projector, fixed-STEPSIZE 1-KS DIIS, OUTER Pulay polish H-carry) is the **staffed** surface — do not re-open precond floor / gap as the primary lever. Residual levers (tip-named): (1) per-KS wall on ngrids=1 RKS water — packed cbg / lazy multi-FFT reuse from UKS land (`3f991946`), XC/Poisson setup tax, avoid redundant KS rebuild; (2) outer polish handoff without thrash so outers stay ~10 and KS/outer → 1; (3) do **not** force ngrids=4 products or restore fixed-H wall hack (prior MG net loss on this cell). **Staff only after** `p0-ot-cp2k-match` merges or is abandoned — exclusive `ot.cpp` + OT path in `gpw_solver.cpp` until then | twin: `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `src/cpu/gpw/gpw_solver.cpp` (OT path), residual `collocate.cpp` / `ks_grid_build.cpp` / `xc_grid.cpp` only if prof proves per-KS mass | gate: OT ratio ≤**1.1×** vs real CP2K pin; energy pin + OT/DIIS parity + FD if energy path; no tol soften |
- [ ] `p0-mg-collint-sphere-parity` | **P0** Close MG MAIN coll/int residual (**~2.0×/2.8×** per-call; **~0.9 s** of the **0.91 s** MAIN gap; ratio **2.33×**) after structure path tip `493b1554` showed **no clear wall win** (best 1.5 / median 1.6 s — **abandon structure lever**). Levers **inside sphere/group contract**, not OMP schedule/task-list restructure and not width-N/lp_z accumulate micro-SIMD (both exhausted): (1) sphere-volume / window parity with CP2K (`shape_radius` in collint; `eps_win`/`eps_drop`/`product_radius` at multigrid product path; reflection cube waste) under pin-safe energy; (2) monom-contract / cab expand cost (group avg_g, shared pol already landed — cut remaining expand+I walks); (3) private-grid reduce bandwidth if still visible under `ATV_GPW_PROF` after (1–2). **Bit-exact plan:** prefer bit-identical vs current energy pin; window/radius changes that reorder or thin spheres land only with **stated** numeric tol vs CP2K pin **and** fresh FD-force, documented in commit + this row; never silent tol soften. Portable xsimd only if a second real hot loop needs it (raw spans, scalar tail; no arch intrinsics; no global `XTENSOR_USE_XSIMD`) | twin: `gpw-multigrid-kpoints` MAIN | files: `src/cpu/gpw/internal/product_collint.cpp`, `src/cpu/gpw/internal/multigrid.cpp` (sphere/group path only — **no** re-litigate task sort/schedule) | gate: MAIN ratio ↓ from **2.33**; coll/int ms ↓ toward CP2K 14.5/17.1; pins + FD force PASS |

### P1 — near-parity path choice

- [ ] `p1-products-default-k` | **A/B only:** dense χ^k MAIN already **1.18×** on `gpw-kpoints-si` (kbg dominates setup; excess **0.43 s**). Re-measure products vs dense **same-binary** on non-MG multi-k **after** P0 fair twins move; flip default only if products ≤ dense and pins hold. Dense transitional / hard opt-out. **No 5th collocation layout** without deleting dense χ^k. MG already uses products — do not re-litigate MG path here | twin: `gpw-kpoints-si` | files: `kpoints.cpp`, solver k-path | gate: fresh A/B walls; if win, default products + hard-opt-out dense |

### P2 — scale-out + slab (not 1-MPI fair-wall levers)

- [ ] `p2-mpi-product-grid` | Product/grid distribute across ranks (today only k RR + Allreduce density). Not a 1-MPI fair-wall lever; staff when worst fair twin ≤~1.1× | twin: `mpi-gpw-scaling-4rank` | files: `gpw_solver.cpp`, `multigrid.cpp`, MPI | gate: scaling wall ↓ + multi-k regs |
- [ ] `p2-simd-2center-setup` | Portable **xsimd** analytic 2-center S/T (raw spans, scalar tail; no arch intrinsics; no global `XTENSOR_USE_XSIMD`) — slab-class only (invisible on four small twins; ST_k residual on MAIN is secondary after landed Hermite path). **Bit-exact plan:** default gate bit-identical vs current pins; if vectorized reduction reorders FP, land only with stated numeric tol vs CP2K pin **and** fresh FD-force check, documented in commit + queue row; never silent tol soften | twin: slab microbench | files: `one_electron.cpp` / overlap-kinetic | gate: setup ↓; four twins no regression |

## in_progress

**cp2k-perf wave-6 Assign** (parent `dev` @ `f8262815`; non-kernel levers only; no product_collint accumulate micro-polish; no capability; no CUDA). Priority **[~] first** — resume two unmerged P0 tips (do **not** re-queue same slugs as new work). File surfaces **disjoint** (OT exclusive on `gpw_solver.cpp` + `ot.cpp`; MG structure only on `multigrid.cpp` + `product_collint.cpp`). `p0-uks-o2-scf-wall` **merged** (`3f991946`, **1.077×**). `p0-stvnl-setup-residual` already done [x]. `p1-products-default-k` held until P0 fair twins move. Pending follow-ons (`p0-ot-match-residual`, `p0-mg-collint-sphere-parity`) stay **unstaffed** until these two land or abandon. Staff = **2** (≤3; no third dep-safe pending without stealing staffed surfaces).

- [~] `p0-ot-cp2k-match` | **P0** Close fair twin `ot-minimizer-water` **3.02× → ≤1.1×** (ATV OT **7.55 s** / CP2K **2.50 s**; excess **5.05 s** — binding worst_ratio). Levers (control path only; ngrids=1 stays — MG was net loss on this micro-cell): (1) outer-iter **25 → ~19** matching CP2K OUTER_SCF MAX_SCF 20 + MINIMIZER DIIS; (2) dense-stable ENERGY_GAP toward CP2K 0.08 and/or FULL_SINGLE_INVERSE projector **when PD-safe** (today floor 1.0 FULL_SINGLE form in `ot.cpp` on `dev`); (3) Armijo / DIIS history so KS rebuilds/outer → ~1 (cut multi-`build_H_and_energy` rejects); (4) OUTER_SCF polish thresholds without restoring Pulay-H + fixed-H wall hack. Per-outer ~302→~131 ms is fewer rebuilds × precond quality. Diagnostic `ot-uks-o2` only after fair twin moves. **Resume** tip `2060c5d3` on `wf/perf-p0-ot-cp2k-match` (FULL_SINGLE_INVERSE + 1-KS DIIS + polish H-carry; tip claims ~3.02→~**1.42×** / 7.55→~**3.55 s** min / 25→**10** it, earlier tip commit ~3.88 s / ~1.56× — re-measure fair wall vs real `cp2k.psmp` pin 2.50 s; finish gate ≤1.1× or report honest residual for `p0-ot-match-residual`) | twin: `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `src/cpu/gpw/gpw_solver.cpp` (OT path only — **exclusive owner of `gpw_solver.cpp` this wave**) | gate: real CP2K twin wall ratio ↓; energy pin + OT/DIIS parity + FD force PASS; no tol soften |
- [~] `p0-mg-collint-structure` | **P0** Non-kernel cut of MG coll+int residual (coll ~2.0×, int ~2.8×; ~0.9 s of the **0.91 s** MAIN gap; MAIN **2.33×**). Task/OMP over `CollintTask` (`dynamic,1`), private-grid reduce, sphere/cache locality, monom walk order, redundant work outside inner SIMD — **not** another width-N accumulate / lp_z / kernel micro-SIMD experiment (exhausted). STVnl setup residual **landed** — this row is **steady-state coll/int only**. Energy-path → FD-force gate. **Resume** tip `493b1554` on `wf/perf-p0-mg-collint-structure` (spatial tasks / fused weights / task angular + monom repack + static spatial slabs — census: tip **no clear wall win**, MAIN best 1.5 / median **1.6 s** still ~**2.33×**; verify fair MAIN vs CP2K **0.688 s**; if still no win, **abandon cleanly** so `p0-mg-collint-sphere-parity` can staff next wave) | twin: `gpw-multigrid-kpoints` MAIN | files: `src/cpu/gpw/internal/multigrid.cpp`, `src/cpu/gpw/internal/product_collint.cpp` (**structure only** — no accumulate micro-polish) | gate: MAIN ratio ↓ from **2.33**; coll/int ms ↓ toward CP2K 14.5/17.1; pins + FD force PASS; no silent tol soften |

## blocked

- [!] `global-xtensor-xsimd` | blocked — never enable global `XTENSOR_USE_XSIMD` (breaks reshape elsewhere)

## done

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
