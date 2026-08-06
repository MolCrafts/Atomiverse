# ES CPU + CUDA optimization queue

Worklist for workflow `es-cpu-cuda-opt`. **Two tracks** on one loop:

| Track | Goal | Gate |
| --- | --- | --- |
| **cpu** | Cut wall on landed GPW/PW CPU (fair twin vs real CP2K) | pins green; `T_atv/T_cp2k` not worse; FD-force if energy path |
| **cuda** | Implement / polish **corresponding** CUDA for landed ES | CPU twin green first; CUDA↔CPU energy/forces tol stated; device-only kernels |

**Not this queue:** new electronic methods (use `cp2k-method-loop` / `pyscf-method-loop`); classical MD CUDA (OpenMM track).

**Disk law:** wipe `build` / `build-*` after every cmake/ctest on every tree. No multi‑GB leftovers on nobackup.

**CUDA runtime:** login nodes have **no** CUDA toolkit. CUDA build/test only via batch on **GH200** (`sbatch -p gpu`, module `GPU/buildenv-gcccuda/2026.03-cu13.0`). CPU track stays on login with `buildenv-gcc/2026.03-mpich`.

Architecture / order: `CLAUDE.md` → Electronic structure optimization order;
`.claude/notes/es-optimization-order.md`; GPW walls `.claude/notes/gpw-perf-baseline.md`;
CUDA RHF `.claude/notes/cuda-rhf-device-only.md`.

## Fair twins (CPU)

| Twin | Role |
| --- | --- |
| `ot-minimizer-water` | OT iter economics (outer-count after g·e DIIS) |
| `ot-uks-o2` | UKS OT fair wall (worst ratio class on `dev`) |
| `gpw-multigrid-kpoints` | multi-k MG products (~1.6× CP2K) |
| `gpw-uks-o2-gamma` | Γ UKS DIIS |

## CUDA correctness twins

| Twin | Role |
| --- | --- |
| `tests/cuda/scf/test_rhf.cu` | device energy + s/p forces (H₂ pure-s + H₂O s/p) |
| H₂O STO-3G RHF | CPU oracle for CUDA energy/forces (same settings) |

## pending

Wave-3 census 2026-08-06 on **`dev` @ `85322f82`** (wave-2 OT g·e DIIS outer + CUDA RHF wave-7 polish **landed**). Rank by residual fair-wall / open-gate value. **Do not invent hybrid/GAPW/PAW/meta-GGA.** Staff only `wf/escuda-<slug>` off **current** `dev`.

### CPU — wall vs CP2K (worst ratio first)

Product path on **`dev`**: Γ OT MINIMIZER DIIS = g·e B-matrix + SAFE_DIIS/Armijo + PR+ CG fallback (`src/cpu/gpw/internal/ot.cpp` / `gpw_solver.cpp`); spectral FFT + OT outer **done**. Anchors 1 MPI × OMP=4 (re-pin 2026-08-06 post wave-2). **Worst class is still OT outer count**, not MG setup.

| Twin | ATV evidence on `dev` | CP2K twin | ratio class |
| --- | --- | --- | --- |
| `ot-uks-o2` | **87–94** OT it / OT wall **~38.5 s** | **8.68 s** / **30** SCF steps (OT-only pin) | **~4.4×** worst (was ~7.1× / 180 it) |
| `ot-minimizer-water` | **32** OT it; reg wall **~11.6–14 s** (OT+DIIS twin) | ~2.7 s older OT class | outer **~3–4×** class (was 93 it / ~6×) |
| `gpw-multigrid-kpoints` | MAIN **~2.1 s** (STVnl ~0.52 s; coll/int ~80–90 ms/iter) | **~1.28 s** quiet | **~1.6×** (≤2× closed) |
| `gpw-uks-o2-gamma` | DIIS ~19–24 s | ~21 s older | **~1×** |

- [ ] `cpu-ot-residual` | **P0** Further cut Γ OT **outer count** and/or KS-per-outer on UKS O₂ then water. CP2K OT O₂ = **30** steps / 8.68 s; ATV **~90 it / ~38.5 s (~4.4×)** after g·e DIIS. Fixed-stepsize pure DIIS (CP2K default) **diverges** on product micro-cases — keep SAFE+Armijo or find a stable CP2K-like step. Candidate levers (measure with `ATV_GPW_OT_PROF`): OUTER_SCF / precond rebuild cadence, energy_gap / FULL_SINGLE family without the dense-inverse regression (water 23→190), fewer Armijo rebuilds when DIIS is healthy, UKS dual-spin DIIS coupling. **Not** a new method. | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `gpw_solver.cpp` | gate: fair ratio ↓ (aim ≤**3×** UKS OT wall vs 8.68 s) with pins + FD force hold; no tol soften |
- [ ] `cpu-stvnl-residual` | **P1** Residual MG setup + measured collint tail toward ~1× (STVnl ~0.52 s + products_mg ~0.3 s + kbg ~0.2 s of ~2.1 s MAIN; coll/int still ~4× CP2K per-iter). Hermite / s–s / first collint polish landed — chase only **measured** hotspots | track cpu | twin: `gpw-multigrid-kpoints` | files: collint / one_electron / STVnl setup | gate: MAIN wall ↓ or flat; pins hold; no layout fork |
- [ ] `cpu-simd-2center-setup` | **P3** xsimd analytic 2-center S/T if still hot on slab-class (invisible on small four twins) — from gpw-perf `p2-simd-2center-setup` | track cpu | twin: slab fragment microbench | files: overlap/kinetic internals | gate: setup ↓; four twins no regression |

*Not promoted (stay on gpw-perf-queue):* `p1-products-default-k` (re-gated until product ≤ dense A/B); `p2-mpi-product-grid` (multi-rank; not 1 MPI fair-wall lever). *Already landed this loop:* `cpu-xc-spectral-fft`, `cpu-ot-diis-outer`.

### CUDA — corresponding ES (user value / open-gate)

Landed on `dev`: device s/p energy + **device s/p forces** (wave 6) + **wave-7 device-only polish** (merge `fb9e7ed8`; GH200 log `.claude/merge-cuda-polish-913612.out` 100% PASS). **No** `src/cuda/gpw/`. Residual host escapes: type[] D2H for path select; `level_shift≠0` → host-fill (`rhf.cu`); higher-L / non-STO-3G host integrals. Open-gate for next RHF slice: CPU SCF regs green + public `run`/`forces` stable + CUDA↔CPU tols stated.

- [ ] `cuda-rhf-level-shift` | **P2** Wire `level_shift` on device s/p path (currently refused → host-fill when `|shift|>0` in `rhf.cu`; host-fill SCFSolver already applies shift) **or** document permanent host fallback | track cuda | twin: shifted SCF if any reg; else unit path on H₂/H₂O | files: `rhf.cu` / `rhf_fock.cu` / mixer | gate: no silent refuse; device path when shift set; pins green; GH200 log evidence |
- [ ] `cuda-gpw-foothold` | **P3 / open-gate only** Thin Γ-point GPW device slice (collocate **or** Poisson/FFT) with same-settings CPU oracle — **not** full method product; staff only if CPU OT water + MG pins still green and public GPW `run`/`forces` frozen | track cuda | twin: water Γ energy CPU oracle | files: new `cuda/gpw/` | gate: energy match tol; Frame-only API; `ATV_BUILD_GPW`+CUDA gated; no hybrid/GAPW |

*Not pending this loop:* CUDA d-shell / higher-L RHF product (expands coverage after level_shift completeness); classical MD CUDA; hybrid/GAPW/PAW/meta-GGA.

## in_progress

*(empty after wave-2 harness — staff next wave from pending)*

## blocked

- [!] `global-xtensor-xsimd` | never enable global `XTENSOR_USE_XSIMD`
- [!] `cuda-gpw-full-product` | blocked until foothold green + CPU architecture frozen for GPW public shape

## done

Wave 2 land 2026-08-06 (`dev` @ `fb9e7ed8`):

- [x] `cpu-ot-diis-outer` | **P0** Tighten Γ OT MINIMIZER DIIS outer count (g·e B-matrix + SAFE_DIIS/Armijo + CG fallback) | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `gpw_solver.cpp` | merge: `25c2268c` / feat `660a50d1` | evidence: water **93→32** OT it pin PASS; UKS O₂ **180→87–94** it, OT wall **~38.5 s** / CP2K **8.68 s** → **~4.4×** (was ~7.1×; land claim 3.74×); FD `gpw-forces-water` PASS max\|dF\|=3.2e-4; \|E_OT−E_DIIS\|~1e-10; no tol soften |
- [x] `cuda-rhf-device-only-polish` | **P1** Wave-7: drop mid-build V_nuc D2H/H2D (energy ← `d_V_nuc`); ERI `__launch_bounds__(256,2)` + occupancy-capped grid-stride | track cuda | twin: `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf_fock.cu`, `rhf_density.cu` | merge: `fb9e7ed8` / feat `1ae27395` | GH200 job log `.claude/merge-cuda-polish-913612.out`: `test_cuda_scf_rhf` **Passed 2.29 s** + `test_cuda_scf_device_integrals` **Passed 0.82 s** (**100%**); need `-DCMAKE_CUDA_ARCHITECTURES=90` on GH200; tols unchanged from wave 6 |

Wave 1 land 2026-08-06 (`dev` @ `8337251b`):

- [x] `cpu-xc-spectral-fft` | **P1** GGA KS spectral gradient/div: fused multi-out FFT + FFTW_MEASURE plans | track cpu | twin: `ot-minimizer-water` | files: `grid_fft.cpp`, `xc_grid_gga.cpp`, `fft_cache.{h,cpp}` | merge: `977710fb` / feat `db132bb8` | evidence: XC ~108→**95** ms/iter; quiet wall ~16.5→**16.2 s** (ratio ~6.11→**~5.99** vs CP2K 2.7 s note); energy pin stable; FD `gpw-forces-water` PASS; non-bit-exact MEASURE FP with stated tol |
- [x] `cuda-rhf-pshell-forces` | **P0** Device analytic forces for l≤1 s/p | track cuda | twin: H₂O STO-3G CUDA↔CPU + `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf.cu`, force kernels | merge: `8337251b` / feat `e6f984d8` | GH200 job log `.claude/merge-cuda-rhf-913053.out`: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS**; tols: H₂ |ΔF| **1e-5** eV/Å, H₂O |ΔF| **1e-4** eV/Å, energy **ha_to_ev(1e-6)**; `last_force_on_device()` fail-closed; no host RHF for covered shells |
