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
| `ot-minimizer-water` | OT iter economics (outer-count after g·e DIIS + OUTER_SCF polish) |
| `ot-uks-o2` | UKS OT fair wall (was worst; now ~1.4× after polish) |
| `gpw-multigrid-kpoints` | multi-k MG products (~1.9× CP2K post collint residual) |
| `gpw-uks-o2-gamma` | Γ UKS DIIS |

## CUDA correctness twins

| Twin | Role |
| --- | --- |
| `tests/cuda/scf/test_rhf.cu` | device energy + s/p forces + level_shift invariant (H₂ pure-s + H₂O s/p) |
| H₂O STO-3G RHF | CPU oracle for CUDA energy/forces (same settings) |
| `tests/cuda/gpw/test_device_poisson.cu` | Γ Poisson/FFT foothold vs CPU oracle |
| `tests/cuda/gpw/test_xc_lda.cu` | Padé LDA XC grid foothold vs CPU oracle |

## pending

Wave-4 land 2026-08-06 (`dev` @ collint merge + CUDA merges). Next staff only
`wf/escuda-<slug>` off **current** `dev`. Do not invent hybrid/GAPW/PAW/meta-GGA.

### Fair-wall rank (1 MPI × OMP=4; after wave-4)

| Twin | ATV on `dev` | CP2K twin | ratio | lever class |
| --- | --- | --- | ---: | --- |
| `gpw-multigrid-kpoints` | MAIN **~2.4 s** remeasure (coll ~63 / int ~76 ms/iter; STVnl ~435 ms; products_mg ~0.3 s) | **~1.28 s** quiet | **~1.9×** | **worst** — primary CPU target (setup residual) |
| `ot-uks-o2` | **27** OT it / OT **~11.8 s** | **8.68 s** / **30** SCF steps | **~1.4×** | ≤3× aim **met**; not primary |
| `ot-minimizer-water` | **25** OT it | ~2.7 s older OT class | improved | outer closed under OT residual |
| `gpw-uks-o2-gamma` | DIIS ~19–24 s | ~21 s older | **~1×** | closed |

Product path landed: collint residual (hybrid SIMD + cab expand + sphere TLS);
STVnl Hermite key reuse; Γ OT g·e DIIS + OUTER_SCF polish; spectral FFT GGA.
Dense `kbg` is **skipped** under MG (`!use_mg`) — do not staff kbg cut for this
twin. Shell-level task list beyond cartesian-prim groups still open (collint
not yet CP2K per-iter class).

### CPU — wall vs CP2K (worst ratio first)

- [ ] `cpu-products-mg-setup` | **P0** Shrink one-shot `build_products` / products_mg (~0.3 s on MG twin) — geometry sort, prim/contrib packing, level assignment. | track cpu | twin: `gpw-multigrid-kpoints` setup mark `products_mg` | files: `multigrid.cpp` (`build_products`), product data structures | gate: products_mg ms ↓; MAIN not worse; pins green; no 5th collocation layout |
- [ ] `cpu-collint-shell-tasks` | **P1** Shell-level task list beyond cartesian-prim groups (still open after residual polish) — chase coll/int toward CP2K ~20 ms class. | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `product_collint.{h,cpp}`, `multigrid.cpp` | gate: per-iter coll/int ↓; MAIN ratio ↓ toward ~1×; pins green; FD if energy path moves |
- [ ] `cpu-simd-2center-setup` | **P3** xsimd analytic 2-center S/T if still hot on slab-class (invisible on small four twins) — from gpw-perf `p2-simd-2center-setup` | track cpu | twin: slab fragment microbench | files: overlap/kinetic internals | gate: setup ↓; four twins no regression |

*Not promoted (stay on gpw-perf-queue):* `p1-products-default-k` (re-gated until product ≤ dense A/B); `p2-mpi-product-grid` (multi-rank; not 1 MPI fair-wall lever); OT further polish (≤3× aim already met on `ot-uks-o2`). *Already landed this loop:* `cpu-xc-spectral-fft`, `cpu-ot-diis-outer`, `cpu-ot-residual`, `cpu-stvnl-residual`, `cpu-collint-residual`.

### CUDA — corresponding ES (open-gate)

Landed on `dev`: device s/p energy + forces + level_shift + device-only polish +
**Γ GPW Poisson/FFT** + **Padé LDA XC grid** footholds + RHF host-Z path select.
Open gate holds for thin device slices with CPU oracle + stated CUDA↔CPU tols.
Public `run`/`forces` shape stable for RHF; GPW full method still out of scope.

*Not pending this loop:* CUDA d-shell / higher-L RHF product; classical MD CUDA;
hybrid/GAPW/PAW/meta-GGA; `cuda-gpw-full-product` (blocked — footholds unit
green, full method product still out of scope / architecture freeze).

## in_progress

*(empty after wave-4 land)*

## blocked

- [!] `global-xtensor-xsimd` | never enable global `XTENSOR_USE_XSIMD`
- [!] `cuda-gpw-full-product` | blocked until foothold green + CPU architecture frozen for GPW public shape — foothold units green; full method product still out of scope for this loop

## done

Wave 4 land 2026-08-06 (`dev` @ collint merge `5fa8b0c6` + CUDA `44a5d858`):

- [x] `cpu-collint-residual` | **P0** Residual collint: hybrid xsimd FMA + 4-wide residual, lp_z≤9, cab-style alpha for large groups, TLS multi-imr sphere, gamma group_idx nthread private grids, integrate \|V(pc)\| once per geometry | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `product_collint.cpp`, `multigrid.cpp` | merge: `5fa8b0c6` / feat `6da97a85` | evidence: MAIN remeasure **~2.4 s** (~**1.9×** vs CP2K 1.28 s); coll **~63** / int **~76** ms/iter (was ~80–90); pin bit-stable \|ΔE\|=5.7e-6 Ha; FD `gpw-forces-water` PASS max\|dF\|=3.2e-4; OT water 25 it + UKS O₂ 27 it / ~11.8 s PASS; shell-level tasks still open |
- [x] `cuda-gpw-xc-lda-foothold` | **P1** Thin device Padé LDA XC grid vs same-settings CPU oracle (closed-shell + LSD + energy reductions) | track cuda | twin: `test_cuda_gpw_xc_lda` | files: `src/cuda/gpw/internal/xc_lda.{cu,cuh}`, unit test | merge: `cc450b59` / feat `90acb1d5` | GH200 worker **915678** + post-merge **915704**: `test_cuda_gpw_xc_lda` **Passed 0.86 s**; tols ~**1e-12** per-point, energy rel **~1e-10**; not full GPW product |
- [x] `cuda-rhf-path-select-host` | **P2** Host-Z path select without setup type[] D2H (`cache_host_types` before upload) | track cuda | twin: `tests/cuda/scf/test_rhf.cu` | files: `rhf.cu`, `rhf.cuh` | merge: `44a5d858` / feat `6f179574` | GH200 worker **915654** + post-merge **915704**: `test_cuda_scf` + `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (0.95 + 2.03 + 0.87 s); tols unchanged |

Wave 3 land 2026-08-06 (`dev` @ `9f2d791c`):

- [x] `cpu-ot-residual` | **P0** OUTER_SCF polish (fixed-H eigen+mix after residual moderate; no DFT+U) | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `gpw_solver.cpp`, `ot.h` | merge: `4249fd3f` / feat `867c1add` | evidence: UKS O₂ **87→27** it, OT wall **~52→13 s** / CP2K **8.68 s** → **~1.53×** (was ~4.4×; aim ≤3× **met**); water **32→25** it; pins + FD `gpw-forces-water` PASS; no tol soften |
- [x] `cpu-stvnl-residual` | **P1** STVnl Hermite geometry-key reuse + lattice phase flatten | track cpu | twin: `gpw-multigrid-kpoints` | files: `one_electron.cpp`, `kpoints.cpp` | merge: `dfe68fdd` / feat `76239cfc` | evidence: STVnl_k **~507→435 ms**; MG MAIN quiet best **~2.6→2.3 s** (~**1.8×** vs CP2K 1.28 s); pin bit-stable; forces-water PASS |
- [x] `cuda-rhf-level-shift` | **P2** Saunders–Hillier λ on device s/p Fock (no host-fill when λ≠0) | track cuda | twin: `tests/cuda/scf/test_rhf.cu` `CUDALevelShiftEnergyInvariant` | files: `rhf.cu`, `rhf_fock.cu`, `scf_solver.cu` | merge: `ba3a8145` / feat `77eb7f79` | GH200 worker job **913709** + post-merge **915584**: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **PASS**; λ=0.5 energy matches λ=0 and CPU @ 1e-8 Ha |
- [x] `cuda-gpw-foothold` | **P3** Thin Γ-point device Poisson/FFT vs CPU oracle | track cuda | twin: `test_cuda_gpw_device_poisson` | files: `src/cuda/gpw/internal/poisson_fft.{cu,cuh}`, unit test | merge: `9f2d791c` / feat `f604577c` | GH200 worker **915518** + post-merge **915584**: `test_cuda_gpw_device_poisson` **Passed 0.97 s** + `test_cuda_pw_poisson` **Passed 1.78 s** (**100%** of 4); tols max\|ΔV\| **1e-9…1e-10**, Hartree **1e-10…1e-12** rel; not full GPW product |

Wave 2 land 2026-08-06 (`dev` @ `fb9e7ed8`):

- [x] `cpu-ot-diis-outer` | **P0** Tighten Γ OT MINIMIZER DIIS outer count (g·e B-matrix + SAFE_DIIS/Armijo + CG fallback) | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `gpw_solver.cpp` | merge: `25c2268c` / feat `660a50d1` | evidence: water **93→32** OT it pin PASS; UKS O₂ **180→87–94** it, OT wall **~38.5 s** / CP2K **8.68 s** → **~4.4×** (was ~7.1×; land claim 3.74×); FD `gpw-forces-water` PASS max\|dF\|=3.2e-4; \|E_OT−E_DIIS\|~1e-10; no tol soften |
- [x] `cuda-rhf-device-only-polish` | **P1** Wave-7: drop mid-build V_nuc D2H/H2D (energy ← `d_V_nuc`); ERI `__launch_bounds__(256,2)` + occupancy-capped grid-stride | track cuda | twin: `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf_fock.cu`, `rhf_density.cu` | merge: `fb9e7ed8` / feat `1ae27395` | GH200 job log `.claude/merge-cuda-polish-913612.out`: `test_cuda_scf_rhf` **Passed 2.29 s** + `test_cuda_scf_device_integrals` **Passed 0.82 s** (**100%**); need `-DCMAKE_CUDA_ARCHITECTURES=90` on GH200; tols unchanged from wave 6 |

Wave 1 land 2026-08-06 (`dev` @ `8337251b`):

- [x] `cpu-xc-spectral-fft` | **P1** GGA KS spectral gradient/div: fused multi-out FFT + FFTW_MEASURE plans | track cpu | twin: `ot-minimizer-water` | files: `grid_fft.cpp`, `xc_grid_gga.cpp`, `fft_cache.{h,cpp}` | merge: `977710fb` / feat `db132bb8` | evidence: XC ~108→**95** ms/iter; quiet wall ~16.5→**16.2 s** (ratio ~6.11→**~5.99** vs CP2K 2.7 s note); energy pin stable; FD `gpw-forces-water` PASS; non-bit-exact MEASURE FP with stated tol |
- [x] `cuda-rhf-pshell-forces` | **P0** Device analytic forces for l≤1 s/p | track cuda | twin: H₂O STO-3G CUDA↔CPU + `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf.cu`, force kernels | merge: `8337251b` / feat `e6f984d8` | GH200 job log `.claude/merge-cuda-rhf-913053.out`: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS**; tols: H₂ |ΔF| **1e-5** eV/Å, H₂O |ΔF| **1e-4** eV/Å, energy **ha_to_ev(1e-6)**; `last_force_on_device()` fail-closed; no host RHF for covered shells |
