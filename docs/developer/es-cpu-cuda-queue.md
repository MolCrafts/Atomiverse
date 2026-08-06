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
| `ot-uks-o2` | UKS OT fair wall (was worst; now ~1.5× after polish) |
| `gpw-multigrid-kpoints` | multi-k MG products (~1.8× CP2K post STVnl residual) |
| `gpw-uks-o2-gamma` | Γ UKS DIIS |

## CUDA correctness twins

| Twin | Role |
| --- | --- |
| `tests/cuda/scf/test_rhf.cu` | device energy + s/p forces + level_shift invariant (H₂ pure-s + H₂O s/p) |
| H₂O STO-3G RHF | CPU oracle for CUDA energy/forces (same settings) |
| `tests/cuda/gpw/test_device_poisson.cu` | Γ Poisson/FFT foothold vs CPU oracle |

## pending

Wave-3 **landed** 2026-08-06 on **`dev` @ `9f2d791c`**. Next residual rank (do not invent hybrid/GAPW/PAW/meta-GGA). Staff only `wf/escuda-<slug>` off **current** `dev`.

### CPU — wall vs CP2K (worst ratio first)

Product path on **`dev`**: Γ OT = g·e DIIS + SAFE/Armijo + PR+ CG + **OUTER_SCF polish** (fixed-H eigen+mix when residual moderate; skip under DFT+U); STVnl Hermite geometry-key reuse + phase flatten. Anchors 1 MPI × OMP=4. **Worst class is now MG MAIN ~1.8×** (OT worst ~1.5× closed under 3× aim).

| Twin | ATV evidence on `dev` | CP2K twin | ratio class |
| --- | --- | --- | --- |
| `gpw-multigrid-kpoints` | MAIN quiet best **~2.3 s** (STVnl ~435 ms) | **~1.28 s** quiet | **~1.8×** |
| `ot-uks-o2` | **27** OT it / OT wall **~13 s** | **8.68 s** / **30** SCF steps | **~1.5×** (was ~4.4× / ~90 it) |
| `ot-minimizer-water` | **25** OT it | ~2.7 s older OT class | outer improved (was 32) |
| `gpw-uks-o2-gamma` | DIIS ~19–24 s | ~21 s older | **~1×** |

- [ ] `cpu-simd-2center-setup` | **P3** xsimd analytic 2-center S/T if still hot on slab-class (invisible on small four twins) — from gpw-perf `p2-simd-2center-setup` | track cpu | twin: slab fragment microbench | files: overlap/kinetic internals | gate: setup ↓; four twins no regression |

*Not promoted (stay on gpw-perf-queue):* `p1-products-default-k` (re-gated until product ≤ dense A/B); `p2-mpi-product-grid` (multi-rank; not 1 MPI fair-wall lever). *Already landed this loop:* `cpu-xc-spectral-fft`, `cpu-ot-diis-outer`, `cpu-ot-residual`, `cpu-stvnl-residual`.

### CUDA — corresponding ES

Landed on `dev`: device s/p energy + forces + **level_shift on device** + **Γ GPW Poisson/FFT foothold**. Residual: higher-L / non-STO-3G host integrals; type[] D2H path select; full GPW method product blocked.

*Not pending this loop:* CUDA d-shell / higher-L RHF product; classical MD CUDA; hybrid/GAPW/PAW/meta-GGA; `cuda-gpw-full-product` (blocked until foothold green — foothold is green, still not a full product).

## in_progress

*(empty after wave-3 land)*

## blocked

- [!] `global-xtensor-xsimd` | never enable global `XTENSOR_USE_XSIMD`
- [!] `cuda-gpw-full-product` | blocked until foothold green + CPU architecture frozen for GPW public shape — foothold unit green; full method product still out of scope for this loop

## done

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
