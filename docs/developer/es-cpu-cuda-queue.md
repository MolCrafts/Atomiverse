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
| `ot-minimizer-water` | OT iter economics (worst ratio class after DIIS land) |
| `ot-uks-o2` | UKS OT fair wall |
| `gpw-multigrid-kpoints` | multi-k MG products (~1.6× CP2K) |
| `gpw-uks-o2-gamma` | Γ UKS DIIS |

## CUDA correctness twins

| Twin | Role |
| --- | --- |
| `tests/cuda/scf/test_rhf.cu` | device energy + s/p forces (H₂ pure-s + H₂O s/p) |
| H₂O STO-3G RHF | CPU oracle for CUDA energy/forces (same settings) |

## pending

Wave-1 census on `dev` (2026-08-05): rank by (cpu) worst \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) on the fair twins, then (cuda) open-gate readiness / user value. **Do not invent hybrid/GAPW/PAW/meta-GGA.** Staff only `wf/escuda-<slug>` off `dev`.

### CPU — wall vs CP2K (worst ratio first)

Measured anchors on **`dev`** after wave-1 land (`cpu-xc-spectral-fft` + prior OT DIIS path; 1 MPI × OMP=4). OT outer-count still the wall; spectral FFT is per-iter polish only.

| Twin | ATV evidence | CP2K twin | ratio class |
| --- | --- | --- | --- |
| `ot-uks-o2` | **180** OT it / **~62 s** (`.uksdiis_out` on pre-outer tip) | **8.68 s** (OT-only pin) | **~7.1×** worst on `dev` |
| `ot-minimizer-water` | **93** OT it (`.otdiis_out`); quiet wall ~**16.2 s** post-spectral | ~2.7 s older OT class | outer-count dominated (~**6.0×** quiet) |
| `gpw-multigrid-kpoints` | MAIN **~2.1 s** | **~1.28 s** quiet | **~1.6×** (≤2× closed) |
| `gpw-uks-o2-gamma` | DIIS ~19–24 s | ~21 s older | **~1×** |

*Branch note (not on `dev`):* `wf/escuda-cpu-ot-diis-outer` @ `59d31079` claims water **32** it / UKS O₂ **94** it / ~**3.9×** — merge gate still open (not wave-1 harness done).

- [~] `cpu-ot-diis-outer` | **P0** Tighten Γ OT MINIMIZER DIIS outer count toward CP2K (~30 class): OS projection / derivative quality / SAFE_DIIS + Armijo polish; water **93** it and UKS O₂ **180** it / ~7.1× are the wall on `dev`. Do **not** restore Pulay-H + exact fixed-H wall hack | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `gpw_solver.cpp` | gate: pins PASS; outer it ↓; \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) ↓; no tol soften; energy-path change ⇒ FD force re-run |
- [ ] `cpu-stvnl-residual` | **P2** Residual STVnl / setup vs CP2K (~0.52 s of ~2.1 s MAIN on MG twin; coll/int already ~80–90 ms/iter) — Hermite / s–s / collint polish landed; chase only **measured** hotspots | track cpu | twin: `gpw-multigrid-kpoints` | files: collint / one_electron / STVnl setup | gate: MAIN wall ↓ or flat; pins hold; no layout fork |
- [ ] `cpu-simd-2center-setup` | **P3** xsimd analytic 2-center S/T if still hot on slab-class (invisible on small four twins) — promoted from gpw-perf `p2-simd-2center-setup` | track cpu | twin: slab fragment microbench | files: overlap/kinetic internals | gate: setup ↓; four twins no regression |

*Not promoted this wave (stay on gpw-perf-queue):* `p1-products-default-k` (re-gated large regression until product ≤ dense A/B); `p2-mpi-product-grid` (multi-rank; not 1 MPI fair-wall lever).

### CUDA — corresponding ES (user value / open-gate)

Landed CUDA ES surface: `cuda::scf::RHF` device s/p energy + **device s/p forces** (`src/cuda/scf/`, wave 6); **no** `cuda/gpw/`. Open gate for a slice: CPU regs green + public `run`/`forces` stable + CUDA↔CPU tols stated. Build/run only via `sbatch -p gpu` (login has no toolkit).

- [ ] `cuda-rhf-device-only-polish` | **P1** Kill residual mid-path H2D/D2H on s/p energy; occupancy / coalescing on ERI launch; load CUDA skills | track cuda | twin: `test_rhf.cu` | files: device integrals + RHF loop | gate: parity green; optional large-system note only (no false "CUDA wins" on H₂O) | *unblocked: pshell forces landed*
- [ ] `cuda-rhf-level-shift` | **P2** Wire `level_shift` on device s/p path (currently refused in `rhf.cu`) **or** document permanent host fallback | track cuda | twin: shifted SCF if any reg | files: `rhf.cu` / mixer | gate: no silent refuse; pins green |
- [ ] `cuda-gpw-foothold` | **P3 / open-gate only** Thin Γ-point GPW device slice (collocate **or** Poisson/FFT) with same-settings CPU oracle — **not** full method product; staff only if CPU OT water + MG pins still green and public GPW `run`/`forces` frozen | track cuda | twin: water Γ energy CPU oracle | files: new `cuda/gpw/` | gate: energy match tol; Frame-only API; `ATV_BUILD_GPW`+CUDA gated; no hybrid/GAPW |

## in_progress

Wave 1 residual (not yet on `dev`):

- [~] `cpu-ot-diis-outer` | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `src/cpu/gpw/gpw_solver.cpp` | branch: `wf/escuda-cpu-ot-diis-outer` @ `59d31079` (g·e B-matrix + CG fallback; claimed 93→32 / 180→94 it; **merge pending**)

## blocked

- [!] `global-xtensor-xsimd` | never enable global `XTENSOR_USE_XSIMD`
- [!] `cuda-gpw-full-product` | blocked until foothold green + CPU architecture frozen for GPW public shape

## done

Wave 1 land 2026-08-06 (`dev` @ `8337251b`):

- [x] `cpu-xc-spectral-fft` | **P1** GGA KS spectral gradient/div: fused multi-out FFT + FFTW_MEASURE plans | track cpu | twin: `ot-minimizer-water` | files: `grid_fft.cpp`, `xc_grid_gga.cpp`, `fft_cache.{h,cpp}` | merge: `977710fb` / feat `db132bb8` | evidence: XC ~108→**95** ms/iter; quiet wall ~16.5→**16.2 s** (ratio ~6.11→**~5.99** vs CP2K 2.7 s note); energy pin stable; FD `gpw-forces-water` PASS; non-bit-exact MEASURE FP with stated tol |
- [x] `cuda-rhf-pshell-forces` | **P0** Device analytic forces for l≤1 s/p | track cuda | twin: H₂O STO-3G CUDA↔CPU + `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf.cu`, force kernels | merge: `8337251b` / feat `e6f984d8` | GH200 job log `.claude/merge-cuda-rhf-913053.out`: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS**; tols: H₂ |ΔF| **1e-5** eV/Å, H₂O |ΔF| **1e-4** eV/Å, energy **ha_to_ev(1e-6)**; `last_force_on_device()` fail-closed; no host RHF for covered shells |
