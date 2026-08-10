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
CUDA RHF / PW envelope `.claude/notes/cuda-rhf-device-only.md`.

## Fair twins (CPU)

| Twin | Role | Baseline ratio (authority) |
| --- | --- | ---: |
| `gpw-multigrid-kpoints` | multi-k MG products | **2.33×** (worst) |
| `gpw-kpoints-si` | multi-k dense χ^k | **1.18×** |
| `gpw-uks-o2-gamma` | Γ UKS DIIS | **1.077×** (≤1.2× closed) |
| `ot-minimizer-water` | OT iter + per-KS residual | OT ~**1.20×**; SCF body **≤1.0×** |

Wall numbers: [gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md)
(arrhenius1, 1 MPI × OMP=4, CP2K 2026.1 quiet TIMING MAXIMUM). **Do not**
quote superseded ~1.25× MAIN figures from wave-7 claims.

## CUDA correctness twins

| Twin | Role |
| --- | --- |
| `tests/cuda/scf/test_rhf.cu` | device energy + s/p forces + level_shift invariant (H₂ pure-s + H₂O s/p) |
| H₂O STO-3G RHF | CPU oracle for CUDA energy/forces (same settings) |
| `regressions/pw/pw-cuda-lda-si` | CUDA↔CPU PW LSDA energy parity (Si; small cell, correctness only) |
| `regressions/pw/cuda-pw-forces` | CUDA↔CPU PW analytic ionic forces (Si dimer stretch; LSDA dual-gate) |
| `regressions/pw/cuda-pw-forces-pbe` | CUDA↔CPU PW analytic ionic forces under **PBE** (Si dimer stretch) |
| `regressions/pw/cuda-pw-stress` | CUDA↔CPU PW Nielsen–Martin cell stress (Si; LSDA dual-gate) |
| `tests/cuda/planewave/test_pw_force_local.cu` | device local HF force kernel vs CPU fragment |
| `tests/cuda/planewave/test_pw_cell_stress.cu` | empty/open-shell reject + H₂ CUDA↔CPU stress parity |
| `tests/cuda/gpw/test_device_poisson.cu` | Γ Poisson/FFT foothold vs CPU oracle |
| `tests/cuda/gpw/test_xc_lda.cu` | Padé LDA XC grid foothold vs CPU oracle |
| `tests/cuda/gpw/test_product_collocate.cu` | device product ρ collocate foothold vs CPU collint |
| `tests/cuda/gpw/test_product_integrate.cu` | device product V·shape integrate foothold vs CPU collint |
| `tests/cuda/gpw/test_ks_chain.cu` + `reg_cuda-gpw-ks-chain` | composed ρ→V_eff stage (LDA) vs CPU `ks_grid_build_rks` |
| `tests/cuda/gpw/test_ks_gga.cu` + `reg_cuda-gpw-ks-gga` | composed ρ→V_eff **PBE-GGA** stage vs CPU `ks_grid_build_rks(PBE)` |
| `tests/cuda/gpw/test_ks_multigrid.cu` + `reg_cuda-gpw-ks-multigrid` | multi-grid product collocate/integrate KS stage vs CPU Γ MG fragment |
| `reg_cuda-pw-collinear-spin` | CUDA↔CPU collinear LSDA O₂ dual-gate (fixed occupation; no smear) |

## pending

**Harness wave 7 land** 2026-08-10 (`dev` @ `018dca0d` after merges
`d2ccc627` / `018dca0d`). Staffed and landed both remaining CUDA pending
slices; **cpu_pending=0** held (no pin-safe fair-wall lever). Fair walls
**unchanged** (authority
[gpw-perf-baseline.md](../../.claude/notes/gpw-perf-baseline.md); worst_ratio
**2.33×** MG MAIN). Confirmed open after land: restricted metallic smear
landed; spin+smear / spin forces / spin stress / spin-PBE still reject;
`KsMultiGridChainDevice` is **stage unit only** (no public `GpwDft`).
Staff only `wf/escuda-<slug>` off current `dev`. Do not invent
hybrid/GAPW/PAW/meta-GGA. **No gpw-perf-queue promote**.
**new_slugs_added=0.** **pending_count=0**; **cpu_pending=0**;
**cuda_pending=0**; **gaps_remaining=5**
(residual-mass + blocked full-product / mg-main-structural — not staffable).

### Fair-wall rank (1 MPI × OMP=4; worst residual first)

| Twin | ATV | CP2K twin | ratio | lever class |
| --- | ---: | ---: | ---: | --- |
| `gpw-multigrid-kpoints` MAIN | **1.6 s** | **0.688 s** quiet | **2.33×** | **worst_ratio** — coll/int ~2.0×/2.8× per-call; pin-safe kernel + structure + sphere **exhausted** |
| `ot-uks-o2` OT | **~11.8 s** / 27 it (post OUTER polish) | **8.68 s** / 30 SCF | **~1.4×** | ≤3× aim **met**; not primary vs MG |
| `ot-minimizer-water` OT-only | **~2.99 s** (10 it) | **2.50 s** | **~1.20×** | control + per-KS residual closed (SCF body **≤1.0×**) |
| `gpw-uks-o2-gamma` | **14.75 s** | **13.691 s** | **1.077×** | ≤1.2× **closed** — do not re-open dual-spin fuse |

Secondary (not in primary four; still baseline): `gpw-kpoints-si` MAIN **1.18×**
— hold products-vs-dense A/B until MG nearer parity.

MG budget (ngrid=37800, ~13 SCF): setup STVnl+products ~0.42 s once +
~13×(coll 30 + int 48 + XC/Poisson ~5) ≈ **1.5 s** → matches 1.6 s pin. CP2K
coll/int alone ~13×(14.5+17.1) ≈ **0.41 s**. Closing MAIN is **blocked**
`mg-main-structural` (non-pin-safe windows or task-list rewrite) — not more
collint micro-opts (`cpu-collint-int-residual-2` wave-8 **no-land**).

### CPU — wall vs CP2K (contribution to worst ratio)

**cpu_pending = 0.** Honest staffable set empty: worst twin has no pin-safe
lever left. Authority walls unchanged vs baseline. Pure-s + mixed-angular
2-center SIMD **landed** (slab-class; invisible on four fair twins). No further
CPU polish staffed here without a pin-safe wall lever. **No promote** from
[gpw-perf-queue.md](gpw-perf-queue.md) P1/P2 holds.

*Not pending (residual mass still counted in `gaps_remaining`, not staffable):*

| Row | Why not pending |
| --- | --- |
| `cpu-collint-int-residual-2` | wave-8 **no-land** — fused segment accumulate noise-flat; CP2K-style per-chunk hsum **regressed** |
| MG MAIN → ~1× (`mg-main-structural`) | **blocked** — pin-safe collint/structure/sphere campaigns closed; needs non-pin-safe windows or task-list rewrite |
| `p1-products-default-k` | stay on gpw-perf-queue; A/B only after MG nearer parity + products ≤ dense |
| `p2-mpi-product-grid` | multi-rank; not 1-MPI fair-wall lever |
| OT / UKS further polish | water OT ~1.20× / SCF body ≤1.0×; UKS DIIS 1.077× closed; OT UKS ≤3× met |

*Already landed this loop (CPU):* `cpu-xc-spectral-fft`, `cpu-ot-diis-outer`,
`cpu-ot-residual`, `cpu-stvnl-residual`, `cpu-collint-residual`,
`cpu-products-mg-setup`, `cpu-collint-shell-tasks`, `cpu-collint-integrate-residual`,
`cpu-simd-2center-setup`, `cpu-simd-2center-angular`
(+ gpw-perf OT match / sphere / UKS fuse / STVnl setup — walls in baseline).

### CUDA — corresponding ES (open-gate readiness × user value)

**Tree fact on `dev` @ `018dca0d` (wave-7 land):** Confirmed present: prior
wave-6 surface **plus** restricted metallic smear on `cuda::pw::PW` (FD/Gaussian/MP
occupations + extra_bands + Mermin free energy) **and** RHF residual/V_nuc D2H
on Evaluate non-default stream. **cuda_pending = 0.**

| Slice on `dev` | Status | Depth gap |
| --- | --- | --- |
| RHF s/p energy + forces + level_shift + path-select + device CDIIS + **non-default stream D2H** | **landed** | higher-L host-fill remains |
| Γ GPW Poisson/FFT + Padé LDA XC + product collocate/integrate + **composed `KsChainDevice`** (LDA + **closed-shell PBE**) + **`KsMultiGridChainDevice`** | **landed** stage units (MG GH200 **1051042** max\|ΔV_eff\|=8.1e-10) | stage only — **≠** full method product |
| `cuda::pw::PW` LSDA + device PBE-GGA + residency + stream + large-cell + **forces (LSDA+PBE)** + **analytic cell stress** + **collinear LSDA spin** + **restricted metallic smear** | **landed** energy (+ spin energy + restricted free energy); closed-shell forces/stress | spin+smear / spin-PBE / spin forces / spin stress still reject |

**Staff order:** queue empty for pin-safe / open-gate CUDA depth in this loop.
Full GPW product stays **blocked**. Higher-L RHF stays product-out-of-scope.

Open gate: CPU twin green + public `run`/`forces`/`stress` stable where owned
+ stated CUDA↔CPU tols + device-only kernels + **GH200 log evidence**.

*Not pending this loop:*

| Row | Why |
| --- | --- |
| `cuda-gpw-full-product` | **staffable** (freeze green on `dev` @ `0147fe02`) — public shape = CPU `GpwDft` ([cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)); footholds / ks-chain / GGA / MG compose under that product; not auto-landed; GH200 + CUDA↔CPU tols |
| CUDA d-shell / higher-L RHF product | out of loop product scope (host-fill remains for non-s/p STO-3G) |
| Further lone GPW kernel footholds | collocate↔integrate + LDA + GGA + MG chain complete; depth is product — not another lone foothold |
| `cuda-pw-pbe-large-cell` | large-cell LSDA ladder **closed**; optional PBE remeasure not a correctness gap |
| Classical MD CUDA | OpenMM track — not this loop |
| hybrid/GAPW/PAW/meta-GGA | method loops |

*Already landed this loop (CUDA):* `cuda-rhf-pshell-forces`,
`cuda-rhf-device-only-polish`, `cuda-rhf-level-shift`, `cuda-rhf-path-select-host`,
`cuda-rhf-cdiis-device`, `cuda-rhf-nondefault-stream`, `cuda-gpw-foothold`,
`cuda-gpw-xc-lda-foothold`, `cuda-gpw-rho-collocate-foothold`,
`cuda-gpw-product-integrate-foothold`, `cuda-gpw-ks-chain`, `cuda-gpw-ks-gga`,
`cuda-gpw-ks-multigrid`, `cuda-pw-method-slice` (+ `cuda-pw-kpoint-hoist`,
`cuda-pw-zheevd`, `cuda-pw-devptr-nrm2` residency polish on `dev`),
`cuda-pw-pbe-device`, `cuda-pw-large-cell-measure`, `cuda-pw-nondefault-stream`,
`cuda-pw-forces`, `cuda-pw-forces-pbe`, `cuda-pw-stress`,
`cuda-pw-collinear-spin`, `cuda-pw-metallic-smear`.

## in_progress

(empty — harness wave 7 closed. Staffable cuda/cpu pending empty; residual
gaps are blocked structural / out-of-scope.)

**Foreign incomplete worktrees on parent (do not steal / do not force-merge):**

| Worktree branch | Loop ownership | Note |
| --- | --- | --- |
| `wf/cuda-md-first-regression` | **OpenMM / MD** — not this loop | leave alone |
| stale `wf/cuda-pw-collinear-spin` (pre-rebase tip) | **superseded** by landed `escuda-cuda-pw-collinear-spin` | delete only if fully-merged or abandoned; no re-staff |
| stale merged `wf/escuda-*` tips | **merged** on `dev` | clearup-owned prune |

## blocked

- [!] `global-xtensor-xsimd` | never enable global `XTENSOR_USE_XSIMD`
- [!] `cuda-gpw-full-product` | product implement under freeze contract (CPU `GpwDft` public shape) — foothold + LDA/GGA/MG `ks_chain` green ≠ method product; freeze **merged** (`0147fe02` / `aea603bb`); staffable under contract; GH200 + stated CUDA↔CPU tols; still a large product slug
- [!] `mg-main-structural` | MG MAIN **2.33×** residual after pin-safe collint + structure + sphere campaigns; further cut needs non-pin-safe windows or task-list rewrite beyond this loop’s pin law — do not staff as soft-tol or self-golden

## done

Harness wave 7 land 2026-08-10 (`dev` @ `018dca0d`):

- [x] `cuda-pw-metallic-smear` | **P2** Lift restricted `smearing != none` — Fermi–Dirac / Gaussian / Methfessel–Paxton occupations + `extra_bands` on device band path; multi-k device Ritz cache; Mermin free energy; spin+smear / forces / stress still reject | track cuda | twin: unit guards + `reg_cuda-pw-metallic-smear` Al FD dual-gate | files: `device_solver.{cu,cuh}`, `pw_solver.{cu,cuh}`, reg, guards | merge: `d2ccc627` / feat `2891e824` | evidence: GH200 **1051119** ctest 9/9; dual-gate E both **−7.5496489573 Ry** n_occ=3 \|ΔE\|=**1.24e-14** Ry (tol 5e-3); **perf_exempt** small cell; not spin+smear / forces / stress |
- [x] `cuda-rhf-nondefault-stream` | **P3** Residual/V_nuc D2H on Evaluate non-default stream (not default stream); host force H2D fallback stream-bound | track cuda | twin: `tests/cuda/scf/test_rhf.cu` + device integrals | files: `scf_solver.cu`, `rhf.cu`, test | merge: `018dca0d` / feat `a8e23747` | evidence: GH200 **1051227** `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (2.98 + 0.86 s); tols unchanged |

Harness wave 6 land 2026-08-10 (`dev` @ `61f873c9`):

- [x] `cuda-pw-collinear-spin` | **P1** Lift `spin_polarized` throw on `cuda::pw::PW` — device nspin=2 density + collinear LSDA (VWN5 spin); fixed-occupation UKS; forces/stress still reject spin; no smear | track cuda | twin: unit XC spin parity + `reg_cuda-pw-collinear-spin` O₂ LSDA dual-gate | files: `device_density.{cu,cuh}`, `xc_lda.{cu,cuh}`, `pw_solver.{cu,cuh}`, `device_solver.{cu,cuh}` | merge: `0482e668` / feat `9c78896b` | evidence: GH200 **1051041** ctest 9/9; E both **−28.6294634700 Ha** mag=2 \|ΔE\|=**1.07e-13** Ha (tol 2e-3); post-merge combined **1051075** same parity; prior COMPLETED **1050941** exit 0; **perf_exempt** small cell; not spin-PBE / spin forces / smear |
- [x] `cuda-gpw-ks-multigrid` | **P2** Extend composed GPW KS stage with multi-grid product collocate/integrate (`MultigridTransferDevice` spectral prolong/restrict + `KsMultiGridChainDevice`) — **stage unit only**, not `GpwDft::run` | track cuda | twin: `test_cuda_gpw_ks_multigrid` + `reg_cuda-gpw-ks-multigrid` vs CPU Γ MG fragment | files: `ks_chain.{cu,cuh}`, `multigrid_transfer.{cu,cuh}` | merge: `61f873c9` / feat `ffd67884` | evidence: GH200 **1051042** unit 3/3 PASS; dual-gate max\|ΔV_eff\|=**8.095e-10**; post-merge combined **1051075** same; prior COMPLETED **1051020** exit 0; not full product |

Harness wave 5 land 2026-08-10 (`dev` @ `721ac286`):

- [x] `cuda-pw-forces-pbe` | **P1** Dual-gate analytic ionic forces under **device PBE-GGA** SCF — stream share `den.stream()` on force path; reg `cuda-pw-forces-pbe` | track cuda | twin: unit `test_pw_force_local` + `reg_cuda-pw-forces-pbe` Si dimer GTH-PBE-q4 | files: `pw_forces.{cu,cuh}`, `pw_solver.cu`, reg | merge: `a92d1af0` / feat `c5ae8fa2` | evidence: GH200 rebased **1050675** max\|ΔF\|=2.634e-13 + post-merge **1050801** max\|ΔF\|=**2.628e-13** eV/Å (tol 5e-3); E both −13.89672028 Ry; **perf_exempt** small cell; not stress/spin/smear |
- [x] `cuda-gpw-ks-gga` | **P1** Extend composed GPW KS stage from Padé LDA to **closed-shell PBE-GGA** (`SpectralFftDevice` + `KsXcKind::PBE`) — **stage unit only**, not `GpwDft::run` | track cuda | twin: `test_cuda_gpw_ks_gga` + `reg_cuda-gpw-ks-gga` vs CPU `ks_grid_build_rks(PBE)` | files: `ks_chain.{cu,cuh}`, `spectral_fft.{cu,cuh}` | merge: `721ac286` / feat `ac713518` | evidence: GH200 rebased **1050680** dual-gate e_H/e_xc ~1e-15 max\|ΔV_eff\|=8.0e-16 max\|Δacc\|=3.9e-14 + post-merge **1050825**; not full product / not multigrid |

Harness wave 4 land 2026-08-10 (`dev` @ `2987d20f`):

- [x] `cuda-pw-stress` | **P0** Device Nielsen–Martin analytic cell stress for `cuda::pw::PW` — public `cuda::pw::pw_stress`; density remap ρ·Ω₀/Ω + NSCF strain-FD twin of CPU | track cuda | twin: unit `test_pw_cell_stress` + `reg_cuda-pw-stress --device=both` | files: `pw_stress.{cu,cuh}`, `device_density.{cu,cuh}`, `pw_solver.{cu,cuh}` | merge: `98bd743a` / feat `2c662eab` | evidence: GH200 worker **1050263** + post-merge unit **1050357** + reg **1050604** — unit PASS; max\|Δσ\| **1.36e-12** eV/Å³ (tol 5e-2); E both **−13.86296636** Ry \|ΔE\|=2.132e-14 Ry; **perf_exempt** small cell; not spin/smear/PBE-force |
- [x] `cpu-simd-2center-angular` | **P3** Portable xsimd mixed-angular 2-center S/T Hermite (L≤d image-axis McMurchie–Davidson; pure-s path retained) | track cpu | twin: unit one_electron angular + `reg_cpu-simd-2center-angular` vs prim image sum | files: `one_electron.cpp` | merge: `2987d20f` / feat `1add075a` | evidence: unit+slab reg PASS; fused vs prim rel **1e-11**; FD `gpw-forces-water` max\|dF\|=3.243e-04; **ratio_before=ratio_after=2.33**; not bit-exact (xsimd exp/reduce); not fair-wall lever |

Harness wave 3 land 2026-08-10 (`dev` @ `1ba1678f`):

- [x] `cuda-pw-forces` | **P0** Device analytic ionic forces for `cuda::pw::PW` — local HF + Ewald + KB nonlocal; public `cuda::pw::pw_forces_analytic`; CPU force-path XC fix (`method.xc`) | track cuda | twin: unit `test_pw_force_local` + `reg_cuda-pw-forces` | files: `pw_forces.{cu,cuh}`, `device_solver.cu`, `pw_solver.{cu,cuh}`, CPU `pw_forces.cpp` | merge: `1ba1678f` / feat `a82a0758` | evidence: GH200 **1050105** max\|ΔF\| **2.86e-13** eV/Å (tol 5e-3); E both −13.85556302 Ry; not stress/spin/smear |
- [x] `cuda-gpw-ks-chain` | **P1** Compose landed GPW device footholds into one ρ→V_eff **stage unit** (collocate → Poisson → Padé LDA → integrate) — **not** `GpwDft::run` | track cuda | twin: `test_cuda_gpw_ks_chain` + `reg_cuda-gpw-ks-chain` | files: `ks_chain.{cu,cuh}` | merge: `9b19991b` / feat `47781d68` | evidence: GH200 **1049858** — unit 5/5; dual-gate e_H/e_xc ~1e-15, max\|ΔV_eff\|=4.4e-16, max\|Δacc\|=6.9e-14; not full product / not GGA |
- [x] `cpu-simd-2center-setup` | **P3** Portable xsimd pure-s analytic 2-center S/T (slab-class; invisible on four fair twins) | track cpu | twin: unit one_electron + `reg_cpu-simd-2center-setup` literature closed form | files: `one_electron.cpp` | merge: `e49e0757` / feat `b363e5e6` | evidence: unit 65/65; slab \|dS\|=2.2e-16 \|dT\|=4.4e-16; microbench ~24–26×; FD `gpw-forces-water` PASS max\|dF\|=3.243e-04; **ratio_before=ratio_after=2.33**; not bit-exact (xsimd exp/reduce) |

Harness wave 2 2026-08-10 (`dev` @ `8280ed89`):

- [x] `cuda-pw-nondefault-stream` | **P2** Non-default stream on PW density+band SCF path — `DeviceDensity` owns `cudaStream_t`; band solver borrows; kernels/cuFFT/cuBLAS/cuSOLVER bound; mid-SCF fences stream-scoped | track cuda | twin: unit stream handle + `pw-cuda-lda-si` + `reg_cuda-pw-nondefault-stream` | files: `device_density.{cu,cuh}`, `device_solver.{cu,cuh}`, `pw_solver.cu`, job shield | merge: `8280ed89` / feat `05b52ba2` | evidence: GH200 **1049681** — ctest **9/9 PASS**; MAX_EIG_DIFF **5.274e-15 Ha**; \|E_cuda−E_cpu\| **1.243e-14 Ry** (both regs); tols unchanged; **perf_exempt** small cell; wipe-shield `BUILD=_pw_stream_work` |

GapCensus wave 4 recovery 2026-08-10 (`dev` @ `e22325b7` / large-cell merge
`948d0d86`):

- [x] `cuda-pw-large-cell-measure` | **P1** ES order step 4 — first large-cell CPU↔CUDA wall (Si 8/64/216 ladder; small cells correctness-only) | track cuda | twin: `tools/bench_cuda_pw_cell.cu` + `pw-cuda-lda-si` | files: harness + `tools/gpu_job_cuda_pw_large_cell.sh` + note | merge: `948d0d86` | evidence: GH200 **1035278** (primary) + **1048264** reconfirm; 64-atom CUDA **~9.65 s** vs CPU **>5400 s** (\(T_{\mathrm{cu}}/T_{\mathrm{cp}} < 0.0018\)); slice \|ΔE\|=1.243e-14 Ry; note [cuda-pw-large-cell.md](../../.claude/notes/cuda-pw-large-cell.md); measurement-only (no energy-path change) |

Harness wave 1 2026-08-10 (`dev` @ `948d0d86`, PBE land `3f901034`):

- [x] `cuda-pw-pbe-device` | **P0** Device PBE-GGA for `cuda::pw::PW` — lift LSDA-only gate; closed-shell PBE-X + PW92-based PBE-C at ζ=0; spectral ∇ρ / −∇·flux in `DeviceDensity`; same-settings CPU oracle | track cuda | twin: unit `test_cuda_pw_xc_pbe` + DeviceDensity stage parity + e2e `reg_cuda-pw-pbe-device` | files: `xc_pbe.{cu,cuh}`, `device_density.{cu,cuh}`, `pw_solver.{cu,cuh}` | merge: `3f901034` | evidence: GH200 job **1048320** — units **4/4 PASS**; dual-gate \(E_\mathrm{cpu}=E_\mathrm{cuda}=-14.84955549\) Ry (7 it); \(\lvert\Delta\rvert=1.421\times10^{-14}\) Ry ≪ 1e-3 Ry; tols **1e-12·max(1,\|ref\|)** pointwise; roadmap PBE CUDA ✓ (PW closed-shell only); **perf_exempt** small cell; not forces/stress/spin/PBE-family |

Wave 7 land 2026-08-06 (`dev` @ `8b76cbe7`):

- [x] `cpu-collint-integrate-residual` | **P1** Residual integrate toward CP2K ~20 ms class — width-4 segment accumulate on AVX-512 (CP2K AVX2 idiom; avoid width-8 ZMM spill), short-segment scalar, Ltot triangular skip on rectangular monom walks, zero-pol reverse, cab shared-alpha from n≥4 | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `product_collint.cpp` | merge: `8b76cbe7` / feat `c6659278` | evidence: int **~58→~45** ms/iter; coll ~29; MAIN **~1.6 s**; post-baseline CP2K quiet **0.688 s** → ratio **2.33×** (supersedes contemporaneous ~1.25× claim); pin \|ΔE\|=2.029e-05 Ha; unit collocate+multigrid PASS; FD `gpw-forces-water` PASS max\|dF\|=3.243e-04; not bit-exact (SIMD order) |

Wave 6 land 2026-08-06 (`dev` @ `a9f87206`):

- [x] `cpu-collint-shell-tasks` | **P0** Shell-level collint task list (`CollintTask` ranges once in `build_products`; collocate/integrate drive contiguous monom runs; cab-style shared-alpha reverse contract on integrate) | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `multigrid.{h,cpp}`, `product_collint.cpp` | merge: `f3571435` / feat `2debae55` | evidence: MAIN **~1.9→~1.6 s**; coll **~63→~30** / int **~76→~58** ms/iter; pin unit multigrid/collocate PASS; FD `gpw-forces-water` PASS max\|dF\|=3.243e-04 (post-merge log) |
- [x] `cuda-gpw-product-integrate-foothold` | **P1** Thin device product V·shape integrate foothold (V·shape → per-product acc) vs same-settings CPU `collint_integrate_product` / continuous host — transpose of landed ρ collocate; reuses `DeviceProduct` / `CollocateGridDesc` | track cuda | twin: `test_cuda_gpw_product_integrate` | files: `product_integrate.{cu,cuh}`, unit test | merge: `a9f87206` / feat `2b317beb` | GH200 post-merge **915825**: `test_cuda_gpw_product_integrate` **Passed 0.95 s** + collocate **2.00 s** + `test_cuda_scf_rhf` **2.25 s** (**3/3 PASS**); tols device≡host continuous **1e-12·peak + 1e-14**, vs collint **1e-10·peak + 1e-14** (p-type **1e-9·peak + 1e-13**); not full GPW product |

Wave 5 land 2026-08-06 (`dev` @ `759ca486`):

- [x] `cpu-products-mg-setup` | **P0** Geometry-batch `build_products` / products_mg — monoms sharing collint geometry compute K/pc once; compact batch sort + flat exclusive-scan contrib packing | track cpu | twin: `gpw-multigrid-kpoints` setup mark `products_mg` | files: `multigrid.cpp` (`build_products`) | merge: `72926013` / feat `bbf1f167` | evidence: products_mg **~285→~18 ms**; MAIN **~2.4→~1.9 s**; pin bit-stable; FD `gpw-forces-water` PASS; no 5th collocation layout |
- [x] `cuda-rhf-cdiis-device` | **P1** Device CDIIS (Pulay) Fock mixer on s/p RHF path — CPU MixStage / DIISMixer twin; e = Xᵀ[F,P]_S X; history on device; tiny Pulay solve host-only | track cuda | twin: `tests/cuda/scf/test_rhf.cu` + H₂O STO-3G CUDA↔CPU | files: `diis_mixer.{cu,cuh}`, `scf_solver.cu`, `rhf_fock.cu` | merge: `d575e50b` / feat `f21a1173` | GH200 worker **915746** + post-merge **915778**: `test_cuda_scf` + `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (0.90 + 2.02 + 0.82 s); tols unchanged (ha_to_ev(1e-6) / 1e-8 Ha class) |
- [x] `cuda-gpw-rho-collocate-foothold` | **P2** Thin device product ρ collocate foothold vs same-settings CPU collint oracle (Γ small grid; s/p + multi-center s–s) | track cuda | twin: `test_cuda_gpw_product_collocate` | files: `product_collocate.{cu,cuh}`, unit test; AVX-512 collint residual gated for aarch64 GH200 | merge: `759ca486` / feat `0d2868e7` | GH200 worker **915754** + post-merge **915778**: `test_cuda_gpw_product_collocate` **Passed 0.80 s** (+ poisson/XC/RHF suite **6/6 PASS**); tols device≡host continuous **1e-14**, vs collint **1e-10·peak + 1e-14** (p-type **1e-9·peak**); not full GPW product |

Wave 4 land 2026-08-06 (`dev` @ collint merge `5fa8b0c6` + CUDA `44a5d858`):

- [x] `cpu-collint-residual` | **P0** Residual collint: hybrid xsimd FMA + 4-wide residual, lp_z≤9, cab-style alpha for large groups, TLS multi-imr sphere, gamma group_idx nthread private grids, integrate \|V(pc)\| once per geometry | track cpu | twin: `gpw-multigrid-kpoints` MAIN | files: `product_collint.cpp`, `multigrid.cpp` | merge: `5fa8b0c6` / feat `6da97a85` | evidence: MAIN remeasure **~2.4 s**; coll **~63** / int **~76** ms/iter; pin bit-stable \|ΔE\|=5.7e-6 Ha; FD `gpw-forces-water` PASS max\|dF\|=3.2e-4; OT water 25 it + UKS O₂ 27 it / ~11.8 s PASS; shell-level tasks still open |
- [x] `cuda-gpw-xc-lda-foothold` | **P1** Thin device Padé LDA XC grid vs same-settings CPU oracle (closed-shell + LSD + energy reductions) | track cuda | twin: `test_cuda_gpw_xc_lda` | files: `src/cuda/gpw/internal/xc_lda.{cu,cuh}`, unit test | merge: `cc450b59` / feat `90acb1d5` | GH200 worker **915678** + post-merge **915704**: `test_cuda_gpw_xc_lda` **Passed 0.86 s**; tols ~**1e-12** per-point, energy rel **~1e-10**; not full GPW product |
- [x] `cuda-rhf-path-select-host` | **P2** Host-Z path select without setup type[] D2H (`cache_host_types` before upload) | track cuda | twin: `tests/cuda/scf/test_rhf.cu` | files: `rhf.cu`, `rhf.cuh` | merge: `44a5d858` / feat `6f179574` | GH200 worker **915654** + post-merge **915704**: `test_cuda_scf` + `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS** (0.95 + 2.03 + 0.87 s); tols unchanged |

Wave 3 land 2026-08-06 (`dev` @ `9f2d791c`):

- [x] `cpu-ot-residual` | **P0** OUTER_SCF polish (fixed-H eigen+mix after residual moderate; no DFT+U) | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `gpw_solver.cpp`, `ot.h` | merge: `4249fd3f` / feat `867c1add` | evidence: UKS O₂ **87→27** it, OT wall **~52→13 s** / CP2K **8.68 s** → **~1.53×** (was ~4.4×; aim ≤3× **met**); water **32→25** it; pins + FD `gpw-forces-water` PASS; no tol soften |
- [x] `cpu-stvnl-residual` | **P1** STVnl Hermite geometry-key reuse + lattice phase flatten | track cpu | twin: `gpw-multigrid-kpoints` | files: `one_electron.cpp`, `kpoints.cpp` | merge: `dfe68fdd` / feat `76239cfc` | evidence: STVnl_k **~507→435 ms**; MG MAIN quiet best **~2.6→2.3 s**; pin bit-stable; forces-water PASS |
- [x] `cuda-rhf-level-shift` | **P2** Saunders–Hillier λ on device s/p Fock (no host-fill when λ≠0) | track cuda | twin: `tests/cuda/scf/test_rhf.cu` `CUDALevelShiftEnergyInvariant` | files: `rhf.cu`, `rhf_fock.cu`, `scf_solver.cu` | merge: `ba3a8145` / feat `77eb7f79` | GH200 worker job **913709** + post-merge **915584**: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **PASS**; λ=0.5 energy matches λ=0 and CPU @ 1e-8 Ha |
- [x] `cuda-gpw-foothold` | **P3** Thin Γ-point device Poisson/FFT vs CPU oracle | track cuda | twin: `test_cuda_gpw_device_poisson` | files: `src/cuda/gpw/internal/poisson_fft.{cu,cuh}`, unit test | merge: `9f2d791c` / feat `f604577c` | GH200 worker **915518** + post-merge **915584**: `test_cuda_gpw_device_poisson` **Passed 0.97 s** + `test_cuda_pw_poisson` **Passed 1.78 s** (**100%** of 4); tols max\|ΔV\| **1e-9…1e-10**, Hartree **1e-10…1e-12** rel; not full GPW product |

Wave 2 land 2026-08-06 (`dev` @ `fb9e7ed8`):

- [x] `cpu-ot-diis-outer` | **P0** Tighten Γ OT MINIMIZER DIIS outer count (g·e B-matrix + SAFE_DIIS/Armijo + CG fallback) | track cpu | twin: `ot-uks-o2` then `ot-minimizer-water` | files: `src/cpu/gpw/internal/ot.cpp`, `gpw_solver.cpp` | merge: `25c2268c` / feat `660a50d1` | evidence: water **93→32** OT it pin PASS; UKS O₂ **180→87–94** it, OT wall **~38.5 s** / CP2K **8.68 s** → **~4.4×** (was ~7.1×; land claim 3.74×); FD `gpw-forces-water` PASS max\|dF\|=3.2e-4; \|E_OT−E_DIIS\|~1e-10; no tol soften |
- [x] `cuda-rhf-device-only-polish` | **P1** Wave-7: drop mid-build V_nuc D2H/H2D (energy ← `d_V_nuc`); ERI `__launch_bounds__(256,2)` + occupancy-capped grid-stride | track cuda | twin: `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf_fock.cu`, `rhf_density.cu` | merge: `fb9e7ed8` / feat `1ae27395` | GH200 job log `.claude/merge-cuda-polish-913612.out`: `test_cuda_scf_rhf` **Passed 2.29 s** + `test_cuda_scf_device_integrals` **Passed 0.82 s** (**100%**); need `-DCMAKE_CUDA_ARCHITECTURES=90` on GH200; tols unchanged from wave 6 |

Wave 1 land 2026-08-06 (`dev` @ `8337251b`):

- [x] `cpu-xc-spectral-fft` | **P1** GGA KS spectral gradient/div: fused multi-out FFT + FFTW_MEASURE plans | track cpu | twin: `ot-minimizer-water` | files: `grid_fft.cpp`, `xc_grid_gga.cpp`, `fft_cache.{h,cpp}` | merge: `977710fb` / feat `db132bb8` | evidence: XC ~108→**95** ms/iter; quiet wall ~16.5→**16.2 s**; energy pin stable; FD `gpw-forces-water` PASS; non-bit-exact MEASURE FP with stated tol |
- [x] `cuda-rhf-pshell-forces` | **P0** Device analytic forces for l≤1 s/p | track cuda | twin: H₂O STO-3G CUDA↔CPU + `tests/cuda/scf/test_rhf.cu` | files: `device_integrals.cu`, `rhf.cu`, force kernels | merge: `8337251b` / feat `e6f984d8` | GH200 job log `.claude/merge-cuda-rhf-913053.out`: `test_cuda_scf_rhf` + `test_cuda_scf_device_integrals` **100% PASS**; tols: H₂ |ΔF| **1e-5** eV/Å, H₂O |ΔF| **1e-4** eV/Å, energy **ha_to_ev(1e-6)**; `last_force_on_device()` fail-closed; no host RHF for covered shells |

Post-loop CUDA PW residency (on `dev`, evidence in `.claude/notes/cuda-rhf-device-only.md`):

- [x] `cuda-pw-method-slice` | `cuda::pw::PW` + device density side; GH200 **948152** parity |
- [x] `cuda-pw-kpoint-hoist` | per-k tables once per run; bit-identical correctness |
- [x] `cuda-pw-zheevd` | cuSOLVER subspace eigen; CUDA↔CPU E at roundoff |
- [x] `cuda-pw-devptr-nrm2` | device-pointer residual norms; bit-identical to prior |
