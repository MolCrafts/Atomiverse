# Architecture notes

## [2026-07-17] SCF is a Driver; methods are concrete — kill the bag

**Binding rule (AGENTS.md / CLAUDE.md).** Three mistakes, one correction:

1. **`SCFSolver` is not a thing.** A solver is *concrete*: `RHF`, `UHF`,
   `RKS`, `UKS`, AM1, PW, GPW. There is no bag that becomes RHF by setting
   `method = HF` / `spin = Restricted`.
2. **SCF does not solve.** SCF is only the self-consistency **loop**
   (Fock → diagonalize → density → mix → residual). A loop is a **Driver**
   (stages + iteration + Handler termination) — same shape as MD / Minimize.
3. **No parallel world models.** Frame is the only system. Public
   `AtomList` / `IntegralPack` / `*Config` / `*System` / free-function
   factories are debt. Parameters are **members of the concrete method**.
   Integrals and AO maps are **private workspace**.

**Wrong (including the transitional "constructor rewrite"):**
```cpp
IntegralPack ints{frame, basis};
SCFConfig cfg{SCFConfig::RHF{}, ints, tol};
SCFSolver{cfg}.build(ints); SCFSolver::run(ints);
```

**Right (target):**
```cpp
cpu::scf::RHF rhf;
rhf.basis = BasisSet::STO3G;
rhf.tol = 1e-10;
rhf.run(frame);   // Frame in/out; SCF loop is the Driver inside/as RHF
```

**Supersedes** the same-day note that stopped at "free functions →
constructors". Turning `build_integrals` into `IntegralPack{…}` was
necessary cleanup of factories, **not** the end state. New work must not
extend `SCFSolver` / `SCFConfig` / public `IntegralPack` / `NDDOSystem`.

**Progress (2026-07-17 cleanup, continued):**

| Delivered | Status |
|---|---|
| Concrete methods `RHF`/`UHF`/`RKS`/`UKS` | **done** — Frame API; `mp2()` / `fci()` / `forces()` |
| Intermediate types off public path | **done** — live under `cpu/scf/internal/`; public tombstones `#error` |
| `AtomList` / `ShellList` | **private** — `internal/basis_workspace.h` only |
| `IntegralPack` / `SCFConfig` / `SCFSolver` / deriv pack | **private** — `scf/internal/*` |
| Public `basis_set.h` | **enum + name only** |
| Examples | **Frame + methods only**; scrub-gated |
| Grep gate | factories + examples intermediate types |

**Further progress:**
- Public `scf_result.h` (results only); `SCFConfig` private in `internal/scf_types.h`
- `methods.h` no longer includes internal gradient/config headers
- `NDDOSystem` moved to `semiempirical/internal/`; public tombstone `#error`
- High-value unit tests (UHF/UKS/spin/evaluate/minimizer/mp2/fci) on method API

**2026-07-17:** `benchmarks/` deleted (rebuild later). Si UPF kept at
`tests/data/pw/Si_lda_nc.upf`. Docs + unit tests updated to method API.

**Still debt:** bag-shaped `PWSolver` / `GPWSolver` / `NDDOSolver` / CUDA SCF;
~200 benchmark lines still use `scf/internal/`; docs tutorials lag.

**Historical note:** an earlier pass deleted a free-floating
`ATV::scf::Phase` and pushed the SCF cycle into `SCFSolver` as an "inner
solver". That fixed "SCF as a top-level category beside MD" but installed
the wrong abstraction (`SCFSolver` bag). The loop belongs on the Driver
node; the method is `RHF`/`UHF`/….

## [2026-06-10] cpu::pw migrated to PWSolver/PWEvaluator + MPI k-point parallelism; SCF audit fixes

**PW now mirrors the gaussian-SCF shape exactly.** `cpu::pw::PWSolver`
(public low-level inner solver, NOT a Driver subclass) owns the
facade-private `internal::PWWorkspace` and six step helpers in
`cpu::pw::internal::` (`PoissonSolver`, `XCBuilder`, `VeffBuilder`,
`DavidsonSolver`, `DensityBuilder`, `ConvergenceCheck`). The
`cpu::pw::PWEvaluator` Evaluator component (`Capability::Energy`)
plugs into `cpu::Evaluate`; `pw_scf()` survives as a thin shim over a
transient `PWSolver`. Forward declarations + `unique_ptr` keep
internal headers out of the public surface (same pattern as
`scf_solver.h`).

**Deleted:** `ATV::scf::Phase` (facade/stage.h), the dead
`Driver<scf::Phase, cpu::device_tag>` instantiation (src/cpu/driver.cpp),
`BuildContext::pw_workspace` (`void*` leak in component.h), the seven
Plan-shaped `PW_*` components (pw_poisson/pw_lda/pw_hamiltonian/
pw_veff/pw_davidson/pw_density/pw_convergence), and the dead
`PW_Density` duplicate — which carried a real bug: its Pulay history
was never trimmed, so after `pulay_depth` iterations it mixed against
the oldest-5 residuals forever. The consolidated `DensityBuilder` uses
a correct sliding window (`while (size > pulay_depth_) erase front`).

**Grep gate widened:** `tests/cuda/scf_phase_scrub.sh` now scans
include/, src/, examples/, tests/ and runs unconditionally as
`test_scf_phase_scrub` (was CUDA-only).

**Convention fixes folded in:** `kAngstrom2Bohr`/`kRy2Hartree` from
core/units.h (no inline magic numbers), lookup tables throw on
unsupported input (`valence_from_z` beyond Z=18, `sph_bessel_for_l`
beyond l=2 — no-fallback rule), `ATV_INFO`/`ATV_DEBUG` logging
replaces std::cout (rank-0 only), PW test targets renamed
`test_cpu_pw_*`, and a latent OOB fixed: `ws.T_G` is resized per
k-point (k-sphere G-count can exceed the Γ-sphere count).

**MPI k-point parallelism** (gated `ATV_HAS_MPI` + runtime
`MPI_Initialized`): `PWSolver::run()` distributes k-points round-robin
over `MPI_COMM_WORLD` (`ws.owned_k` / per-k `ws.gvec_k`), Allreduces
the density accumulation (`DensityBuilder::accumulation()`, MPI_SUM,
IN_PLACE) plus the band energy each SCF iteration, then finishes the
Pulay mix identically on every rank. Davidson's initial-guess noise
uses a deterministic per-subspace-index LCG (`TinyRng`, replaces
`std::rand`) so the SCF trajectory is rank-layout invariant — energy
parity across n=1/2/4 is exact, gated by `test_mpi_pw_kpoint_n{1,2,4}`
(tests/mpi/test_pw_kpoint.cpp, own MPI_Init_thread main).

**Gaussian-SCF audit fixes:** toy solver wrapped in
`cpu::scf::toy::` (ODR clash with production `SCFResult`);
`scf_solver.h` switched to forward decls + `unique_ptr` and the
isolation gate revived via `ATV_REPO_ROOT` compile definition
(`test_cpu_scf_solver_internal_isolation` actually scans the tree
again); `SCFEvaluator` pre-builds its ASPC corrector `SCFSolver` in
`build()` (no allocation in `launch()`); `unwrap_for_scf` uses
`Box::min_image` with a zero-box gas-phase guard; build() doc
corrected (only n_basis-sized prealloc is per-build; X, shell pairs,
KS grid are per-run because atoms move).

**Four pre-existing issues surfaced by the verification run:**

1. *OpenBLAS 0.3.32 / LAPACK 3.12.1 xSYGVD workspace-query regression*
   (Arrhenius `buildenv-gcc/2026.03` module — the only OpenBLAS on the
   system, all threading variants affected): the LWORK=-1 query returns
   `info=0` but writes `WORK(1)=0`, so the follow-up call aborts with
   "parameter number 11 had an illegal value". Every gaussian-SCF test
   (anything through `diagonalize_fock_generalized`) aborted. Fixed in
   `scf_utils.h` by calling `dsygvd_` directly with the documented
   workspace sizes (LWORK = 1+6N+2N², LIWORK = 3+5N) — same routine,
   same numerics, no query. A 12-line standalone repro confirmed the
   bug is environmental, not Atomiverse code.

2. *Davidson subspace-expansion heap overflow* (verbatim in the
   historical `pw_davidson.cpp`, faithfully ported): the expansion
   wrote the new column at slot `n_sub` BEFORE growing
   `sub_vectors`/`H_sub_vectors` (resize happened after the band
   loop) — one full ng-sized column OOB per added residual. Latent
   under the old allocator layout; surfaced as "double free or
   corruption" in `test_cpu_pw_si`/`test_mpi_pw_kpoint_*` after the
   refactor shifted heap layout. Located with ASan (`build-asan`,
   davidson.cpp:503); fixed by resizing before the write. NB:
   valgrind cannot run these binaries — `-march=native` emits AVX-512
   the installed valgrind doesn't decode (SIGILL at the first Mat3
   copy); use ASan on this machine.

3. *Becke weight-derivative catastrophic cancellation*
   (`src/cpu/scf/grid.cpp`, `get_smooth_function_derivative` — verbatim
   from PySCF's `grids_response_becke`): the iterated smoothing
   polynomial p3 was used as `s_uab = 0.5*(1 - p3 + 1e-200)`. For
   far-tail grid points (|mu| → 1) p3 rounds to exactly ±1, `1-p3`
   collapses to 0, and the log-derivative ratio `pt = -t/s` — whose
   mathematical divergence is cancelled downstream by multiplying with
   `pbecke ∝ s` — blows up against the 1e-200 floor instead, leaving
   ~1e150 garbage in dE_xc/dR. Reproduced 6/6 at -O0 (xc_terms
   max_abs 6.2e150, net translational force 4e149 Ha/Bohr in
   `test_cpu_scf_ks_grid_response_gradient`); -O2's FMA kept p3 at
   1-1ulp by luck, hiding the same defect. ASan/valgrind useless here
   — it is pure FP, not memory. Fixed with the cancellation-free
   recurrences `1-f(x) = (1-x)²(x+2)/2`, `1+f(x) = (1+x)²(2-x)/2`
   (s exact down to denormals, t/s a faithful log-derivative, the
   pbecke cancellation restored). After the fix: -O0 3/3 pass with
   xc_terms bitwise equal to the -O2 value.

4. *Unoptimized default build* — the documented configure line had no
   `CMAKE_BUILD_TYPE` and the top-level CMakeLists set no default, so a
   bare `cmake -B build ...` compiled everything at **-O0** (with
   `-march=native` but no -O level). Measured: the same PBE/def2-SVP
   SCF takes 97 s at -O0 vs 9.3 s in the *ASan-instrumented* -O2 tree.
   This, stacked on login-node contention, is why "second-level" tests
   ran for minutes-hours. Fixed: CMakeLists now defaults
   `CMAKE_BUILD_TYPE` to Release when unset (single-config gens), and
   the CLAUDE.md configure lines pass `-DCMAKE_BUILD_TYPE=Release`
   explicitly.

**Test-contract update — `test_cpu_scf_evaluator_no_alloc` (ac-014):**
the old gate asserted *exact* launch-to-launch alloc-count equality.
That held only while `SCFEvaluator::launch()` constructed a transient
`SCFSolver` per step (identical cold-start allocation sequence). With
the pre-built persistent solver (the no-alloc-in-launch rule), buffer
capacities and DIIS history carry across launches and OpenMP workers
grow per-thread buffers lazily, so the count wobbles ~1% (e.g.
647–657) without any leak — measured: iteration count constant at 3,
wobble persists even single-threaded (409/413). The gate now asserts
what ac-014 actually guards: bounded spread (≤5%) + no upward trend
over a 12-launch window.

**Deferred:** Phase 5b IntegralPack/ShellList hot-path split, PW
augmentation charges (E_aug = 0 for the norm-conserving UPF v1 path),
PW forces (PWEvaluator is Energy-only).

## [2026-07-02] Streaming molrs analysis accumulators + recorder/IO unification

Post-review fixes on the observer/recorder/io refactor (multi-axis review, REQUEST CHANGES):
- **Streaming accumulators** replace whole-trajectory buffering in the RDF/MSD/VACF observers. molrs core gains `compute::{RDFAccumulator, MSDAccumulator}` and `compute::fit::VACFAccumulator` (each unit-tested streaming==batch; RDF is bit-identical, VACF matches FFT to round-off); the batch `RDF::compute` is reimplemented ON the accumulator (one source of truth). FFI: `molrs::{rdf,msd,vacf}_accumulator_new` opaque types; C++ wrappers (PIMPL, CUDA-safe) in `cpu/observer/trajectory_metrics.h`. Memory: O(n_bins) for g(r), O(window·n_atoms) / O(resolution·n_atoms) for the correlation rings.
- **API changes:** `ATV::MSD` gains `window` ctor param (default 256, caps the Einstein-D windowed fit lag); `ATV::VACF` `resolution` now must be >= 1 (default 256; `0 = full trajectory` is gone — impossible to stream). Both throw `std::invalid_argument` on bad values (no-fallback).
- **Shared measure-side skeleton:** `cpu/observer/constant_count_gate.h` (`ConstantCountGate` + `pack_blocked`) replaces the triplicated on_snapshot boilerplate; dropped frames are counted and warned at finalize (no more silent data loss).
- **One g(r) emitter:** `cpu::rdf_write_curve` in rdf_histogram.{h,cpp}; legacy `rdf_write` normalizes then delegates. Legacy histogram pair carries `TODO(roykid, 2026-07-02)` pending the CUDA observer's molrs migration.
- **One XYZ emitter:** molrs `trajectory_append` FFI deleted (with recorder/trajectory.{h,cpp}); everything routes through `write_frame_xyz`. New `write_frame_xyz_meta` FFI serializes key=value pairs into the extxyz comment line — `TrajectoryWriter` re-attaches `step`/`energy_eV` through it (restores the pre-refactor MolRec scalars).
- **Driver infra scan is capability-typed:** `scan_pending_infra_()` keys on `provides_neighbor_list()/provides_halo_send()/provides_halo_recv()/neighbor_cutoff_hint()` + `role()==Observer && capabilities()&Stress` — no `name()` matching, no `dynamic_cast` (mirrors `cuda::md::MD::install_infra_`). `cpu::md::Virial` now declares `capabilities()==Stress`.
- **Stored-ERI RHF Fock:** `build_coulomb_exchange_fused` adapts the contiguous ERI storage zero-copy (`xt::adapt`) instead of materializing an n⁴ copy per SCF iteration.

## [2026-07-10] arch-refactor 链收口（01–09）

九篇链一次过（用户裁决：**保留 Driver 命名**并在 harness 强化定位——Driver=任务类型，
Gaussian job type 类比 SP/MD/Opt/Freq/MC；方法=inner Solver+Evaluator 组件，物性=Capability，
永不上 Driver 层）。各篇一句：

- **01-bindings**：`DefaultBinding` 加 `PhaseKind` 判别（`make_binding` 经 ADL `kind_of`），
  driver 注册期校验 kind、跨 workflow 误绑定直接 throw 并点名两个 workflow。
- **02-halo**：ComponentBase 的 8 个 halo 虚函数拆成 opt-in `HaloParticipant` 协议
  （声明式 `CommField` + 引擎侧打包），`halo_participant()` 探测。
- **03-neighbor**：`NeighborConsumer`/`NeighborProvider` 双协议（`as_*` 只读探测），
  infra scan 去 `dynamic_cast`/`name()` 匹配。
- **04-engines**：GPW 补齐 `GPWSolver` 类（build fail-fast/run 可重入），cuda SCF 解耦
  driver Context → 窄 `SolverContext{stream, frame, step}`（shell 门禁）；
  `eigen_solver`→`hessian_diagonalizer` 归位。
- **05-context**：Context 分级——flat core + `ctx.md/mc/opt/analysis` service slice
  （纯聚合、非多态，static_assert 锁定）。
- **06-frame**：cpu::Frame 死 shim（flat Bonds/Angles/Dihedrals）删除；**write_back
  round-trip 契约**钉死（x/v/f/charge/image/tag 回写、tag checked-i32、type/mass/topology/box
  钦定 per-run 静态、scalars 只写通道），golden 双向断言。
- **07-molrs**：cuda::RDF 迁移 molrs streaming accumulator（ring 冻结 positions+per-frame
  cell，NPT-ready），全仓最后的手写分析数学 `rdf_histogram.{h,cpp}` 删除 + 双 grep 门。
  **顺带实证既有隐患**：`CoulLong` 的 cluster CSR 只在 `flat_pairs_requested` 下填充，
  而迁移后无人置位（此前唯一 setter 是旧 RDF）——Coulomb 路径需要独立的 CSR 请求
  （TODO 已留 coul_long.cu / neighbor_list.cu，GPU 验证 owed）。
- **08-python**：Python 假可用面清除（scf.py/optimizer.py/bind_scf/bind_optimizer/5 死
  examples），canonical LAMMPS 名 clean-break（PairLJCut 等，无 alias），pw_scf fail-loud，
  `test_surface.py` 钉契约（70 pass）。唯一全本地闭环 done 的一篇。
- **09-harness**：CLAUDE.md Architecture 章重写（五类 taxonomy 表+Gaussian 类比列、方法
  接入表、五协议 opt-in 表、Context 分层表、Frame 多后端规范章删 "identical field set"、
  molrs 边界）；SamplingDriver/`cpu::mc::MC` 补录（此前实现了却缺席 harness）；
  three-category/fourth-category 计数漂移全树归零；本条目即其归档交付。

关键横向决策沉淀：PhaseKind 判别、opt-in 协议表（HaloParticipant/NeighborConsumer/
NeighborProvider/PairEvaluator/AsyncComponentBase）、Context service 分组、write_back
契约、cuda RDF 归 molrs（engine measures, molrs analyzes 两后端无例外）。
GPU-owed 验证集中在 01–07 的台账（GH200 sbatch + Linux CI + /mol:bench）。

## [2026-07-15] CPU breadth 01 — architecture foundation subset

- Component registration 已硬切为显式 lifetime：rvalue/`unique_ptr` 的
  `add` 转移 ownership，`add_ref` 是唯一 borrowing spelling，Driver、
  PhaseBuilder 与 CPU categorical setters 的普通 lvalue overload 均删除；
  全部 in-tree CPU caller 已明确迁移，compile-time/runtime ownership 门禁通过。
- molrs CXX boundary 由 `molrs-cxxapi/CXX_API_VERSION`、capability mask、
  Atomiverse configure-time exact version check 和 runtime exact contract 共同守门；
  mismatch 会报告 expected/found。当前 molrs 是 dirty 未发布 worktree，因此
  release tag/SHA 与 CI pin 保持未完成，禁止把本地状态伪装成发布基线。

## [2026-07-15] CPU breadth 02 — typed Minimize task cutover

- `TaskSpec` 硬切到 v3 typed sum：`MDTask` 只接受 `md::Phase`，
  `MinimizeTask` 只接受 `minimizer::Phase`，`SinglePointTask` 保持独立。
  不存在通用 raw-phase IR、v2 decoder、alias 或 translation branch。
- 经过两个真实 staged task 后的边界裁决：只有 wire envelope、
  explicit `ComponentId`、materializer 入口和 `TaskResult` 是共享的；
  phase enum、component allow-list、budget 和 handler params 必须留在每个
  task alternative 内，不继续泛化。
- MPI task protocol 硬切 v2，`DispatchResult{Frame, TaskResult}` 将状态与
  loop-control 分离；MD/Minimize 都通过同一 DD owner-gather 路径，
  SinglePoint 保持 full replica。
- Minimize 新增明确的 detached teardown：先 destroy components，再移交
  仍持有 Cartesian communicator 的 DD，最后 shutdown Runtime；避免 gather
  在 communicator 释放后发生。
- 4-rank protocol 结构门禁抓到 zero-owner rank 在 `LBFGS::build` 拒绝
  `dim==0`，导致其他 rank 已进入 optimizer collective 而空 rank 进入
  protocol status collective（`MPI_ERR_TRUNCATE`）。LBFGS 现在为空分区保留一个
  inert storage slot，launch 仍以 local dim 0 参加完全相同的 collective 序列。
- Python 公开面为 `atomiverse.minimizer.{Minimize,LBFGS,ForceConverged,
  MinimizeResult}`，仍是 `add`/`on` 组合和唯一 `molrs.Frame` 状态货币。
  GPU 明确 deferred，`device!="cpu"` fail loud。

## [2026-07-15] CPU breadth 03 — long-range contract and cell lifecycle cutover

- `CoulLong`、`KSpaceEwald`、`KSpacePME` 通过窄的
  `LongRangeParticipant` 暴露 real/reciprocal role 与共享 `alpha`；
  `CPUDriver` 在所有 typed CPU loop build 前统一拒绝缺半边、重复半边、
  `alpha` 不一致，以及 Stress consumer 配到无 Stress reciprocal 的 Plan。
  校验按 component identity 去重，不依赖 RTTI、名称或 MD 私有分支。
- `Box` 现在提供单调 mutation revision；cell-dependent cache 以
  `(Box identity, revision)` 为键，既捕获 NPT/deform 原地变胞，也不会把
  两个 revision 相同的 replica Frame 误认为同一个 cell。
- Ewald/PME 公开 `KSpaceLifecycleStats{builds,cell_refreshes,
  capacity_rebuilds}`。PME 的 per-atom spline workspace 已从 launch 局部
  vector 移到 component-owned capacity；同 shape 变胞只 refresh，网格 shape
  或 atom capacity 跨界才 rebuild/replan，destroy 明确释放所有 workspace；
  FFTW plan ownership 硬切为 move-only，禁止复制裸 plan 句柄。
- PME 尚未写 reciprocal Stress，因此删除其 `Capability::Stress` 声明；
  压力耦合 Plan 现在 fail loud。应力算法本身不在本次架构重构范围内。

## [2026-07-16] cpu-breadth campaign bulk implementation push

Closed/advanced the remaining CPU campaign surface in one push:

- **01**: `acceptance-cpu.yml` (dev/master), `ci.yml` includes `dev`,
  `tools/verify_cpu_test_manifest.py`, GPW⇒RHF∧PW CMake fatal, harmonic
  `reg_harmonic_driver_matrix`.
- **02**: `reg_lj_dimer` public Minimize literature gate.
- **03**: already CPU-complete earlier (PME stress, NPT, LAMMPS, 2-rank).
- **04**: `PairCoulCut`, `ForceFieldSnapshot` + `CompiledForceField`,
  `reg_tip3p_two_water` (LJ+bond+angle install path; coul special-bonds
  follow-up; molrs ForceFieldRef CXX snapshot still release-gated).
- **05**: `cpu::path::Remd` — N owned `MD` programs, Metropolis exchange,
  velocity rescale, `reg_remd_harmonic`.
- **06**: `bias::DistanceCV`, `bias::HarmonicBias`,
  `LaunchContext::requested_capabilities`, `reg_umbrella_harmonic`.
- **07**: `docs/developer/cpu-architecture-breadth-verdict.md`.

GPU criteria remain deferred. frame-block-04 formal ASan/perf still open.

## [2026-07-16] open-item close-out (specials / stale / WHAM / pins)

- **PairCoulCut special-bonds**: same filtered-CSR path as `PairLJCut`
  (`set_special_bonds(0,0,0)` rebuilds exclusion map on PairList version).
  `CompiledForceField` applies `special_coul[]`. Unit tests exclude 1-2 and
  keep intermolecular; `reg_tip3p_two_water` runs full LJ+coul/cut+bonded
  with specials (`E≈0.66 eV`, no intramolecular coulomb blow-up). Tags in
  `add_bond`/`add_angle` are global tags, not local indices.
- **Evaluate/MD/Minimize install**: `test_cpu_compiled_forcefield` installs
  one snapshot into all three workflows.
- **molrs pin**: `cmake/MolrsContract.cmake` records
  `ATV_MOLRS_CXX_API_VERSION=1` and `ATV_MOLRS_GIT_REF=v0.7.0`; CI checkouts
  use `ref: v0.7.0`. Path overrides still cannot bypass CXX API version.
- **Stale handles**: `Span`/`MetaRef` generation rejection unit-tested
  (grow, column insert, topology restride, meta insert → `stale_handle_error`).
- **Umbrella WHAM**: multi-window synthetic Boltzmann samples + in-process
  WHAM recovers flat PMF (RMSE < 0.3 kBT). Ownership: ATV = CV/bias/samples;
  production WHAM analysis remains molrs-owned.
- **REMD**: exchange stats + deterministic seed gated; analytic T-marginal
  literature gate still open (needs 1-particle spring evaluator).
- Remaining open: molrs ForceField CXX snapshot, molpy cross-repo fixture,
  alloc-intercept perf gates, frame-block ASan/perf matrices, REMD
  T-histograms.

## [2026-07-16] Component model hard cut (Role + kinds + ForceField)

**Deleted:** `Capability` write-set; exclusive `ComponentRole`;
`capabilities()` / `reads()` / `requested_capabilities`; free-function
`is_energy_force_source()`; intermediate kinds `ElectronicStructure` and
`Infrastructure`; constant presets `kRole*`.

**Role** is a `uint32_t` bitset of `RoleFlag` only:
- `FrameImmutable` — does not mutate x/v/f/box/topology
- `Deferred` set — optional / may leave the hot path; **unset** ⇒ main-loop required
- `ExternalIO` — writes outside the process

No named role presets. Compose flags where needed
(`Observer` = `FrameImmutable | Deferred`;
`Recorder` = those + `ExternalIO`; default `0` = required).

**Kinds are classes, one header each** under `cpu/component/`:
`component.h` holds only `cpu::Component`. Marker kinds:
`potential.h`, `hf.h`, `semiempirical.h`, `plane_wave.h`, `gpw.h`,
`bias.h`, `path_coupling.h`, `observer.h`, `recorder.h`,
`integrator.h`, `thermostat.h`, `barostat.h`, `optimizer.h`,
`constraint.h`, `mover.h`. Depth ≤ 2: `Concrete → Kind → Component`.

Electronic structure is **method-named**, not a bag kind:
`SCF : HF`, `AM1Evaluator : Semiempirical`, `PWEvaluator : PlaneWave`,
`GPWEvaluator : GPW`.

Pipeline glue (NeighborList, Halo*, Migrate, Reduce, Zero*) inherits
`Component` directly — name the job, no `Infrastructure` parent.

**Renames:** `ComponentBase` → `Component`; `cpu::scf::SCFEvaluator` →
`cpu::scf::SCF`. Pair path: `PairLJCut → PairPotential → Potential → Component`.

**ForceField** is **data only** (molrs-aligned `def_pairstyle` /
`def_bondstyle` / …). It is not a Component and does not install into a
Driver. Classical terms register as Components via
`add_potential(Potential&&)` (replaces singular `set_pair` / `set_bond` /
`set_angle` / `set_dihedral`). Neighbor remains `set_neighbor`.

**Validation:** presets `dynamic_cast` to concrete kinds (Potential,
Bias, PathCoupling, HF, Semiempirical, PlaneWave, GPW, Integrator, …).
No free-function component classifier. Stress assembly uses
`requires_system_stress()` / `provides_system_stress()` on Component.
