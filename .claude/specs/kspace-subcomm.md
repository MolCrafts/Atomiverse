---
title: PPPM k 空间 FFT 子通信域（选择器第三臂）
slug: kspace-subcomm
status: code-complete
created: 2026-08-26
revised: 2026-08-26
grilled: true
---

# PPPM k 空间 FFT 子通信域（选择器第三臂）

## Summary

在高 rank 数下，32³ SPME 网格的倒空间求解是 24k 原子 GAFF 体系的唯一主瓶颈：np=256 时 replicated 臂 Kspace 9.87 s（占 Loop 15.22 s 的 64.9%），brick 臂 9.85 s（占 63.7%），两臂同墙——因为 32 个 x 平面挂在 256-rank 通信域上，224 个 rank 持零平面却仍然参与 `fftw_mpi` 集合与全域 Alltoallv。本 spec 给 `KSpacePPPM` 现有的两臂选择器加**第三臂**：把 slab FFT 收进一个由 `dd->cart_comm` 派生的、大小受 K0 平面数上界约束的 MPI 子通信域，非成员 rank 只做 spread / mesh halo / remap 与插值，完全不进入 FFT 集合；同时修正现行选择器在 np∈[16,~64] 区间已被测量证伪的判据（np=64 实测 brick 4.78 s 优于 replicated 5.43 s，而现行规则在该区间选 replicated）。用户已定：纯 MPI，无 OpenMP 混合，无 Kokkos。交付判据只含正确性与不变量，性能数字只进 `tools/bench_gaff_lammps/README.md` 快照。

## Domain basis

倒空间能量是 Ewald 分解的 SPME 网格近似（`src/cpu/longrange/pppm.cpp:361-384` / `:776-805` 已实现，本 spec 不改动其数学）：

- Ewald 倒空间项：`E_rec = (C/2πV) Σ_{m≠0} H(m) |S(m)|²`，其中 `H(m) = exp(-π²m²/α²)/m²`，`S(m)` 为 B 样条插值后的网格电荷结构因子，`C = 14.399645 eV·Å/e²`（`core/units.h`，`ATV::units::coulomb`）。
- 自能 `E_self = -C α/√π Σ_i q_i²` 按 owned 原子分摊；净电荷项 `E_net = -C π q_tot²/(2 V α²)` 只在 rank 0 累加（现状，不变）。
- 单位：能量 eV，长度 Å，电荷 e，`α` 为 Å⁻¹（实测基线 α = 0.25232811 Å⁻¹，spacing 1.98 Å，网格 32³）。

参考文献（新增代码不引入新物理，仅引入新的通信划分；REFERENCE.md 逐条落 DOI）：

- U. Essmann et al., *A smooth particle mesh Ewald method*, J. Chem. Phys. **103**, 8577 (1995), DOI 10.1063/1.470117 —— SPME 影响函数与 B 样条插值，本实现的 `H(m)` 与 `bsp_mod_` 来源。
- R. W. Hockney & J. W. Eastwood, *Computer Simulation Using Particles*, CRC Press (1988), DOI 10.1201/9780367806934 —— PPPM 的 mesh/brick 分解语义。
- S. Plimpton, J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039 —— 空间分解与 "分解是同一个和式的划分" 这一恒等式，正是本 spec 全部 parity 判据的物理依据。
- M. Frigo & S. G. Johnson, Proc. IEEE **93**, 216 (2005), DOI 10.1109/JPROC.2004.840301 —— FFTW 的 MPI slab（dim-0）分解：**每个 rank 至少要有一个平面才有活干**，是子组大小上界 `P_fft ≤ K0` 的出处。
- NaCl 岩盐 Madelung 常数 `M = 1.747564594633182`：C. Kittel, *Introduction to Solid State Physics*, 8th ed., Wiley (2005), ch. 3；J. Sherman, Chem. Rev. **11**, 93 (1932), DOI 10.1021/cr60038a002 —— regression 的文献 oracle（`oracle: literature`）。

物理不变量（判据的根）：**总能量与每原子力是 Hamiltonian 的泛函，与通信域划分无关**。换 FFT 通信域只允许改变浮点求和次序，不允许改变任何物理量。

## Design

### 落位与形态

第三臂是**同一个选择器的第三个分支**，不是新组件、不是新类：`cpu::Driver::build()` 对同一 role 的重复注册会 throw（`include/atomiverse/cpu/driver/driver.h:528`），新加一个 KSpace 组件类会直接撞上它。全部改动落在 `include/atomiverse/cpu/longrange/pppm.h` + `src/cpu/longrange/pppm.cpp`（CLAUDE.md 钉死：倒空间求解器及其数学在各后端 `longrange/`）。CUDA 后端不动。

### 新增公共符号（三个自由函数 + 两个只读访问器 + 一个构造参数）

```cpp
enum class PppmMeshArm { Replicated, BrickWorld, BrickSubgroup };

int pppm_fft_ranks(int planes_k0, int nprocs, int min_planes_per_rank = 1);
PppmMeshArm pppm_select_mesh_arm(int planes_k0, int nprocs, int fft_ranks);
```

- `pppm_fft_ranks` 是**纯算术**（无 MPI、无 env、无状态）：返回 `planes_k0` 的最大因子 `d`，满足 `d ≤ min(nprocs, planes_k0 / min_planes_per_rank)`，且 `d ≥ 1`。取因子而非直接取 `min` 是为了让 FFTW-MPI 的 dim-0 划分整除、无 ragged slab。任一入参 ≤ 0 抛 `std::invalid_argument`（无 fallback 值）。样例：`(32,256)→32`、`(32,64)→32`、`(32,16)→16`、`(32,256,2)→16`、`(34,4)→2`、`(K,1)→1`。
- `pppm_select_mesh_arm` 是**纯策略**：`nprocs ≤ 1 → Replicated`；`fft_ranks == nprocs → BrickWorld`；`fft_ranks < nprocs → BrickSubgroup`；`fft_ranks > nprocs` 抛。FFTW-MPI 编译开关与 env 强制留在调用点（`pppm.cpp`），使这两个函数可以在**不启动 MPI** 的情况下被秒级单测覆盖。
- `KSpacePPPM` 构造参数新增 `int fft_ranks = 0`（`0 = 由网格推导`，负值构造期抛；`> min(nprocs, K0)` 在 shape 重建处抛并给出 how-to）。这是构造参数/公有成员形态，不是 Config 袋，也不是第四个环境变量（iron law 3）。`from_accuracy` 签名不变。
- 只读访问器 `int fft_ranks() const noexcept` 与 `PppmMeshArm mesh_arm() const noexcept`（首次 `compute()` 前分别为 0 / `Replicated`）。它们存在的唯一理由是**防假绿**：regression 必须能断言 "np=8 那一腿真的走了子组臂、子组大小真的是 6"，否则一个悄悄退回 world 臂的实现也会打印 PASS（`regressions/md/multihop-widebox-parity` 的 hop 断言是同一教训）。

### 子通信域的所有权与生命周期

私有成员镜像现有 `mpi_*` / `fft_m0_*` 组：`MPI_Comm fft_comm_`、`bool owns_fft_comm_`、`int fft_ranks_requested_`、`int fft_ranks_`。

- **归属**：`KSpacePPPM` 私有（`cpu/longrange/kspace.h:11-14` 明言多 rank 归并由具体 wrapper 拥有）。`ctx.md.dd->cart_comm` 仍是组件唯一的通信域入口，`Context` / `DynamicsServices`（`include/atomiverse/cpu/component/context.h:56`，有 `trivially_copyable` 断言）**不动**。
- **建**：lazy，建在既有的 shape-keyed 重建块内（`pppm.cpp:650` 的 `!mpi_fft_ready_ || shape 变化` 分支），**绝不在逐步路径上**。`MPI_Comm_split(dd->cart_comm, rank < P_fft ? 0 : MPI_UNDEFINED, rank, &fft_comm_)`——非成员得到 `MPI_COMM_NULL`。cart-rank 0 依构造恒为成员（`first_rank` 的能量累加靠它）。
- **不建的情况**：`fft_ranks_ == nprocs` 时直接**别名** `dd->cart_comm` 且 `owns_fft_comm_ = false`，一次 split 都不做。这既省掉一次集合，也规避 `src/cpu/asynchronous/analysis_executor.cpp:49-58` 记录的 OpenMPI 5.0.x 多次 comm 派生的运行时坑；全树至今**没有任何** `MPI_Comm_split/create/group`（`src/cpu/communication/domain.cpp:127` 的 `cart_comm` 是 world 别名），本 spec 是树上第一个真派生通信域，所以"能不派生就不派生、派生了只派生一次"是硬形态。
- **拆**：`release_storage()`（`pppm.cpp:1131-1156`）同一 `MPI_Finalized` 守卫下 `MPI_Comm_free`，仅当 `owns_fft_comm_`；随后置 `MPI_COMM_NULL`、`owns_ = false`。shape 变化触发重建时先按同样守卫释放旧 comm。所有权标志 + finalize 守卫的形状照抄 `analysis_executor.cpp:119-129`。
- **集合一致性**：split 前用 `cpu::comm::require_collective_fingerprint(version, digest, dd->cart_comm, "KSpacePPPM::fft_comm")`（`include/atomiverse/cpu/communication/schema_mpi.h:24`，签名已通用，直接调用即可），digest 混入 `{K0,K1,K2,fft_ranks_,arm}`。划分参数若在 rank 间发散，`MPI_Comm_split` 是**挂死**不是报错；这条 Allreduce 让它变成一次集合性失败。同理，`long_range_contract()`（`include/atomiverse/cpu/component/long_range_protocol.h:22`，只含 `{role, alpha, real_cutoff, accuracy}` 构建期元数据）在**每个** rank 上返回值必须与子组无关——`fft_ranks` 不得进入 contract。

### FFT 组与 remap 的改造

- FFTW-MPI plan 组（`pppm.cpp:665/:673/:677`）三处的 `comm` 实参换成 `fft_comm_`：`fftw_mpi_local_size_3d` / `plan_dft_r2c_3d` / `plan_dft_c2r_3d` 全部以通信域为参数，故 slab-in-subgroup **零新 FFT 代码**。非成员不建 plan、不分配 `mpi_data_`，`mpi_local_n0_ = 0`。
- 平面归属表 `fft_m0_counts_/fft_m0_starts_`（`:684-692`）**保持按父通信域 rank 索引、长度 nprocs**，仍用父域 `MPI_Allgather` 填充，非成员天然写入 `count = 0`。这样 `:710`/`:809` 的两次 Alltoallv remap（装填 / 去重求和 / 位移构造）**原样复用**，`fft_owner` 的像自动被限制到子组成员。
- `fft_owner`（`:701-708`）末尾的 `return nprocs - 1;` 必须删除：在子组臂下 `nprocs-1` 往往是非成员，这是一个会静默算错的 fallback（iron law "无 fallback 值"）。改为 throw，消息给出 `m0 / K0 / fft_ranks_` 与如何复现。既有 world 臂下该分支本就不可达，改动无行为风险。
- 能量归并（`:804`）从父域 Allreduce 改为**子组内** Allreduce：非成员 `e_local = 0` 且不使用 `e_recip`，rank 0 恒为成员。少一次全域集合。
- mesh halo（`:507` reverse_comm / `:908` forward_comm，逐维 `dd->face_neighbors` Sendrecv）与 brick 簿记（`pppm.h:95` 的 `nxlo_in_…nzhi_out_` / `setup_brick_partition_` / `brick_index_`）与子组正交，**一行不改**；`owned_range.h:29` 的 "spread/interpolate 只迭代 owned 原子、ghost 经 reverse_comm 汇入" 不变量也必须保持，它是电荷汇入良定义的前提。

### 选择器与 env 语义

`pppm.cpp:318` 的 `use_brick_mpi` 布尔量泛化为三值：编译期无 MPI/FFTW-MPI 或无 DD → `Replicated`；`ATV_PPPM_ALLREDUCE=1` → 强制 `Replicated`；`ATV_PPPM_BRICK=1` → 强制 `BrickWorld`（旧两臂语义保持，调试用，且这正是三臂对比矩阵能用**同一个冻结二进制**测出来的原因）；否则 `pppm_select_mesh_arm(K0, nprocs, pppm_fft_ranks(K0, nprocs))`。不新增第四个环境变量。现行 `mesh ≤ 64³ && ranks ≥ 16 → replicated` 判据被删除——它在 np∈[16,~64] 已被 np=64 的实测（brick 4.78 s vs replicated 5.43 s）证伪。新策略中的每一个常数（`min_planes_per_rank`、是否保留 replicated 的区间）必须在代码注释里引用 `tools/bench_gaff_lammps/README.md` 三臂矩阵的具体一行，不许拍脑袋。

### 计时

`pppm.cpp:331-346/:438-445` 的 `ATV_PPPM_PROFILE` 临时剖析器（thread_local chrono 累加 + 每 500 次 fprintf）与 `include/atomiverse/core/timer.h:185` 的 `ATV_PROFILE_SCOPE` → `TimerRegistry` 重复，且只测 replicated 臂。删除它，两个 brick 臂与 replicated 臂统一用 `ATV_PROFILE_SCOPE(ctx, ...)`（`ctx.runtime.timer` 由 Backend 发布），子键区分 spread / halo / remap / fft / interp。这同时给三臂矩阵提供每阶段拆分。

### Reuse decision（逐条落实 librarian 报告）

| 候选 | 判定 |
| --- | --- |
| `pppm.cpp:318` `use_brick_mpi` 选择器 | **generalize** —— 布尔 → 三臂枚举，策略抽成纯函数 `pppm_select_mesh_arm`（任务 5） |
| `pppm.cpp:642` FFTW-MPI plan 组 | **generalize** —— 三处 `comm` 实参改 `fft_comm_`，零新 FFT 代码（任务 4） |
| `pppm.cpp:710`/`:809` Alltoallv remap 对 | **generalize** —— 装填/求和/位移原样复用；`fft_owner` 的像限制到成员，删掉 `nprocs-1` fallback 改 throw（任务 4） |
| `pppm.cpp:507`/`:908` mesh halo | **reuse** —— 与子组正交，不动 |
| `pppm.h:95` brick 簿记 | **reuse** —— 子组路径从同一批 brick 汇入 |
| `pppm.cpp:331`/`:438` `ATV_PPPM_PROFILE` | **generalize** —— 删除，改用 `core/timer.h:185` `ATV_PROFILE_SCOPE` → TimerRegistry（任务 6） |
| `src/cpu/communication/domain.cpp:127`（全树无 comm 派生） | **pattern** —— 本 spec 是首个真派生通信域：一次 split / 可别名 / finalize 守卫，形态在上文钉死 |
| `src/cpu/asynchronous/analysis_executor.cpp:45`（`owns_*_comm_` + `MPI_Finalized` 守卫） | **pattern** —— 子组 comm 的所有权与释放照此形状 |
| `pppm.cpp:1137` `release_storage` 的 FFTW-MPI 拆除 | **reuse** —— 子组 comm 的 free 放同一函数同一守卫 |
| `schema_mpi.h:24` `require_collective_fingerprint` | **reuse**（无需泛化）—— 现签名 `(version, fingerprint, comm, op)` 已通用，直接传 mesh+arm digest；不改签名 |
| `component/context.h:56`（`ctx.md.dd->cart_comm` 唯一通路） | **reuse** —— 子组 comm 为 solver 私有，Context / DynamicsServices 不动 |
| `communication/domain_runtime.h:30` DomainRuntime | **pattern（推迟）** —— 只有当子组不再私有时才是它的归属；迁移 trigger 见 Out of scope |
| `component/long_range_protocol.h:22` `LongRangeContract` | **reuse（不动）** —— `fft_ranks` 不得进 contract；build() 期任何 rank 发散的 throw 是挂死不是报错 |
| `component/owned_range.h:29` owned-only 迭代 | **reuse（不得破坏）** —— 电荷汇入良定义的前提 |
| `tests/mpi/test_pppm_brick_parity.cpp:80` | **pattern** —— 新臂 MPI 门（min/max Allreduce 一致性、skip-gate）照此模板，但**不**再往 tests/ 里搬文献 oracle |
| `src/cpu/longrange/pme.cpp:192` KSpacePME 的 Allreduce+rank0+Bcast | **new（本 spec 不做）** —— 子组机制的第二个消费者；按 "inline until the second real use" 只登记 trigger，不预抽取 |
| `docs/developer/kspace-long-range.md:72-76` 每风格 MPI 策略表 | **reuse** —— 增第三臂行（任务 8） |

## Files to create or modify

- `include/atomiverse/cpu/longrange/pppm.h` —— `PppmMeshArm`、`pppm_fft_ranks`、`pppm_select_mesh_arm`、`fft_ranks` 构造参数、`fft_ranks()`/`mesh_arm()` 访问器、`fft_comm_`/`owns_fft_comm_`/`fft_ranks_` 私有成员、文件头策略表补第三臂
- `src/cpu/longrange/pppm.cpp` —— 三臂选择器、子组 comm 生命周期、FFTW-MPI plan 组与 m0 表重指向、`fft_owner` throw、子组内能量归并、`ATV_PROFILE_SCOPE` 计时、移动构造/`release_storage` 同步新成员
- `tests/cpu/longrange/test_pppm_coul.cpp` —— 新增纯函数单测（无 MPI，毫秒级）
- `tests/mpi/test_pppm_subcomm_parity.cpp` (new) —— 子组臂 vs 默认臂的单次静态启动 parity 门（n=1/2/4，skip-gated）
- `tests/CMakeLists.txt` —— 在既有 `ATV_BUILD_KSPACE` + `ATV_USE_MPI` 嵌套内注册 `test_mpi_pppm_subcomm_parity` 的 n1/n2/n4
- `regressions/longrange/pppm-subcomm-parity-mpi/reg_pppm-subcomm-parity-mpi.cpp` (new) —— 自重启 e2e（np=1 文献 oracle；np=8 子组臂 vs np=1；np=8 `ATV_PPPM_BRICK=1` world 臂 vs 子组臂）
- `regressions/longrange/pppm-subcomm-parity-mpi/REFERENCE.md` (new) —— `oracle: literature` + DOI + 钉死的数值表 + 容差推导 + falsification 表 + `perf_exempt: yes`
- `regressions/CMakeLists.txt` —— `if(ATV_USE_MPI AND MPIEXEC_EXECUTABLE)` 下 `atv_add_regression(longrange pppm-subcomm-parity-mpi)` + `ATV_MPIEXEC` 编译期钉入
- `docs/developer/kspace-long-range.md` —— `## MPI: PME vs PPPM FFT` 表增第三臂行与选择器新策略说明
- `tools/bench_gaff_lammps/README.md` —— 三臂 × np∈{64,125,200,256} 的带日期快照（唯一的性能落点）

（文件数：8 个改动 + 2 个新建，全部落在 CPU 后端 `longrange/` 这一个架构层及其 tests/regressions/docs 附属，未跨层、未引入新的顶层概念，故按 `large-spec-split.md` 不拆。）

## Tasks

- [x] Write failing unit tests for `pppm_fft_ranks` / `pppm_select_mesh_arm` in `tests/cpu/longrange/test_pppm_coul.cpp` (hard-coded: (32,256)→32, (32,64)→32, (32,16)→16, (32,256,2)→16, (34,4)→2, (K,1)→1; throws on 0-args; arm map Replicated/BrickWorld/BrickSubgroup)
- [x] Implement `PppmMeshArm`, `pppm_fft_ranks`, `pppm_select_mesh_arm`, the `fft_ranks` constructor argument and the `fft_ranks()` / `mesh_arm()` accessors in `include/atomiverse/cpu/longrange/pppm.h` with Doxygen `@brief`/`@param`(units)/`@return`
- [x] Write failing MPI parity test `tests/mpi/test_pppm_subcomm_parity.cpp` and register `test_mpi_pppm_subcomm_parity` n1/n2/n4 in `tests/CMakeLists.txt` (skip-gated on `ATV_HAS_FFTW_MPI`)
- [x] Implement the subgroup communicator in `src/cpu/longrange/pppm.cpp`: lazy `MPI_Comm_split` at the shape-keyed rebuild guarded by `require_collective_fingerprint`, alias when `fft_ranks == nprocs`, repoint `fftw_mpi_local_size_3d` / both `fftw_mpi_plan_dft_*` and the subgroup-scoped energy Allreduce, replace `fft_owner`'s `nprocs - 1` fallback with a throw, and free the comm in `release_storage()` under the existing `MPI_Finalized` guard
- [x] Generalize the `use_brick_mpi` selector in `src/cpu/longrange/pppm.cpp` into the three-arm choice, preserving `ATV_PPPM_ALLREDUCE` / `ATV_PPPM_BRICK` forcing and adding no fourth environment variable
- [x] Replace the `ATV_PPPM_PROFILE` ad-hoc profiler in `src/cpu/longrange/pppm.cpp` with `ATV_PROFILE_SCOPE` (`core/timer.h:185`) sub-keyed spread / halo / remap / fft / interp across all three arms
- [x] Add regression example `regressions/longrange/pppm-subcomm-parity-mpi/reg_pppm-subcomm-parity-mpi.cpp` + `REFERENCE.md` and register it in `regressions/CMakeLists.txt` (public headers only; hard-coded Madelung literal and pinned energies; no third-party engine at build or run time)
- [x] Verify against the 256-rank `tools/bench_gaff_lammps/widebox_parity.sh` window with the subgroup arm as default, then record the three-arm × np∈{64,125,200,256} matrix in `tools/bench_gaff_lammps/README.md` and the third-arm row in `docs/developer/kspace-long-range.md`
- [x] Run full check + test suite

## Testing strategy

**单测（`tests/`，秒级，无文献 oracle）**

- `tests/cpu/longrange/test_pppm_coul.cpp`（镜像 `src/cpu/longrange/pppm.cpp`，沿用该目录既有的 pppm 单测归属）：
  - `pppm_fft_ranks` happy path（硬编码期望值，见 Tasks 第 1 条）；
  - edge：`nprocs = 1` → 1；`min_planes_per_rank > planes_k0` → 1；`planes_k0` 为素数（如 31）且 `nprocs = 8` → 1；
  - throw：`planes_k0 ≤ 0`、`nprocs ≤ 0`、`min_planes_per_rank ≤ 0`、`fft_ranks > nprocs`（`std::invalid_argument`）；
  - `pppm_select_mesh_arm` 的三值映射，含被证伪区间的回归点 `(32, 64, 32) == BrickSubgroup`（旧策略在此选 Replicated）。
  - 每个 TEST 只针对一个函数、无 MPI、无 FFT。

- `tests/mpi/test_pppm_subcomm_parity.cpp`（沿用 `tests/mpi/test_pppm_brick_parity.cpp:80` 的既有 MPI 门形态：一次静态启动、无收敛循环、无计时、无外部引擎）：
  - `world_size ∉ {1,2,4}` → `GTEST_SKIP`；`world_size > 1 && !ATV_HAS_FFTW_MPI` → `GTEST_SKIP`；
  - n=4、`spacing` 选到 `K0 = 6`（`good_fft_size` 为 {2,3,5,7}-smooth，6 = 2·3 合法；立方盒 K2 = 6 为偶）时 `mesh_arm() == BrickSubgroup` 且 `fft_ranks() == pppm_fft_ranks(6,4) == 3`（真的有 1 个非成员 rank；grilling 修正——原稿的 K=12/fft_ranks=6 在 world=4 下不可能）；
  - 同一 Frame 上 `KSpacePPPM{α, spacing, order, /*fft_ranks=*/world_size}`（world 臂；4 ∤ 6 产生 ragged slab，fftw_mpi 自行分割 local_n0=2,2,1,1，即现行 world 臂行为）与默认（子组臂）的总能量相对差 < 1e-9、tag 对齐的每原子力差 < 1e-9 eV/Å；
  - 归约后的能量在 rank 间一致（min/max Allreduce，< 1e-10 相对）；
  - n=1 时 `mesh_arm() == Replicated` 且构造 `fft_ranks=1` 不改变结果。
  - **不**在此文件断言 Madelung——文献 oracle 只属于 `regressions/`。

**Regression example（唯一 e2e，`regressions/longrange/pppm-subcomm-parity-mpi/`，≤ 60 s，退出码即判据）**

一个自重启二进制（`ATV_MPIEXEC` 编译期钉入，形态照 `regressions/md/multihop-widebox-parity`），只消费公共头 + 链接 `atomiverse`：

- 父进程按序拉起三条腿：`np=1`（默认臂 → Replicated）、`np=8`（默认臂 → BrickSubgroup）、`np=8 ATV_PPPM_BRICK=1`（world 臂）。
- 体系：n=6 岩盐 NaCl（216 原子，d = 2.82 Å，L = 16.92 Å），`CoulLong` + `KSpacePPPM`，`spacing` 取到 `K = 12`（`pppm_fft_ranks(12, 8) = 6`，子组 6 成员 / 2 非成员），单次静态求值。
- 判据（全部**硬编码字面量**，无运行时第三方）：
  1. `|E(np=1) / E_Madelung − 1| < 2e-3`（网格离散化窗口；`E_Madelung = −½·N·M·C/d`，`M = 1.747564594633182` 为文献常数字面量）；
  2. `|E(np=8, 子组) − E(np=1)| / |E(np=1)| < 1e-9`；
  3. `|E(np=8, 子组) − E(np=8, world 臂)| / |E| < 1e-9`，tag 对齐的 `max|ΔF| < 1e-9 eV/Å`；
  4. `‖Σ_i F_i‖ < 1e-10 eV/Å`，且岩盐对称性下 `max|F_i| < 5e-3 eV/Å`；
  5. **防假绿**：np=8 腿必须自报 `mesh_arm() == BrickSubgroup` 且 `fft_ranks() == 6`，world 腿必须自报 `BrickWorld`——臂没走到就判失败，而不是让一个悄悄退回旧臂的实现打印 PASS。
- `REFERENCE.md` 落 `oracle: literature`、`perf_exempt: yes`、上表 DOI、钉死的 E 数值、容差推导（差异只允许来自求和次序）、以及 falsification 表（至少：把子组大小人为设成 world、破坏 `fft_owner` 的成员限制、去掉 mesh reverse_comm 三种注入各自打在哪条判据上）。

**实机窗口（不进 ctest）**：`tools/bench_gaff_lammps/widebox_parity.sh` NP=256（该脚本已在 replicated 臂上验证 dev=0.0），以子组臂为默认重跑，全部窗口须绿。

## Out of scope

- **Alltoallv 形状改造**：全域 `MPI_Alltoall`（counts）+ `MPI_Alltoallv`（payload、多数 count=0）保持不变。改成 `reduce_scatter`/邻域集合是紧邻的下一个 spec，本 spec 只把 FFT 集合从 P 缩到 P_fft，不动 remap 的通信形状。
- **子组 comm 迁到 `DomainRuntime`**：迁移 trigger 是第二个消费者出现（`src/cpu/longrange/pme.cpp:192` 的 `KSpacePME` Allreduce+rank0+Bcast）。按 "inline until the second real use"，本 spec 只登记 trigger，不预抽取。
- **成员选择的局部性优化**：成员固定为 cart-rank `[0, P_fft)`；按 brick 的 x 区间对齐成员以压 remap 体积是后续优化。
- **删除 replicated 臂**：`ATV_PPPM_ALLREDUCE=1` 与无 FFTW-MPI 场景仍需要它。
- **OpenMP 混合 / Kokkos**：用户已定，纯 MPI。
- **CUDA 后端**：`cuda/longrange/` 一行不改。
- **性能判据**：仓库法则——性能不进验收；三臂矩阵与 steps/s 只进 `tools/bench_gaff_lammps/README.md` 快照。
- **已发现但本 spec 不修的既有债（报告而非忽略）**：`tests/mpi/test_pppm_brick_parity.cpp:98-114` 在 `tests/` 内断言 Madelung 文献值，与 iron law 1（文献 oracle 只进 `regressions/`）冲突。本 spec 不扩大它，并且新建的 `regressions/longrange/pppm-subcomm-parity-mpi` 已经在 `regressions/` 内覆盖了同一 Madelung 判据；把该断言从单测里摘掉留给一个独立的清理 slug（摘除会删掉一个现存 gate，不宜在本 spec 里顺手做）。
