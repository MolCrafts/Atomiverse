---
title: 移植 LAMMPS CommBrick 的每步通信纪律到 ATV halo
slug: lammps-comm-port
status: done
created: 2026-08-26
grilled: true
---

# 移植 LAMMPS CommBrick 的每步通信纪律到 ATV halo

## Summary

把 LAMMPS `CommBrick` 的每步通信纪律移植进 ATV 的 CPU halo：步进路径上零集合通信、前向载荷只带坐标（`comm_x_only` 语义）、接收缓冲区不再逐步清零。两棵源码树的对比读已经证明点对点部分不是瓶颈——两边每 rank 每步都是同样的 40 次 P2P、同样约 5500 个 ghost——差距全部来自挂在 halo 旁边的每步 6 次 `MPI_Allreduce`（np=256 下约 2.3–3.0 s）、56 B 对 24 B 的前向行载荷，以及每 rank 每步约 440 KB 的接收缓冲 memset。本 spec 只删除冗余的集合通信与拷贝，不改变任何物理量：所有已存在的 MPI 单元套件与 regression 必须在 pin 不变的前提下保持绿，np=256 的 widebox 单点 parity 必须仍然逐位/窗口内一致。用户可见结果是 np=256、24048 原子、5000 步的 `Comm` 段从 3.54 s 降到约 1 s，公开 API 一个字节都不变。

## Domain basis

本 spec 的"领域"是并行分解的几何契约，不是新的物理项：

- **每轴跳数（uniform 分区闭式）**：`hops_d = ceil(shell · dims_d / L_d)`，`shell = max(pair cutoff, bonded extension)` [Å]，`L_d` 为全局盒子边长 [Å]。生产几何 `L = 63.0016 Å`（立方）、`dims = 8,8,4`、`shell = 11 Å` ⇒ `{2, 2, 1}`。
- **最小镜像下界（拒绝条件）**：周期轴上 `shell ≤ L_d − w_self`，等价于 `hops_d ≤ dims_d − 1`；越界即原子会与自身周期镜像作用，必须抛出而不是降级。
- **分解不变量**：域分解是同一个求和的划分，能量/受力/`Σᵢ Fᵢ` 不得随 rank 数或处理器网格改变（除双精度求和顺序外）。这是本 spec 全部 runtime 验收的判据来源。
- 单位：长度 Å、能量 eV、力 eV/Å，与 `core/units.h` 一致。

参考（移植来源与文献）：

- S. Plimpton, *J. Comput. Phys.* **117**, 1 (1995), DOI 10.1006/jcph.1995.1039 —— swap-list 前向/反向通信（`maxneed`、`sendlist`、`firstrecv`、`sendself`）。
- A. P. Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022), DOI 10.1016/j.cpc.2021.108171 —— 现代 LAMMPS `borders` / `forward_comm` / `reverse_comm` 语义。
- LAMMPS 源码基线 `/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-src` @ `78a7ddf` (2026-08-25)：`src/comm_brick.cpp:485` `CommBrick::forward_comm`、`:545` `reverse_comm`、`:720` `borders`（步进路径完全无集合通信，校验只在 `setup()`/`borders()`）；`src/atom_vec.cpp:354` `AtomVec::pack_comm`（每 ghost 仅 x/y/z）对 `:796` `pack_border`（v/mass/q/type/tag/image）。

## Design

移植的是**通信纪律**，不是文件；ATV 的 Frame/Driver/Component 形状与公开 API 不变，`HaloSend` / `HaloRecv` / `HaloRefresh` 的构造参数、注册路径、`ctx.md.swap_list` 发布契约全部保持原样。改动集中在三处纪律：

**1. 集合通信离开步进路径（D1）。**
`require_collective_halo_schema`（`src/cpu/communication/halo_exchange.cpp:461`，内部两次 `MPI_Allreduce`，`schema_mpi.h:31-32`）目前在 `HaloSend::launch`（:1238）、`HaloRefresh::launch`（:1360）、`HaloRecv::launch`（:1477）每步各调用一次，`HaloRecv` 另有一次 ghost 布局守卫 Allreduce（:1491-1497）。降级依据：plan 指纹是 atoms-Block 列集合的纯函数（`atom_pack.cpp:94-119`，含 `flags`/`cols`），只有 `Block::insert`/`erase` 能改变它（`frame.cpp:320-339`），而没有任何 Component 在 `launch()` 里做这件事；运行中的 `set_rows`/restride 只 bump `generation_`（本地令 `PackPlan::valid_for` 失效，`atom_pack.h:36-39`），指纹不变。因此：

- `HaloSend`：schema 校验保留在 `build()` 与 `full_rebuild == true` 分支内（该分支已经 bump `dd->halo_generation`，:1312）。
- `HaloRecv`：schema 校验与布局守卫都改为"仅当 `dd->halo_generation` 相对上次校验发生变化时执行"，新增一个私有 `std::uint64_t last_checked_halo_generation_` 成员。该触发条件是 rank 一致的：`halo_generation` 由 `HaloSend` 在所有 rank 的同一步同时 bump。
- `HaloRefresh`：`build()` 里做一次性校验（当前 `build` 为空，`:1333`），`launch()` 不再有任何集合调用。

`setup_swaps`（:599-620）的 rounds `MPI_Allreduce(MAX)` 直接删除，替换为 swap_list 侧的新纯函数 `SwapList::derive_uniform_rounds(dd, box, shell)`：对每个轴 `d` 扫过该轴的全部坐标 `c ∈ [0, dims[d])`，逐个复制 `dd` 并置 `coords[d] = c` 调用既有 `derive_hops`，取 `max(minus[d], plus[d])` 的全局最大值——复杂度 `O(Σ_d dims[d])`，与 rank 数无关。所有输入都是 rank 一致的：`dims`/`periods` 一致（`domain.h:62,86`），`cuts` 按契约复制（`domain.h:191-193`），`box` 是全局盒子（`domain.cpp:44-56`），`shell = max(cutoff_, bonded_extension_)` 而 `bonded_extension_` 已在 `build()` 里一次性 Allreduce(MAX) 归约（`halo_exchange.cpp:639-644`）。**约束（必须写进注释）：`bonded_extension_` 只能停留在 build 期；一旦它变成每步量，它的 Allreduce 会跟着回到步进路径，本 spec 的前提失效。** 最小镜像拒绝（`refuse_minimum_image`，`swap_list.cpp:87-105`）在本地扫描内抛出，因为每个 rank 都看过每个坐标，拒绝天然 rank 一致——严格优于今天"未触发的 rank 收到降级泛化消息"的行为（:612-618）。`SwapList` 仍然**不含任何 MPI**（`swap_list.cpp:1-6` 的构造性约束不破）。

`mpi::Reduce`（`reduce.cpp:70`，每步 1 次、20 个 double）**保留、不是本 spec 的交付项**：NVT 每步需要全局动能，LAMMPS `fix_nh` 做同样的温度 Allreduce，只是记在 Modify 段。

**2. `comm_x_only` 载荷（D2，是成员关系的纠正，不是新开关）。**
`frame_schema.h:96-111` 中 `vx`/`vy`/`vz`/`charge` 去掉 `kColComm`，`kColComm` 精确等于 `x,y,z`（默认 double 存储下 24 B/ghost），与其 LAMMPS 孪生 `pack_comm` 一致；`kColBorder` 原样保留 v/mass/charge/type/tag/image，所以 ghost 的 v 与 q 仍然在每次 full rebuild（border）落地，和 `pack_border` 完全同构。pack/unpack **无需改动**：PBC 位移判定键于 `col.wire_id <= AtomFieldId::z`，不是列表位置（`atom_pack.cpp:291`）。

配套修复（承重，不可省）：`refresh_columns`（:1102-1139）从传入的 plan 里挑字段三元组，挑不到就抛 `"halo refresh plan is missing its field set"`（:1135-1137）；因此 `HaloRefresh::launch` 的 plan 源（:1349）必须从 `make_comm_plan` 改为 `make_border_plan`。否则 `HaloRefresh{velocities}`（约束运行自动挂载，`md.h:460-467`）与 `HaloRefresh{charges}`（电极栈，`md.h:478-482`，五个 `updates_owned_charges()` 声明者）在任何多 rank 运行上都会当场抛错。refresh 的线上字节数不变（只由选中的列决定），传输行为不变。

力/能量路径上的 ghost 速度消费者：**没有**（负结果）。`PairDpd` 是 rank-local 全对、nlocal 作用域、未实现 `NeighborConsumer`；`FixRattle` 在 `final_integrate` 读 ghost v，由约束门控的 `HaloRefresh{velocities}` 供给。将来若出现第二个消费者，指定的挂钩是 InfraScan 式声明探测（`driver.h:1223-1265` + `component.h:112` 的 `updates_owned_charges()` 为模板）——在第二个消费者出现前不做。ghost 电荷同理：静态电荷在 rebuild 之间不变；改写自有电荷的组件必须声明 `updates_owned_charges()`，MD 据此自动挂载 charge refresh。顺带说明：`md.h:450-452` 里"前向 halo 只带坐标、ghost 速度根本不存在"的注释今天其实是错的，本改动之后才成立。

**3. 每步接收缓冲不再清零（D3）。**
`rbuf.assign(n, byte{0})`（:776-781）与 `recv_force[...].assign(..., byte{0})`（:1530-1534）改为只设尺寸不清内容（`resize` 保持容量语义即可）：LAMMPS 的 `buf_recv` 从不清零，字节数在 unpack 侧本来就校验。**必须区分**：`zero_ghost_forces`（:1090-1099，:1316）是反向部分和累加器的物理要求，原样保留——本条只删线上字节缓冲的 memset。

**溯源头（CLAUDE.md § License & LAMMPS porting 强制）。** `swap_list.h:2,20` 已经是"孪生 + DOI"形态；把 `halo_exchange.h`/`.cpp` 的文件头扩写为同一形态，点名 `src/comm_brick.cpp` 的 `forward_comm` / `reverse_comm` / `borders` 为被移植的纪律，并保留 Sandia 版权声明。使用孪生+DOI 形态（按结构重新实现），**不用** `rys_roots.h` 的"命名空间副本"形态——本 spec 不誊抄任何 LAMMPS 源码。同时按第 9 条法则修正 `halo_exchange.h:92-93`：那段自述的稳态契约今天被每步集合通信自我否定，重写为新契约（refresh 路径上零集合通信、零重分配、零 memset）。

**Reuse decision**（逐条落定 librarian 的候选）：

- `reuse` `SwapList::derive_hops`（`swap_list.h:226`）—— 新函数把它当作逐坐标核，不重写任何跳数数学，非均匀 `cuts` 链行走逻辑原样复用。
- `reuse` `make_border_plan`（`atom_pack.h:45-46`）—— `HaloRefresh` 直接改指到它；**不新增** `make_*_plan` 自由函数（该组是已钉住的例外集，不得扩张）。
- `reuse` `require_collective_halo_schema`（`halo_exchange.cpp:461`）—— 校验逻辑一字不改，只改调用时机。
- `reuse` `dd->halo_generation`（`domain.h:128`）—— 作为 `HaloRecv` 的 rank 一致 full-rebuild 信号；不新增标志位、不新增 Config、不新增环境变量。
- `new` `SwapList::derive_uniform_rounds` —— 现有符号里没有任何一个能在无 MPI 前提下给出 rank 一致的 rounds；命名、抛错风格（带 how-to 的 `std::runtime_error`）、Doxygen 单位标注一律照抄最近的 pattern `derive_hops` / `max_hops`，使新代码读起来与旧代码同源。
- `DomainDecomposition::coords_of`（`domain.h:79-83`）—— 本 spec 不需要它：扫描直接遍历 `coords[d]` 而非 rank id；它继续是 rank↔coords 的唯一权威，保持原样。

## Files to create or modify

- `include/atomiverse/core/frame_schema.h`
- `include/atomiverse/cpu/communication/swap_list.h`
- `src/cpu/communication/swap_list.cpp`
- `include/atomiverse/cpu/communication/halo_exchange.h`
- `src/cpu/communication/halo_exchange.cpp`
- `tests/cpu/communication/test_swap_list_uniform_rounds.cpp` (new)
- `tests/cpu/communication/test_atom_pack_comm_plan.cpp` (new)
- `tests/CMakeLists.txt`
- `tests/mpi/test_halo_refresh_charges.cpp`
- `tools/bench_gaff_lammps/README.md`

## Tasks

- [x] Write failing unit tests for `SwapList::derive_uniform_rounds` (tests/cpu/communication/test_swap_list_uniform_rounds.cpp → `TestSwapListUniformRounds`; register `test_cpu_comm_swap_list_uniform_rounds` in tests/CMakeLists.txt beside line 200)
- [x] Implement `SwapList::derive_uniform_rounds` in include/atomiverse/cpu/communication/swap_list.h + src/cpu/communication/swap_list.cpp (pure, no MPI, Doxygen `@param`/`@return` with [Angstrom]) and switch `setup_swaps` (src/cpu/communication/halo_exchange.cpp:599-620) to it, deleting the rounds `MPI_Allreduce`
- [x] Write failing unit test pinning comm-plan membership (tests/cpu/communication/test_atom_pack_comm_plan.cpp → `TestCommPlanMembership`; register `test_cpu_comm_atom_pack_comm_plan` in tests/CMakeLists.txt)
- [x] Narrow `kColComm` to x/y/z in include/atomiverse/core/frame_schema.h:96-111 and repoint `HaloRefresh`'s plan source to `make_border_plan` at src/cpu/communication/halo_exchange.cpp:1349, correcting the now-false `kColComm` comment in tests/mpi/test_halo_refresh_charges.cpp:94
- [x] Demote the three `require_collective_halo_schema` calls (src/cpu/communication/halo_exchange.cpp:1238, :1360, :1477) and the `HaloRecv` layout-guard Allreduce (:1491-1497) to `build()` / full-rebuild, adding the `last_checked_halo_generation_` member in include/atomiverse/cpu/communication/halo_exchange.h
- [x] Remove the per-step recv-buffer zeroing in src/cpu/communication/halo_exchange.cpp:776-781 and :1530-1534 (size-set without clearing; keep `zero_ghost_forces` untouched)
- [x] Add the `comm_brick.cpp` provenance header (twin + DOI form, Sandia notice) to include/atomiverse/cpu/communication/halo_exchange.h + src/cpu/communication/halo_exchange.cpp and rewrite the contradicted steady-state contract paragraph at halo_exchange.h:92-93
- [x] Verify regression examples regressions/md/comm-swaplist-halo-mpi, comm-topology-multihop, multihop-widebox-parity, const-pot-charge-halo-mpi, regressions/longrange/pppm-subcomm-parity-mpi exit 0 with unchanged pins, run `widebox_parity.sh` NP=256 DIMS=8,8,4, and record the measured `Comm` section in tools/bench_gaff_lammps/README.md as a dated snapshot (no threshold)
- [x] Run full check + test suite
- [x] `/mol:simplify` clang-format on touched C++ + stale 7-column test comments; no other apply-list hygiene

## Testing strategy

**单元测试（`tests/`，秒级，单函数/单方法，镜像 `src/` 布局）**

- `tests/cpu/communication/test_swap_list_uniform_rounds.cpp` → `TestSwapListUniformRounds`，目标只有 `SwapList::derive_uniform_rounds`：
  - happy path / 生产几何：`dims = {8,8,4}`、`box = Box::cubic(63.0016)`、`shell = 11.0` ⇒ rounds `{2, 2, 1}`（硬编码期望值，来自 `ceil(shell·dims/L)`）。
  - 一致性：在均匀网格与非均匀 `cuts` 网格上，`derive_uniform_rounds` 逐轴等于**遍历该轴全部坐标**的 `derive_hops` 的最大值（测试内自行循环 coords 求 max，作为旧行为的等价 pin）。
  - 边界：`dims[d] <= 1` 的轴恒为 0（豁免所有宽度检查）；`shell` 恰好等于一个 slab 宽度仍为 1 跳；`shell = 0` 全零；非周期轴在全局边界截断而不拒绝。
  - 拒绝：`dims = {2,1,1}`、`L = 10 Å`、`shell = 9 Å`（`L − w_self = 5 Å`）⇒ 抛 `std::runtime_error`，且**在该轴的每一个坐标上**都抛出同样含"minimum-image""axis 0"与数值界的消息（证明拒绝 rank 一致，不再需要 Allreduce 携带 flag）。
- `tests/cpu/communication/test_atom_pack_comm_plan.cpp` → `TestCommPlanMembership`，目标只有 `make_comm_plan`（无需 MPI）：规范 schema 下恰好 3 列、`wire_id` 为 `x,y,z`、`comm_bytes(plan, 1) == 3 * sizeof(StorageReal)`（默认 double 下 24 B）；同一 frame 的 `make_border_plan` 仍含 `vx,vy,vz,mass,charge,type,tag,image`（证明 ghost v/q 只是搬到 border 期，不是被删掉）。

**MPI 行为门（`tests/mpi/`，全部保持不变而绿，是本 spec 的行为判据）**：`test_mpi_halo_dual_image_ghosts`（n1/n2/n4/n8）、`test_mpi_halo_exchange`、`test_mpi_halo_refresh_charges`、`test_mpi_evaluate_halo_parity`、`test_mpi_pppm_brick_parity`、`test_mpi_pppm_subcomm_parity`、`test_mpi_atom_migration`、`test_mpi_lj/coul/bonded_energy_bruteforce`。除 `test_halo_refresh_charges.cpp:94` 的注释更正外，不允许改动任何断言——改断言即等于改行为。

**Regression（本仓库法则：e2e 只住 `regressions/<domain>/<case>/`，一个交付特性一例、带文献引用）**。本 spec 交付的是通信纪律移植，**不引入任何新物理**，因此不新建 oracle 案例（新建会制造无来源的 pin，违反 pin 溯源法）。指定的 regression example 是已存在的 `regressions/md/comm-swaplist-halo-mpi`（`oracle: literature`，判据是分解不变量恒等式，pins 已硬编码在 `REFERENCE.md`：`|E_dd − E_brute|/N ≤ 1e-9 eV/atom`、`maxᵢ|ΔF| ≤ 1e-9 eV/Å`、`‖ΣF‖ ≤ 1e-10 eV/Å`、`nswap = 6`、`Σnrecv = n_ghost = 384`、rebuilds ≥ 2），并连带 `comm-topology-multihop`、`multihop-widebox-parity`、`const-pot-charge-halo-mpi`（**HaloRefresh 重指向的承重门**：`max|q_ghost − q_owner| = 0`，tol 1e-15 e）、`regressions/longrange/pppm-subcomm-parity-mpi`，全部 exit 0 且 pin 不变。所有 golden 都是文件内硬编码或文件内 O(N²) 复算，运行期不导入、不 spawn 任何第三方引擎。

**机器判据**：`DATA=<MIL0PEGM18 eq.data> FF=<bench.ff> NP=256 DIMS=8,8,4 tools/bench_gaff_lammps/widebox_parity.sh`，bond/angle/dihedral/pair+kspace/E/fmax/fnorm 全窗口 PASS，且脚本自检的 `hops ≥ 2` 多跳几何断言成立（当前 `dev` 基线是每一位都 dev=0.0；本移植只删冗余集合通信与拷贝，数值必须逐位一致或落在窗口内）。

**性能**：`tools/bench_gaff_lammps/README.md` 只记录快照（预期 np=256 `Comm` 3.54 s → ~1 s），**永远不作为验收判据**。

**已知债务（第 9 条法则：点名，不在此修）**：`src/cpu/dynamics/qeq.cpp:231` 写 `kAtom_charge` 却未声明 `updates_owned_charges()`，且 nlocal 作用域、无 MPI —— 今天就是 rank-local 破的（其 regression 为单 rank，不受本 spec 影响），走独立 `/mol:fix` slug。

## Out of scope

- `QEq` 的 `updates_owned_charges()` 声明与 DD 化（独立 slug）。
- 反向 `comm_f_only` 零拷贝（先测量：SoA → 3 条消息的交换需要数据支撑）。
- 稠密 tag 映射替换 `tag_to_local` 哈希抖动。
- 只在 reneighbor 时做坐标 wrap（受 `atom_migration.cpp:162-163` 的 Migrate slab 比较依赖阻塞）。
- kspace remap 集合形状（reduce_scatter / neighborhood）—— 下一个 slug。
- ghost 速度消费者的 InfraScan 式声明探测：在出现第二个真实消费者之前不建抽象。
- `bonded_extension_` 的任何每步化改造（会把它的 Allreduce 带回步进路径）。
- CUDA halo、`cuda/communication/` 任何改动；本 spec 是 CPU 单层。
- 任何性能门槛型验收；任何新 Config bag / 工厂函数 / 环境变量。
