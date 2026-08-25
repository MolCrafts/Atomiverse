---
title: 多跳 halo — maxneed 展开与窄子域正确性
status: in-progress
created: 2026-08-25
grilled: true
deps: comm-topology-01-swaplist
---

# 多跳 halo — maxneed 展开与窄子域正确性

## Summary

在 `comm-topology-01-swaplist` 交付的 swap-list 架构上放开"每维一跳"的限制：每个分解轴按 LAMMPS `Comm::setup` 语义派生 `maxneed`（均匀切分为 `ceil(need / w)`，RCB 非均匀切分沿发送方向逐个累加对端 slab 宽度直至覆盖 `need`），并在一维内执行多轮 swap，每轮把上一轮已收到的 ghost 行继续转发给更远的 rank。交付后，子域宽度小于 `cutoff + skin` 的分解（窄盒、高 rank 数、RCB 极端切分）不再抛异常拒绝，而是正确运行：能量与受力与单 rank 在数值窗口内一致。两处几何拒绝点（build 时与 launch 时）被 `SwapList::setup()` 的重派生取代，NPT/deform 盒收缩使 `maxneed` 升高时自动增加轮数；cell-list 的扩展轴垫层同步跟随真实 ghost 壳宽度，避免多跳 ghost 落到网格外被 `bin_atoms` 静默钳位。

## Domain basis

**ghost 壳宽度（不变量，本 spec 不改动）**

```
need_d = max(r_cut_halo, bonded_extension)      [Å]
```

`bonded_extension` 由 `HaloSend::compute_bonded_extension` 在 `build()` 扫描 per-atom 键接拓扑得到，并经 `MPI_Allreduce(MAX)` 全局归约（`src/cpu/communication/halo_exchange.cpp:478-485`）。多跳不改变 `need` 的定义，只改变覆盖 `need` 所需的轮数。

**每维跳数（LAMMPS `Comm::setup`）**

均匀切分（`dd->cuts[d]` 为空，`layout = UNIFORM`），子域宽度 `w_d = L_d / dims_d`：

```
maxneed_d = ceil(need_d / w_d)
```

非均匀切分（`dd->cuts[d]` 非空，RCB / `Rebalance` 产生，`layout = NONUNIFORM`）：对方向 `s ∈ {−1, +1}`，沿发送方向依次访问 rank 坐标 `c − s·1, c − s·2, …`（周期回绕），累加其 slab 宽度 `w`，

```
need^{(s)}_d = min{ k : Σ_{i=1..k} w(c − s·i) ≥ need_d }
maxneed_d    = MPI_Allreduce(MAX) over all ranks and both directions
```

即"发送方向上最薄 slab 链"决定轮数；全局取 MAX 使所有 rank 的轮数一致，轮次对称、天然无死锁。均匀切分下该式退化为闭式解，无需归约。

**最小镜像上界（保留的唯一拒绝条件）**

周期轴上 ghost 壳最多覆盖除本 rank 自身 slab 外的全部盒长，超过即需要原子与自身周期像相互作用，违反最小镜像约定：

```
maxneed_d ≤ dims_d − 1        等价于   need_d ≤ L_d − w_d(self)
```

不满足时抛异常，信息指向"cutoff 超过最小镜像上限"，而不是指向"子域太薄"。非周期轴（`periods[d] == 0`）同样截断到 `dims_d − 1`，超出部分没有对端。

**验证量**：同一构型的总势能 `E [eV]` 与逐原子受力 `F_i [eV/Å]` 与 rank 网格无关（空间分解不变性）。

参考文献：

- S. J. Plimpton, *Fast Parallel Algorithms for Short-Range Molecular Dynamics*, J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039 —— 空间分解、ghost 壳与逐跳转发（每轮消息携带上一轮已收数据）。
- A. P. Thompson et al., *LAMMPS — a flexible simulation tool …*, Comput. Phys. Commun. **271**, 108171 (2022), DOI 10.1016/j.cpc.2021.108171 —— `Comm::setup` 的 `maxneed`/`nswap` 派生与 UNIFORM / NONUNIFORM layout 语义。
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed., Oxford University Press (2017), DOI 10.1093/oso/9780198803195.001.0001 —— 最小镜像约定与 `r_c ≤ L/2` 上界。

## Design

### 跳数派生（`SwapList`，-01 交付的类型）

`SwapList` 新增一个纯派生入口：输入 `dims[3]`、`periods[3]`、`coords[3]`、`cuts[3]`（分数）、盒长 `L[3]`、壳宽 `need`，输出每维 `max_hops(d)` 与 `nswap(d) = 2 · max_hops(d)`。均匀路径为闭式，非均匀路径走"发送方向 slab 链累加 + `MPI_Allreduce(MAX)`"。派生本身是纯函数（可脱离 MPI 单测），归约由 `setup()` 包装。

`setup()` 在每次 **full rebuild** 开头调用（LAMMPS 在每次 reneighbor 调用 `comm->setup()`）。开销为 O(dims) 算术，非均匀时多一次 1-int 归约，与既有的 per-face count 交换同量级。NPT/deform 收缩盒子 → `w_d` 变小 → `max_hops` 升高 → 本轮直接多发几轮，不抛异常、不静默截断。

命名避让：`NeighborConsumer::halo_cutoff()`（`include/atomiverse/cpu/component/neighbor_protocol.h:55`）与 `HaloSend::halo_cutoff()`（`include/atomiverse/cpu/communication/halo_exchange.h:139`）已经同名不同物，新类型上一律取 `SwapList::shell_width()` / `SwapList::max_hops(d)`，不再增加第三个 `halo_cutoff`。

### 多跳前向（`HaloSend::launch`）

一维内的轮次沿用 `Migrate::exchange_along_dim`（`src/cpu/communication/atom_migration.cpp:154-250`）的形状：逐轮 `Sendrecv`、轮间 `Waitall` 串行、lo/hi 两条流用不同 tag（`dims[d] == 2` 时 `lo_nbr == hi_nbr`，共用 tag 会撞）。第 `k` 轮的候选源是 `[0, slot_k)`——包含前面轮次收到的 ghost 行，因此 sendlist 可含 ghost 行（-01 的 pack 原语已支持任意 frame 行，无需改动）。

每轮的挑选判据是几何 slab：把对端 rank 的近侧平面按周期位移映射到本 rank 坐标系，凡落在其 `need` 深度带内的行入 sendlist；`k = 1` 时退化为当前的"距面 `need` 以内"判据。几何判据保证一行不会被回送到它的来源方向，也不会出现自 ghost（`max_hops ≤ dims − 1` 已排除完整回绕）。

**分解轴绝不做最小镜像**：`CellList::min_image_unextended_`（`include/atomiverse/cpu/neighbor/cell_list.h:181`，调用点 `src/cpu/neighbor/cell_list.cpp:777`、`:953`）与 `axis_extended_` 判据保持原样。ghost 已被放在真实周期像位置，折叠会把同一 tag 的两个像折成一条分离，正是 `tests/mpi/test_halo_dual_image_ghosts.cpp` 钉住的 ~16% LJ 静默双计数。`dims[d] == 2` 的多跳恰好落在这个几何里，属于本 spec 必须保住的硬约束。

`++dd->halo_generation`（`src/cpu/communication/halo_exchange.cpp:1076`）保持在每次 full rebuild 末尾 bump：多跳下 ghost 槽位布局变化更频繁，`NeighborList` 的 `halo_stale`（`include/atomiverse/cpu/neighbor/neighbor_list.h:191-193`）依赖它作废 CSR 中的 ghost 行索引；漏 bump 的后果记录在 `domain.h:107-124`（NPT 守恒量每周期 1–3 eV 跑飞）。

### 反向（`HaloRecv`）

反向按前向轮次的**逆序**逐轮归约：最远一轮的 ghost 力先回传到中间 rank 的 ghost 行上，再随下一轮回到 owner。槽位与计数复用前向记录的 per-swap `recv_counts`，语义与现有单跳反向一致，仍然"恰好执行一次"。

### cell-list 垫层联动（耦合点，必须同批落地）

`CellList::init_extended` 当前忽略传入的 `cutoff_skin` 形参（`src/cpu/neighbor/cell_list.cpp:47`），扩展轴网格窗口固定为 `[sublo − (cutoff+skin), subhi + (cutoff+skin)]`（`:96-107`），只垫了一个 cutoff。`maxneed > 1` 时 ghost 壳深达 `need`，超出窗口的行会被 `bin_atoms` 的钳位 `cx = max(0, min(cx, ncx−1))`（`:215-217`）**静默**折进边缘格——不报错、不缺原子，只是配对错误。

因此：`dd` 上新增 `double ghost_shell_width`（Å，`HaloSend::build`/`launch` 写入 `max(cutoff_, bonded_extension_)`），`compute_grid` 的扩展轴垫层取 `max(cutoff_skin_, dd_->ghost_shell_width)`，格边长仍由全局盒长整除派生（保持跨 rank 的格点晶格对齐，见 `:76-88` 的 home/forward 分类论证）。`grid_stale()`（`:129-143`）的比较集合加入捕获的壳宽，使壳宽升高时重派生网格。`ghost_shell_width == 0`（HaloSend 尚未 build）退化为今天的行为。

顺带修掉一处既有债：`bonded_extension > cutoff + skin` 的体系（GAFF 类长键接跨度）今天就已经可能让 ghost 落到窗口外，同一处改动一并覆盖。

### Why（决策记录）

- 多跳按 LAMMPS `maxneed` 展开，而不是自造"按需扩壳"：与 `Comm::setup` 语义一致，轮次全局一致因而无死锁，且与既有的 `exchange_along_dim` 轮次形状同构。
- RCB 非均匀 slab 按发送方向最薄链推导：拒绝点原来的判据（`need > 本 rank 自身 sub`）本身就问错了问题——决定能否覆盖的是**对端**的宽度链，不是自己的宽度。
- 两处拒绝点换成 `setup()` 重派生，而不是删除：盒收缩必须触发 `nswap` 上调；静默截断是本条改动要根除的东西，不是要保留的。
- cell-list 垫层与 halo 宽度联动：否则多跳 ghost 被 `bin_atoms` 钳位，得到的是"能跑但配对错"的静默错误，比抛异常更糟。

### Reuse decision

1. `reuse` `HaloSend::compute_bonded_extension` + `MPI_Allreduce(MAX)`（`src/cpu/communication/halo_exchange.cpp:478-485`，访问器 `halo_exchange.h:139`）—— `need = max(pair cutoff, bonded skin)` 的语义原样保留，多跳只改轮数。
2. `generalize` `HaloSend` 的几何守卫（`src/cpu/communication/halo_exchange.cpp:499-517` build 时 + `:1032-1043` launch 时复查）—— 两处一起改写为"调用 `SwapList::setup()` 重派生跳数"，仅保留 `maxneed_d > dims_d − 1` 的最小镜像上界抛出。不是删除，也不是静默截断。
3. `generalize` `CellList::init_extended` / `compute_grid` / `grid_stale`（`src/cpu/neighbor/cell_list.cpp:47`、`:69-127`、`:129-143`）—— 扩展轴垫层从"一个 cutoff_skin"推广为"`max(cutoff_skin, ghost 壳宽)`"，同时服务单跳与多跳两类调用者；入口 `include/atomiverse/cpu/neighbor/neighbor_list.h:315` 随之传参。
4. `reuse` `CellList::min_image_unextended_`（`cell_list.h:181`，调用点 `cell_list.cpp:777`/`:953`）与 `axis_extended_` 判据 —— 一行不改；分解轴上的 ghost 永不折叠是硬约束。
5. `reuse` `++dd->halo_generation`（`halo_exchange.cpp:1076`）与 `NeighborList::halo_stale`（`neighbor_list.h:191-193`）—— 多跳 borders 每次 sendlist 重建继续 bump，语义不变。
6. `reuse` `tests/mpi/test_halo_dual_image_ghosts.cpp`（注册于 `tests/CMakeLists.txt:560-562`）—— 扩展该文件而非新开：它已拥有窄盒 / `dims==2` 几何与 O(N²) 最小镜像暴力 oracle（`:71`）；更深 `maxneed` 用 `atv_add_mpi_run` 增注册 `_n4` / `_n8`。
7. `reuse` `ATV_FORCE_DIMS`（唯一解析点 `src/cpu/communication/domain.cpp:74`）强制窄切分；in-tree 先例 `regressions/md/const-pot-charge-halo-mpi/reg_const-pot-charge-halo-mpi.cpp:200` 的 `::setenv("ATV_FORCE_DIMS", …)`。
8. `reuse` `Migrate::exchange_along_dim`（`src/cpu/communication/atom_migration.cpp:154-250`）作为轮次转发的形状范本（逐轮 Sendrecv、`dims==2` 时 lo/hi 不同 tag）——迁移路径本身**已经**支持多跳，本 spec 不改动它。
9. `reuse` `SwapList`（-01 交付）—— 多跳挂在其 `setup()` 上，不新建并行类型。
10. `new` — `SwapList::max_hops(d)` / `nswap(d)` / `shell_width()` 与 `DomainDecomposition::ghost_shell_width`：既有类型上没有承载"跳数"和"壳宽"的字段，且必须避开 `halo_cutoff()` 的既有命名冲突。

## Files to create or modify

- `include/atomiverse/cpu/communication/swap_list.h` —— `max_hops` / `nswap` / `shell_width` 与 `setup()` 重派生（由 -01 交付，路径以 -01 实际落地为准）
- `src/cpu/communication/swap_list.cpp` —— 均匀闭式 + 非均匀 slab 链派生
- `include/atomiverse/cpu/communication/domain.h` —— 新增 `ghost_shell_width` 字段（Å）
- `include/atomiverse/cpu/communication/halo_exchange.h` —— `HaloSend` 多跳状态与 doxygen 契约更新
- `src/cpu/communication/halo_exchange.cpp` —— 多跳轮次前向/反向、两处守卫改写、`ghost_shell_width` 写入
- `include/atomiverse/cpu/neighbor/cell_list.h` —— 捕获的壳宽成员与 `init_extended` 契约
- `src/cpu/neighbor/cell_list.cpp` —— 扩展轴垫层与 `grid_stale` 联动
- `include/atomiverse/cpu/neighbor/neighbor_list.h` —— `reinit_strategy` 调用点传壳宽
- `tests/cpu/communication/test_swap_list_maxneed.cpp` (new)
- `tests/cpu/dynamics/test_cell_list_halo_pad.cpp` (new)
- `tests/mpi/test_halo_dual_image_ghosts.cpp` —— 新增窄子域用例
- `tests/CMakeLists.txt` —— 两个串行单测注册 + `_n4` / `_n8` 的 `atv_add_mpi_run`
- `regressions/md/comm-topology-multihop/reg_comm-topology-multihop.cpp` (new)
- `regressions/md/comm-topology-multihop/REFERENCE.md` (new)
- `regressions/CMakeLists.txt` —— 在 `ATV_USE_MPI AND MPIEXEC_EXECUTABLE` 块内注册并定义 `ATV_MPIEXEC`

## Tasks

- [x] Write failing unit tests for SwapList hop derivation (tests/cpu/communication/test_swap_list_maxneed.cpp → TestSwapListMaxneed; register in tests/CMakeLists.txt)
- [x] Implement uniform + non-uniform hop derivation and `setup()` re-derivation in `SwapList` (include/atomiverse/cpu/communication/swap_list.h, src/cpu/communication/swap_list.cpp)
- [x] Write failing unit tests for the halo-coupled cell grid pad (tests/cpu/dynamics/test_cell_list_halo_pad.cpp → TestCellListHaloPad; register in tests/CMakeLists.txt)
- [x] Generalize `CellList::init_extended`/`compute_grid`/`grid_stale` to pad extended axes by `max(cutoff+skin, ghost_shell_width)` (src/cpu/neighbor/cell_list.cpp, include/atomiverse/cpu/neighbor/cell_list.h), add `ghost_shell_width` to include/atomiverse/cpu/communication/domain.h and pass it at include/atomiverse/cpu/neighbor/neighbor_list.h:315
- [x] Write failing narrow-subdomain MPI cases in tests/mpi/test_halo_dual_image_ghosts.cpp (ATV_FORCE_DIMS 4,1,1 and 8,1,1) and register `_n4` / `_n8` via `atv_add_mpi_run` in tests/CMakeLists.txt
- [ ] Implement multi-round forward swaps with ghost forwarding and reverse-order accumulation in `HaloSend`/`HaloRecv` (src/cpu/communication/halo_exchange.cpp, include/atomiverse/cpu/communication/halo_exchange.h)
- [ ] Generalize the two `HaloSend` geometry guards (src/cpu/communication/halo_exchange.cpp:499-517 and :1032-1043) into `SwapList::setup()` re-derivation, keeping only the `maxneed > dims-1` minimum-image throw
- [ ] Add regression example regressions/md/comm-topology-multihop/ (reg_comm-topology-multihop.cpp + REFERENCE.md; register in regressions/CMakeLists.txt) with hard-coded tolerances and an in-file O(N²) oracle
- [ ] Add doxygen `@brief`/`@param`(with units)/`@return` for `SwapList::max_hops`/`nswap`/`shell_width`, `DomainDecomposition::ghost_shell_width`, and the updated `HaloSend` geometry contract
- [ ] Run full check + test suite

## Testing strategy

单元测试全部位于 `tests/`，路径镜像 `src/`，每个测试针对单一函数/方法，无 e2e、无收敛回路、无第三方读取。

**`tests/cpu/communication/test_swap_list_maxneed.cpp`（串行，不初始化 MPI；目标：跳数派生纯函数）**

- happy path（均匀）：`L = 15.78 Å`、`need = 6.6 Å`；`dims = 4 → w = 3.945 Å → max_hops = 2`；`dims = 8 → w = 1.9725 Å → max_hops = 4`；`dims = 2 → w = 7.89 Å → max_hops = 1`。
- 边界：`dims = 1 → max_hops = 0`（该轴未分解，不产生 swap）；`need` 恰等于 `w` → `max_hops = 1`（不因浮点误差跳到 2）。
- 非均匀（RCB）：`L = 20 Å`、`dims = 4`、`cuts = {0, 0.5, 0.6, 0.7, 1.0}`（宽度 `{10, 2, 2, 6} Å`）、`need = 5 Å`。`coords = 3` 向 `−` 方向链 `2 → 4 → 14` ⇒ 3 跳，向 `+` 方向 `10` ⇒ 1 跳；`coords = 2` 向 `−` 为 2 跳；全局 MAX = **3**。全部为硬编码期望整数。
- 拒绝：`need > L − w_self` 抛 `std::runtime_error`，消息含最小镜像上界字样（不含"子域太薄，请减少 rank"）。
- `setup()` 重派生：同一实例上把 `L` 由 `15.78 → 10.0`（NPT 收缩）后 `max_hops` 由 2 升到 3，且再次以相同参数调用为幂等。

**`tests/cpu/dynamics/test_cell_list_halo_pad.cpp`（串行，`dd` 桩：`dims = {4,1,1}`、`sublo/subhi` 手填；每个 TEST 针对一个方法）**

- `compute_grid`（经 `init_extended`）：`cutoff = 2.0 Å`、`skin = 0.2 Å`（`cutoff_skin = 2.2 Å`）、`ghost_shell_width = 6.6 Å` ⇒ x 轴窗口覆盖 `[sublo − 6.6, subhi + 6.6]`，格数 `ncx ≥ ceil((w + 13.2) / cs)`；`ghost_shell_width = 0` ⇒ 退化为 `2.2 Å` 垫层（今日行为）。格边长仍为全局 `L / floor(L / cutoff_skin)`，原点为其整数倍（晶格对齐不被垫层破坏）。
- `bin_atoms`：位于 `sublo − 6.5 Å` 与 `sublo − 6.5 + cs` 的两行落入**不同**格且格索引 `> 0`（即未被 `cell_list.cpp:215-217` 的钳位折叠到边缘格）。
- `grid_stale`：仅把 `dd->ghost_shell_width` 由 `6.6 → 9.0` ⇒ 返回 `true`。

**`tests/mpi/test_halo_dual_image_ghosts.cpp`（扩展；单步冻结 dt = 0，无回路，秒级）**

- 既有 `n1` / `n2` 用例与 `1e-9` 相对窗口保持不变（回归护栏）。
- 新增 `_n4`：`ATV_FORCE_DIMS = 4,1,1`，3³ 晶格 `a0 = 5.26 Å`（`L = 15.78 Å`）、`cutoff = 6 Å`、自动 halo `6.6 Å` ⇒ 子域 `3.945 Å < 6.6 Å`（今日抛异常），`max_hops = 2`；总能量对 O(N²) 最小镜像暴力 oracle 相对误差 `< 1e-9`，且断言 `dd->n_ghost > 0`。
- 新增 `_n8`：`ATV_FORCE_DIMS = 8,1,1`，同盒 ⇒ 子域 `1.9725 Å`、`max_hops = 4`；同一 oracle、同一窗口；并覆盖"某些 rank owned 数为 0"的边界（27 原子 / 8 rank）。

**Regression example（`regressions/md/comm-topology-multihop/`，公共头 + link `atomiverse`，`oracle: literature`，≤ 60 s）**

父进程经编译期固定的 `ATV_MPIEXEC` 自启 `np = 4`，子进程 `ATV_FORCE_DIMS = 4,1,1`；4³ 晶格 `a0 = 3.945 Å`（`L = 15.78 Å`，64 原子）、LJ `ε = 0.0104 eV`、`σ = 3.4 Å`、`cutoff = 6 Å`（halo `6.6 Å` ⇒ `max_hops = 2`）。硬编码判据（全部为文件内字面量，运行时不调用任何第三方）：

| 判据 | 阈值 |
| --- | --- |
| `|E(np=4) − E_bruteforce| / |E_bruteforce|` | ≤ 1e-9 |
| `max_i |F_i(np=4) − F_i^bruteforce|` | ≤ 1e-9 eV/Å |
| 观测到的 x 轴 `max_hops` | ≥ 2（非空洞：否则退化为单跳） |
| `|E(np=4) − E(np=1)| / |E(np=1)|` | ≤ 1e-12（分解不变性） |

`E_bruteforce` 由文件内 O(N²) 最小镜像闭式求和给出（第一方、确定性）；`REFERENCE.md` 以字面量表格钉住交付时测得的 `E(np=1)` 值，并引用 Plimpton 1995 / Thompson 2022 作为多跳 ghost 语义的文献依据（与 `regressions/md/const-pot-charge-halo-mpi/REFERENCE.md` 的自洽不变量先例同形）。

## Out of scope

- swap-list 基础架构本身（`comm-topology-01-swaplist` 已交付；本 spec 只在其 `setup()` 上放开跳数）。
- 256-rank 实机验证与任何性能数字 / 加速比（属 `comm-topology-03` 的 bench 记录）；本 spec 的验收不含时间量。
- OpenMP 混合并行、kspace 子通信域、Kokkos 后端。
- CUDA 后端 halo（`cuda/communication`）：CPU 正确性先行，CUDA 切片不在本链内。
- `Migrate` 的多跳迁移：`exchange_along_dim`（`src/cpu/communication/atom_migration.cpp:154-250`）已逐轮转发跨多子域的 owner，本 spec 不改动它，也不重写为 swap-list。
- 重复 tag 的 ghost 在 `dd->tag_to_local` 上"后写覆盖"的既有语义：多跳使双像更常见，但键接元组解析行为不变，改动它需要独立 spec。
- 非正交盒（三斜）下的倾斜 slab 链推导：当前 `compute_sublo_subhi` 为正交实现，三斜是独立能力。

## Notes

- 依赖链：本 spec 假定 `comm-topology-01-swaplist` 已落地并提供 `SwapList` 类型。
- 顺带发现（iron law 9，已折进 Design）：`CellList::init_extended` 的 `cutoff_skin` 形参当前被忽略（`src/cpu/neighbor/cell_list.cpp:47`），且 `bonded_extension > cutoff + skin` 的体系今天就可能让 ghost 落到网格窗口外被 `:215-217` 静默钳位——同一处垫层改动一并覆盖。
- `Migrate::exchange_along_dim` 已支持多跳 owner 迁移，因此窄子域下的迁移路径无需改动；它同时是本 spec 轮次转发的 in-tree 形状范本。
