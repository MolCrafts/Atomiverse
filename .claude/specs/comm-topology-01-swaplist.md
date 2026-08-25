---
title: Swap-list halo — LAMMPS CommBrick 形状重构（行为等价，maxneed ≡ 1）
status: in-progress
created: 2026-08-25
grilled: true
---

# Swap-list halo — LAMMPS CommBrick 形状重构（行为等价，maxneed ≡ 1）

## Summary

把 `cpu/communication` 现有的「每步逐轴扫描 + 六面级联」halo 重构为 LAMMPS CommBrick 的 swap-list 架构：一张在 reneighbor 时刻派生一次的有序 swap 表（`sendproc`/`recvproc`/`pbc`/`sendlist`/`nrecv`/`firstrecv`），步内的 forward 与 post_force 的 reverse 都退化成「按表打包 / 按表回累加」，不再有任何逐步的几何重推导。本期严格 **行为等价**：`maxneed` 恒为 1（每维仅与直接邻居交换），ghost 槽位布局、PBC 镜像语义、dims==2 双镜像 ghost、`HaloRefresh` 的公开契约与自动注入点全部逐位保持不变；用户可见面（`HaloSend(cutoff, every)`、组件名、默认绑定 `pre_neighbor@0` / `post_force@-1000`）一字不改。旧级联路径整体删除，不留 fallback、不留开关——本仓无兼容层。多跳（maxneed>1）与大 rank 数验证不在本期。

Why（决策记录）：方案 B（按 LAMMPS 重构）由用户点名——每步重扫描是 comm 开销的结构性来源；`setup()`/`borders()` 的时机沿用 LAMMPS 约定（每次 reneighbor 调用 `setup` + `borders`，步内只 `forward_comm`）；reverse 同表逆序一并落地，避免留下半新半旧的两套索引；`Migrate` 的 `Alltoallv` 路径不动。本 spec 不做任何性能声明——它只把结构换掉并证明数值一致，性能量化留给后续 slug。

## Domain basis

- **Ghost 壳层几何（LAMMPS CommBrick `setup()`）**：沿维 `d` 需要的跳数
  `need_d = ceil(r_halo · dims_d / L_d)`，`nswap = 2 · Σ_d need_d`。约定说明：LAMMPS 原式是 `floor(x)+1`，与 ceil 仅在 `r_halo` 恰整除子域宽度的测度零边界不同；本仓取 **ceil**，使 `r_halo == sub` 保持单跳——这与今天守卫的严格大于（`need > sub` 才拒绝，等值放行，`halo_exchange.cpp:498-517`）逐位一致，也与 -02 单测「need 恰等于 w → 1 跳」的断言一致。
  **未分解轴豁免（等价性关键）**：今天的守卫先 `if (dims[d] <= 1) continue` 再比较（`halo_exchange.cpp:498-499`）——`dims_d == 1` 的轴强制 `need_d = 1` 且 `sendself`，即使 `r_halo > L_d` 也合法运行（该轴无 ghost，周期性由 cell-list 对非 extended 轴的 min-image 承担）。`SwapList::setup` 必须在 ceil 之前复刻这条短路，否则 EAM 型 `2·cutoff` 壳宽在薄轴（如 2×1×1 分解的 20×10×16 盒）上会把今天合法的运行改成硬抛。本期对**已分解轴**约束 `need_d ≡ 1`，即 `r_halo ≤ L_d / dims_d`；违反即 **throw**（与现有 build 守卫代数等价）。
  单位：`r_halo`、`L_d`、`sublo/subhi` 均为 Å；`r_halo = max(cutoff, bonded_extension)`。
- **周期镜像位移**：跨全局盒边界发出的原子在打包时加上整数镜像偏移 `pbc_d ∈ {−1, 0, +1}` 乘以当前 `L_d`，使接收方在自己的坐标系里看到该原子。速度、电荷不加位移（平移不变量）——`HaloRefresh` 的既有契约。
- **Reverse 累加 = 牛顿第三定律的分布式补全**：ghost 行是 `f_x/f_y/f_z`（及任何 `kColReverse` 列）的部分和，必须 **恰好一次** 沿同一张表逆序回加到 owner；漏一次丢力，多一次泵能量（Σ F ≠ 0）。单位：eV/Å。
- 参考文献：
  - S. Plimpton, *Fast Parallel Algorithms for Short-Range Molecular Dynamics*, J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039 —— swap-list forward/reverse comm 的原始形式（`maxneed`、`sendlist`、`firstrecv`、`sendself`）。
  - A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022), DOI 10.1016/j.cpc.2021.108171 —— 现代 LAMMPS 的 borders/forward/reverse 语义（本仓 `halo_participant.h:25` 已引用同一条）。
  - A. Rahman, Phys. Rev. **136**, A405 (1964), DOI 10.1103/PhysRev.136.A405 —— 回归示例采用的 LJ 参数（ε/k_B = 120 K，σ = 3.4 Å，r_c = 2.5σ）。

## Design

### 新类型：`ATV::cpu::comm::Swap` / `SwapList`

`include/atomiverse/cpu/communication/swap_list.h` + `src/cpu/communication/swap_list.cpp`。术语用 LAMMPS 原名 snake_case。

```
struct Swap {
    int  dim;                 // 0/1/2
    int  sign;                // -1 = lo 面, +1 = hi 面
    int  sendproc, recvproc;  // Cartesian 邻居 rank
    bool sendself;            // sendproc == my_rank —— 见下方硬约束
    int  pbc[3];              // 整数镜像标志，打包时 × box.length(dim)
    double slablo, slabhi;    // borders 资格带 [Å]（±BIG 哨兵）
    std::vector<int> sendlist;
    int  nrecv, firstrecv;    // 本 swap 收到的 ghost 数 / 首槽位
};
```

`SwapList` 是 **纯本地** 数据结构（不含任何 MPI 调用，不依赖 `ATV_HAS_MPI`），三个原语：

- `setup(dd, box, halo_cutoff, my_rank)` —— 派生 `maxneed(d)`、展开 `nswap = 2·Σ maxneed` 个有序 swap（顺序 `(0,lo),(0,hi),(1,lo),(1,hi),(2,lo),(2,hi)`，与现有 `face_neighbors` 面序逐位一致），填 `sendproc/recvproc/sendself/pbc/slablo/slabhi`。`need_d > 1` → `throw std::runtime_error`，消息复用现有守卫的 how-to（换更大盒 / 更少 rank / 更小 cutoff）。**每次全量重建都调用**（LAMMPS 在每个 reneighbor 步调 `comm->setup()`），因此 NPT/deform 缩盒后 `slablo/slabhi` 与守卫自动跟随。
- `rebuild_sendlist(iswap, frame, sources_end)` —— 按 `slablo/slabhi` 带扫描 `[0, sources_end)`，重填该 swap 的 `sendlist`（容量保留）。级联性质由调用方给出的 `sources_end` 表达（axis 0 扫 owners，axis 1 扫 owners+axis0 ghosts，…），与今天完全相同。
- `commit_recv(iswap, nrecv, next_slot) -> int` —— 写回 `nrecv`、`firstrecv = next_slot`，返回 `next_slot + nrecv`。前缀和成为类型不变量，取代 `HaloRecv` 里手算的 `slot_start[6]`。
  另有 `size()`、`maxneed(int d)`、`operator[]`、`total_recv()`、`invalidate()`（清空各 swap 的 sendlist/nrecv/firstrecv、保留 swap 几何——atom sort/迁移后的失效语义）。不提供 `memory_bytes()`（无树内调用点；需要时随 `HaloSend::memory_bytes()` 覆盖一并加）。

**通信留在 Component 侧**：count 握手、`Isend/Irecv`、pack/unpack 仍在 `HaloSend`/`HaloRecv`/`HaloRefresh` 内——`SwapList` 之于 halo，正如 `CellList`/`PairList` 之于 `NeighborList`。用户请求的四件套映射为：`setup()` = `SwapList::setup`；`borders()` = `HaloSend` 全量重建路径（驱动 `rebuild_sendlist` + count 握手 + border 打包 + `commit_recv`，末尾 `++dd->halo_generation`）；`forward_comm()` = `HaloSend` 步内刷新路径；`reverse_comm()` = `HaloRecv::launch`。

### 所有权与发布

`SwapList` 由 `HaloSend` **拥有**（私有成员 `swaps_`），以非拥有 `const comm::SwapList* swap_list` 发布到 `DynamicsServices`（`include/atomiverse/cpu/component/context.h:52`），完全照 `NeighborList → ctx.md.pair_list`（`context.h:64`，CLAUDE.md § Frame 点名认可的形状）。**发布时机与空指针契约**（architect 审定）：`HaloSend` 在 `build()` **和** `launch()` 都发布（镜像 `neighbor_list.h:133/:234`）；消费者（`HaloRecv`/`HaloRefresh`）只在 `launch()` 读取，`swap_list == nullptr ⇒ 统一的集合式 no-op`（各 rank 一致判空，不可能死锁）——注意 `HaloRefresh{positions}` 绑定在 `post_integrate`，位于 `pre_neighbor` **之前**，step 0 的 nullptr 读取是必然路径，不是假设。`HaloSend::destroy()` 将发布指针置空（为 component-plugin-05/06 的运行时 unfix 预付，避免悬垂）。单 rank 下 `HaloSend` 提前返回、指针保持 `nullptr`。`domain.h:141` 的 `send_lists` 与 `:148` 的 `recv_counts` 从 DD **删除**（热路径结构不持派生索引；且它们此前迫使每次 comm 形状变化都重编整个含 `domain.h` 的世界）。无 Config 袋、无工厂：`HaloSend(double cutoff, int every = 10)` 签名不变，`maxneed` 在 `setup()` 内派生。错误一律 `throw std::runtime_error` 带 how-to。缓冲区 PIMPL 沿用 `detail::FaceBufs` 形状，但由固定 `array<...,6>` 泛化为按 `nswap` 尺寸的 per-swap 向量（`nswap` 是派生量，不是常数 6）。

### 三条必须逐位保住的行为（等价性的全部风险都在这里）

1. **`sendself` 是「跳过」，不是 LAMMPS 的本地 memcpy。** 当 `dims[d] == 1` 时 `MPI_Cart_shift` 给出 `neighbor == my_rank`，今天的代码跳过收发（`halo_exchange.cpp:588/628/1160`），该轴不产生任何 ghost，周期性由 pair kernel 的 min-image 承担。`src/cpu/neighbor/cell_list.cpp:55` 把 `axis_extended_[d]` 定义为 `dims[d] > 1`，并且 **只对非 extended 轴做 min-image**（`:620-632`、`:777`、`:953`）；若按 LAMMPS 引入 self-image ghost，这些 ghost 会落在仍被 min-image 的轴上，直接双计——即 `tests/mpi/test_halo_dual_image_ghosts.cpp` 记录的那类静默 16% 能量误差。故 `Swap::sendself == true` ⇒ 该 swap 不发不收、`nrecv = 0`，并在头文件注释里写明「与 LAMMPS 的偏离及其理由」，留给 -02 复议。
2. **dims==2 的死锁与双镜像。** 该情形下一维的 lo/hi 邻居是 **同一个别的 rank**：顺序 per-swap `Sendrecv` 会交叉 tag 并死锁（`halo_exchange.cpp:1127-1131` 记录的真实事故）。因此 swap 顺序只是 **表的属性**；**执行按 `dim` 分组并发**（同轴两个 swap 一起 `Isend/Irecv` + `Waitall`，tag 仍由 swap 序号派生，等价于今天的 `100 + face`）。同时，同一原子经两个 swap 到达同一 rank、形成一个 tag 的两份周期镜像，是 **正确的现行行为**——严禁引入任何按 tag 去重。
3. **槽位布局与资格判据逐位不变。** ghost 槽位按 swap 序追加，`firstrecv[i] = n_local + Σ_{j<i} nrecv[j]`，与今天的 `slot_start[f]` 完全同式；`HaloRecv` 的 `MPI_Allreduce(MIN)` 布局守卫（`:1236-1243`）保留，改判 `total_recv() == n_ghost`。资格判据保持 **半开** 比较（今天是 `q − lo < cutoff` / `hi − q < cutoff`，严格小于），不得改成 LAMMPS 的 `<=`，否则边界上的 ghost 集合会漂。`pbc` 存整数标志、打包时乘 **当前** `frame.box.length(dim)`（NPT/deform 会改 `L_d`，setup 时把米制位移算死会发陈旧位移——这是对「setup 时算好存表」的一处刻意收紧：被消灭的是每面每步的 `coords/dims` 边界推导，不是那一次乘法）。

### 保持不动的接缝

`HaloParticipant` 协议（`halo_participant.h:37`）、`kColComm/kColBorder/kColReverse` 选列（swap 只改消息条数，不改每条消息的形状）、`require_collective_halo_schema` 指纹校验与各路 discriminator（`:339`、`:1108-1118`）、`zero_ghost_forces`（`:834`）、bonded skin 加宽与 `MPI_Allreduce(MAX)`（`:478-485`）、`HaloRefresh::Field` 三值枚举与 `md.h:449/452/467` 三处自动注入的 phase/order、`Migrate` 的 `ghost_tags` 回滚纪律（`atom_migration.cpp:350`）。

### 层次触达声明

设计面只在 `cpu/communication` 一个 pillar 内。`cpu/component/context.h`（新增一个已发布指针字段）与 `cpu/dynamics/atom_sort.h`（`notify_sorted` 增加一个必填 `comm::SwapList*` 形参，使每个调用点被迫表态而不是静默漏掉失效）是字段搬家强制的 **单行接缝**，不构成第二个架构层；把它们切成独立子 spec 会产出无法单独编译的中间态。**const 不对称的裁定**（architect 审定）：`notify_sorted` 需要非 const 指针（要调 `invalidate()`）；`halo_exchange.cpp` 的调用点传 `&swaps_`（拥有者自己的非 const 成员），`neighbor_list.h:217` 的调用点传字面 `nullptr`（该路径为串行、无 swap 表）——**全仓不得出现 `const_cast`**，也不得增加第二个非 const 发布。

### Reuse decision（逐条解决 librarian 报告）

- **reuse** `pack_comm` / `unpack_comm` / `pack_border` / `unpack_border`（`atom_pack.h:62/65/68/71`）—— 已是 LAMMPS 形状（index list + 三元位移 + `firstrecv`），直接作为每 swap 的打包原语调用，签名不动。
- **reuse** `pack_extra_fields` / `unpack_extra_fields`（`halo_exchange.cpp:368/384`）—— `HaloParticipant::CommField` 附载仍按「行表 + first」驱动，改的只是行表来源。
- **reuse** `require_collective_halo_schema` / `halo_payload_fingerprint`（`:339`/`:318`）、`zero_ghost_forces`（`:834`）、`compute_bonded_extension` + `Allreduce(MAX)`（`:478-485`）—— 原样调用，后者的结果成为 `setup()` 的 `halo_cutoff` 入参。
- **generalize** `dd->send_lists`（`domain.h:141`）+ `dd->recv_counts`（`:148`）→ 收进 `SwapList`，以 `ctx.md.swap_list` 发布（不新建平行结构）。
- **generalize** `rebuild_send_lists_for_axis`（`halo_exchange.cpp:280`）→ `SwapList::rebuild_sendlist`：单轴双面改为单 swap，资格判据由「单 cutoff 带」改为 `slablo/slabhi` 带（数值同集合）。
- **generalize** `forward_axis_stage`（`:557`）→ `HaloSend` 的 borders/forward 表驱动主体，保留其全部实质（count 交换、pack、`Isend/Irecv`、容量增长 + `set_capacity` 后重取 span、unpack + tag 注册）。
- **generalize** `HaloRecv` 的 `slot_start` 前缀和（`:1229`）→ `Swap::firstrecv` + `commit_recv` 不变量，布局守卫保留。
- **generalize** `detail::FaceBufs`（`:56` 声明的 PIMPL）→ 按 `nswap` 尺寸的 per-swap 缓冲，仍是 per-instance PIMPL，`send_buf_capacity_for_test()` 语义改为 per-swap 容量并继续可用。
- **delete, no new symbol** `pbc_shift_for_face`（`:259`）—— 其判断被 `setup()` 的 `pbc[3]` 表取代，不做泛化，直接删除。
- **new — `Swap` / `SwapList`**：仓内无任何既有类型持有「有序 swap + 每 swap 索引表 + 收包槽位」这一结构；`PairList` 是原子对 CSR，`DomainDecomposition` 是所有权几何，二者都不匹配。新类型的命名、构造与错误处理照 `NeighborList` + `CellList` 的 setup/rebuild-on-cadence/step-apply 三分与 `mode_stale`/`size_stale`/`halo_stale` 具名布尔惯例（`neighbor_list.h:202`），使新代码读起来像既有代码。

## Files to create or modify

- `include/atomiverse/cpu/communication/swap_list.h` (new)
- `src/cpu/communication/swap_list.cpp` (new)
- `include/atomiverse/cpu/communication/halo_exchange.h`
- `src/cpu/communication/halo_exchange.cpp`
- `include/atomiverse/cpu/communication/domain.h`
- `include/atomiverse/cpu/component/context.h`
- `include/atomiverse/cpu/dynamics/atom_sort.h`
- `include/atomiverse/cpu/neighbor/neighbor_list.h`
- `CMakeLists.txt`
- `tests/cpu/communication/test_swap_list.cpp` (new)
- `tests/CMakeLists.txt`
- `tests/mpi/test_halo_exchange.cpp`
- `tests/cpu/dynamics/test_domain_decomposition.cpp`
- `tests/cpu/dynamics/test_atom_sort.cpp`
- `regressions/md/comm-swaplist-halo-mpi/reg_comm-swaplist-halo-mpi.cpp` (new)
- `regressions/md/comm-swaplist-halo-mpi/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`
- `docs/developer/kspace-long-range.md`

## Tasks

- [x] Write failing unit tests for SwapList (`tests/cpu/communication/test_swap_list.cpp` → `TestSwapList`; register `test_cpu_comm_swap_list` in `tests/CMakeLists.txt`): swap 展开顺序/`nswap`/`maxneed`、`pbc[3]` 镜像标志、`sendself` 判定、`slablo/slabhi` 半开带、`rebuild_sendlist` 索引集合、`commit_recv` 前缀和、`need_d > 1` 抛错
- [x] Implement `Swap` + `SwapList` in `include/atomiverse/cpu/communication/swap_list.h` and `src/cpu/communication/swap_list.cpp` (setup / rebuild_sendlist / commit_recv / total_recv / maxneed / invalidate，无 MPI 依赖) and add the source to `CMakeLists.txt`
- [x] Generalize `dd->send_lists` + `dd->recv_counts` into `SwapList` ownership: 从 `include/atomiverse/cpu/communication/domain.h` 删除两个字段，在 `include/atomiverse/cpu/component/context.h` 发布 `const comm::SwapList* swap_list`，给 `include/atomiverse/cpu/dynamics/atom_sort.h` 的 `notify_sorted` 加必填 `comm::SwapList*` 形参并更新 `include/atomiverse/cpu/neighbor/neighbor_list.h` 的调用点
- [x] Generalize `rebuild_send_lists_for_axis` + `forward_axis_stage` + `detail::FaceBufs` into 表驱动的 `HaloSend` borders/forward in `src/cpu/communication/halo_exchange.cpp` and `include/atomiverse/cpu/communication/halo_exchange.h`（`swaps_` 成员 + 每次全量重建调 `setup`/`borders` 并 bump `halo_generation`；按 `dim` 分组并发收发；删除 `pbc_shift_for_face` 与旧级联，无 fallback 开关）
- [x] Generalize `HaloRecv` 的 `slot_start` 前缀和 into `Swap::firstrecv` and reverse over the reversed swap table in `src/cpu/communication/halo_exchange.cpp`（保留 `MPI_Allreduce(MIN)` 布局守卫，改判 `total_recv() == n_ghost`）
- [x] Rewrite `HaloRefresh::launch` in `src/cpu/communication/halo_exchange.cpp` onto the published swap table（`Field` 枚举、幂等契约、positions 加位移 / velocities+charges 不加，全部不变）
- [x] Update existing consumers to the new accessors in `tests/mpi/test_halo_exchange.cpp`, `tests/cpu/dynamics/test_domain_decomposition.cpp`, `tests/cpu/dynamics/test_atom_sort.cpp`
- [ ] Add Doxygen `@brief`/`@param`(with units)/`@return` for every public symbol in `include/atomiverse/cpu/communication/swap_list.h`, and sync the stale prose in `include/atomiverse/cpu/communication/halo_exchange.h`, `include/atomiverse/cpu/communication/domain.h`, `docs/developer/kspace-long-range.md`
- [ ] Add regression example `regressions/md/comm-swaplist-halo-mpi/reg_comm-swaplist-halo-mpi.cpp` + `REFERENCE.md` and register it in `regressions/CMakeLists.txt`（public API only；就地解析 O(N²) min-image 判据，运行时不触第三方）
- [ ] Run full check + test suite（含 `ctest -L mpi` 的 n=1/2/8 全部用例）

## Testing strategy

**单元测试（`tests/cpu/communication/test_swap_list.cpp`，镜像 `src/cpu/communication/swap_list.cpp`，套件 `TestSwapList`，每例只打一个方法，无 mpirun）**——手工构造 `DomainDecomposition`（`dims`/`coords`/`periods`/`sublo`/`subhi`/`face_neighbors`）与 `Box`，直接断言：

- happy path：`dims = {2,1,1}`、box 20×10×16 Å、`halo_cutoff = 3.0` Å ⇒ `nswap == 6`；`maxneed(d) == 1`（d = 0,1,2）；swap 顺序为 `(0,lo),(0,hi),(1,lo),(1,hi),(2,lo),(2,hi)`。
- `pbc[3]`：`coords[0] == 0` 的 lo swap 得 `{+1,0,0}`，`coords[0] == dims[0]-1` 的 hi swap 得 `{-1,0,0}`，其余全 0（硬编码期望值）。
- `sendself`：`dims[1] == dims[2] == 1` ⇒ swap 2..5 的 `sendself == true`；断言这类 swap 的 `nrecv` 保持 0。
- `slablo/slabhi` 半开带：`rebuild_sendlist` 在一个手排的一维原子链上给出的索引集合，与「`q − sublo < cutoff` / `subhi − q < cutoff`」逐元素相同（硬编码索引数组）；`sources_end` 增大时新增的 ghost 行被正确纳入（级联语义）。
- `commit_recv` 前缀和：给定 `nrecv = {3,2,0,0,0,0}`、`n_local = 10` ⇒ `firstrecv == {10,13,15,15,15,15}`，`total_recv() == 5`（硬编码）。
- edge/throw：`halo_cutoff` 大于 `L_d/dims_d` ⇒ `EXPECT_THROW(std::runtime_error)` 且消息含 how-to；`dims = {1,1,1}` ⇒ 全部 swap `sendself`，`total_recv() == 0`。

**既有多 rank 用例是本 spec 的等价性核心**（继承债声明：`tests/mpi/` 中若干用例本身以 `cpu::md::MD::run()` 驱动、违反 iron law 1——这是**被依赖而非被扩张**的既有债，本 spec 不在 `tests/` 下新增任何 driver `run()`；迁移这些 e2e 到 `regressions/` 属独立 slug）：`tests/mpi/` 全部 20 个用例（label `mpi`，n = 1/2/8）必须零改判地通过，特别是 `test_mpi_halo_exchange`（forward/reverse 对称性、PBC ghost 落点、`HaloParticipant` 附载往返）、`test_mpi_halo_dual_image_ghosts`（dims==2 双镜像不得被去重）、`test_mpi_lj_energy_bruteforce` / `test_mpi_coul_energy_bruteforce` / `test_mpi_bonded_energy_bruteforce`（对 O(N²) 暴力解）、`test_mpi_halo_refresh_charges` 与 `tests/cpu/communication/test_halo_refresh_charges.cpp`（HaloRefresh 接缝）。其中读 `dd->send_lists`/`dd->recv_counts` 的断言改读发布的 swap 表，判据不变。

**回归示例（本 spec 唯一的 e2e）**：`regressions/md/comm-swaplist-halo-mpi/`。仅用公开 API：`cpu::Frame` + `PairLJCut` + `NeighborList` + `VelocityVerlet` + `cpu::md::MD`（halo 由 `install_defaults` 自动注入）。系统：8³ = 512 原子简单立方晶格，`a0 = 3.617 Å`（Rahman 数密度 0.02071 Å⁻³），立方盒 28.94 Å，`ε = 120 k_B = 0.010341 eV`、`σ = 3.4 Å`、`r_c = 2.5σ = 8.5 Å`；一个近 rank 边界的原子预位移 0.35 Å 打破对称性；`mpiexec -n 2` 自重启（照 `regressions/CMakeLists.txt:179-182` 的 `ATV_MPIEXEC` 形状）⇒ `dims = {2,1,1}`，子域 14.47 Å > halo 9.35 Å（`need = 1`），且 `2×9.35 > 14.47` 使双镜像路径被真实走到。跑 21 步（`every = 10`，跨 ≥2 次全量重建，既走 borders 也走步内 forward），末步把位置 allgather 到 rank 0，就地做 O(N²) min-image LJ 求和作为解析判据（无第三方、无引擎快照）。硬编码门：`|E_dd − E_bruteforce| / N ≤ 1e-9 eV/atom`；`max_i |F_i,dd − F_i,bruteforce| ≤ 1e-9 eV/Å`；`|Σ_i F_i| ≤ 1e-10 eV/Å`；`nswap == 6` 且 `Σ nrecv == n_ghost > 0`；重建次数 ≥ 2。`REFERENCE.md` 记 `oracle: literature`（判据是「域分解是同一求和的划分」这一恒等式 + Rahman 模型参数），引 Plimpton 1995 / Thompson 2022 / Rahman 1964，并声明 `perf_exempt`——本 slug 不做任何性能声明。运行时间 ≤ 60 s。

## Out of scope

- **`maxneed > 1` 多跳**（子域比 ghost 壳薄时的多跳 swap 链、`sendself` 本地 memcpy 语义的复议）—— 归 `-02`；本期在 `setup()` 里以 throw 明确挡住，不做静默截断。
- **256 rank 级验证与强扩展曲线** —— 归 `-03`；本期不提交任何 wall-clock 数字、不设 `performance` 判据（结构换完之前的加速比不可信）。
- **OpenMP 混合并行、kspace 子通信域、Kokkos 后端** —— 本期不碰。
- **`Migrate` / `Alltoallv` 交换路径、RCB 负载均衡、`atom_sort` 的 binning 策略** —— 只跟随字段搬家做最小改动，算法不动。
- **CUDA 侧 `cuda/communication/`** —— CPU 与 CUDA 的 halo 是两套引擎，本期只动 CPU。
- 已考虑并否决的替代：把 `send_lists` 长成 `vector<Swap>` 留在 `domain.h`（违反热路径数据铁律，且 `domain.h` 是宽重编面）；新建 `cpu/communication/halo/` 子目录（违反「后端根下只放 pillar 目录」）；保留旧级联作为 `#ifdef` fallback（本仓无兼容层铁律）。
