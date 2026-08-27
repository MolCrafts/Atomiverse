---
title: Split the DomainDecomposition grab-bag
status: approved
created: 2026-08-27
chain: comm-maint
---

# Split the DomainDecomposition grab-bag

## Summary

`DomainDecomposition` 是一个 296 行头、22 个公开字段、**无 `private:`、无访问器**的 struct，以裸指针 `ctx.md.dd` 发给全树，被目录外 67 个文件 include。它把 LAMMPS 明确分开的三个类合并成一个：`Domain`（`sublo/subhi`、盒、周期性）、`Comm`（`procgrid/myloc/procneigh`）、`Atom`（tag→row 映射、排序调度）。本步做**最小可验证的切口**而不是一次性大拆：把 6 个 `atom_sort_*` 字段从 DD 上移走并改用真实步数，处置生产中从无写者的 `periods`，并为其余字段确立"谁写"的边界。约束 (a)：不得把一个大 bag 换成三个小 bag。

## Domain basis

不声明新物理。排序节奏与周期性标志都不是物理量；判据是行为等价（同一 seed 下轨迹逐位相同）加上"排序时钟按步推进而非按力评估次数推进"。

## Design

**对照 LAMMPS**（checkout `78a7ddf`）：`Domain` 拥有 `boxlo/boxhi`、`xprd`、`sublo/subhi`、`periodicity`、`pbc()`、`remap()`（`domain.h:35-91`）；`Comm` 只拥有 `procgrid[3]`、`myloc[3]`、`procneigh[3][2]`、splits（`comm.h:53-58`），且 `comm.h`/`comm_brick.h` 里 grep `sublo|subhi|boxlo|xprd|periodicity` **零命中**；`Atom` 拥有 `map_style`/`map_array` 与 `sortfreq`/`nextsort`（`atom.h:291,297-301`）。依赖方向是 Domain → Comm（`domain.cpp:345-374` 的 `set_local_box` 读 `comm->xsplit`），不是反向。

**切口一：`atom_sort_*`（6 个字段）。**

现状：`NeighborList::build()` 在 `neighbor_list.h:169-174` 一次性把 6 个值写进 DD（只在 `build()` 里，`if (is_multi_rank(...))` 内，此后不再发布）；`CommBrick::borders` 在 `comm_brick.cpp:1092-1103` 读它们并回写推进调度。这 6 个字段**目录外零读者**——它们是 `NeighborList → CommBrick` 的单向信道，却借一个全树可见的公开 struct 传递。

更糟的是步数是手搓的：`++dd->atom_sort_step` 出现在 `borders`（`:1075`）**和** `forward_comm`（`:1139`）两处，靠"每步只走其中一个"成立。已知反例：Minimize 的 line-search 经 `run_stage → comm_before_force_ → forward_comm` 也会自增（`exchange` 有 `line_search_trial` 挡板，`forward_comm` 没有），于是排序时钟按力评估次数而不是按步推进。

**做法：这个类型已经存在，不用发明。** `include/atomiverse/cpu/dynamics/atom_sort.h:56` 的 `md::AtomSortState` 正好就是这 6 个字段加三个谓词——`resolved_binsize()`（`:66`）、`due()`（`:74`）、`mark_sorted()`（`:80`）、`configure()`（`:85`）。`NeighborList` 已经**拥有**一个 `sort_state_`；`neighbor_list.h:169-174` 是把它手工摊平进 DD，`comm_brick.cpp:1092-1103` 是把三个谓词手工重实现一遍。两边一起删掉，改为按 `ctx.md.pair_list`（`context.h:66`）的先例，在 `DynamicsServices` 上发布一个非拥有指针 `md::AtomSortState* atom_sort`，由 `NeighborList::build()` 发布。裸指针保持 `context.h:284` 的 `static_assert(std::is_trivially_copyable_v<DynamicsServices>)` 成立。DD 上的 6 个字段删除。

**约束 (a) 的一处直接后果**：`AtomSortParams`（`atom_sort.h:41`）本身就是一个参数包，而且在公开面上。它经 `configure(AtomSortParams)` 传入而非构造函数，按 CLAUDE.md 的字面读法勉强合规；按操作者的平铺约束则应改为 `configure(int nfreq, double binsize, bool enabled)` 并删掉 `AtomSortParams`。**本 spec 采后者**——既然在拆 bag，就不要在同一步里保留一个。

**时钟的选择是一个行为改变，不是纯重构——必须明说。** `LaunchContext` 上的字段名是 `step`（`component.h:52`），由 `Runtime` 从 `step_idx` 填入（`runtime.cpp:338`、`:446`）。手搓计数器在 `borders`（`:1075`）和 `forward_comm`（`:1139`）各自 `++`，即**每次通信机会**加一。Verlet 下每步恰好触发其一，所以计数器 == 步数；但 `step_linear_`（Evaluate/Minimize）下 `comm_before_force_` 每次**力评估**触发一次，于是 L-BFGS 线搜索期间计数器跑在 `step_idx` 前面。改用 `ctx.step` 会**改变 Minimize 下的排序节奏**——这正是要修的东西，但它意味着本 spec 的判据不能是"轨迹逐位相同"，只有 MD 路径可以这样要求。另注意 `setup_verlet_comm_` 把 step 硬编码为 `0`（`runtime.cpp:375`）。

**切口二：`periods`。**

`init_topology`（`domain.cpp:115-164`）**从不给 `periods` 赋值**，其 `rank_at` lambda（`:149-155`）三轴一律周期性回绕。生产中该字段永远是类内默认 `{1,1,1}`，唯一生产读点是 `halo.cpp:191`。于是 `Halo::derive_hops` 里 `reach_minus = periodic ? dims-1 : coords[d]` 的非周期分支在生产中不可达——不是死代码那么简单，而是**声称支持非周期分解但实际不支持**。

**做法**：二选一，必须选一个。(a) 在 `init_topology` 里真正从 `Box`/调用者接入周期性并让 `rank_at` 遵守它，同时为非周期轴补测试；或 (b) 删除 `periods` 与 `Halo` 里所有非周期分支，并在 `init_topology` 的文档里写明"仅支持全周期"。**倾向 (b)**——没有需求驱动 (a)，而 (a) 会引入一条无人使用、无法验证的路径，正是本链在清理的东西。若选 (b)，必须在 CLAUDE.md 或 `domain.h` 里显式记下这个限制。

**其余字段：确立边界，不搬家。** 本步**不**动 `sublo/subhi`、`tag_to_local`、生成计数器，也**不**动 `ghost_shell_width`（`domain.h:146`）——后者是真正的分解几何，且有**库外消费者**：`src/cpu/neighbor/cell_list.cpp:79` 用它给网格加边，三个 regression 直接读它再喂给 `Halo::derive_uniform_rounds`（`reg_comm-swaplist-halo-mpi.cpp:221`、`reg_comm-topology-multihop.cpp:223`、`reg_multihop-widebox-parity.cpp:436`）。regression 是走公开头的库外消费者，动它就是公开面破坏。但要把已知的越界写者记录成注释契约：目录外只有两个生产写者——`neighbor_list.h:169-174`（本步消除）与 `atom_sort.h:140-154` 的 `notify_sorted`（写 `tag_to_local`/`ghost_tags`/`n_ghost`/两个 generation）。后者是合法的：排序确实使这些失效。把它写进 `domain.h` 的类注释，让下一个写者知道边界在哪。

**下一步（不在本 spec）**：真正把 DD 按 LAMMPS 的三分法拆开。本 spec 交付后应重新评估——`atom_sort_*` 与 `periods` 拿掉后，剩下的 14 个字段是否还构成一个 bag，可能已经不构成。

## Files

- `include/atomiverse/cpu/communication/domain.h` — 删 6 个 `atom_sort_*`；处置 `periods`；类注释写明合法写者
- `src/cpu/communication/domain.cpp` — `init_topology` 依 (a)/(b) 调整
- `include/atomiverse/cpu/neighbor/neighbor_list.h` — 删除向 DD 的发布；提供平铺的排序查询
- `include/atomiverse/cpu/component/neighbor_protocol.h` — `NeighborProvider` 加查询（若走协议路线）
- `src/cpu/communication/comm_brick.cpp` — `borders` 改用 `step_idx` + 查询；删两处 `++dd->atom_sort_step`
- `src/cpu/communication/halo.cpp` — 依 (b) 删非周期分支
- `tests/cpu/communication/`、`tests/cpu/dynamics/` — 排序节奏与 line-search 不推进时钟的测试

## Tasks

- [ ] Publish the existing md::AtomSortState as a non-owning pointer on DynamicsServices, owned by NeighborList; do not invent a new type
- [ ] Delete the six atom_sort_* fields from DomainDecomposition and the hand-flattening block in neighbor_list.h
- [ ] Replace the hand-reimplemented due/resolved_binsize/mark_sorted in comm_brick.cpp with calls on the published state
- [ ] Flatten AtomSortState::configure to (nfreq, binsize, enabled) and delete AtomSortParams
- [ ] Replace both ++dd->atom_sort_step sites with LaunchContext::step, and state in the spec which clock Minimize is now on
- [ ] Add a test that a Minimize line-search force re-evaluation does NOT advance the sort clock
- [ ] Add a test that the sort fires on the requested step cadence under plain MD
- [ ] Decide and execute the periods disposition: wire it through init_topology with tests, or delete it and the unreachable non-periodic branches and document the all-periodic limitation
- [ ] Record the legal out-of-directory writers of DomainDecomposition in its class comment

## Testing

**MD 路径**行为等价：同一输入下轨迹逐位相同，`ctest` 全绿，`widebox_parity.sh` dev=0.0（8,8,4 与 16,4,4）。**Minimize 路径的排序节奏会改变**（这是修复目标，不是回归）——判据改为"节奏等于步数而不是力评估次数"，并记录改动前后的排序次数。新增两个针对性测试：line-search 不推进排序时钟；MD 下排序按 `nfreq` 触发。若走 (a)，非周期轴需要一个多 rank 测试；若走 (b)，需要一个"非周期请求会抛或被文档明确拒绝"的测试。

## Out of scope

- 不把 DD 拆成三个新结构体——那正是约束 (a) 禁止的形状。
- 不动 `sublo/subhi`、`tag_to_local`、生成计数器的归属。
- 不动 `cuts` / `Rebalance`。
