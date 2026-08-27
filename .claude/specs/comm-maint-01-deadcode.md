---
title: Comm layer dead code and stale names
status: approved
created: 2026-08-27
chain: comm-maint
---

# Comm layer dead code and stale names

## Summary

清除 CPU 通信层已确认为死的代码与已失效的名字，并装上让它们不再悄悄堆积的护栏。本步删除 `atom_migration.{h,cpp}`（306 行，全树零调用者）、`comm_brick.cpp` 内两个零调用点的 `pack_atom`/`unpack_atom`、零实现者的 `HaloParticipant` 与 `AtomExchange` 协议及其 `Component` 访问器、零调用者的 `CommBrick::send_buf_capacity_for_test()`、零消费者的 `kColWriteBack` 标志位；把注释与文档里指向不存在类型（`HaloSend` / `Migrate` / `HaloRecv` / `SwapList` / `halo_exchange`）的引用改写成现存类型；并为 `atomiverse` 目标打开 `-Wall -Wunused-function`——当前构建**一个 `-W` 标志都没有**，这正是上述死代码能静默存活的机制。本步不改任何运行时行为。

## Domain basis

本 spec 不声明新物理，不引入任何物理量或文献。被删符号全部经 grep 与 `nm -C build/libatomiverse.a` 双重确认无引用；被改写的仅为注释、文档与构建标志。判据是"链接产物的行为字节不变"，不是物理窗口。

## Design

**这不是"顺手清理"，而是一条有证据的清单。** 每一项都在本 spec 里给出确认方式，实现时必须逐项复核而不是照抄。

**实体与符号（删除）**

- `src/cpu/communication/atom_migration.cpp` + `include/atomiverse/cpu/communication/atom_migration.h` —— 整文件删除。唯一符号 `scatter_atoms_from_root` 零调用者；`atom_migration.h` 在全树的唯一 `#include` 是它自己的 `.cpp`。最后一个调用点随 `benchmarks/md/pe/performance/polyethylene_tension.cpp` 于 `2613daf2`（2026-07-17）删除。同步删 `CMakeLists.txt:302` 的源文件条目。`bcast_frame.h:28,31` 与 `task/task_protocol.h:43` 三处把它当"先例"引用的注释改写为 `bcast_frame` 自身或 `Comm::exchange`。
- `src/cpu/communication/comm_brick.cpp` 的 `pack_atom` / `unpack_atom` —— 本翻译单元内零调用点（`exchange_along_dim` 走 `pack_to` 和 `comm::unpack_exchange`）。它们是 `atom_migration.cpp` 同名函数的逐字节副本，随重命名一起被复制进来。连同 `:207` 那条已失效的注释（"atom_migration's `MPI_Alltoallv` needs uniform per-atom strides"）一并删除。
- `include/atomiverse/cpu/component/halo_participant.h` 整文件 + `Component::halo_participant()`（`component.h:93`）—— 零实现者、零生产调用点。全树唯一调用是 `tests/cpu/test_install_infra_hook.cpp:139` 断言它返回 `nullptr`；该断言随协议一起删除。头注释里描述的"driver infrastructure scan"并不存在。
- `include/atomiverse/cpu/component/atom_exchange.h` 整文件 + `Component::atom_exchange()`（`component.h:95`）—— 零实现者、零调用点；其头注释自己写明迁移由 `ctx.md.comm->exchange()` 承担。
- **CUDA 侧的孪生体一并删除**：`include/atomiverse/cuda/component/halo_participant.cuh` 的 `cuda::HaloParticipant`（零实现者）、其 `bind_participants`（`:57`，零调用者）、以及 `cuda/component/component.cuh:204` 的 `halo_participant()`（零覆盖）。只删 CPU 侧会让 CLAUDE.md 的 cpu/cuda 镜像律在另一个方向上破掉。
- `CommBrick::send_buf_capacity_for_test()`（`comm_brick.h:69`，`comm_brick.cpp:130`）—— 零调用者。
- `kColWriteBack`（`core/frame_schema.h:68`）—— 在 27 行字段表里被 11 行设置，却在 `src/`、`include/`、`tests/`、`python/`、CUDA 全无消费者，也无解释性注释。删除该常量并从所有字段行摘掉。**若实现时发现它其实承载某个未被 grep 到的意图**（例如预留给 `to_molrs` 写回），则改为保留 + 补一条说明它由谁消费的注释，二者择一，不得原样留下。

**实体与符号（改写，不删）**

- 指向不存在类型的注释：`CommBrick` 出现在 23 个文件，`CommBrick` 8 处，`Comm::exchange ` 3 处（`comm_brick.cpp:823`、`domain.h:183`、`rcb.h:19`），`Halo::` 在 6 个 regression `REFERENCE.md`，`halo.cpp:191` 被 `tests/mpi/test_halo_dual_image_ghosts.cpp:353` 与 `regressions/md/multihop-widebox-parity/REFERENCE.md:112` 引用为文件行号——该文件已删除。逐处改写为现存类型（`CommBrick` / `Halo` / `Comm::exchange`）。`domain.h` 的字段注释大量以 "Written by `CommBrick`" 解释语义（`:111-176`），改写为 `CommBrick`。
- `.claude/notes/architecture.md:82` 把 `cpu/communication` 描述为 `CommBrick`/`CommBrick::forward_comm`/`Comm::exchange`——全部不存在。`:48` 的协议表列着 `HaloParticipant`。这是 `librarian` 在 spec 阶段读的文件，错的地图会让下一个 spec 照着不存在的类规划。改写为真实清单。
- `CLAUDE.md`：MD phase 列表缺 `post_integrate` / `pre_exchange` / `pre_neighbor` / `neighbor`（真值 `core/stage.h:24-36`）；"Opt-in protocols" 一节列着本步删除的 `HaloParticipant`，并列着 `AsyncComponent`——该类型不存在，真实类型是 `asynchronous/async_component_base.h:53` 的 `AsyncComponentBase`（一个基类，不是协议）。三处同步。
- `.claude/notes/metalwalls-capability-map.md:118` 声称 `kAtom_charge` 带 `kColComm`；真实行是 `frame_schema.h:110-111` 的 `kColState | kColBorder | kColWriteBack`。改正。

**`ForwardPacker` 的形状（本步做决定并执行）**

现状 `class Component : public ForwardPacker`（`component.h:61`）给每个 CPU 组件的 vtable 挂三个空虚函数，与 CLAUDE.md "no dead default virtuals" 直接冲突；而被 CLAUDE.md 点名为标准形状的 `halo_participant()` 恰恰是本步要删的死协议。真实实现者只有三个：`ChargePacker`（`charge_packer.h:27`）、`FixSettle`（`fix_settle.h:58`）、`FixRattle`（`fix_rattle.h:98`）。

**决定：直接摘掉基类，且不加访问器。** `Component` 不再继承 `ForwardPacker`；`FixRattle` 与 `FixSettle` 各自把 `ForwardPacker` 加为第二基类；`ChargePacker` 本来就是独立实现（`charge_packer.h:27`），零改。`Comm::forward_comm(Packer&)` 签名不变。

**为什么不加 `as_forward_packer()` 访问器。** 与 `NeighborConsumer` 不同，**引擎从不扫描 packer**——调用是插件自己发起的（`fix_settle.cpp:413,420`、`fix_rattle.cpp:421`、`ChargePacker::forward` 经 7 个电荷求解器）。既然没有探测方，就不需要探测入口。于是这一步从 `Component` 摘掉 3 个虚函数、**新增 0 个**，是纯减法。这个不对称正是本项的全部论据。

**已知破坏面**：`tests/cpu/component/test_pack_forward_comm.cpp:102-106` 有 `static_assert(is_base_of_v<ForwardPacker, Component>)`，`:65,:69,:78,:85` 四个 TEST 在裸 `StubComponent` 上调用 packer 三件套——五处要改指到 `ChargePacker`/`FixSettle` 或删除。`tests/cpu/communication/test_comm.cpp:28-30`（`Packer == ForwardPacker` 别名）不受影响。

**护栏（本步的持久产出）**

`build-gaff/CMakeFiles/atomiverse.dir/flags.make` 当前是 `-O3 -DNDEBUG -std=gnu++20 -march=native -fopenmp`——**没有任何 `-W` 标志**，所以 `-Wunused-function` 从不触发，上述两个死函数才能编译无声。为 `atomiverse` 目标打开 `-Wall`。

**分级执行，不要一次性打开再淹没在告警里**：先只加 `-Wunused-function -Wunused-variable`（本 spec 的直接因），测量剩余告警数；若为零则同一步加 `-Wall`，再测量。`-Wall` 的告警若超过 50 条，则本步只落 `-Wunused-*`，把 `-Wall` 连同它的告警清单记进 INDEX 的 ticket 区，不在本 spec 里扩围。**不加 `-Werror`**——那是另一个决定，且会让任何编译器升级变成阻塞。

## Files

- 删除：`include/atomiverse/cpu/communication/atom_migration.h`、`src/cpu/communication/atom_migration.cpp`
- 删除：`include/atomiverse/cpu/component/halo_participant.h`、`include/atomiverse/cpu/component/atom_exchange.h`
- `CMakeLists.txt` — 摘掉 `atom_migration.cpp`；`atomiverse` 目标加警告标志
- `src/cpu/communication/comm_brick.cpp` — 删 `pack_atom`/`unpack_atom` + `:207` 注释、`send_buf_capacity_for_test`
- `include/atomiverse/cpu/communication/comm_brick.h` — 删 `send_buf_capacity_for_test` 声明
- `include/atomiverse/cpu/component/component.h` — 删两个访问器；`ForwardPacker` 继承改为 `as_forward_packer()`
- `include/atomiverse/cpu/dynamics/charge_packer.h`、`fix_settle.h`、`fix_rattle.h` — 各加 `as_forward_packer()` 覆盖
- `include/atomiverse/core/frame_schema.h` — `kColWriteBack` 处置
- `include/atomiverse/cpu/communication/domain.h`、`bcast_frame.h`、`cpu/task/task_protocol.h`、`rcb.h` 及其余 23 个含 `CommBrick` 的文件 — 注释改写
- `tests/cpu/test_install_infra_hook.cpp` — 删 `halo_participant()` 断言
- `CLAUDE.md`、`.claude/notes/architecture.md`、`.claude/notes/metalwalls-capability-map.md`

## Tasks

- [ ] Verify each deletion target still has zero references at implementation time (grep + `nm -C` on the built archive); record any that regained a caller and stop rather than delete
- [ ] Delete atom_migration.{h,cpp} and its CMakeLists.txt entry; rewrite the three comments that cite it as precedent
- [ ] Delete pack_atom/unpack_atom and the stale MPI_Alltoallv comment in comm_brick.cpp
- [ ] Delete send_buf_capacity_for_test from comm_brick.{h,cpp}
- [ ] Delete halo_participant.h, atom_exchange.h, both Component accessors, and the nullptr assertion in test_install_infra_hook.cpp
- [ ] Remove the ForwardPacker base from Component; add it as a second base on FixRattle and FixSettle only; add no accessor
- [ ] Retarget or delete the five sites in test_pack_forward_comm.cpp that assume every Component is a ForwardPacker
- [ ] Delete the CUDA HaloParticipant twin, bind_participants, and cuda Component::halo_participant()
- [ ] Resolve kColWriteBack: delete it from frame_schema.h and all 11 rows, or keep it with a comment naming its consumer
- [ ] Rewrite every comment referencing CommBrick / CommBrick / Migrate / Halo / halo_exchange to a type that exists
- [ ] Refresh .claude/notes/architecture.md for cpu/communication and cpu/component protocols
- [ ] Correct CLAUDE.md MD phase list, protocol list, and the AsyncComponent name
- [ ] Correct .claude/notes/metalwalls-capability-map.md:118
- [ ] Enable -Wunused-function -Wunused-variable on the atomiverse target; measure and report the remaining warning count; add -Wall in the same step only if that count is zero and -Wall stays under 50

## Testing

单元测试不变更覆盖面——本步不改行为。判据是 `ctest` 全绿（当前 237/237）加上"删除后仍能链接"。`test_install_infra_hook.cpp` 少一条断言。新增警告标志后必须完整重建一次以证明零新告警（或把告警数记进报告）。

## Out of scope

- 不动 `exchange_along_dim` 本身（属 `comm-maint-06-transport`）。
- 不动 `atom_pack.cpp` 的 fast/generic 分支（属 `comm-maint-03-genericpath`）。
- 不加 `-Werror`。
- 不重命名 CUDA 的 `DomainDecomposition`（属 `comm-maint-07-wire`）。
