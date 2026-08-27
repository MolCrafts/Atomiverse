---
title: One round-exchange primitive and one comm schedule
status: approved
created: 2026-08-27
chain: comm-maint
---

# One round-exchange primitive and one comm schedule

## Summary

`comm_brick.cpp` 里有**五个各自为政的手写 MPI 循环**——`forward_cascade`/`forward_round`、`forward_comm(ctx)`、`reverse_comm(ctx)`、`exchange_along_dim`、`forward_comm(Packer&)`——分属三套约定：前四个用 `halo_tag(kind, iswap)`（4096 stride）+ 预投递 `Irecv` + `Isend` + 每轮 `Waitall` + 常驻 `FaceBufs`；第五个用 **tag 0** + 阻塞 `MPI_Send` + **每次调用新建 `std::vector<double>`**，还带一个不可达的 self 分支。LAMMPS 的对照是：所有路径共用**一对** `buf_send`/`buf_recv` 和**恒为 0** 的 tag（`comm_brick.h:73-76`），缓冲区在 `borders()` 末尾按 `maxforward`/`maxreverse` 一次性定尺（`comm_brick.cpp:889-894`），此后整个重邻居区间不再分配。本步抽出一个平铺签名的 round-exchange 原语让五处实例化，统一 tag 与缓冲区，并顺带把写了三份的 Verlet 通信时序收成一份。

## Domain basis

不声明新物理。判据是线上字节与所有既有数值逐位相同，加上"每步的 MPI 调用数不增加"。

## Design

**原语的归宿是 `detail::FaceBufs` 的成员函数，不是自由函数。** 五个循环全在同一个 TU 里，而 `FaceBufs`（`comm_brick.cpp:73`）已经拥有原语需要的一切——`requests`、`send_pos`、`recv_pos`、`send_count`、`recv_count`。做成自由函数的话签名要摊到 `(MPI_Comm, sendproc[], recvproc[], send_tag, recv_tag, sbuf, rbuf, n, reqs, …)` 十来个参数，那个宽度必然招来一个 `struct RoundExchange`——正是约束 (a) 要挡的东西。做成成员函数是**平铺的出路**：缓冲区和 request 成为 `this`，每次调用只传 swap 区间、方向、tag kind 和两个可调用体。零新增公开面，也符合 CLAUDE.md "natural owning type → method on it"。

**不引入 `ExchangeConfig` 之类的参数包**——差异项直接作为参数传，多则拆成两个重载而不是塞进结构体。

**tag 统一。** 两处绕过 `halo_tag`：`exchange_along_dim`（`:304-326`）用裸字面量 `0,1,2,3`，`forward_comm(Packer&)`（`:1470`）用裸 `0`。给 `HaloTagKind`（`:449`）加 `exchange` 与 `packer` 两个 arm，两处改走 `halo_tag`。容量核对：`kHaloTagStride = 4096` × 5 kind = 20480，仍在 `:454` 引用的 MPI 32767 保证之内。`forward_comm(Packer&)` 的 tag 0 目前碰巧安全（严格串行 Irecv/Send/Wait，每个 peer 同时至多一条消息），但读者要自己推一遍才知道，而任何把它改成非阻塞的改动都会静默破坏。

**缓冲区统一。** `forward_comm(Packer&)` 每次调用 `sbuf.resize`/`rbuf.resize` 两个局部 `std::vector<double>`；它被 7 个电荷求解器经 `ChargePacker::forward` 间接调用，还有 `FixRattle`/`FixSettle` 各两处——这是每步多次的路径。改为共用 `FaceBufs`。同时采纳 LAMMPS 的定尺纪律：**在 `borders()` 末尾按最大 packer 宽度一次性定尺**，此后区间内不再分配。这需要一个 LAMMPS `Comm::init()` 式的"拉取"——遍历已挂载组件读出各自的 `comm_forward()` 取最大值。当前 ATV 没有这一步，所以 packer 路径的分配才落在每次调用里。

**`Comm::setup` 的契约。** `setup` 在生产中**从无外部调用者**——唯一调用点是 `CommBrick::borders()` 自己（`comm_brick.cpp:1069`）。而 `comm_brick.cpp:1088-1091` 抛出的错误信息叫用户"rebuild the communicator (run setup again)"，这是通过 `Comm` 接口做不到的事。二选一：让 orchestrator 在 box 变化时真正调用 `setup`（LAMMPS `Verlet::run` 就是这么做的，`verlet.cpp:279-287`），或把 `setup` 从公开接口撤下改为私有并改写那条错误信息。**倾向前者**——box 变化时重建 swap 表本来就该由循环驱动，而不是藏在 `borders` 里。

**时序去重（本 spec 的第二部分）。** exchange→borders→neighbor→force→reverse 的顺序在 `src/cpu/driver/runtime.cpp` 写了三份：`comm_before_force_`/`comm_after_force_`（`:332-365`，供 `step_linear_` 与 `run_stage`）、`setup_verlet_comm_`（`:367-390`）、`step_verlet_`（`:438-484`）。**经核实三者当前行为都正确**——它们服务不同的 phase 枚举（`md::Phase` 有 `pre_exchange`/`pre_neighbor`/`neighbor`，`evaluate::Phase`/`minimizer::Phase` 没有），`run_stage` 的包裹也被 `is_linear_force_slot_` 挡住。所以这是**复制风险，不是现存缺陷**，优先级相应靠后。做法：把 `comm_before_force_` 拆成两个私有成员，参数平铺、不引入 `CommSchedule` 之类的结构体：`comm_refresh_(int slot, int step)`（`!reneighbor` 的 forward-only 臂，现 `:340-343`）与 `comm_rebuild_(int exchange_slot, int borders_slot, int step)`（exchange + borders 臂，现 `:345-346`）。`comm_before_force_` 保留现有语义（refresh 或 rebuild + neighbor）；`step_verlet_`（`:449-462`）直接调这两个，从而能在中间插进自己的 `walk_slot_(pre_exchange)` / `walk_slot_(pre_neighbor)`——这正是它现在无法复用 `comm_before_force_` 的原因；`setup_verlet_comm_`（`:377-381`）调 `comm_rebuild_`。声明放在 `runtime.h:158-170` 现有那几个旁边。`timed_comm`（`:319`）继续沿用，不新增计时面。

## Files

- `src/cpu/communication/comm_brick.cpp` — round-exchange 原语；五处改为实例化；`forward_comm(Packer&)` 并入 tag 方案与 `FaceBufs`
- `include/atomiverse/cpu/communication/comm.h` / `comm_brick.h` — `setup` 契约处置；packer 宽度拉取入口
- `src/cpu/driver/runtime.cpp` / `include/atomiverse/cpu/driver/runtime.h` — 时序 helper；box 变化时调用 `setup`（若采纳）
- `tests/cpu/driver/test_verlet_comm.cpp` — `SpyComm` 断言调用序列

## Tasks

- [ ] Add the round-exchange primitive as a member of detail::FaceBufs so the buffers and requests are `this` rather than parameters; no config struct; Doxygen brief states its contract
- [ ] Re-express forward_cascade/forward_round, forward_comm(ctx), reverse_comm(ctx) and exchange_along_dim on it
- [ ] Add HaloTagKind::exchange and ::packer; route exchange_along_dim's literal 0,1,2,3 and forward_comm(Packer&)'s literal 0 through halo_tag; verify 5 kinds x 4096 stays under 32767
- [ ] Fold forward_comm(Packer&) onto the same primitive: FaceBufs instead of per-call vectors, and delete its unreachable self branch
- [ ] Add a LAMMPS Comm::init-style pull of the maximum packer width, and size the shared buffers once at the end of borders()
- [ ] Resolve the Comm::setup contract: drive it from the loop on box change, or make it private and rewrite the misleading error message
- [ ] Split comm_before_force_ into comm_refresh_(slot, step) and comm_rebuild_(exchange_slot, borders_slot, step) with flat int args, and route all three runtime.cpp sites through them
- [ ] Extend SpyComm coverage to assert the call sequence each of the three paths produces

## Testing

主判据是逐位不变：`ctest` 全绿、`widepbox_parity.sh` dev=0.0（8,8,4 与 16,4,4）、MIL0PEGM18 np=256 单点能量与力与改动前一致。性能判据是**不退步**：np=256 wall 不劣于改动前的 3.44–3.46 s / 5000 步（当前 1447–1460 steps/s，同节点 LAMMPS 1521）；这不是加速 spec，Comm 分段不得变慢超过 3%。`SpyComm` 断言三条路径的调用序列，是时序去重的直接闸门。

## Out of scope

- 不追求 Comm 提速——本链是可维护性，性能只做"不退步"约束。
- 不引入 `CommTiled`（`comm-maint-07-wire` 决定基类去留）。
- 不改 round 的划分或波段（`comm-maint-04-roundscan`）。
