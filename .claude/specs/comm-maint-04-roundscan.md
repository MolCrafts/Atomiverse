---
title: Make the multi-hop round scheduling pure and tested
status: code-complete
created: 2026-08-27
chain: comm-maint
---

# Make the multi-hop round scheduling pure and tested

## Summary

本链最高价值的一步。2026-08-27 修掉的一个缺陷——`Halo::expand` 里 `if (round > 1 && false)` 关掉了 LAMMPS `ineed >= 2` 的 slab 波段，让每个转发轮把收到的整个 slab 而不是仍在接收方 shell 内的部分往前送，三轴叠加后 ghost 行数为应有的 **1.76 倍**——活在一片**完全没有测试**的区域里：`test_halo_maxneed.cpp` 的 9 个 TEST 只测 hop *计数*，全文 grep `slablo|slabhi|sendlist` 零命中；`test_halo.cpp` 只覆盖 round-1 波段。它一路活到 256-rank 基准，靠 ghost 膨胀才暴露。本步把"轮次/窗口调度"从 `comm_brick.cpp` 的传输代码里抽成 `Halo` 上的纯函数，纳入既有的纯逻辑测试边界，并补齐 round>1 的行为测试。

## Domain basis

不声明新物理。被测的是几何完备性命题：给定子域宽 \(w\)、shell \(c\)、跳数 \(h = \lceil c/w \rceil\)，转发链在第 \(h\) 跳后必须覆盖 \([\text{sublo}-c, \text{subhi}+c]\) 且不重复投递同一原子的同一周期像。深度裁剪界 \(\text{sublo}+c\)（低面）/ \(\text{subhi}-c\)（高面）来自 LAMMPS `CommBrick::setup` 的 `ineed >= 2` 分支（`src/comm_brick.cpp`，checkout `78a7ddf`）；其正确性论证是归纳的：下一跳的裁剪界严格更低，所以在本跳裁掉的行不会是后续任何一跳所需。参考 S. Plimpton, J. Comput. Phys. **117**, 1 (1995)。

## Design

**现状。** 调度逻辑内联在 `comm_brick.cpp` 的 `forward_cascade`（外层：轴 stage → 轮）与 `forward_round`（内层：每轮的扫描窗口）里，与 MPI 收发交织，因此只能通过多 rank 端到端测试触及。`Halo` 本身是纯的、无 MPI、无 `Context`（`halo.h:21-24` 明文约束），并因此拥有 1218 行单元测试——**这就是要扩展的边界**，`CellList`/`PairList` 对 `NeighborList` 是同一先例。

**抽出什么。** 三个纯查询，全部只读 `Halo` + 标量，不引入聚合类型（约束 a）：

1. **轮的划分** —— `stage_end` / `round_end` / `round_begin`（`comm-maint-02-dedup` 已把它们移到 `Halo`）。
2. **每轮的扫描窗口** —— 当前是 `forward_cascade:830-841` 里的 `win_first[k]` / `win_last[k]` 计算：round 1 取 `[0, sources_end)`，round r>1 取**对面**上一轮的收行区间 `[firstrecv, firstrecv+nrecv)`。抽成 `Halo` 上的查询，签名平铺：`void scan_window(int iswap, int sources_end, int& first, int& last) const`。
3. **深度裁剪** —— 已经在 `Halo::expand` 里（`slablo`/`slabhi`），本步不移动，但要让它可被独立断言。

**为什么这三个而不是整个 `forward_round`。** `forward_round` 还负责 MPI 收发、容量增长、ghost tag 登记，那些不是纯的。分界线就是 `Halo` 已有的那条：决定**发哪些行、发给谁、收到哪里**是纯的；真正搬字节不是。

**必须补的测试**（这是本 spec 的主体，不是附属）：

- round>1 的 `sendlist` **内容**，不只是长度。用 `x_chain_frame` 那样的手造 DD + 逐行坐标，断言 hop-2 的低面 swap 恰好发出 \([\text{subhi}, \text{sublo}+c]\) 区间内的行——这条断言会直接抓住 `&& false`。
- 深度裁剪界本身：`slablo`/`slabhi` 在 round>1 时等于 `(mid, sublo+c)` / `(subhi-c, mid)`。
- mid-plane 边界：构造一个恰好落在 mid 上的行，断言它被发出（闭区间）。
- **3 跳链的完整性**：`16,4,4` 网格（子域 3.94 Å vs shell 11 Å ⇒ hops 3,1,1）下，三轮的并集覆盖整个 shell 且无重复。这是唯一"转发轮必须把行再交给下一跳"的几何，深度裁剪切多了会先在这里丢对。
- 扫描窗口本身：round 1 取 `[0, sources_end)`；round r>1 取对面上一轮的收行区间，而**不是**同面的。

参考几何用已验证过的两组：63.0016 Å 立方盒 / 8,8,4 ⇒ hops 2,2,1；16,4,4 ⇒ hops 3,1,1。

**不做的事。** 不改 `forward_cascade` 的传输结构（属 `comm-maint-06-transport`），不改波段公式本身——它已在 2026-08-27 修复并经 np=256 三组几何 dev=0.0 验证。本步是把那次修复**锁进单元测试**，让它不能再被一个 `&& false` 悄悄关掉。

## Files

- `include/atomiverse/cpu/communication/halo.h` / `src/cpu/communication/halo.cpp` — 新增 `scan_window` 纯查询；`slablo`/`slabhi` 保持
- `src/cpu/communication/comm_brick.cpp` — `forward_cascade` 改为调用 `Halo::round_scan_window`，删去内联的窗口推导
- `tests/cpu/communication/test_halo_rounds.cpp` — 新文件，本 spec 的主体

## Tasks

- [x] Add Halo::round_scan_window(round, width, sources_end, prev_first, prev_last, win_first, win_last) as a pure query with flat out-parameters and a Doxygen brief
- [x] Rewrite forward_cascade to call it instead of deriving win_first/win_last inline; assert the two agree during a transition build
- [x] Write tests/cpu/communication/test_halo_rounds.cpp asserting round>1 sendlist CONTENTS at the 8,8,4 geometry
- [x] Assert the round>1 slab bounds are (mid, sublo+cutoff) and (subhi-cutoff, mid)
- [x] Assert a row sitting exactly on the mid-plane and on sublo+cutoff is shipped (inclusive band)
- [x] Assert 3-hop chain completeness and non-duplication at the 16,4,4 geometry
- [x] Assert scan_window takes the OPPOSITE face's previous receipts for round>1, not the same face
- [x] Add a regression-style test that an open round>1 band inflates the ghost set, so the 1.76x defect cannot return silently

## Testing

新文件是判据本体。反向验证是硬要求：把 `Halo::expand` 的 round>1 波段临时改回开放区间，新测试必须**红**；恢复后必须绿。没有这个反向验证，测试可能只是重述实现。端到端仍要过 `widebox_parity.sh`（8,8,4 与 16,4,4，dev=0.0），但那是既有闸门，不是本步新增覆盖。

## Out of scope

- 不改传输结构、不改 tag 方案、不动缓冲区（`comm-maint-06-transport`）。
- 不改波段公式。
- 不为非周期轴补测试——`periods` 在生产中从无写者（见 `comm-maint-05-dd-split`）。

## Outcome

Landed as `Halo::round_scan_window` (static, pure, flat raw-array
out-parameters matching `derive_hops(const int (&coords)[3])` — no window
aggregate) plus `tests/cpu/communication/test_halo_rounds.cpp`, 11 tests.
`forward_cascade` calls the query instead of deriving the bounds inline, and
the `width > 2` guard moved to live with the logic it protects.

**Reverse-verified.** Reopening the round>1 band in `Halo::expand` — i.e.
reproducing the exact 2026-08-27 defect — makes `test_cpu_comm_halo_rounds`
fail while `test_cpu_comm_halo_maxneed` and `test_cpu_comm_halo` both still
pass. That is the coverage gap, demonstrated rather than asserted: the two
pre-existing halo test files cannot see this class of bug at all.

241/241 green; np=16 MIL0PEGM18 single point unchanged at
`E=1725.258348 pair=-200.208398 eV`.
