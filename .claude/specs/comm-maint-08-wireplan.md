---
title: Give the wire plan its verbs
status: approved
created: 2026-08-27
chain: comm-maint
---

# Give the wire plan its verbs

## Summary

操作者 2026-08-27：*"尽量移除各种巨大的 packer/bag 这样的结构体，太丑了不适合维护，或者给更好的解决方案"*。

调查结论：`PackPlan` **不是** CLAUDE.md 禁的那种参数包——没人把它塞进构造函数，它是从 Frame schema 编译出来的产物，与 `PairList` / `CellList` 同类。真正难维护的是它**只有数据没有动词**：14 个自由函数以 `const PackPlan&` 为参数，每个开头都重新确认 plan 自己已经知道的事，调用方要先在 6 个同形函数里挑对一个、被调函数再运行时检查你挑对了没有。最长的签名 9 个参数，其中 3 个是一个向量、2 个是一个缓冲区、1 个（`frame`）已经在 plan 里。

本步把动词搬到类型上。不改线格式、不改字段表、不动 `ColumnRef`。

## Domain basis

不声明新物理。判据是线上字节与所有既有数值逐位不变。

## Design

**为什么不是"删掉这个结构体"。** 删它就得回到 LAMMPS 的形状：每个 atom style 在构造函数里手写 `fields_border = {...}` 并重复 7 次，30 来个 `atom_vec_*.cpp` 各一份（`atom_vec_charge.cpp:35-42`）。ATV 的声明式字段表（`core/frame_schema.h:89`）加一个通信字段只要改一行，这是本层做得**比 LAMMPS 好**的地方，`PackPlan` 是它的运行时投影。要守住。

**改成什么。** 六个 kind 专用自由函数 + 五个尺寸函数塌缩成类型上的方法，kind 分派变成内部细节：

| 现在 | 之后 |
| --- | --- |
| `pack_comm(frame, plan, list, n, sx, sy, sz, out, cap)` | `plan.pack(frame, list, n, shift, out, cap)` |
| `pack_border(frame, plan, list, n, sx, sy, sz, out, cap)` | 同上——plan 知道自己是哪条线 |
| `unpack_comm(frame, plan, n, first, data, size)` | `plan.unpack(frame, n, first, data, size)` |
| `unpack_border(...)` | 同上 |
| `comm_bytes(plan, n)` / `border_bytes(plan, n)` / `reverse_payload_bytes(plan, n)` / `forward_wire_bytes(plan, n)` | `plan.bytes(n)` |
| `plan_row_bytes(plan)` | `plan.row_bytes()` |
| `plan_wire_fingerprint(plan, d)` | `plan.wire_fingerprint(d)` |
| `pack_reverse_rows(plan, first, count, buf)` | `plan.pack_rows(first, count, buf)` |
| `accumulate_reverse_rows(plan, rows, buf)` | `plan.accumulate(rows, buf)` |
| `pack_exchange(frame, plan, row, out, cap)` | `plan.pack_row(frame, row, out, cap)` |
| `unpack_exchange(frame, plan, row, data, size)` | `plan.unpack_row(frame, row, data, size)`（非 const，可重新 stride） |

**三个位移标量合成一个平铺数组**：`const double shift[3]`，与 `Halo::derive_hops(const int (&coords)[3])`（`halo.h:252`）同形。**不引入 `Shift` 结构体**——操作者约束 (a)。

**`frame` 参数保留。** plan 里有 `frame_owner`，技术上可以省掉，但它是 `plan.validate(frame)` 的对照物：显式传入让"这个 plan 属于这个 frame"在调用点可见，而不是靠一个缓存指针默认成立。这是有价值的检查，不是冗余。

**`kind` 从公开面消失。** `PackKind` 仍在，但只作为 `make_*_plan` 的内部选择器；调用方不再需要知道，因为它不再挑函数。`if (plan.kind != PackKind::Comm) throw` 这类检查全部删除——它们防的是调用方挑错函数，而挑错函数这件事已经不可能。

**不动的东西**：`ColumnRef`（12 字段的类型擦除是字段表的实现基础，删它等于放弃声明式）；线格式；`kCol*` 标志位；快/通用双臂（`comm-maint-03` 已定：两臂都活，generic 承载插件运行时列）。

**~~顺带：`ColDType` 5 路 switch 去重~~ —— 撤销。** 这条来自 grep `case ColDType::Int8:` 的计数（4 处），没有逐个读。实际上四处做的是四件不同的事：`frame_schema.h:28` 是 enum→sizeof 的查表，`block.h:112` 是 enum→`std::is_same_v` 的类型谓词，`frame.cpp:186` 是按 dtype 构造 variant 存储，只有 `atom_pack.cpp:738` 是"按 dtype 分派函数模板"。而它的孪生（`gather_column` 那份）已在 `comm-maint-03` 随 border 死臂一起删掉，所以现存的类型分派 switch 只剩一处，**没有可去的重**。造一个 `visit_dtype<F>` 把查表、类型谓词、variant 构造和模板分派统一起来是增加间接层而不是减少——操作者要的是少写丑结构体，不是多写模板。

## Files

- `include/atomiverse/cpu/communication/atom_pack.h` — 方法化；删自由函数声明
- `src/cpu/communication/atom_pack.cpp` — 实现搬进方法；kind 分派内部化；`visit_column`
- `src/cpu/communication/comm_brick.cpp`、`bcast_frame.cpp` — 调用点
- `include/atomiverse/cpu/communication/schema_mpi.h` — `plan.wire_fingerprint()`
- `tests/cpu/communication/*`、`tests/mpi/*` — 调用点

## Tasks

- [x] Add pack / unpack / bytes / row_bytes / wire_fingerprint / pack_rows / accumulate / pack_row / unpack_row as members, dispatching on the plan's own kind
- [x] Collapse sx/sy/sz into a flat `const double shift[3]`; introduce no shift struct
- [x] Delete the kind-mismatch runtime checks, which the dispatch makes unreachable
- [x] Delete the free functions and update all 38 call sites
- [~] ColDType switch dedup — WITHDRAWN after reading the four sites: they are a size table, a type predicate, a variant constructor and a template dispatch. Only one is typed dispatch and its twin was already deleted in comm-maint-03. Nothing to deduplicate.
- [x] Keep ColumnRef, the field table and both wire arms untouched

## Testing

行为保持：`ctest` 全绿（当前 240/240），np=16 MIL0PEGM18 单点逐位不变
（`E=1725.258348 pair=-200.208398 eV`），`widebox_parity.sh` 8,8,4 与 16,4,4 dev=0.0。
方法化本身由既有的 pack/unpack 往返测试覆盖；新增一个"同一个 plan 用错 kind 已
不可能表达"的编译期说明即可，不需要新运行时测试。

## Out of scope

- 不删 `ColumnRef`，不改字段表形状。
- 不动 `FaceBufs` 的并行数组（`comm-maint-06-transport`）。
- 不动 `DomainDecomposition`（`comm-maint-05-dd-split`）。
