---
title: Collapse duplicated comm helpers and rehome pure functions
status: approved
created: 2026-08-27
chain: comm-maint
---

# Collapse duplicated comm helpers and rehome pure functions

## Summary

`comm_brick.cpp` 的匿名命名空间里有 27 个函数，其中 **6 个是 `Halo`/`Swap` 的纯函数**、**4 个是 `PackPlan` 的纯函数**、**4 个是 reverse 线编解码器**（它的 forward 同胞 `pack_comm`/`pack_border` 是 `atom_pack.cpp` 的公开符号），另有 **3 个与 `atom_migration.cpp` 逐字节重复**（`rank_along_dim` 还有第三份在 `domain.cpp:175`）。层内还有 4 份溢出检查乘法、2 份 FNV 哈希、2 份 root-rank 校验前言、3 份 Frame 重建收尾。本步把它们各自归位到唯一的家，`comm_brick.cpp` 只保留真正需要文件作用域的 6 个。不改行为、不改线格式。

## Domain basis

不声明新物理。所有被移动的函数是纯函数或纯编解码器，判据是移动前后线上字节与所有既有数值逐位相同。

## Design

**归位表**（来源：本会话的耦合扫描，行号为改动前）

| 现址 | 去处 | 依据 |
| --- | --- | --- |
| `stage_end` `:389`、`round_end` `:409`、`round_begin` `:424`、`mirror_swap` `:442` | `Halo` 成员/`halo.h` 自由函数 | 纯粹是 swap 表的查询，只读 `Halo` |
| `pbc_shift_for_swap` `:477` | `Swap` 成员（取 `const Box&`） | 纯粹是 `Swap::pbc` × 盒长 |
| `swap_is_silent` `:491` | `Swap::is_silent()` | 纯谓词 |
| `halo_payload_fingerprint` `:509`、`forward_wire_bytes` `:532`、`plan_row_bytes` `:537`、`reverse_payload_bytes` `:881` | `atom_pack.{h,cpp}` | 纯粹是 `PackPlan` 的函数，与已在那里的 `comm_bytes`/`border_bytes` 同类 |
| `plan_is_fxyz_reverse` `:889`、`pack_reverse_rows` `:898`、`accumulate_column` `:932`、`accumulate_reverse_rows` `:950` | `atom_pack.{h,cpp}` | **这是 reverse 线编解码器**；它的三个同胞 `pack_comm`/`pack_border`/`pack_exchange` 都是 `atom_pack.h` 的公开 API。reverse 独自留在 `comm_brick.cpp` 是重命名时的残留，不是设计 |
| `checked_mul` `:526` | 与 `atom_pack.cpp:68` 合并为一处 | 层内共 4 份同体：`gather.cpp:32`（`checked_product`）、`bcast_frame.cpp:258`（`checked_product`）、`comm_brick.cpp:526`、`atom_pack.cpp:68` |
| `hash_wire_value` `:495,:505` | 与 `atom_pack.cpp:80/:90`（`hash_bytes`/`hash_value`）合并 | 同体 |
| `require_collective_halo_schema` `:518` | `schema_mpi.h`，紧邻 `require_collective_schema` | 4 行薄包装 |
| `setup_swaps` `:604` | 就是 `Halo::setup` 的四参重载 | 3 行包装：`derive_uniform_rounds` 后调 `Halo::setup` |
| `rank_along_dim` `:152`、`slab_along_dim_from_cuts` `:162`、`dest_rank_for` `:187` | `DomainDecomposition` 成员（`domain.{h,cpp}`） | 纯粹是 DD + `Box` 的函数；`domain.cpp:175` 已有第三份 `rank_along_dim`。`atom_migration.cpp` 的那份随 slug 01 删文件一起消失 |

留在 `comm_brick.cpp` 的 6 个：`facebufs_bytes`、`prealloc_bufs`（两者触碰 PIMPL 的 `FaceBufs`）、`pack_forward_for_swap`、`pack_to`、`forward_round`、`forward_cascade`、`exchange_along_dim`、`zero_ghost_forces`。

**约束 (a)——不写 bag 类。** 归位不得引入任何新的聚合类型。`pbc_shift_for_swap` 成为 `Swap` 成员后签名是 `void pbc_shift(const Box&, double out[3]) const`，不引入 `PbcShift` 结构体。`dest_rank_for` 成为 `DomainDecomposition::owner_of(double x, double y, double z, const Box&) const`，三个坐标平铺，不引入 `Point3`。

**匿名命名空间的三块嵌套要一并理顺。** 现有 Block C（`:617`）在语法上嵌套在 Block B（`:148`）内部——`:604` 与 `:617` 之间没有闭合括号，两者一起在 `:1023`/`:1024` 关闭。嵌套不产生任何作用域效果，纯属编辑事故。合并为一块。`using detail::FaceBufs;` 在同一块里出现两次（`:150`、`:377`），删一处。

**四命名空间问题记录但不动。** 本目录横跨 `ATV::cpu`（`comm.h`）、`ATV::cpu::comm`（7 文件）、`ATV::cpu::md::mpi`（5 文件）、`ATV::cpu::communication`（`domain_runtime`）。CLAUDE.md 已把"命名空间尚未对齐目录"列为有意的后续事项，本步**不**触碰——归位时函数跟随它的宿主类型所在的命名空间，不做统一。

## Files

- `include/atomiverse/cpu/communication/halo.h` / `src/cpu/communication/halo.cpp` — 接收 6 个 `Halo`/`Swap` 纯函数
- `include/atomiverse/cpu/communication/atom_pack.h` / `src/cpu/communication/atom_pack.cpp` — 接收 4 个 `PackPlan` 纯函数 + 4 个 reverse 编解码器；合并 `checked_mul` 与哈希
- `include/atomiverse/cpu/communication/domain.h` / `src/cpu/communication/domain.cpp` — 接收 3 个归属函数；删掉 `:175` 的重复 `rank_along_dim`
- `include/atomiverse/cpu/communication/schema_mpi.h` — 接收 `require_collective_halo_schema`
- `src/cpu/communication/comm_brick.cpp` — 变薄；合并匿名命名空间
- `src/cpu/communication/gather.cpp`、`bcast_frame.cpp` — 改用合并后的 `checked_mul`

## Tasks

- [ ] Move the six pure Halo/Swap helpers onto Halo/Swap with flat parameters and Doxygen briefs
- [ ] Move the four pure PackPlan helpers into atom_pack.{h,cpp} beside comm_bytes/border_bytes
- [ ] Move the reverse codec (plan_is_fxyz_reverse, pack_reverse_rows, accumulate_column, accumulate_reverse_rows) into atom_pack.{h,cpp} so all four wire codecs live together
- [ ] Move rank_along_dim, slab_along_dim_from_cuts and dest_rank_for onto DomainDecomposition; delete the duplicate rank_along_dim at domain.cpp:175
- [ ] Collapse the four overflow-checked-multiply copies and the two hash-helper copies to one each
- [ ] Move require_collective_halo_schema next to require_collective_schema in schema_mpi.h
- [ ] Replace setup_swaps with the existing Halo::setup overload
- [ ] Merge the three anonymous-namespace blocks in comm_brick.cpp into one and delete the duplicate using-declaration
- [ ] Add unit tests for the newly public Halo/Swap and PackPlan helpers that had none while file-scoped

## Testing

移动是行为保持的，所以主判据是既有测试全绿且**数值逐位不变**：`ctest` 237/237，加上 `widebox_parity.sh` 在 8,8,4 与 16,4,4 两个几何 dev=0.0。新公开的符号各补一个最小单元测试——这些函数此前因文件作用域而无法直接测，正是本步的附带收益。reverse 编解码器移动后必须有一个"pack 后 accumulate 回原值"的往返测试。

## Out of scope

- 不改任何线格式或字段集合。
- 不统一四个命名空间。
- 不动 fast/generic 双分支（`comm-maint-03-genericpath`）。
- 不动 `forward_round`/`forward_cascade` 的内部结构（`comm-maint-04-roundscan`、`comm-maint-06-transport`）。
