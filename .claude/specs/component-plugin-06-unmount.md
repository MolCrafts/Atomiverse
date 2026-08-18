---
title: 运行中卸载与再挂载
status: in-progress
created: 2026-08-18
chain: component-plugin
grilled: true
---

# 运行中卸载与再挂载

## Summary

CPU `Driver` 在已经 `build()`、甚至已经跑过若干步之后，可以用 `unfix(name_or_id)` 卸下一个已挂节点，语义对齐 LAMMPS `unfix`：先 `destroy` 被卸节点，再按 05 活表（即未清空的 `pending_`）重编 `Plan`，再跑与 `build()` 同一套闭包检查（04 `require_*` + `validate_long_range_contract_`）。依赖被卸能力的插件不得继续保持 Active；把节点经现有 `on(phase).add` / `set_evaluator` 家族再挂回去、依赖重新满足后，合法组合可重建并继续跑。Recorder 卸载只结束其生命周期，不假装撤回已经写出的文件。不改任何具体势能或方法子类。

## Domain basis

本 spec 不声明新物理。

## Design

本规格只在 `cpu::Driver` 上补一条 LAMMPS 形 setter 家族成员，不引入 `register_plugin`。05 活表就是 `pending_`（`build()` 不再清空）；06 不另建花名册。

**解析。** `unfix` 按 `name_or_id` 解析表项：先精确匹配可选的 id 字符串（若 05 只提供指针身份，则本步可用 `name()` 精确匹配），再精确匹配 `Component::name()`。零命中 → `std::runtime_error`，how-to 列出活表里已有的 name；多于一命中 → 同形 how-to，要求调用方用更窄的身份（`PluginIdentity` 或先保证 `name()` 唯一）。

**卸载。** 命中后：若该节点已经 `build()` 过，用一条只含该节点的临时 `Plan` 调用已有 `Runtime::destroy_plan`，**禁止** `destroy_components()` / `flush_live_()` / `teardown_pass()`。然后从 `pending_` 删除；owned 节点再从 `owned_components_` 丢掉；若它占着单槽 setter（`neighbor_set_` 等），清空该槽。borrowed 节点只退表、不 `delete` 调用方对象。

**重编与闭包。** 私有 `rebind_from_live_table_()` 按剩余 `pending_` 走与 `build()` 相同的 `PlanBuilder` 路径写回 `plan_`，对仍活着的 Runtime 调用 `Runtime::load`，**不**对仍在表里的节点再 `build()`。随后跑闭包：`validate_long_range_contract_()` → `validate_components()`。缺硬 require 时 `throw_missing`，信息含 how-to。`unfix` 在表变更后**立即** rebind+闭包，所以卸掉最后一个 evaluator 时异常发生在 `unfix` 本身。

**依赖失效。** 04 matcher 是闭包的一部分。被卸能力的依赖者不得继续被 launch（从 Plan 中拿掉，或标成非 Active）。本步若 05 尚未有 Active/WaitingDeps 枚举，最小实现是：依赖不满足则 `unfix`/`rebind` 抛与 `build()` 相同的缺依赖异常；再挂后下一次 `run` 段重建合法组合。不另做 `register_plugin` 查询类型。

**再挂。** 不新开注册入口。`on(phase).add` / `set_evaluator` 写回 `pending_`。`begin_run_` 在 `live_` 且表脏时只对**新**节点 `Runtime::build_plan`，再 `load` 全表重编的 Plan，再跑同一套闭包。`unfix` / 再挂只允许出现在步与步之间、或 `run()` 段与段之间；从某个 `launch()` 里调用则 throw how-to。

**Recorder。** `RoleFlag::ExternalIO` 同样走 destroy → 退表 → 重编。`destroy` 可以关句柄；不得 unlink / truncate 已写出字节。

**生命周期。**

1. `unfix`：解析 →（若已 build）`destroy_plan` 单节点 → 退表 → `rebind` → `load` → 闭包。
2. 再挂：`on(phase).add` → 下一次 `begin_run_` 合并新节点、`build_plan` 仅新节点、`load`、闭包。
3. `live_` 与 `step_counter_` 不变。

### Reuse decision

- reuse `Runtime::destroy_plan` / `load` / `build_plan`
- reuse 04 matcher / `validate_components` / `throw_missing`
- reuse 05 活表（`pending_` + `PluginIdentity` / `mounted()`）
- new — `cpu::Driver::unfix(std::string_view)`

## Files to create or modify

- `include/atomiverse/cpu/driver/driver.h`
- `tests/cpu/driver/test_unfix.cpp` (new)
- `tests/CMakeLists.txt`
- `regressions/architecture/component-plugin-06-unmount/reg_component-plugin-06-unmount.cpp` (new)
- `regressions/architecture/component-plugin-06-unmount/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`

## Tasks

- [ ] Write failing unit tests for Driver::unfix (tests/cpu/driver/test_unfix.cpp → TestDriverUnfix)
- [ ] Implement Driver::unfix and rebind_from_live_table_ in include/atomiverse/cpu/driver/driver.h
- [ ] Add docstring per doxygen on unfix (no physical units; @param name_or_id, @throws how-to)
- [ ] Register test_cpu_driver_unfix in tests/CMakeLists.txt
- [ ] Add regression example regressions/architecture/component-plugin-06-unmount/
- [ ] Register atv_add_regression(architecture component-plugin-06-unmount) in regressions/CMakeLists.txt
- [ ] Verify against mid-run unfix of last evaluator (throw how-to) and remount via on(phase).add
- [ ] Run full check + test suite

## Testing strategy

`TestDriverUnfix`：假 Component + `StubIntegrator` / `StubForceEvaluator`。禁止完整物理 MD。

- 卸非硬依赖 Observer：退表，`destroy==1`，下一步不再 launch。
- 未知 name：throw how-to。
- 重名：按 name throw，身份可区分时卸其中一个。
- 卸最后一个能量源：`unfix` 立即 throw，节点已 destroy 且已退表。
- borrowed `add_ref`：调用方对象仍在。
- 单槽 neighbor 卸后可再 `set_neighbor`。
- 假 Recorder 缓冲 `"rec-1\n"` 在 unfix 后仍在。
- `launch()` 内 unfix：throw how-to。

回归：`cpu::md::MD` 两段 `run(frame, 2)`；`unfix("fake_eval")` throw 含 `missing required` 与 `set_evaluator`；再 add 后 `launches==4`，打印 `ok: unfix-remount launches=4`。

## Out of scope

- CUDA unfix。
- `register_plugin`。
- 改 `PairLJCut` / `RHF` / `XYZ` 等子类。
- 撤回 Recorder 已写文件。
- Python / 任务编解码。
- `launch()` 热路径卸载。
- 子类构造函数 / `install_defaults` 里 `on().add` 的默认件与用户插件同一条路径，可按名字或身份 `unfix`；卸完走与 `build()` 同一套闭包。
