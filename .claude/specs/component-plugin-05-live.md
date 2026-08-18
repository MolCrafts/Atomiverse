---
title: 组件插件活挂载表
status: in-progress
created: 2026-08-18
chain: component-plugin
grilled: true
---

# 组件插件活挂载表

## Summary

`Driver::build()` / `compile()` 编完 Plan 之后不再把挂载表丢掉。调用方可以按**实例身份**（指针，不是类型名）枚举仍挂在该 Driver 上的插件，并在不析构整个 Driver 的情况下再编一次，得到指针级等价的 Plan。`Runtime::load` 继续负责换整张 Plan；本步不提供运行中卸载。

## Domain basis

本步不声明新物理。

## Design

现有 `pending_` 是编 Plan 前的一次性草稿：`ATV::Driver::build()` 扫一遍后立刻 `pending_.clear()`。本步把 `pending_` **就地提升**为活挂载表，而不是另起一张平行表。

**实体与生命周期**

- `detail::PendingBinding<Component>` 仍是一行挂载。03 落地后只持 `component*`；同一实例因 `default_bindings()` 出现多行仍然合法；去重按实例指针。
- 所有权不变：`add(rvalue)` 转移 / `add_ref` 借用。禁止新工厂。
- 活表键是实例身份，不是 `name()`、不是 `typeid`。

**新符号**

- `ATV::PluginIdentity`：平凡值类型，唯一字段 `const void* instance`。两个同类型实例必须是两个身份。
- `detail::PendingBinding::identity()`：该行的 `PluginIdentity`。
- `Driver::mounted()`：返回活表的 `const` 引用，`build()` 之后仍可读。
- `Driver::is_mounted(PluginIdentity)`：按实例身份查询。

**`build()` 语义**

`Driver::build()` 每次从活表重编 `plan_`，**删除** `pending_.clear()`。表未变时第二次 `build()` 的 Plan 指针与第一次逐槽相等。`cpu::Driver` 里「`Base::build()` 消费 pending」的注释改为「活表保留，Plan 是它的编译快照」。

本步不给活表加 Active / WaitingDeps / Failed 状态机（留给 06 若需要再补）；本步只保证表在 compile 后仍可寻址、可重编。

**子类默认插件就是注册。** 预设构造函数里的 `on(phase).add(ZeroForces{}, …)` 和 `install_defaults()` 里按已挂载集合补上的 `Virial` / `Halo` / `Reduce`，走的是同一条 `on().add` 路径，不是第二条注入器。它们出现在活表里，身份与用户挂的插件相同。`install_defaults` 必须幂等：已 `is_mounted` / 已有同类供给则不再 `add`，这样第二次 `build()` 不会把 `Reduce` 挂两遍。条件默认（「有人要应力才挂 Virial」）写在子类钩子里，因为构造时还看不见用户后挂的件；无条件默认写在构造函数。两种都是插件注册。

### Reuse decision

- generalize `Driver::build` / `pending_.clear()` — 提升为活表；禁止另写第二张 mount map。
- reuse `Runtime::load` / `build_plan` / `destroy_plan` — 本步不重跑组件生命周期。
- reuse `PhaseBuilder::add` / `add_ref`。

## Files to create or modify

- `include/atomiverse/core/driver_template.h`
- `include/atomiverse/cpu/driver/driver.h`
- `tests/core/test_driver_live_table.cpp` (new)
- `tests/CMakeLists.txt`
- `regressions/architecture/component-plugin-05-live/reg_component-plugin-05-live.cpp` (new)
- `regressions/architecture/component-plugin-05-live/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`

## Tasks

- [ ] Write failing unit tests for Driver (tests/core/test_driver_live_table.cpp → TestDriver)
- [ ] Generalize `Driver::build` in include/atomiverse/core/driver_template.h to keep `pending_` as the live mount table
- [ ] Add `PluginIdentity`, `PendingBinding::identity`, `Driver::mounted` and `Driver::is_mounted` with Doxygen in include/atomiverse/core/driver_template.h
- [ ] Rewrite consume-and-clear comments on `cpu::Driver::build` and `requires_tag_index` in include/atomiverse/cpu/driver/driver.h
- [ ] Register TestDriver in tests/CMakeLists.txt as `test_driver_live_table`
- [ ] Add regression example regressions/architecture/component-plugin-05-live/reg_component-plugin-05-live.cpp
- [ ] Register the regression in regressions/CMakeLists.txt
- [ ] Verify two same-type instances stay distinct identities and a second compile matches the first Plan pointer-for-pointer
- [ ] Run full check + test suite

## Testing strategy

单测只覆盖 `Driver::build` / `mounted` / `is_mounted`，禁止在 `tests/` 里跑 `run()`。Happy：`add_ref` 两个同名 probe，`build()` 后 `mounted().size()` 含二者且身份不等；再 `build()` Plan 指针相等。回归：`cpu::Evaluate` + zero 三件套 + 两个 `live_probe`，`mounted_count=5`，二次 compile 指针复现。

## Out of scope

- 运行中 unfix（06）。
- PW / GpwDft / MeanFieldMethod 内层 `pending_.clear()`。
- 重编后重建组件生命周期。
- CUDA `Driver::build` 包装层、整数插件 id、按 `name()` 寻址。
- 把无条件默认从构造函数搬到别的 API（它们已经是 `on().add`）。
