---
title: 插件槽只持 Component*
status: in-progress
created: 2026-08-18
chain: component-plugin
grilled: true
---

# 插件槽只持 Component*

## Summary

在假定 `cpu::NestedLoopAdapter`（链上 02）已存在的前提下，CPU 的 `PhaseBuilder::add` / `add_ref` 在接到纯 `cpu::DriverNode` 时于注册边界自动包成该适配器，使 `pending_` 与编译后的 `Plan::Operation` 插件槽只持 `Component*`。宿主查看槽位时只能看见正在跑的插件，不再以循环节点身份暴露内层阶段枚举；公开的 `on(phase).add` / `add_ref` 签名与所有权规则不变。CUDA `DeviceTag` 路径保持现状（无 `DriverNode` 重载）。方法类与插件 unmount API 均不改。

## Domain basis

本规格不声明新物理：无方程、无 DOI/arXiv、无量纲预算（迭代次数，不是 eV/Å/fs）。

## Design

本步只改**注册边界与槽位占用模型**，不引入新的公开类型。

**实体与符号**

- `ATV::PhaseBuilder<PhaseEnum, DeviceTag>`（`include/atomiverse/core/driver_template.h`）——复用既有 `on` / `add` / `add_ref` 表面。CPU 上纯 `DriverNode` 且非 `Component` 的 rvalue `add` 与 `add_ref` 在函数体内构造 `cpu::NestedLoopAdapter`，把适配器指针写入 `pending_`；`budget < 0` 仍取 `child.default_budget()`。`add(lvalue)` 继续 `= delete`。双身份类型（`Component` + `DriverNode`）继续走 Component 重载，**不**包适配器。
- `ATV::detail::PendingBinding<Component>` —— 去掉 `nested_driver` / `nested_budget` / `is_driver()`；只保留 `phase`、`order`、`component`。
- `ATV::Operation<Component>` 与 `PlanBuilder::Entry`（`include/atomiverse/core/plan.h`）——同样去掉 `nested_driver` / `nested_budget` / `is_driver()`；编译后的槽只持 `Component*`。
- `Driver::build()` —— 只向 `PlanBuilder` 推 component 条目。
- `cpu::NestedLoopAdapter`（02）—— rvalue 路径把节点交给适配器拥有；`add_ref` 路径适配器借入节点。适配器进 `owned_components_`。`owned_drivers_` 本步不再 push。
- `cpu::Driver` 的 `for_each_pending` / `for_each_pending_kind` / `requires_tag_index` —— 签名不变。`for_each_pending_kind` 若遇到适配器，用 02 已有的非拥有 `DriverNode` 访问器展开后再调 `DrvFn`。
- `cpu::Runtime` —— 槽上不再有循环节点身份，所有 `op.is_driver()` 分支改为只 `build` / `launch` / `destroy` `op.component`。嵌套 `run_nested_erased` 由适配器的 `launch` 承担。

**生命周期 / 所有权**

- 拥有：`add(DriverNode&&)` → 适配器拥有节点 → 适配器由父 `owned_components_` 拥有。
- 借入：`add_ref(DriverNode&)` → 适配器借入节点（调用方仍可在 `run` 后读 `child.converged()`）。
- CUDA：不增加、不删除 `DriverNode` 重载。

**为何单测放在 `tests/cpu/driver/`：** `tests/core/` 没有 CPU 侧 `driver_template` 镜像。本步要实例化 CPU `DriverNode` + `NestedLoopAdapter`，只能从 CPU 后端测。

### Reuse decision

- reuse `PhaseBuilder::on` / `add` / `add_ref` —— 公开签名不动。
- generalize `detail::PendingBinding` —— 只持 `Component*`。
- generalize `Plan::Operation` / `PlanBuilder::Entry` —— 去掉 component|driver 双占。
- reuse `cpu::NestedLoopAdapter`（02）—— 禁止再写第二个包装类型。

## Files to create or modify

- `include/atomiverse/core/driver_template.h`
- `include/atomiverse/core/plan.h`
- `include/atomiverse/cpu/driver/driver.h`
- `src/cpu/driver/runtime.cpp`
- `tests/cpu/driver/test_driver_nesting.cpp`
- `regressions/architecture/component-plugin-03-slot/reg_component-plugin-03-slot.cpp` (new)
- `regressions/architecture/component-plugin-03-slot/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`

## Tasks

- [x] Write failing unit tests for plugin-slot occupancy (tests/cpu/driver/test_driver_nesting.cpp → DriverNesting)
- [x] Generalize `detail::PendingBinding` and `Driver::build` in include/atomiverse/core/driver_template.h to Component*-only occupancy
- [x] Generalize `Plan::Operation` and `PlanBuilder::Entry` in include/atomiverse/core/plan.h to Component*-only occupancy
- [x] Implement PhaseBuilder DriverNode `add` / `add_ref` wrap via `cpu::NestedLoopAdapter` in include/atomiverse/core/driver_template.h
- [x] Update `for_each_pending*` / `requires_tag_index` in include/atomiverse/cpu/driver/driver.h to unwrap NestedLoopAdapter
- [x] Update plan walks in src/cpu/driver/runtime.cpp to build/launch/destroy `op.component` only
- [x] Add Doxygen per doxygen style on wrap overloads (`@param budget` = dimensionless iterations)
- [x] Add regression example regressions/architecture/component-plugin-03-slot/ (public API only; hard-coded goldens, no third-party runtime)
- [x] Verify against DriverNesting / MinimizeNestedNode / PathValidation cases
- [ ] Run full check + test suite

## Testing strategy

单测只增 `tests/cpu/driver/test_driver_nesting.cpp` 的单函数用例。禁止在 `tests/` 做 e2e / 全 SCF。

- 快乐路径：`outer.on(phase).add_ref(inner)` 后 `build()`，该槽 `Operation::component` 非空且是 `NestedLoopAdapter`。
- 边沿：rvalue `add(InnerLoop{})` 同样包适配器；`budget = -1` 走 `default_budget()`；双身份仍走 Component 重载。
- 既有钉：`NestedConvergenceDoesNotStopTheParent`；`RunStageReplaysNestedDriver`；`RunsOnTheParentsRuntime`。
- 回归：`outer.run(frame, 5)` 后 `outer_iters == 5`、`outer.converged() == false`、`inner.converged() == true`。

## Out of scope

- 插件 unmount / 热插拔。
- 改任何方法类或丢掉双身份（若仍存在）。
- CUDA `DriverNode`。
- 新工厂或第二个 Adapter 类型。
- 删除 `owned_drivers_` 数据成员。
- 重写 `run_nested` 状态隔离语义。
