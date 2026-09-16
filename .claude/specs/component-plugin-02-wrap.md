---
title: NestedLoopAdapter 将 DriverNode 包成 Component
status: in-progress
created: 2026-08-18
chain: component-plugin
grilled: true
---

# NestedLoopAdapter 将 DriverNode 包成 Component

## Summary

本步在 `cpu/driver` 落地唯一通用宿主包装 `ATV::cpu::NestedLoopAdapter`：它是 `cpu::Component`，拥有或借用一个既有 `cpu::DriverNode`，把内层循环的 `build_on` / `run_nested_erased` / `destroy_on` 接到外层 Component 生命周期上，预算在构造时确定。调用方用现有 `on(phase).add(adapter)` / `add_ref(adapter)` 把它挂进外层 Plan；Component 面上只看得到适配器，看不到内层 Phase 枚举。若内层节点带 `cpu::HF` 标签，适配器按 01-probe 已落地的 `provides_energy` / `provides_force` 原样转发，否则两者皆为 false。`RHF` / `PW` / `GpwDft` 与各 `*Evaluator` 的身份不变，`PhaseBuilder` 本步不改。

## Domain basis

本 spec 不声明新物理。

## Design

本步只新增宿主适配器，不新增 kind、不改方法类、不改 Plan 编译器。假设链上 `component-plugin-01-probe` 已落地：`cpu::Component` 与 `cpu::HF` 均已有虚函数 `provides_energy()` / `provides_force()`。

**实体。** 新类型 `ATV::cpu::NestedLoopAdapter` 放在 driver 柱（`include/atomiverse/cpu/driver/nested_loop_adapter.h`），因为它是宿主对 `DriverNode` 的适配，不是产品势/方法。它公开继承 `cpu::Component`，**不**继承 `cpu::DriverNode`。被包对象是已有节点：`cpu::Driver<Phase>`、`cpu::scf::MeanFieldMethod<D>`（因而 `RHF`/`UHF`/`RKS`/`UKS`）、`cpu::pw::PW`、`cpu::gpw::GpwDft`。产品族既有的 Component 包装仍是 `PWEvaluator` / `GPWEvaluator` / `AM1Evaluator` / `RM1Evaluator`——本步不写第三条产品包装，也不让方法类变成 `Component`。

**构造与所有权（对标 `PhaseBuilder::add` / `add_ref`）。** 两个公开构造，参数即成员，无 `*Config`、无 `make_*`：

- `NestedLoopAdapter(DriverNode& inner, int budget = -1)` — 借用；调用方保证 `inner` 活过 `build`/`launch`/`destroy`。
- `NestedLoopAdapter(std::unique_ptr<DriverNode> inner, int budget = -1)` — 拥有；`inner == nullptr` 抛 `std::invalid_argument`，消息写明应传入非空节点或改用借用构造。
- 额外允许 `template <class Node> NestedLoopAdapter(Node&&, int budget = -1)`，仅在 `Node` 为 `DriverNode` 右值且**不是** `Component` 时启用（与 `PhaseBuilder` 禁止双重身份的 SFINAE 同形）。禁止隐式把左值当成拥有。

`budget < 0` 时在构造里一次性写成 `inner.default_budget()`，之后不再回读。拷贝删除，移动默认。适配器内部只持 `std::unique_ptr<DriverNode> owned_`（借用时为空）与 `DriverNode* inner_`。

**生命周期。** 单职责三方法，全部转发，不在 `launch` 里分配：

- `build(const BuildContext&)`：从 `ctx.runtime.exec.runtime` 取 `cpu::Runtime*`；空则抛 `std::runtime_error`（how-to：适配器必须挂在已经 `setup_pass` / 父 Runtime 已发布的外层循环上，与 `Driver::run_nested` 同句）。然后 `inner_->build_on(*rt)`。
- `launch(const LaunchContext&)`：`inner_->run_nested_erased(ctx.runtime, ctx.frame, budget_)`。不改写传入的 `ctx.stage` / `ctx.step`；内层 Phase 只存在于 `run_nested_erased` 为自己 Plan 构造的 LaunchContext。
- `destroy(const DestroyContext&)`：同样取 Runtime，空则抛，然后 `inner_->destroy_on(*rt)`。

`name()` 返回 `"nested_loop_adapter"`。`default_bindings()` 为空——相位由调用方 `on(phase).add` 声明，本步不抢 03-slot 的 PhaseBuilder 自动包装。

**能力谓词。** 不新增 `NestedLoop` kind。适配器在 Component 面上覆盖 01-probe 谓词：

```
provides_energy() / provides_force():
  if (auto* hf = dynamic_cast<const HF*>(inner_)) return hf->provides_energy/force();
  return false;
```

`dynamic_cast<HF*>` 只允许出现在这个宿主适配器内部，不是插件去认其他族。禁止对 `PlaneWave` / `GPW` / `Semiempirical` 再写一套 cast；那些产品继续走既有 `*Evaluator`。

**双重身份禁令。** `std::is_base_of_v<cpu::DriverNode, NestedLoopAdapter>` 为 false；`dynamic_cast<DriverNode*>(&adapter)` 为空。被包方法类保持 `!std::is_base_of_v<cpu::Component, …>`——本步不得改 `RHF`/`PW`/`GpwDft`/`*Evaluator`/`PhaseBuilder` 使它们获得或失去另一身份。外层 Plan 里适配器是一条 `Operation::component`，`is_driver() == false`；内层节点仍走自己的 Plan。

**错误。** 无回退值。空节点、空 Runtime 均抛；不静默跳过内层。

### Reuse decision

- reuse `cpu::DriverNode::build_on` / `run_nested_erased` / `destroy_on` — 适配器只转发这三步。
- reuse `cpu::Driver<Phase>`、`MeanFieldMethod`、`PW`、`GpwDft` as nodes — 不把方法类改成 `Component`。
- reuse `PWEvaluator` / `GPWEvaluator` / `AM1Evaluator` / `RM1Evaluator` — 不新增第三条产品包装。
- new — `cpu::NestedLoopAdapter`：唯一通用的 DriverNode→Component 宿主包装。

## Files to create or modify

- `include/atomiverse/cpu/driver/nested_loop_adapter.h` (new)
- `tests/cpu/driver/test_nested_loop_adapter.cpp` (new)
- `tests/CMakeLists.txt`
- `regressions/architecture/component-plugin-02-wrap/reg_component-plugin-02-wrap.cpp` (new)
- `regressions/architecture/component-plugin-02-wrap/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`

## Tasks

- [x] Write failing unit tests for NestedLoopAdapter (tests/cpu/driver/test_nested_loop_adapter.cpp → TestNestedLoopAdapter)
- [x] Implement NestedLoopAdapter in include/atomiverse/cpu/driver/nested_loop_adapter.h
- [x] Add docstring per doxygen with units on NestedLoopAdapter public symbols
- [x] Register test_cpu_nested_loop_adapter in tests/CMakeLists.txt
- [x] Add regression example regressions/architecture/component-plugin-02-wrap/reg_component-plugin-02-wrap.cpp (public API only; hard-coded goldens, no third-party runtime)
- [x] Register atv_add_regression(architecture component-plugin-02-wrap) in regressions/CMakeLists.txt and add REFERENCE.md
- [x] Verify against inner budget=3 and Component-hidden-phase case
- [ ] Run full check + test suite

## Testing strategy

单元测试只放 `tests/cpu/driver/test_nested_loop_adapter.cpp`，夹具 `TestNestedLoopAdapter`，每个 `TEST_F` 只打一个方法。禁止在 `tests/` 里跑完整 SCF/MD。内层用本地 `cpu::Driver` 桩，外层经 `on(phase).add_ref(adapter)` 走 Component 槽。

- `BuildCallsBuildOn` — 外层 `run` 一次后内层 `builds == 1`。
- `LaunchRunsInnerLoopToBudget` — 预算 3，内层 `launches == 3`。
- `LaunchKeepsOuterStageOnComponentContext` — 适配器收到的 `stage` 保持外层绑定相；外层 Plan 上 `component != nullptr && !is_driver()`。
- `DestroyCallsDestroyOn` — `destroys == 1`。
- `provides_*`：内层 `+ HF` 时转发；非 HF 为 false。
- 边沿：`budget < 0` 用 `default_budget()`；空 `unique_ptr` 抛；`static_assert` 双重身份禁令。

回归 `regressions/architecture/component-plugin-02-wrap/`：硬编码金值 — 内层 `launches == 3`、adapter 不是 DriverNode、外层 op `is_driver() == false`。

## Out of scope

- 改 `PhaseBuilder` / 自动包装（`component-plugin-03-slot`）。
- 改 `RHF` / `PW` / `GpwDft` / 任一 `*Evaluator`。
- 让 adapter 再继承 `DriverNode`，或让方法类再继承 `Component`。
- 对 `PlaneWave` / `GPW` / `Semiempirical` 做 cast 转发。
- 把 preset 校验改成只认 `provides_*`（04-match）。
- CUDA 侧适配器。
