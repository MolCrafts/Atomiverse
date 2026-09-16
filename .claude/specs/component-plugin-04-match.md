---
title: Component plugin provide/require matcher
status: in-progress
created: 2026-08-18
chain: component-plugin
grilled: true
---

# 组件插件 provide/require 匹配器

## Summary

本链第四步把 MD / Evaluate / Minimize / McBase / PathDriver 里复制的族 `dynamic_cast` 清单、`name()=="langevin"` / `"fix_langevin"` 字符串匹配，以及 `DriverNode` 上的 `HF` cross-cast，抽到 `cpu::Driver` 基类的通用 provide/require 扫描。五个 preset 的 `validate_components` 只调用 `require_energy` / `require_force` / `require_energy_or_force` 以及已有 `Integrator` / `Optimizer` / `Mover` kind 对应的 `require_*`。新增一种能量插件只需在 kind 上把 `provides_energy` / `provides_force` 置为 `true`，不必再改五个 preset 源文件。`validate_long_range_contract_` 保持原配对算法。不复活 `Capability` 位集。

## Domain basis

本 spec 不声明新物理。

## Design

前提：01-probe 已落地 `provides_energy()` / `provides_force()`；03 槽单一占位已落地。本 spec 只消费这些探针，不在 `PairLJCut` / `RHF` / `FixLangevin` 上补声明。

**拥有者。** 匹配器属于 `cpu::Driver<PhaseEnum>`，与现有 `scan_pending_infra_`、`throw_if_missing_neighbor_list_`、`throw_missing`、`validate_long_range_contract_` 同级。

**实体。** 扩展已有 `Driver::InfraScan`：

- `has_energy` / `has_force`：来自 `Component::provides_energy()` / `provides_force()`。
- `has_integrator` / `has_optimizer` / `has_mover`：来自已有 kind `dynamic_cast`（若已有同类布尔探针则优先用探针）。
- 既有 infra 字段保持语义。

**扫描。**

1. 增加 **const** 重载 `scan_pending_infra_() const`：用 `for_each_pending_kind` 走完 provide/require 事实。能量源 **禁止** 再扫族类型名，也禁止 `c.name() == "langevin"`。
2. 嵌套节点：经 02/03 适配器展开后读 `provides_*`；若仍需 `HF` 探查，**仅在此 const 扫描里** 保留一处 `dynamic_cast<const HF*>(d)`。五个 preset 不得再写该 cast。
3. 既有 non-const `scan_pending_infra_()` 先调用 const 重载，再补 halo send/recv。
4. `validate_long_range_contract_()` 一字不改算法。

**新符号（protected）。**

| 方法 | 通过条件 | `throw_missing` 文案 |
| --- | --- | --- |
| `require_energy()` | `has_energy` | `"energy source"` |
| `require_force()` | `has_force` | `"force source"` |
| `require_energy_or_force()` | `has_energy \|\| has_force` | `"energy or force source"` |
| `require_integrator()` | `has_integrator` | `"Integrator"` |
| `require_optimizer()` | `has_optimizer` | `"Optimizer"` |
| `require_mover()` | `has_mover` | `"Mover"` |

**Preset 收缩。**

- `MD`：`require_integrator()` + `require_force()`
- `Evaluate`：`require_energy_or_force()`
- `Minimize`：`require_optimizer()` + `require_energy_or_force()`
- `McBase`：`require_mover()` + `require_energy()`
- `PathDriver`：`require_optimizer()` + `require_force()`

五个头里仅为族 cast 而存在的 `#include` 一并删除。

### Reuse decision

- reuse `Component::provides_energy` / `provides_force`
- generalize `Driver::scan_pending_infra_`
- generalize 五个 preset 的 `validate_components`
- reuse `validate_long_range_contract_`
- reuse `throw_missing` / `for_each_pending_kind`
- new — `require_*` 原语

## Files to create or modify

- `include/atomiverse/cpu/driver/driver.h`
- `include/atomiverse/cpu/dynamics/md.h`
- `include/atomiverse/cpu/evaluate/evaluate.h`
- `include/atomiverse/cpu/minimize/minimize.h`
- `include/atomiverse/cpu/montecarlo/mc_base.h`
- `include/atomiverse/cpu/path/path_driver.h`
- `tests/cpu/driver/test_require_match.cpp` (new)
- `tests/cpu/driver_test_helpers.h`
- `tests/cpu/test_cpu_driver_role_validation.cpp`
- `tests/cpu/path/test_path_validation.cpp`
- `tests/cpu/test_mc_scaffold.cpp`
- `tests/CMakeLists.txt`
- `regressions/architecture/component-plugin-04-match/reg_component-plugin-04-match.cpp` (new)
- `regressions/architecture/component-plugin-04-match/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`

## Tasks

- [ ] Write failing unit tests for `require_*` / const `scan_pending_infra_` (tests/cpu/driver/test_require_match.cpp → TestRequireMatch; register via `atv_add_cpp_test` in tests/CMakeLists.txt; add Component-only probe stubs in tests/cpu/driver_test_helpers.h)
- [ ] Write failing updates for throw needles and probe-only Evaluate in tests/cpu/test_cpu_driver_role_validation.cpp, tests/cpu/path/test_path_validation.cpp, tests/cpu/test_mc_scaffold.cpp
- [ ] Generalize `scan_pending_infra_` in include/atomiverse/cpu/driver/driver.h to a provide/require scan and implement `require_*` with doxygen
- [ ] Generalize `validate_components` in md.h, evaluate.h, minimize.h, mc_base.h, path_driver.h to call `require_*` only; drop copied kind-list includes
- [ ] Add regression example regressions/architecture/component-plugin-04-match/
- [ ] Verify against probe-only Evaluate::build, nested HF node, `provides_force` Langevin without `name()` match, and missing-role throws
- [ ] Run full check + test suite

## Testing strategy

`TestRequireMatch`：直接继承 `Component` 且 `provides_energy()==true` 的桩，经 `Evaluate::build()` 通过；裸 Component 抛 `"energy or force source"`。回归打印字面量 `ok: probe-only energy source accepted by Evaluate::build`。

## Out of scope

- Hessian / Phonon 里仍复制的族清单。
- CUDA preset。
- unmount。
- 改具体求值器。
- 复活 Capability 位集。
- 重写 long-range 配对。
