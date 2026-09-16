---
title: Component energy/force probes
status: in-progress
created: 2026-08-18
chain: component-plugin
grilled: true
---

# Component energy/force probes

## Summary

本步在 CPU `Component` 探针面上落地两个布尔谓词 `provides_energy()` / `provides_force()`：基类默认 `false`，能量/力 kind（`Potential`、`Bias`、`PathCoupling`、`Semiempirical`、`PlaneWave`、`GPW`）在 kind 基类给出默认 `true`，纯标签 `HF` 带同样的默认 `true` 虚函数供下一步 adapter 转发，`FixLangevin` 作为唯一具体覆盖声明写力。本步不改 `validate_components`、不改 Plan、不改 Runtime，也不改 `PairLJCut` / `VelocityVerlet` / `RHF` / `PW` / `GpwDft`；后续步骤才能用探针替换 kind 的 `dynamic_cast` 列表和 `name()=="langevin"` 字符串洞。

## Domain basis

本 spec 不声明新物理。探针是无量纲布尔谓词，不引入能量、力、温度或其它物理量，也不引用文献。`FixLangevin` 已有的 Schneider–Stoll 力公式保持原样，本步只改它对外声明的探针。

## Design

现有 `cpu::Component` 已有 `as_*` / `requires_*` / `provides_*` 谓词（`provides_system_stress`、`requires_tag_index` 等）：无载荷的能力用谓词，有协议对象的用 `as_*`。本步按同一形状增加两个谓词，**禁止**复活 `Capability` 位集，**禁止**复活 `reads()` / `capabilities()`。

**实体与符号**

- `cpu::Component`（`component.h`）：新增
  `virtual bool provides_energy() const { return false; }`
  `virtual bool provides_force() const { return false; }`
  与现有 `provides_system_stress()` 并列。Doxygen `@brief` / `@return` 标明无量纲布尔；无物理单位。
- kind 基类覆盖为默认 `true`（两边都 true）：`Potential`、`Bias`、`PathCoupling`、`Semiempirical`、`PlaneWave`、`GPW`。形状照抄 `Observer::role()`（kind 给默认）和 `PairPotential::as_neighbor_consumer()`（kind 声明协议，具体 pair 零改）。
- `cpu::HF`（`hf.h`）**不是** `Component`，保持纯多态标签。在标签上放同样的默认 `true` 虚函数，供下一步 adapter 从嵌套 `DriverNode` 转发。**不要**把 `MeanFieldMethod` 改成 `Component`。`RHF` 已是 `MeanFieldMethod<RHF> : Driver + HF + …`，继承标签虚函数即可，本步零改。
- 其余 kind（`Integrator`、`Thermostat`、`Barostat`、`Optimizer`、`Constraint`、`Mover`、`Observer`、`Recorder`、`Criterion`）不覆盖，保持基类 `false`。因此 `VelocityVerlet : Integrator` 无需改文件即为 `false`/`false`；`PairLJCut : Pair → PairPotential → Potential` 无需改文件即为 `true`/`true`。
- `PW` / `GpwDft` 本身是 `Driver<pw::Phase>` / `Driver<gpw::Phase>`，**不**继承 `PlaneWave` / `GPW`；继承 kind 的是 `PWEvaluator` / `GPWEvaluator`。本步不给这两个方法 Driver 加探针，也不改它们的继承。下一步 adapter 再接线。

**唯一具体覆盖（证明“抽到 kind，而不是每个子类”）**

`include/atomiverse/cpu/dynamics/fix_langevin.h` 的 `cpu::md::FixLangevin : Thermostat` 是本步允许改动的**唯一**具体类。`Thermostat` 默认 `false`（温控不是能量/力源）；Langevin 在 `post_force` 写入拖曳 + 随机力，所以覆盖 `provides_force() → true`。它不写势能（无 `tally`），**不**覆盖 `provides_energy()`，保持 `false`。这正是“抽到 kind，而不是每个子类”的反例：不为了一个写力的温控把整个 `Thermostat` 改成 `true`，也不给每个 pair / 积分器加覆盖。现有注释里的 `Capability::Force` 改为指向 `provides_force()`。本步仍不改 `MD::validate_components` 里的 `c.name() == "langevin" || c.name() == "fix_langevin"`；探针落地后，后续 spec 才能删掉那条字符串洞。

**生命周期 / 所有权**

无新类型、无工厂、无 `*Config`、无插件注册表。谓词是 `const`、无分配、不进入 `build`/`launch`/`destroy`。本步不碰 `scan_pending_infra_`、`validate_long_range`、MD/Evaluate 的校验列表、`pending_.clear()`。

**Reuse decision**

- reuse `cpu::Component` 上已有的 `as_*` / `requires_*` / `provides_*` 探针形状（`provides_system_stress`、`requires_tag_index`）。
- reuse `PairPotential::as_neighbor_consumer()` 的 kind 默认模式（具体 `PairLJCut` 零改）。
- reuse `Observer::role()` 的 kind 默认模式。
- reuse `cpu::HF` 纯标签模式（虚析构 + 虚函数，不继承 `Component`）。
- new — `scan_pending_infra_`、`validate_long_range`、MD/Evaluate 校验列表、`pending_.clear()` 属于后续链（本步明确不改校验 / Plan / Runtime）。
- new — `molrs_contract::required_capabilities()` 是 CXX FFI 版本掩码，不是 Component 能量/力探针。
- new — 不发明 plugin registry；探针是现有类型上的虚函数。

## Files to create or modify

- `include/atomiverse/cpu/component/component.h`
- `include/atomiverse/cpu/component/potential.h`
- `include/atomiverse/cpu/component/bias.h`
- `include/atomiverse/cpu/component/path_coupling.h`
- `include/atomiverse/cpu/component/semiempirical.h`
- `include/atomiverse/cpu/component/plane_wave.h`
- `include/atomiverse/cpu/component/gpw.h`
- `include/atomiverse/cpu/component/hf.h`
- `include/atomiverse/cpu/dynamics/fix_langevin.h`
- `tests/cpu/component/test_provides_probes.cpp` (new)
- `tests/CMakeLists.txt`
- `regressions/architecture/component-plugin-01-probe/reg_component-plugin-01-probe.cpp` (new)
- `regressions/architecture/component-plugin-01-probe/REFERENCE.md` (new)
- `regressions/CMakeLists.txt`

## Tasks

- [x] Write failing unit tests for `provides_energy` / `provides_force` (tests/cpu/component/test_provides_probes.cpp → TestComponent, TestPotential, TestBias, TestPathCoupling, TestSemiempirical, TestPlaneWave, TestGPW, TestHF, TestFixLangevin) and register `test_cpu_component_provides_probes` in tests/CMakeLists.txt
- [x] Implement `Component::provides_energy` / `Component::provides_force` default false in include/atomiverse/cpu/component/component.h with Doxygen `@brief` / `@return` (dimensionless bool)
- [x] Implement kind-default true overrides in include/atomiverse/cpu/component/potential.h, bias.h, path_coupling.h, semiempirical.h, plane_wave.h, gpw.h
- [x] Implement the same default-true virtuals on the `HF` tag in include/atomiverse/cpu/component/hf.h (do not inherit Component; do not change MeanFieldMethod)
- [x] Override `FixLangevin::provides_force` to true in include/atomiverse/cpu/dynamics/fix_langevin.h; leave `provides_energy` false; rewrite the `Capability::Force` comment to name the probe
- [x] Add regression example regressions/architecture/component-plugin-01-probe/reg_component-plugin-01-probe.cpp (public API only; hard-coded goldens, no third-party runtime) plus REFERENCE.md, and register `atv_add_regression(architecture component-plugin-01-probe)` in regressions/CMakeLists.txt
- [ ] Run full check + test suite

## Testing strategy

单元测试只打探针、单函数，放在 `tests/cpu/component/test_provides_probes.cpp`（镜像 `include/atomiverse/cpu/component/`）。用本地桩实例化抽象 kind（实现 `name()` + `launch()`），不要拉 MD/SCF 回路。每个 `TEST` 只断言一个方法。gtest 套件名即类型：`TestComponent`、`TestPotential`、…。

- 快乐路径：`Component` 桩 `provides_energy()` / `provides_force()` 均为 `false`；`Potential` / `Bias` / `PathCoupling` / `Semiempirical` / `PlaneWave` / `GPW` 桩两者均为 `true`。
- 边沿：`HF` 桩（只继承 `cpu::HF`，不是 `Component`）两者均为 `true`；编译期确认 `HF` 不是 `Component` 子类。`FixLangevin{T, T, damp, dt, seed}`（`#ifdef ATV_HAS_MD`）`provides_force()==true` 且 `provides_energy()==false`。一个非能量 kind 桩（直接继承 `Component` 或 `Integrator`）保持双 `false`，证明未抽到每个子类。
- 域校验：无。期望值是硬编码布尔字面量 `false` / `true`，不是物理量。
- 禁止：在 `tests/` 下跑 Evaluate/MD/SCF、改 `validate_components` 的行为测试、构造完整 `PairLJCut` / `RHF` / `PW` / `GpwDft` 求解。

回归例子：`regressions/architecture/component-plugin-01-probe/reg_component-plugin-01-probe.cpp`。最小公共 API：用公开头构造 kind 桩 + `HF` 桩 +（`ATV_HAS_MD` 时）`cpu::md::FixLangevin`，打印并核对硬编码布尔。不是 MD e2e。期望字面量：

```
component energy=0 force=0
potential energy=1 force=1
bias energy=1 force=1
path_coupling energy=1 force=1
semiempirical energy=1 force=1
plane_wave energy=1 force=1
gpw energy=1 force=1
hf energy=1 force=1
langevin energy=0 force=1
```

（无 MD 时省略 `langevin` 行。）`REFERENCE.md` 记录上述字面量；第三方软件不参与。

## Out of scope

- 改 `validate_components`（MD / Evaluate / Minimize / Path / MC）或删 `name()=="langevin"` 字符串洞——下一步。
- 改 Plan、Runtime、`scan_pending_infra_`、`validate_long_range`、`pending_.clear()`。
- 改 `PairLJCut`、`VelocityVerlet`、`RHF`、`PW`、`GpwDft`（含给 `PW`/`GpwDft` 加 `PlaneWave`/`GPW` 继承）。
- 把 `MeanFieldMethod` 改成 `Component`。
- 复活 `Capability` 位集、`reads()`、`capabilities()`。
- 发明 plugin registry；复用 `molrs_contract::required_capabilities`。
- CUDA `cuda::Component` / `cuda::HF` 镜像（本步仅 `cpu/component` + 一个 `FixLangevin` 例外）。
- 给整个 `Thermostat` kind 默认 `true`，或给其它具体子类加覆盖。
- 前端 / UI。
