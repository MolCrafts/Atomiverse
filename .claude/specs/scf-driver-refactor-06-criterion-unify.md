---
title: SCF 驱动化重构 · 06 终止即组件统一（Handler→Criterion，删 LoopHandler/add_handler）
slug: scf-driver-refactor-06-criterion-unify
status: code-complete
chain: scf-driver-refactor
link: 06/08
created: 2026-07-20
---

# SCF 驱动化重构 · 06 终止即组件统一

## Summary
落实新用户裁定：**终止判据是 Component（Criterion），Handler 概念退役**。把三个内置终止判据 `ForceConverged`/`StopOnNaN`/`AllReplicasConverged`（现 `loop_handler.h:91/160/138`）迁为 `Criterion` 组件（verdict-only、无 launch、无 phase 绑定），迁移 preset 默认（`Minimize` 构造器挂 `ForceConverged`、`PathDriver` 挂 `AllReplicasConverged`）与全部 `add_handler` 调用点（测试/示例/文档）；**删除 `LoopHandler` 基 + `add_handler` + `poll_handlers_` 遗留轮询**，让 01 引入的 criterion 轮询成为唯一 verdict 通道；删 `loop_handler.h`。`ScfConverged`（02）本已是 Criterion，不涉。`OutcomePolicy`（accept/reject 插槽）**保持不动**（未来统一候选，出局）。CUDA 自有 `cuda::LoopHandler` 保持不动（CUDA 无 Criterion 机制，延后）。终止行为逐位保持：Minimize 照旧收敛、NEB band 照旧终止、StopOnNaN 照旧中止。

## Domain basis
无物理；领域锚点是**行为保持**——迁移前后终止时机逐位一致（五判据几何收敛、整 band 收敛、能量非有限中止）。

架构裁定（新用户规则，记入设计根据）：**"该不该停"从来不是一个独立的类层级，而是一个判据组件**。旧设计 `LoopHandler` 是一个非组件的策略对象、由 driver 私有 `poll_handlers_` 轮询；新设计里判据就是 `Criterion : Component`（verdict-only），与其它组件同生命周期、同所有权规则，由 driver 每迭代拉取 `verdict()`（severity 组合 Stop>Converged>Continue、单次 MPI 归约）。这消除了"终止是一等公民但不是组件"的最后一处例外——`ForceConverged` 服务 Minimize、`StopOnNaN` 服务 MD/Minimize/Path/MC、`AllReplicasConverged` 服务 Path，现皆为组件。沿用整链决策日志①（一切驱动循环皆 Driver、终止是可组合策略）与决策⑨。

## Design

**验证内部：** `loop_handler.h` 有 `LoopHandler` 基（`:66`）、`LoopState`（`:56`）、三个内置判据（`ForceConverged:91`、`AllReplicasConverged:138`、`StopOnNaN:160`），均读 `LoopMetrics`（`facade/loop.h`）返回 `Verdict`。01 已在 `cpu/component/criterion.h` 落 `class Criterion : public Component`（纯虚 `Verdict verdict(const LoopState&)`），CPUDriver 已有 `add_criterion`/`add_criterion_ref` + 每迭代 criterion 轮询，与遗留 handler 轮询**并存至本环**。

**1. 三内置判据迁为 Criterion（置于 `cpu/driver/criteria.h`，新）：** 每个由 `LoopHandler` 子类改为 `Criterion` 子类——`verdict(const LoopState&)` 体照搬（读同样的 `LoopMetrics`，返回同样的 `Verdict`），并实现 `Criterion` 契约：`launch()` 空体、`default_bindings()` 返回 `{}`（verdict-only、不入 phase plan）。**置于 `cpu/driver/` 而非 `cpu/component/`——理由：** `Criterion` 的**基类（kind）**在 `cpu/component/criterion.h`（组件种类声明，与 Potential/Observer 等并列）；但这三个是 **loop 节点的内置终止默认**（Minimize/Path 在构造器里挂它们），归属与旧 `loop_handler.h` 相同的 driver 层最自然，且保住其"一个判据服务所有工作流"的跨工作流通用性（`StopOnNaN` 对 MD/Minimize/Path/MC 一视同仁）。`ScfConverged`（SCF 专属、flag-gated）仍留 `cpu/scf/`。

**2. `LoopState` 迁移：** 从 `loop_handler.h:56` 迁入 `cpu/component/criterion.h`（`Criterion::verdict` 的入参），随 `loop_handler.h` 删除。

**3. preset 默认迁移：** `Minimize` 构造器 `add_handler(ForceConverged{...})`→`add_criterion(ForceConverged{...})`；`PathDriver` 构造器 `add_handler(AllReplicasConverged{})`→`add_criterion(...)`；任何 `StopOnNaN` 默认挂载同改。

**4. 全部 `add_handler` 调用点迁移：** 测试（`test_loop_handler`→`test_criterion`、`test_optimize_driver_early_stop`、`test_dynamics_full_budget`、`test_minimize_driver_swap_optimizer`、minimizer 测试、04a 已迁的 `test_driver_nesting`）、示例（`05_geom_opt`、`15_am1` 等含自定义 handler 处）全部 `add_handler`→`add_criterion`、自定义 `LoopHandler` 子类改 `Criterion` 子类。

**5. 删除遗留通道 + 统一轮询（`cpu_driver.h`）：** 删 `LoopHandler` 基、`add_handler`/`add_handler_ref`、`poll_handlers_` 及遗留 handler 轮询；criterion 轮询（01）成为唯一 verdict 通道（severity 组合 + 单 MPI 归约不变）；删 `loop_handler.h`，把其消费方 `#include` 重指 `criterion.h`。criterion 列表 `criteria_` 由 loop 节点自持，`run_nested` 的 `ctx.loop` 重指隔离（child 只轮询自己的 `criteria_`，不能停父）与旧 handler 轮询同构，原样保住。

**6. `LoopMetrics` → `ConvergenceMetrics` 诚实更名（用户裁定"是什么就用什么命名"）：** `facade/loop.h` 的 `LoopMetrics` 承载的就是收敛判定量（`max_force`/`rms_force`/位移/ΔE 等，字段名本已诚实），类型名更为 `ConvergenceMetrics`，全部消费方（三内置 Criterion、LBFGS/CG 的发布侧、`ctx.loop.metrics`）机械同步；字段不增不删。SCF 不使用它（`ScfConverged` 自持真名成员，02 已定）。

**7. 文档（flagged docs）：** `CLAUDE.md`/`AGENTS.md` 的 `StopWhenTooHot`/`StopWhenEnergyAbove` 自定义示例改写为自定义 `Criterion`；`docs/user-guide/concepts.md`、相关 tutorial（`ch12-geometry-optimization`、`ch00-architecture` 等）与 `docs/developer/*` 里 Handler 叙述改 Criterion。

**Reuse decision：**
- `reuse` `Criterion` 基 + `add_criterion` + criterion 轮询（01）、`LoopMetrics`/`Verdict`（facade/loop.h）—— 三内置判据只换基类、`verdict` 体照搬。
- `reuse` `run_nested` 的 `ctx.loop` 隔离 —— criterion 轮询继承同一 per-loop 所有权。
- `generalize` `cpu_driver.h` 轮询 —— 双通道（handler+criterion）收敛为单 criterion 通道。
- `delete`：`LoopHandler`、`add_handler`、`poll_handlers_`、`loop_handler.h`。
- `new` —— `cpu/driver/criteria.h`（三内置 Criterion）+ `test_criterion.cpp`；无 `make_*`、无新 `.sh` 门。
- **不动**：`OutcomePolicy`（`outcome_policy.h`/`mc_base.h`/`gcmc.h`，accept/reject 插槽，未来候选）；CUDA `loop_handler.cuh`/`cuda_driver.cuh`（CUDA 无 Criterion，延后）；`ScfConverged`（02 已是 Criterion）。

## Files to create or modify
- `include/atomiverse/cpu/driver/criteria.h` (new) — `ForceConverged`/`StopOnNaN`/`AllReplicasConverged` 作 `Criterion` 子类。
- `include/atomiverse/cpu/component/criterion.h` (01) — 接纳 `LoopState`。
- `include/atomiverse/cpu/driver/loop_handler.h` (delete)。
- `include/atomiverse/cpu/driver/cpu_driver.h` — 删 `LoopHandler`/`add_handler`/`poll_handlers_` + 遗留轮询；统一 criterion 轮询。
- `include/atomiverse/cpu/minimizer/*`（Minimize preset 构造器）、Path preset 构造器 — `add_handler`→`add_criterion`。
- 消费方 `#include` 重指：`cpu/md/md.h`、`cpu/minimizer/lbfgs.h`、`src/cpu/minimizer/lbfgs.cpp` 等含 `loop_handler.h` 处 → `criterion.h`/`criteria.h`。
- 测试：`tests/cpu/driver/test_loop_handler.cpp`→`test_criterion.cpp`、`tests/cpu/test_optimize_driver_early_stop.cpp`、`tests/cpu/test_dynamics_full_budget.cpp`、`tests/cpu/test_minimize_driver_swap_optimizer.cpp`、`tests/cpu/minimizer/test_cg.cpp`、`tests/cpu/minimizer/test_trajectory_writer.cpp` — `add_handler`→`add_criterion`。
- 示例：`examples/05_geom_opt.cpp`、`examples/15_am1.cpp` 等 `add_handler` 处。
- 文档（docs）：`CLAUDE.md`/`AGENTS.md`（自定义 handler 示例）、`docs/user-guide/concepts.md`、`docs/tutorial/{ch00-architecture,ch12-geometry-optimization}.md`、`docs/developer/*`。
- `tests/CMakeLists.txt` — `test_criterion` 注册。

## Tasks
- [x] Migrate tests/cpu/driver/test_loop_handler.cpp to a failing tests/cpu/driver/test_criterion.cpp asserting the three built-in Criterion verdicts + severity composition + Minimize/Path terminate identically; update tests/CMakeLists
- [x] Migrate ForceConverged/StopOnNaN/AllReplicasConverged to Criterion components in cpu/driver/criteria.h (verdict body verbatim, empty launch, no bindings) and relocate LoopState into cpu/component/criterion.h
- [x] Migrate preset defaults (Minimize ctor ForceConverged, PathDriver ctor AllReplicasConverged, any StopOnNaN default) from add_handler to add_criterion
- [x] Migrate every remaining add_handler call site + custom LoopHandler subclass in tests and examples to add_criterion / Criterion
- [x] Delete LoopHandler + add_handler + poll_handlers_ and the legacy handler poll, unify cpu_driver.h to the single criterion poll, delete loop_handler.h, and reroute includes to criterion.h/criteria.h
- [x] Rename LoopMetrics to ConvergenceMetrics (facade/loop.h + all consumers, mechanical, fields unchanged)
- [x] Update the docs (CLAUDE.md/AGENTS.md custom-handler example, concepts + tutorials + developer docs) from Handler to Criterion — docs
- [x] Run full check + test suite
- [x] Hygiene cleanup (/mol:simplify): 19 janitor fixes applied — stale handler prose, test-name drift, include precision, formatting; deferred greens are pre-existing debt

## Testing strategy
- **判据 verdict + 组合（单元，秒级）**：`test_criterion` 断言三内置 Criterion 的 `verdict()` 与旧 handler 逐位一致；severity 组合 Stop>Converged>Continue、单 MPI 归约不变。
- **行为保持**：`test_optimize_driver_early_stop`/`test_dynamics_full_budget`/`test_minimize_driver_swap_optimizer` 下 Minimize 收敛时机不变；NEB band（`regressions/path/neb_muller_brown` + path 测试）终止不变；`StopOnNaN` 仍中止发散 run。
- **嵌套隔离**：内层 criterion 收敛仍不能停外层（`test_driver_nesting`，继承 `ctx.loop` 重指）。
- **债务终结（grep）**：`LoopHandler`/`add_handler`/`poll_handlers_` 在 CPU `include/ src/ tests/ examples/` 计数为零；`loop_handler.h` 已删。
- **不动项**：`OutcomePolicy`（MC accept/reject 不受影响，`test_ensemble_policies` 绿）；CUDA `loop_handler.cuh`（CUDA 测试不受影响）；`ScfConverged`（02，已 Criterion）。
- **Domain validation**：终止时机逐位不变即验证。
- **Regression example**：本环**不**含新 `regressions/` 案例——纯统一；既有 `neb_muller_brown` 回归即行为守卫。runtime 关口是完整 ctest 快套件。

## Out of scope
- `OutcomePolicy`（accept/reject 插槽）统一 —— 未来候选，不动。
- CUDA `cuda::LoopHandler`/`cuda_driver.cuh` → Criterion —— CUDA 无 Criterion 机制，延后（GPU 节点工作）。
- `ScfConverged` —— 02 已是 Criterion。
