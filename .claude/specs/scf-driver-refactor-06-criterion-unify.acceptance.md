---
slug: scf-driver-refactor-06-criterion-unify
criteria:
  - id: ac-001
    summary: Three built-in judges are Criterion components (verdict-only)
    type: code
    evaluator_hint: "inspect: cpu/driver/criteria.h"
    pass_when: |
      ForceConverged, StopOnNaN, and AllReplicasConverged are Criterion subclasses in
      cpu/driver/criteria.h with the verdict() body carried over verbatim, an empty launch(),
      and default_bindings() returning {}; LoopState now lives in cpu/component/criterion.h.
    status: verified
    last_checked: 2026-07-20
  - id: ac-002
    summary: The criterion poll is the only verdict channel
    type: code
    evaluator_hint: "inspect: cpu_driver.h poll"
    pass_when: |
      cpu_driver.h polls only criteria_ per iteration (severity-composed Stop>Converged>Continue,
      single MPI reduce); LoopHandler, add_handler, and poll_handlers_ are deleted along with the
      legacy handler poll path; loop_handler.h is removed and includes reroute to criterion.h/criteria.h.
    status: verified
    last_checked: 2026-07-20
  - id: ac-003
    summary: Preset defaults attach Criteria
    type: code
    evaluator_hint: "inspect: Minimize/PathDriver ctors"
    pass_when: |
      The Minimize ctor attaches ForceConverged and the PathDriver ctor attaches
      AllReplicasConverged via add_criterion (not add_handler); any StopOnNaN default is attached
      the same way.
    note: Minimize's default was already attached in install_defaults() pre-change (guarded by
      has_criterion(), formerly has_handler()); that placement is preserved. PathDriver attaches in
      its ctor. No StopOnNaN default existed.
    status: verified
    last_checked: 2026-07-20
  - id: ac-004
    summary: All add_handler call sites migrated; grep-count-zero
    type: code
    evaluator_hint: "grep: add_handler|LoopHandler"
    pass_when: |
      An evaluator grep for add_handler or LoopHandler over CPU include/, src/, tests/, examples/
      returns zero matches (CUDA loop_handler.cuh excluded); custom LoopHandler subclasses in tests
      and examples are rewritten as Criterion subclasses.
    status: verified
    last_checked: 2026-07-20
  - id: ac-005
    summary: Termination behavior preserved
    type: code
    evaluator_hint: "target: optimize/dynamics/path tests"
    pass_when: |
      test_optimize_driver_early_stop, test_dynamics_full_budget, and test_minimize_driver_swap_optimizer
      show Minimize converging at the same iteration as before; the NEB band terminates identically
      (path tests + neb_muller_brown regression); StopOnNaN still aborts a diverged run.
    status: verified
    last_checked: 2026-07-20
  - id: ac-006
    summary: Nested criterion isolation preserved
    type: code
    evaluator_hint: "target: test_driver_nesting"
    pass_when: |
      An inner loop whose criterion converges early does not stop the outer loop (test_driver_nesting),
      because criteria_ is per-loop and run_nested re-points ctx.loop as before.
    status: verified
    last_checked: 2026-07-20
  - id: ac-007
    summary: Docs migrated from Handler to Criterion
    type: docs
    evaluator_hint: "inspect: CLAUDE.md + concepts + tutorials"
    pass_when: |
      The CLAUDE.md/AGENTS.md custom-handler example (StopWhenTooHot/StopWhenEnergyAbove) is rewritten
      as a custom Criterion, and docs/user-guide/concepts.md + the architecture/geometry tutorials +
      developer docs describe termination as Criteria.
    status: pending
  - id: ac-008
    summary: OutcomePolicy socket untouched
    type: code
    evaluator_hint: "inspect: outcome_policy.h/mc_base.h/gcmc.h"
    pass_when: |
      outcome_policy.h, mc_base.h, and gcmc.h are unchanged; MC accept/reject (test_ensemble_policies)
      is unaffected; the OutcomePolicy socket remains a separate, still-present mechanism.
    status: verified
    last_checked: 2026-07-20
  - id: ac-009
    summary: CUDA termination untouched (deferred)
    type: code
    evaluator_hint: "inspect: cuda/driver/loop_handler.cuh"
    pass_when: |
      cuda/driver/loop_handler.cuh and cuda_driver.cuh handler poll are unchanged; no CUDA source is
      edited by this link (CUDA has no Criterion kind; its criterion migration is deferred).
    status: verified
    last_checked: 2026-07-20
  - id: ac-011
    summary: LoopMetrics renamed ConvergenceMetrics, fields unchanged
    type: code
    evaluator_hint: "grep: LoopMetrics vs ConvergenceMetrics"
    pass_when: |
      facade/loop.h names the metrics struct ConvergenceMetrics with the exact pre-rename
      field set; an evaluator grep finds zero LoopMetrics tokens under CPU include/, src/,
      tests/, examples/ (CUDA excluded); all consumers (built-in criteria, LBFGS/CG publish
      side, ctx.loop.metrics) compile against the new name; SCF code does not reference it.
    status: verified
    last_checked: 2026-07-20
  - id: ac-010
    summary: Whole suite green in seconds
    type: runtime
    evaluator_hint: "target: full ctest"
    pass_when: |
      `cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON
      && cmake --build build -j && ctest --test-dir build --output-on-failure` passes the whole suite
      in seconds; test_criterion passes and no termination-dependent test regresses.
    status: verified
    last_checked: 2026-07-20
---

# Acceptance criteria — scf-driver-refactor-06-criterion-unify

终止即组件：三内置判据 → Criterion、删 `LoopHandler`/`add_handler`、统一 criterion 轮询，终止行为逐位保持。`code`/`docs`/`runtime`。

- **ac-001..004** —— 三判据为 Criterion、单一 criterion 轮询、preset 默认迁移、全 `add_handler` 迁移 grep 清零。
- **ac-005 / ac-006** —— Minimize/NEB/StopOnNaN 终止不变；嵌套隔离保住。
- **ac-007** —— 文档 Handler→Criterion。
- **ac-011** —— `LoopMetrics` 更名 `ConvergenceMetrics`（是什么就叫什么），字段不变，SCF 不引用。
- **ac-008 / ac-009** —— `OutcomePolicy` 与 CUDA 终止不动（未来/延后）。
- **ac-010** —— 整套快套件秒级全绿。

无 `scientific`/`performance`/`ui_runtime`：纯统一、无端到端功能（本链回归在 03b）、无性能预算、无前端。
