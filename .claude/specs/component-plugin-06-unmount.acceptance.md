---
spec: component-plugin-06-unmount
created: 2026-08-18
criteria:
  - id: ac-001
    summary: Unit tests cover Driver::unfix before implementation
    type: code
    pass_when: |
      tests/cpu/driver/test_unfix.cpp exists, is registered as
      test_cpu_driver_unfix in tests/CMakeLists.txt, and defines
      TestDriverUnfix cases that each target a single method
      (unfix or the remount rebind).
    status: pending
  - id: ac-002
    summary: unfix destroys one node and rebinds Plan from live table
    type: code
    pass_when: |
      cpu::Driver<PhaseEnum>::unfix(std::string_view) is declared on
      include/atomiverse/cpu/driver/driver.h; after a successful unfix
      of a built node, that node’s destroy ran once, it is absent from
      pending_/mounted(), and Runtime::destroy_components was not used
      to implement the removal.
    status: pending
  - id: ac-003
    summary: Unfix last energy source throws how-to after committed unmount
    type: code
    pass_when: |
      TestDriverUnfix shows unfix of the sole energy-source stub throws
      std::runtime_error whose what() contains "missing required" and
      either "set_evaluator" or "on("; the stub is already destroyed
      and removed from the live table (no rollback).
    status: pending
  - id: ac-004
    summary: Dependents of an unfixed capability cannot keep running
    type: code
    pass_when: |
      After unfix of a required provider, rebind/closure either throws
      the same missing-require as build() or leaves the dependent out
      of the compiled Plan so it is not launched; remount via
      on(phase).add of a satisfying provider followed by the next run
      chunk restores a legal composition.
    status: pending
  - id: ac-005
    summary: Recorder unfix does not retract already-written bytes
    type: code
    pass_when: |
      TestDriverUnfix uses an in-test fake Recorder that wrote the
      literal "rec-1\n"; after unfix the buffer still equals "rec-1\n"
      and retract_count is 0.
    status: pending
  - id: ac-006
    summary: Live table remains host-observable after unfix and remount
    type: code
    pass_when: |
      After unfix and after remount, Driver::mounted() / is_mounted()
      still answer who is present without a new public register_plugin
      or parallel query type.
    status: pending
  - id: ac-007
    summary: unfix has doxygen and no register_plugin symbol
    type: docs
    pass_when: |
      Driver::unfix has @brief, @param name_or_id, and @throws; grep of
      include/atomiverse/cpu/driver and src/cpu/driver finds no
      register_plugin identifier.
    status: pending
  - id: ac-008
    summary: Regression mid-run unfix then remount reproduces literals
    type: runtime
    pass_when: |
      regressions/architecture/component-plugin-06-unmount/reg_component-plugin-06-unmount.cpp
      (public headers only, fake Component, no third-party) after
      run(frame, 2) has launches==2; unfix("fake_eval") throws with
      "missing required" and "set_evaluator" or "on("; after
      on(md::Phase::force).add of the fake evaluator, run(frame, 2)
      succeeds with launches==4 and prints
      "ok: unfix-remount launches=4".
    status: pending
out_of_scope:
  - CUDA unfix
  - register_plugin
  - retracting Recorder files
  - unfix of auto-injected Zero*/Migrate
---

# Acceptance — component-plugin-06-unmount

步间 unfix 先 destroy 再按活表重编；卸掉最后一个能量源当时给出 how-to；再 add 后同一 Driver 能继续；外部 I/O 不假装撤回。
