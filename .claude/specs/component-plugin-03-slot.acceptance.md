---
spec: component-plugin-03-slot
created: 2026-08-18
criteria:
  - id: ac-001
    summary: Add failing DriverNesting slot-occupancy unit tests
    type: code
    pass_when: |
      tests/cpu/driver/test_driver_nesting.cpp contains new single-function
      TEST(DriverNesting, …) cases that inspect plan slots after
      on(phase).add_ref(inner) / add(inner) and fail until wrap +
      Component*-only occupancy land
    status: pending
  - id: ac-002
    summary: PendingBinding holds only Component*
    type: code
    pass_when: |
      include/atomiverse/core/driver_template.h detail::PendingBinding has
      phase, order, component only; no nested_driver, nested_budget, or
      is_driver(); Driver::build pushes only component PlanBuilder entries
    status: pending
  - id: ac-003
    summary: Plan Operation and Entry hold only Component*
    type: code
    pass_when: |
      include/atomiverse/core/plan.h Operation and PlanBuilder::Entry have
      no nested_driver, nested_budget, or is_driver(); build() always
      reads component->name/scheduling/frequency
    status: pending
  - id: ac-004
    summary: PhaseBuilder wraps DriverNode as NestedLoopAdapter
    type: code
    pass_when: |
      CPU PhaseBuilder::add/add_ref DriverNode overloads construct
      cpu::NestedLoopAdapter (02) and store it as Component* in pending_;
      public signatures stay add(child, order=0, budget=-1) /
      add_ref(child, order=0, budget=-1); no new factory; CUDA DeviceTag
      still has no DriverNode overload
    status: pending
  - id: ac-005
    summary: Pending walks unwrap NestedLoopAdapter for HF
    type: code
    pass_when: |
      include/atomiverse/cpu/driver/driver.h for_each_pending_kind still
      has CompFn+DrvFn signature and calls DrvFn on the adapter's inner
      DriverNode so Driver+HF nodes still count as force sources
    status: pending
  - id: ac-006
    summary: Runtime launches only op.component
    type: code
    pass_when: |
      src/cpu/driver/runtime.cpp set_frame/build/step/run_slot_/destroy
      paths compile with Component*-only Operation and never cast
      op.nested_driver
    status: pending
  - id: ac-007
    summary: Document wrap overloads with Doxygen units
    type: docs
    pass_when: |
      PhaseBuilder DriverNode add/add_ref carry @brief, @param order,
      @param budget (dimensionless iterations) in
      include/atomiverse/core/driver_template.h
    status: pending
  - id: ac-008
    summary: Run architecture regression with hard-coded nest counts
    type: runtime
    pass_when: |
      regressions/architecture/component-plugin-03-slot/reg_component-plugin-03-slot.cpp
      (public headers + link atomiverse) exits 0 when the nested plan slot
      is a NestedLoopAdapter Component* and outer.run(frame, 5) yields
      outer_iters==5, outer.converged()==false, inner.converged()==true;
      no third-party import or subprocess
    status: pending
  - id: ac-009
    summary: Existing nesting and HF-validation unit tests stay green
    type: runtime
    pass_when: |
      ctest -R 'test_cpu_driver_nesting|test_cpu_minimizer_line_search_nested_node|test_cpu_path_validation'
      passes (inner cannot stop outer; run_stage re-enters; default
      budget 20 launches; Minimize/Path still accept Driver+HF)
    status: pending
  - id: ac-010
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      cmake --build build and ctest --test-dir build --output-on-failure
      complete with zero failures after the occupancy change
    status: pending
out_of_scope:
  - unmount API
  - method class changes
  - CUDA DriverNode
---

# Acceptance — component-plugin-03-slot

注册边界自动包装；pending/Plan 只持 Component*；Runtime 只 launch 插件；既有嵌套钉保持绿。
