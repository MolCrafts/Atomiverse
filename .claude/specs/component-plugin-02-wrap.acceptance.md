---
spec: component-plugin-02-wrap
created: 2026-08-18
criteria:
  - id: ac-001
    summary: Unit tests cover each NestedLoopAdapter method
    type: code
    pass_when: |
      tests/cpu/driver/test_nested_loop_adapter.cpp defines fixture
      TestNestedLoopAdapter and one TEST_F per method: BuildCallsBuildOn,
      LaunchRunsInnerLoopToBudget, LaunchKeepsOuterStageOnComponentContext,
      DestroyCallsDestroyOn, ProvidesEnergyWhenInnerIsHf,
      ProvidesEnergyFalseWhenInnerIsNotHf (and matching ProvidesForce*).
    status: pending
  - id: ac-002
    summary: Adapter is Component, never DriverNode
    type: code
    pass_when: |
      NestedLoopAdapter publicly derives cpu::Component and
      std::is_base_of_v<cpu::DriverNode, cpu::NestedLoopAdapter> is false
      in the unit test and the regression binary.
    status: pending
  - id: ac-003
    summary: launch runs the inner loop to the ctor budget
    type: code
    pass_when: |
      TestNestedLoopAdapter.LaunchRunsInnerLoopToBudget is green: wrapping
      a stub Driver with NestedLoopAdapter(inner, 3) yields exactly 3
      inner-component launches via the Component slot.
    status: pending
  - id: ac-004
    summary: build and destroy forward to the node once
    type: code
    pass_when: |
      TestNestedLoopAdapter.BuildCallsBuildOn and DestroyCallsDestroyOn
      are green: one outer run produces inner builds==1 and destroys==1;
      Driver::build() alone does not increment builds.
    status: pending
  - id: ac-005
    summary: provides_energy/force forward only through HF
    type: code
    pass_when: |
      provides_energy() and provides_force() are true iff
      dynamic_cast<const cpu::HF*>(inner) succeeds and the HF predicates
      return true; a non-HF DriverNode yields false/false. No PlaneWave,
      GPW, or Semiempirical cast appears in the adapter.
    status: pending
  - id: ac-006
    summary: Component LaunchContext never shows inner Phase
    type: code
    pass_when: |
      TestNestedLoopAdapter.LaunchKeepsOuterStageOnComponentContext is
      green: the LaunchContext passed to NestedLoopAdapter::launch keeps
      stage == slot_of(the outer binding) for the whole call; the outer
      Plan operation for the adapter has component != nullptr and
      is_driver() == false.
    status: pending
  - id: ac-007
    summary: Own and borrow constructors match add/add_ref
    type: code
    pass_when: |
      NestedLoopAdapter(DriverNode&, int) borrows and
      NestedLoopAdapter(unique_ptr<DriverNode>, int) owns; a null
      unique_ptr throws std::invalid_argument. budget < 0 is stored as
      inner.default_budget() at construction.
    status: pending
  - id: ac-008
    summary: Regression reproduces budget=3 Component-wrap goldens
    type: runtime
    pass_when: |
      regressions/architecture/component-plugin-02-wrap/reg_component-plugin-02-wrap.cpp
      exits 0 and reproduces the hard-coded literals: inner launches == 3,
      is_base_of DriverNode == false, outer adapter Operation::is_driver()
      == false. No third-party binary is invoked.
    status: pending
  - id: ac-009
    summary: Public adapter symbols have Doxygen docs
    type: docs
    pass_when: |
      NestedLoopAdapter and each public constructor/method in
      include/atomiverse/cpu/driver/nested_loop_adapter.h carry @brief
      and @param (budget is dimensionless iteration count) / @return
      as required by the project Doxygen style.
    status: pending
out_of_scope:
  - PhaseBuilder auto-wrap (03-slot)
  - method class identity changes
  - CUDA adapter
---

# Acceptance — component-plugin-02-wrap

宿主适配器是 Component、不是 DriverNode；launch 跑满预算且外层看不到内层 Phase；HF 探针只在适配器内转发。
