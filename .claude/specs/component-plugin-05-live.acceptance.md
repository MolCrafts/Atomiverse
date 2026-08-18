---
spec: component-plugin-05-live
created: 2026-08-18
criteria:
  - id: ac-001
    summary: Driver::build keeps pending_ as the live mount table
    type: code
    pass_when: |
      include/atomiverse/core/driver_template.h Driver::build compiles plan_
      from pending_ and does not call pending_.clear().
    status: pending
  - id: ac-002
    summary: Expose instance-identity live-table queries on Driver
    type: code
    pass_when: |
      PluginIdentity keys on const void* instance; Driver::mounted() and
      Driver::is_mounted(PluginIdentity) are public; lookup does not use
      Component::name() or typeid.
    status: pending
  - id: ac-003
    summary: Unit test keeps mounts after build and recompiles equal Plan
    type: code
    pass_when: |
      tests/core/test_driver_live_table.cpp TestDriver.BuildRetainsMountsAndSecondCompileMatchesPlan
      is registered as test_driver_live_table; after build mounted() lists
      both add_ref probes by PluginIdentity{&probe}; second build matches
      saved plan op pointers; two probes share name live_probe but not identity.
    status: pending
  - id: ac-004
    summary: cpu::Driver comments no longer say pending_ is consumed
    type: docs
    pass_when: |
      include/atomiverse/cpu/driver/driver.h build() and requires_tag_index()
      comments describe pending_ as the live table that survives compile,
      not as a list emptied by Base::build().
    status: pending
  - id: ac-005
    summary: Regression compile twice reproduces hard-coded Plan
    type: runtime
    pass_when: |
      regressions/architecture/component-plugin-05-live/reg_component-plugin-05-live.cpp
      exits 0 after cpu::Evaluate::compile(); mounted_count equals 5;
      prepare names are zero_forces, zero_energy, zero_stress; evaluate
      holds two distinct live_probe pointers; second compile() reproduces
      those pointers slot-for-slot (no third-party runtime).
    status: pending
  - id: ac-006
    summary: Inner method pending_.clear sites stay untouched
    type: code
    pass_when: |
      src/cpu/planewave/pw_solver.cpp, src/cpu/gpw/gpw_solver.cpp and
      src/cpu/scf/mean_field_method.cpp still contain their own
      this->pending_.clear() inside reset_built_driver_ (or equivalent).
    status: pending
  - id: ac-007
    summary: No runtime unfix API is added
    type: code
    pass_when: |
      No public Driver/PhaseBuilder unfix, remove, or erase-by-identity
      method is declared in include/atomiverse/core/driver_template.h.
    status: pending
out_of_scope:
  - unfix (06)
  - method-inner pending_.clear()
  - CUDA live table
---

# Acceptance — component-plugin-05-live

pending_ 在 compile 后仍可按实例身份寻址；二次 compile Plan 指针相等；不在本步做 unfix。
