---
spec: component-plugin-04-match
created: 2026-08-18
criteria:
  - id: ac-001
    summary: TestRequireMatch covers require_* and probe-only Evaluate
    type: code
    pass_when: |
      tests/cpu/driver/test_require_match.cpp exists, is registered
      with atv_add_cpp_test in tests/CMakeLists.txt, and contains
      single-method cases for require_energy, require_force,
      require_energy_or_force, require_integrator, require_optimizer,
      and require_mover, including a Component-only stub that is not
      Potential/Bias/HF/Semiempirical/PlaneWave/GPW/PathCoupling.
    status: pending
  - id: ac-002
    summary: Existing validation tests expect energy/force wording
    type: code
    pass_when: |
      tests/cpu/test_cpu_driver_role_validation.cpp and
      tests/cpu/path/test_path_validation.cpp assert missing-source
      messages that mention energy and/or force (or Integrator /
      Optimizer) rather than the old Potential/Bias/HF type list.
    status: pending
  - id: ac-003
    summary: Driver scan records provide/require flags via probes
    type: code
    pass_when: |
      include/atomiverse/cpu/driver/driver.h InfraScan has has_energy,
      has_force, has_integrator, has_optimizer, has_mover; const
      scan_pending_infra_() sets energy/force from provides_* (and at
      most one DriverNode HF cross-cast); it does not dynamic_cast
      Potential/Bias/PathCoupling/Semiempirical/PlaneWave/GPW and does
      not compare name() to langevin or fix_langevin.
    status: pending
  - id: ac-004
    summary: Driver exposes require_* helpers using throw_missing
    type: code
    pass_when: |
      include/atomiverse/cpu/driver/driver.h declares protected const
      require_energy, require_force, require_energy_or_force,
      require_integrator, require_optimizer, require_mover; each
      throws via throw_missing with the messages energy source,
      force source, energy or force source, Integrator, Optimizer,
      Mover respectively; each has a doxygen @brief.
    status: pending
  - id: ac-005
    summary: Five presets validate only by calling require_*
    type: code
    pass_when: |
      validate_components in include/atomiverse/cpu/dynamics/md.h,
      evaluate/evaluate.h, minimize/minimize.h, montecarlo/mc_base.h,
      and path/path_driver.h call the matching require_* only
      and contain no family dynamic_cast list and no name()==langevin match.
    status: pending
  - id: ac-006
    summary: Long-range pairing algorithm stays untouched
    type: code
    pass_when: |
      Driver::validate_long_range_contract_ still enforces exactly
      one real-space plus one reciprocal LongRangeParticipant with
      matching alpha; require_* does not implement or replace that
      pairing.
    status: pending
  - id: ac-007
    summary: Capability bitset is not reintroduced
    type: code
    pass_when: |
      No new Capability enum, bitset, or capabilities() write-set
      appears in include/atomiverse/cpu/driver/driver.h or the five
      preset validate_components bodies.
    status: pending
  - id: ac-008
    summary: Nested HF node still satisfies Path and MC validation
    type: runtime
    pass_when: |
      ctest --test-dir build -R
      'test_cpu_path_validation|test_cpu_mc_scaffold|test_cpu_driver_role_validation|test_cpu_require_match'
      --output-on-failure exits 0.
    status: pending
  - id: ac-009
    summary: Probe-only fake energy component builds Evaluate
    type: runtime
    pass_when: |
      regressions/architecture/component-plugin-04-match/reg_component-plugin-04-match.cpp
      binds a local Component that is not on the old kind list and only
      overrides provides_energy() to true, Evaluate::build() succeeds, and
      the process prints the literal
      "ok: probe-only energy source accepted by Evaluate::build"
      and exits 0.
    status: pending
out_of_scope:
  - Hessian/Phonon copied lists
  - CUDA presets
  - unmount
  - Capability bitset
---

# Acceptance — component-plugin-04-match

五个 preset 只调 require_*；新能量插件只需 kind 探针，不必改 preset。
