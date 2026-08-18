---
spec: component-plugin-01-probe
created: 2026-08-18
criteria:
  - id: ac-001
    summary: Component energy/force probes default false
    type: code
    pass_when: |
      include/atomiverse/cpu/component/component.h declares
      virtual bool provides_energy() const and
      virtual bool provides_force() const, both defaulting to false;
      a local Component stub reports false/false.
    status: pending
  - id: ac-002
    summary: Energy/force kinds default both probes true
    type: code
    pass_when: |
      Potential, Bias, PathCoupling, Semiempirical, PlaneWave, and GPW
      in include/atomiverse/cpu/component/{potential,bias,path_coupling,semiempirical,plane_wave,gpw}.h
      override both probes to return true; local stubs of each kind
      report true/true without any concrete-subclass override.
    status: pending
  - id: ac-003
    summary: HF tag exposes default-true probes and is not a Component
    type: code
    pass_when: |
      include/atomiverse/cpu/component/hf.h declares the same two
      virtuals defaulting to true; HF does not inherit Component;
      MeanFieldMethod is unchanged and is not made a Component;
      a local HF-only stub reports true/true.
    status: pending
  - id: ac-004
    summary: FixLangevin is the sole concrete force-probe override
    type: code
    pass_when: |
      include/atomiverse/cpu/dynamics/fix_langevin.h overrides
      provides_force() to true and does not override provides_energy();
      constructed FixLangevin reports energy=false force=true;
      the Capability::Force comment is rewritten to name provides_force().
    status: pending
  - id: ac-005
    summary: No Capability bitset or reads/capabilities revival
    type: code
    pass_when: |
      cpu::Component gains no Capability enum, no reads(), and no
      capabilities(); the new API is only the two bool predicates.
    status: pending
  - id: ac-006
    summary: Named concretes and validate/Plan/Runtime stay untouched
    type: code
    pass_when: |
      git diff is empty for PairLJCut, VelocityVerlet, RHF, PW, and
      GpwDft sources/headers; validate_components, Plan, and Runtime
      are not edited.
    status: pending
  - id: ac-007
    summary: Unit tests probe one method each under tests/cpu/component
    type: code
    pass_when: |
      tests/cpu/component/test_provides_probes.cpp exists, is registered
      as test_cpu_component_provides_probes in tests/CMakeLists.txt,
      contains single-method TESTs for TestComponent, TestPotential,
      TestBias, TestPathCoupling, TestSemiempirical, TestPlaneWave,
      TestGPW, TestHF, and TestFixLangevin, and introduces no Driver loop.
    status: pending
  - id: ac-008
    summary: Regression prints hard-coded probe bools
    type: runtime
    pass_when: |
      regressions/architecture/component-plugin-01-probe/reg_component-plugin-01-probe.cpp
      (registered via atv_add_regression(architecture component-plugin-01-probe))
      constructs public kind/HF stubs and FixLangevin, and exits 0 only
      when it reproduces the hard-coded literals
      component 0/0, potential 1/1, bias 1/1, path_coupling 1/1,
      semiempirical 1/1, plane_wave 1/1, gpw 1/1, hf 1/1,
      langevin 0/1 (langevin row omitted if ATV_HAS_MD is off).
      No third-party process, no MD/SCF e2e.
    status: pending
out_of_scope:
  - validate_components / name()==langevin deletion
  - Plan / Runtime / scan_pending_infra_ / pending_.clear()
  - PairLJCut / VelocityVerlet / RHF / PW / GpwDft
  - MeanFieldMethod as Component
  - Capability bitset revival
  - CUDA Component/HF mirror
---

# Acceptance — component-plugin-01-probe

探针落在基类与 kind 上；`HF` 仍是纯标签；`FixLangevin` 是唯一具体覆盖。单元测试只打探针；回归打印硬编码布尔。
