---
slug: hf-capability-gaps-15-embed
criteria:
  - id: ac-001
    summary: Internal env classes exist as ctor-validated classes; no build_* factories; public header untouched by internals
    type: code
    pass_when: |
      include/atomiverse/cpu/scf/internal/point_charges.h and external_field.h define
      internal::PointChargeEmbedding / internal::ExternalField with validating constructors;
      `rg 'build_point|build_external' include/ src/` is empty; mean_field_method.h does not
      include either new internal header.
    status: verified
    last_checked: 2026-07-21
  - id: ac-002
    summary: Point-charge core matrix equals nuclear-attraction machinery with Z=q; classical sums exact
    type: runtime
    pass_when: |
      test_cpu_scf_embedding passes: PointChargeEmbedding::core_matrix for one unit charge
      matches the nuclear-attraction machinery with a Z=1 pseudo-atom elementwise to 1e-12,
      and nuclear_energy/self_energy match hand-computed sums to 1e-12 Ha.
    status: verified
    last_checked: 2026-07-21
  - id: ac-003
    summary: External-field origin convention identical to link 03 dipole origin (center of nuclear charge)
    type: runtime
    pass_when: |
      test_cpu_scf_embedding asserts, on an asymmetric molecule, that ExternalField's R0 equals
      the link-03 dipole origin to 1e-12 Bohr, and that E(F)-E(0) at F=1e-4 a.u. equals
      -mu·F to first order (relative error < 1e-2).
    status: verified
    last_checked: 2026-07-21
  - id: ac-004
    summary: Finite-field dipole equals analytic dipole for H2O/STO-3G to 1e-5 a.u.
    type: scientific
    pass_when: |
      regressions/scf/h2o_embedding_field: central-difference dE/dF at F=1e-4 a.u. agrees with
      the link-03 analytic dipole componentwise within 1e-5 a.u.; runtime <= 60 s.
    status: pending
  - id: ac-005
    summary: Point-charge-embedded H2O total energy matches pinned PySCF qmmm.mm_charge to 1e-8 Ha
    type: scientific
    pass_when: |
      regressions/scf/h2o_embedding_field: RHF/STO-3G H2O with the spec's fixed charge set
      matches the externally pinned PySCF value within 1e-8 Ha, with the classical MM-MM
      constant handled explicitly in the assertion.
    status: pending
  - id: ac-006
    summary: forces() with non-empty environment throws; empty environment is a bitwise no-op
    type: runtime
    pass_when: |
      test_cpu_scf_embedding: forces() throws std::runtime_error when point_charges or
      external_field is non-zero; with empty/zero members, Hcore and e_total are unchanged
      (elementwise identical) vs a run without the members set.
    status: verified
    last_checked: 2026-07-21
  - id: ac-007
    summary: Tutorial ch20 environment section with finite-field table and collapse hazard
    type: docs
    pass_when: |
      docs/tutorial/ch20-hf-properties-environment.md contains the 外场与点电荷嵌入 section:
      Fock-operator narrative, finite-field vs analytic dipole table, an explicit finite-field
      variational-collapse warning (~1e-4 a.u. discipline), and chapter-end PySCF scalar rows.
    status: pending
  - id: ac-008
    summary: Full build and unit suite pass; unit suite stays seconds-fast, no e2e in tests/
    type: runtime
    pass_when: |
      cmake --build build succeeds and ctest --test-dir build passes with test_cpu_scf_embedding
      registered; no SCF-to-convergence-vs-external-engine comparison lives under tests/.
    status: verified
    last_checked: 2026-07-21
---

# Acceptance criteria

ac-002/ac-003 钉住积分机器与原点约定(单元级、秒级);ac-004 是本 link 的交叉核验恒等式
(dE/dF == 解析偶极,link 03);ac-005 是外部 oracle(PySCF 数值外部运行后钉定,非运行时依赖);
ac-006 保证无静默错误力与零扰动基线;ac-001/ac-007/ac-008 分别守住架构规则、教程承诺与三层测试纪律。
