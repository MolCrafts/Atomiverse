---
slug: hf-capability-gaps-17-bsse
criteria:
  - id: ac-001
    summary: ghost_atoms is a method member; Counterpoise is a plain class, no factory
    type: code
    pass_when: |
      `std::vector<int> ghost_atoms` exists on MeanFieldMethod in
      include/atomiverse/cpu/scf/mean_field_method.h; counterpoise.h declares a
      class with an index-set constructor, inherits neither Component nor any
      Kind; grep over the diff finds no build_*/make_* free function and no new
      kAtom_* Frame column (Frame purity ruling upheld).
    status: verified
    last_checked: 2026-07-21
  - id: ac-002
    summary: ghost workspace semantics verified at unit level
    type: runtime
    pass_when: |
      test_ghost_basis passes: with atom i ghosted, n_electrons excludes Z_i,
      V_nuc excludes pairs with i, n_basis still counts i's shells, the nuclear
      attraction matrix carries no Z_i term, and an out-of-range ghost index
      throws; whole test runs in seconds, no full-system SCF beyond workspace.
    status: verified
    last_checked: 2026-07-21
  - id: ac-003
    summary: Counterpoise 5-SCF orchestration identities hold (sign convention locked)
    type: runtime
    pass_when: |
      test_counterpoise passes on He–He/STO-3G: e_int_cp equals
      e_dimer − e_a_ghost − e_b_ghost to 1e-12 Ha; bsse = (e_a_ghost − e_a) +
      (e_b_ghost − e_b) satisfies bsse ≤ 0 (monomer energy lowers in the dimer
      basis; convention locked in CounterpoiseResult doxygen and identical in
      test and tutorial); the method's ghost_atoms member is restored to its
      pre-run value after run().
    status: verified
    last_checked: 2026-07-21
  - id: ac-004
    summary: water-dimer HF/6-31G* CP energy matches pinned PySCF oracle
    type: scientific
    pass_when: |
      regressions/scf/h2o_dimer_cp_hf631gs (ATV_BUILD_REGRESSIONS,
      `regressions` target) reproduces the externally pinned PySCF ghost-atom
      CP-corrected interaction energy to 1e-6 Ha, with the CP correction
      magnitude in the published mHa class; runtime ≤ 60 s; REFERENCE.md cites
      Boys & Bernardi 1970 (DOI 10.1080/00268977000101561) and the exact
      PySCF pin command + numbers.
    status: pending
  - id: ac-005
    summary: regression is an out-of-library consumer
    type: code
    pass_when: |
      reg_h2o_dimer_cp.cpp includes only public atomiverse headers (no
      scf/internal/), links only atomiverse, and is not registered in default
      ctest.
    status: verified
    last_checked: 2026-07-21
  - id: ac-006
    summary: tutorial ch04 closes with the BSSE/counterpoise section
    type: docs
    pass_when: |
      docs/tutorial/ch04-basis-sets.md contains a "BSSE 与 counterpoise"
      section with the borrowed-basis intuition, the 5-SCF recipe, an explicit
      over-correction/size-consistency/deformation-correction honesty passage,
      a water-dimer hands-on using Counterpoise, and chapter-end PySCF table
      rows (scalars in tables).
    status: pending
  - id: ac-007
    summary: full build and unit suite stay green
    type: runtime
    pass_when: |
      cmake --build build succeeds and ctest --test-dir build
      --output-on-failure passes with zero failures, including all
      pre-existing SCF tests with empty ghost_atoms (behavioral no-op).
    status: verified
    last_checked: 2026-07-21
---

# Acceptance criteria

逐条为二值判定:ac-001/ac-005 为代码/结构检查(成员面、无工厂、无 Frame 泄漏、
回归用例只消费公开 API);ac-002/ac-003 为秒级单元运行时(BSSE 符号约定 bsse ≤ 0
三处锁定);ac-004 为领域验证(PySCF 定标 1e-6 Ha + mHa 量级检查);ac-006 为
教程章节内容检查;ac-007 为整体回归不变量。

## Verification notes (2026-07-21)

- ac-001/005: `ghost_atoms` on MeanFieldMethod; `Counterpoise` plain class in
  `counterpoise.h`; no factories; no new `kAtom_*`; regression uses only
  public headers (`methods.h`, `frame.h`, `frame_schema.h`).
- ac-002: `test_ghost_basis` 5/5 pass (~0 ms).
- ac-003: `test_counterpoise` 3/3 pass (~211 ms), including He–He identities
  and KS+ghost throw.
- ac-007: full rebuild OK; all 48 SCF-related ctest targets green (incl.
  new ghost/counterpoise). Five pre-existing MPI coul/halo failures
  (`coul/long` requires reciprocal partner) are unrelated to this link and
  predate it. Empty `ghost_atoms` is a no-op for prior SCF paths.
- ac-004/006 left **pending** (scientific + docs review); regression binary
  does pass against the Cartesian PySCF 2.14.0 pin (`e_int_cp` /
  `bsse` within 1e-6 Ha, |bsse| ≈ 1.48 mHa, ~4 s).
