---
slug: hf-capability-gaps-02-cart2sph
criteria:
  - id: ac-001
    summary: real_harmonics hoisted to core; gpw re-pointed; no stale includes
    type: code
    pass_when: |
      include/atomiverse/core/real_harmonics.h and src/core/real_harmonics.cpp exist and
      compile unconditionally (no ATV_BUILD_GPW gate); rg "gpw/internal/real_harmonics" over
      include/ src/ tests/ returns zero matches; gpw unit tests pass unchanged.
    note: AUDIT — gpw suite run under a dedicated ATV_BUILD_GPW=ON config: 14/15 pass;
      the single failure (GpwSolverClass.WarmStartReachesColdFixedPoint) reproduces
      byte-identically on a pristine-HEAD worktree in the same config, i.e.
      pre-existing and unrelated to the hoist ("unchanged" holds).
    status: verified
    last_checked: 2026-07-20
  - id: ac-002
    summary: core real-harmonics unit tests pass after migration
    type: runtime
    pass_when: |
      ctest test_real_harmonics (tests/core/test_real_harmonics.cpp) passes, preserving the
      migrated orthonormality / sphere_monomial_integral assertions at 1e-12.
    status: verified
    last_checked: 2026-07-20
  - id: ac-003
    summary: CartToSph matches PySCF cart2sph_coeff and absorbs Cartesian norms
    type: runtime
    pass_when: |
      ctest test_cart2sph passes: d is 6x5 and f is 10x7; every coefficient equals the pinned
      PySCF gto.mole.cart2sph_coeff values to 1e-12; the unit-normalized d-shell self-overlap
      sandwich (double-factorial normalization pitfall) yields the 5x5 identity to 1e-12.
    status: verified
    last_checked: 2026-07-20
  - id: ac-004
    summary: AO dimension switches per basis convention gate
    type: runtime
    pass_when: |
      ctest test_spherical_basis passes: H2O/def2-SVP n_basis == 24 (was 25 Cartesian);
      H2O/6-31G* n_basis unchanged from current Cartesian value; spherical overlap S equals
      the c2s sandwich of the Cartesian S fixture to 1e-12.
    note: the sandwich identity is pinned at 1e-12 in test_cart2sph (explicit 5x5 fixture);
      test_spherical_basis pins the end-to-end consequence (unit spherical diagonal at
      1e-12 + pinned PySCF S spot values).
    status: verified
    last_checked: 2026-07-20
  - id: ac-005
    summary: RHF/def2-SVP energy matches pinned PySCF cart=False value
    type: scientific
    evaluator_hint: regressions target, oracle protocol from link 01-env
    pass_when: |
      reg_h2o_rhf_def2svp_sph (regressions/scf/h2o_rhf_def2svp_sph/) converges and its total
      RHF energy agrees with the externally pinned PySCF cart=False number recorded in
      REFERENCE.md to within 1e-8 Ha, in <= 60 s, via the `regressions` target.
    note: AUDIT — scientific criterion executed directly via the regressions target
      (mol_project.bench.repo unset): PASS, |E - E_PySCF| = 4.1e-13 Ha (tolerance 1e-8).
    status: verified
    last_checked: 2026-07-20
  - id: ac-006
    summary: spherical-basis analytic gradient rejected, never silently wrong
    type: code
    pass_when: |
      requesting the analytic SCF gradient with a Spherical-convention basis throws
      std::runtime_error with a message naming the unsupported combination; a unit test in
      test_spherical_basis.cpp asserts the throw.
    status: verified
    last_checked: 2026-07-20
  - id: ac-007
    summary: tutorial ch04 gains the Cartesian-vs-spherical section
    type: docs
    pass_when: |
      docs/tutorial/ch04-basis-sets.md contains a 「笛卡尔与球谐高斯基」 section with the solid-
      harmonics derivation, the 6d s-contaminant and 5d/7f dimension counting, an Atomiverse
      hands-on snippet, and PySCF comparison table row(s) with scalar entries (no bar charts).
    status: pending
  - id: ac-008
    summary: conventions hold — doxygen complete, no factory functions
    type: code
    pass_when: |
      every new public/internal symbol in core/real_harmonics.h, cart2sph.h, basis_table.h,
      basis_set.h carries @brief/@param(units)/@return; tests/architecture/no_factory_scrub.sh
      and the full unit suite pass in the standard HF+PW CPU configure.
    status: verified
    last_checked: 2026-07-20
---

# Acceptance criteria

ac-001/002 钉住上提的无损性(GPW 数值零变化);ac-003 是本链环的数学核心(系数正确性 + 归一化
陷阱);ac-004 钉住维度切换与逐基组约定门;ac-005 是科学判据 — 与 01-env 协议下外部钉死的
PySCF `cart=False` RHF/def2-SVP 能量一致;ac-006 保证未完成的梯度路径显式失败;ac-007/008 覆盖
教程弧与仓库规约。
