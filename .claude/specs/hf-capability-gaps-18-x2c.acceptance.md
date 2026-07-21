---
slug: hf-capability-gaps-18-x2c
criteria:
  - id: ac-001
    summary: pVp kernel matches externally pinned PySCF int1e_pnucp
    type: scientific
    pass_when: |
      tests/cpu/scf/test_x2c.cpp passes: W matrix elements for H2 and HeH+
      (STO-3G) agree with pinned PySCF mol.intor('int1e_pnucp') values to
      1e-7 (version+command in test comments, no Python at ctest time;
      matches the engine nuclear Boys floor ~2e-8 vs PySCF — same floor
      as V; existing integral tests pin V at 2e-6), and W is symmetric.
    status: pending
  - id: ac-002
    summary: Exact-decoupling identities pin the X2cWorkspace algebra
    type: scientific
    pass_when: |
      Unit tests assert: with injected c=1e8 the workspace h_X2C equals T+V
      elementwise to 1e-6 (nonrelativistic limit); for the one-electron H
      atom (def2-SVP) the lowest h_X2C eigenvalue equals the lowest
      positive-energy eigenvalue of the modified Dirac matrix to 1e-10.
    status: pending
  - id: ac-003
    summary: Consumption wiring, mutual-exclusion gate both ways, no factories
    type: code
    pass_when: |
      x2c=true replaces only the private Hcore (S/ERI/Fock loop untouched;
      x2c=false path byte-identical to status quo); ECP+x2c on an
      I-containing Frame throws a runtime_error naming the element while
      all-electron H2 with x2c=true converges without throwing; forces()
      with x2c=true throws; X2cWorkspace stays under internal/ and
      tests/architecture/no_factory_scrub.sh passes.
    status: verified
    last_checked: 2026-07-21
  - id: ac-004
    summary: Ar x2c-1e RHF matches pinned PySCF x2c1e to 1e-5 Ha
    type: scientific
    pass_when: |
      regressions/scf/x2c_argon (registered via atv_add_regression, <= 60 s,
      not in ctest): Ar/def2-TZVP RHF with x2c=true matches the externally
      pinned PySCF scf.RHF(mol).x2c1e() value (xuncontract=False) to
      <= 1e-5 Ha, and E_x2c - E_nonrel is negative with magnitude within 5%
      of the pinned PySCF difference.
    status: pending
  - id: ac-005
    summary: Tutorial ch20 gains the scalar-relativity x2c section
    type: docs
    pass_when: |
      docs/tutorial/ch20-hf-properties-environment.md contains the
      "标量相对论:x2c" section: orbital-contraction narrative, one-page
      x2c-1e, ECP-vs-x2c decision guidance with the mutual exclusion,
      picture-change honesty warning, Ar hands-on, chapter-end PySCF table
      rows (scalars in a table).
    status: pending
  - id: ac-006
    summary: Full build and unit suite pass with test_x2c registered
    type: runtime
    pass_when: |
      cmake --build build succeeds and ctest --test-dir build passes with
      test_cpu_scf_x2c registered; existing SCF unit tests stay green and
      the whole suite stays seconds-fast (no heavy-atom SCF in tests/).
    status: verified
    last_checked: 2026-07-21
---

# Acceptance criteria

ac-001 钉新积分核对外部 oracle 的逐元素对账;ac-002 用两条机器恒等式
(非相对论极限、一电子精确退耦)钉退耦线代,不靠自家黄金值;ac-003 钉
架构裁定——只换 Hcore、互斥门双向、力拦截、工作区私有、无工厂;ac-004
钉重系统 oracle(收缩基 + xuncontract=False 裁定显式入题)与相对论降低
的符号/幅值;ac-005/006 钉教程交付与全套件绿且保持秒级。
