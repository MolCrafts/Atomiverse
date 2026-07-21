---
slug: hf-capability-gaps-21-cfmm
criteria:
  - id: ac-001
    summary: HARD GATE first — CFMM J reproduces Schwarz direct J within tolerance
    type: scientific
    pass_when: |
      regressions/scf/c10h22_cfmm_j (C10H22/STO-3G) asserts, at the first SCF
      iteration, max_ij |J_CFMM(i,j) − J_direct(i,j)| <= 1e-8 Hartree against the
      in-tree Schwarz-screened direct J with documented cfmm_order/cfmm_threshold;
      this criterion carries no timing assertion and precedes any speed claim.
    status: pending
  - id: ac-002
    summary: Converged RHF total energy with CFMM-J matches direct to 1e-7 Ha
    type: scientific
    pass_when: |
      the same regression converges RHF twice (coulomb=Direct vs coulomb=CFMM) and
      asserts |E_CFMM − E_direct| <= 1e-7 Hartree; runs <= 60 s; REFERENCE.md cites
      WJGH 1994 (DOI 10.1016/0009-2614(94)01128-1) and states the baseline is our
      own PySCF-pinned direct path because PySCF ships no FMM.
    status: pending
  - id: ac-003
    summary: longrange facade consumed; no cpu/longrange include in cpu/scf
    type: code
    pass_when: |
      include/atomiverse/longrange/gaussian_extent.h exists (header-only, WJGH
      extent formula documented in Bohr); rg '"atomiverse/cpu/longrange' over
      include/atomiverse/cpu/scf src/cpu/scf returns zero matches while cfmm_j.*
      includes only top-level longrange/ facade headers.
    status: pending
  - id: ac-004
    summary: Third J route wired member-driven; default bitwise-unchanged; no factories
    type: code
    pass_when: |
      scf_enums.h defines Coulomb{Direct,RI,CFMM}; MeanFieldMethod exposes coulomb,
      cfmm_order, cfmm_threshold members passed through FockStage into
      HamiltonianBuilder; with default Coulomb::Direct all pre-existing scf unit
      tests pass unmodified; setting Coulomb::CFMM together with RI::J/JK throws;
      tests/architecture/no_factory_scrub.sh passes (no new build_* free function).
    status: pending
  - id: ac-005
    summary: Unit tests for extent classification and CfmmJ pass in seconds
    type: runtime
    pass_when: |
      ctest runs test_gaussian_extent (r_ext vs hand-computed erfc^-1 values <=1e-12,
      well-separated truth table) and test_cfmm_j (s|s moment equals overlap charge
      <=1e-12, near/far pair lists complementary with no double counting, single
      H2O/STO-3G j(P) vs dense direct J <=1e-8) green, with no converged-SCF loop
      inside tests/.
    status: pending
  - id: ac-006
    summary: Alkane crossover table pinned — CFMM J-build beats direct beyond crossover
    type: performance
    evaluator_hint: benchmarks/scf/validation/cfmm_scaling/
    pass_when: |
      benchmarks/scf/validation/cfmm_scaling/ contains run script + README
      (methodology, machine env, cfmm_order/threshold) and a produced table of
      J-build wall-clock direct vs CFMM over an alkane series CnH2n+2/STO-3G with
      an identified crossover size beyond which CFMM is strictly faster; the same
      numbers appear in ch19; never registered in ctest.
    status: pending
  - id: ac-007
    summary: ch19 closing section with honest PySCF statement and both tables
    type: docs
    pass_when: |
      docs/tutorial/ch19-hf-large-systems.md ends with a "长程库仑 III:CFMM 加速 HF"
      section that states PySCF ships no FMM (direct SCF + Schwarz, optional DF),
      closes the direct→RI→CFMM narrative, and embeds the accuracy table and the
      crossover system-size table as scalar tables (no bar charts; molplot scaling
      curves permitted) plus a hands-on rhf.coulomb = Coulomb::CFMM snippet.
    status: pending
  - id: ac-008
    summary: Doxygen complete on all new public/internal symbols
    type: docs
    pass_when: |
      every new symbol in longrange/gaussian_extent.h, cpu/scf/internal/cfmm_j.h,
      and the new Coulomb/cfmm_* members carries @brief plus @param/@return with
      units (Bohr for extents, Hartree for J/energies), including the extent
      formula and the CFMM-order/threshold error-control semantics.
    status: pending
---

# Acceptance criteria

ac-001 是链尾硬门:在任何速度论断(ac-006)被评估之前,ac-001 的 J 矩阵逐元素复现
必须先 verified;ac-002 把同一容差论断提升到收敛总能;ac-003/ac-004 钉住架构裁定
(门面消费、成员驱动接缝、无工厂);ac-005 保 tests/ 秒级纪律;ac-006 为唯一
performance 判据(交叉点表);ac-007/ac-008 收束教程叙事与文档纪律。
