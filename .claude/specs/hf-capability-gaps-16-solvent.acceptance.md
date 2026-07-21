---
slug: hf-capability-gaps-16-solvent
criteria:
  - id: ac-001
    summary: internal::Pcm exists as SCF-private workspace, no factories
    type: code
    pass_when: |
      include/atomiverse/cpu/scf/internal/pcm.h defines ATV::cpu::scf::internal::Pcm
      with constructor-set parameters (no make_*/build_* free function anywhere in the
      diff); Bondi radii are a static in-header table that throws on unsupported Z;
      tests/architecture/no_factory_scrub.sh passes.
    status: verified
    last_checked: 2026-07-21
  - id: ac-002
    summary: Born-ion analytic oracle validates cavity + conductor solve
    type: code
    pass_when: |
      test_pcm.cpp: for a single sphere (radius R) enclosing charge Q at eps=78.4,
      sum(q_k) = -f(eps)*Q within 1e-3*|Q| and DG_el matches -0.5*f(eps)*Q^2/R within
      1%; f(eps) checked for both x=0.0 and x=0.5 conventions; runs in seconds.
    status: verified
    last_checked: 2026-07-21
  - id: ac-003
    summary: solvent_eps=0 default is bit-identical to gas phase
    type: code
    pass_when: |
      test_pcm.cpp: RHF H2O/STO-3G with default members yields e_total bitwise equal
      to a run on a build without touching solvent members (same iterations, same
      energy to the last ulp), and e_solvent == 0.
    status: verified
    last_checked: 2026-07-21
  - id: ac-004
    summary: Self-consistent reaction field lowers the SCF energy
    type: code
    pass_when: |
      test_pcm.cpp: RHF H2O/STO-3G with solvent_eps=78.4 converges,
      e_solvent < 0, and total energy is strictly below the gas-phase
      total; whole unit suite still completes in seconds.
    status: verified
    last_checked: 2026-07-21
  - id: ac-005
    summary: forces() with solvent on throws instead of returning wrong gradients
    type: code
    pass_when: |
      test_pcm.cpp: calling rhf.forces(frame) with solvent_eps > 0 throws
      std::runtime_error (or a documented subclass); no force columns are written.
    status: verified
    last_checked: 2026-07-21
  - id: ac-006
    summary: H2O solvation energy matches pinned PySCF C-PCM to 0.1 kcal/mol
    type: scientific
    pass_when: |
      regressions/scf/h2o_rhf_631gs_cpcm passes under the `regressions` target in
      <= 60 s: RHF/6-31G* (Pople631Gs), eps=78.4, Bondi*1.2, x=0, Lebedev settings
      matching REFERENCE.md; |DG_el - pinned PySCF solvent.PCM (C-PCM) value|
      <= 0.1 kcal/mol, with the pinned value in the -6..-8 kcal/mol range and
      PySCF version + full settings recorded in REFERENCE.md.
    status: pending
  - id: ac-007
    summary: Tutorial ch20 implicit-solvent section is complete and honest
    type: docs
    pass_when: |
      docs/tutorial/ch20-hf-properties-environment.md contains a 隐式溶剂 section
      with: conductor/cavity narrative, the x=0 (C-PCM) vs x=0.5 (COSMO) convention
      divergence, explicit statements that non-electrostatic terms are excluded and
      the cavity surface is not smoothed, a hands-on solvation-energy run, a molvis
      cavity/surface-charge rendering, and a chapter-end PySCF comparison table
      (scalars in tables).
    status: pending
  - id: ac-008
    summary: Doxygen with units on every new public symbol; scrubs green
    type: docs
    pass_when: |
      Every new public/internal symbol in pcm.h, mean_field_method.h, fock_stage.h,
      scf_result.h carries @brief and @param/@return with units (Hartree/Bohr
      internal, eV external); test_stage_typing_scrub and no_factory_scrub pass.
    status: pending
---

# Acceptance criteria

ac-001/002 钉住私有工作区形态与解析 Born oracle;ac-003 是零开销关断保证(默认行为逐位不变);
ac-004/005 钉住 fock 缝自洽与"不给错梯度";ac-006 是对外部钉定 PySCF C-PCM 的科学验收
(0.1 kcal/mol);ac-007/008 保证教程叙事与文档诚实性(非静电项、表面不连续均如实声明)。
