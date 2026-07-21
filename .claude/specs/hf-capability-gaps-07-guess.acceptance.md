---
slug: hf-capability-gaps-07-guess
criteria:
  - id: ac-001
    summary: New guess/mixer headers are classes; no factory free functions
    type: code
    pass_when: |
      internal/hueckel_guess.h, sap_guess.h, sap_data.h, ediis_adiis.h exist with
      constructor-based classes (HueckelGuess/SapGuess/EDIISMixer/ADIISMixer);
      tests/architecture/no_factory_scrub.sh passes; no cpu/scf build_* free
      function added.
    status: verified
    last_checked: 2026-07-21
    note: |
      hueckel_guess.h/sap_guess.h/sap_data.h/ediis_adiis.h all constructor-based
      (HueckelGuess, SapGuess, SapAtomicPotential, EDIISMixer, ADIISMixer);
      tests/architecture/no_factory_scrub.sh PASS; no new build_* free function
      (dispatch reuses pre-existing build_molecular_grid only as a caller).
  - id: ac-002
    summary: Selection is method members; enums live in scf_enums.h
    type: code
    pass_when: |
      InitGuess in scf_enums.h has Hueckel and SAP; MixerKind{CDIIS,EDIIS_CDIIS,
      ADIIS_CDIIS} in scf_enums.h; MeanFieldMethod has members mixer (default
      CDIIS) and mixer_switch_tol; no new *Config type appears.
    status: verified
    last_checked: 2026-07-21
    note: |
      scf_enums.h: InitGuess{Hcore,MINAO,Hueckel,SAP}, MixerKind{CDIIS,
      EDIIS_CDIIS,ADIIS_CDIIS}; MeanFieldMethod members mixer (default
      MixerKind::CDIIS) + mixer_switch_tol (1e-2); no new *Config type
      anywhere in the diff.
  - id: ac-003
    summary: Unit tests pin Hueckel/SAP guess matrix math
    type: runtime
    pass_when: |
      test_cpu_scf_init_guess passes: Hueckel F_uv = 0.5*1.75*(H_uu+H_vv)*S_uv
      off-diagonal and VSIP diagonal element-wise; SAP F symmetric and guess
      density satisfies Tr(PS)=N_elec within 1e-8; unsupported Z throws.
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_init_guess: 10/10 PASS in 0.05 s. Hueckel diag == VSIP and
      off-diag 0.5*1.75*(Hii+Hjj)*S_ij element-wise (H2O/STO-3G + def2-SVP
      spherical); SAP V symmetric, H-atom closed-form V_00 matched to 1e-6;
      Tr(PS)=10 to 1e-8 for both guesses; vsip(19,0)/SapAtomicPotential(119)
      and in-constructor unsupported-Z all throw.
  - id: ac-004
    summary: Unit tests pin EDIIS/ADIIS simplex solutions
    type: runtime
    pass_when: |
      test_cpu_scf_ediis_adiis passes: two-point closed-form optimal c matches
      to 1e-10 for both mixers; coefficients satisfy sum=1, c_i>=0; degenerate
      history returns latest F without NaN.
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_ediis_adiis: 11/11 PASS in 0.04 s. EDIIS two-point interior
      optimum c=(0.75,0.25) and ADIIS c=(1/3,2/3) matched to 1e-10 (KKT polish
      gives ~1e-14); sum(c)=1, c>=0 asserted; EDIIS vertex case exact; ADIIS
      objective({0.5,0.5})=-0.005 to 1e-14; duplicate-F history returns latest
      F finite; depth-1 returns input unchanged.
  - id: ac-005
    summary: Unit suite stays fast and E2E-free; defaults bit-identical
    type: runtime
    pass_when: |
      Full ctest passes with no SCF-to-convergence run added under tests/;
      existing test_cpu_scf_solver and test_cpu_scf_accelerators pass unchanged
      with default mixer=CDIIS.
    status: verified
    last_checked: 2026-07-21
    note: |
      Full ctest: 141 passed, 5 failed == exactly the pre-existing MPI baseline
      (test_mpi_halo_exchange_n{1,2,8}, test_mpi_coul_energy_bruteforce_n{1,8}).
      test_cpu_scf_solver, test_cpu_scf_accelerators, test_cpu_scf_stages,
      test_cpu_scf_method_driver_parity all pass with default mixer=CDIIS
      (CDIIS branch byte-identical; new tests are single-call math only).
  - id: ac-006
    summary: Regression pins iteration counts and hard-case convergence
    type: scientific
    pass_when: |
      regressions/scf/h2o_scf_convergence (<=60 s, regressions target only):
      H2O/def2-SVP iterations Hueckel<Hcore and SAP<Hcore; hard case where
      CDIIS-alone exceeds budget but ADIIS->CDIIS converges to the PySCF energy
      pinned in REFERENCE.md within 1e-6 Ha.
    status: verified
    last_checked: 2026-07-21
    note: |
      reg_h2o_scf_convergence PASS in 3.1 s. Part 1 (H2O/def2-SVP RHF,
      tol=1e-9): hcore 16, minao 16, hueckel 14, sap 14 iterations -> both
      Hueckel<Hcore and SAP<Hcore hold; every guess at pinned E=-75.96098398708
      within 1e-7. Part 2 (2.6x-stretched H2O UHF): CDIIS-alone NOT converged
      in 60 iters (stuck ~-75.357, the PySCF-confirmed unstable symmetric
      solution); ADIIS->CDIIS converged in 38 iters to -75.71896632254 vs
      PySCF pin -75.71896632261 (|dE|~7e-11 << 1e-6 Ha).
  - id: ac-007
    summary: ch18 half-chapter with domain-of-use pitfall and PySCF table
    type: docs
    pass_when: |
      docs/tutorial/ch18-scf-techniques.md contains the 初猜与 DIIS 家族 half:
      EDIIS/ADIIS far-from-convergence-only pitfall with ADIIS/EDIIS->CDIIS
      recipe, a molplot convergence-curve figure, and a chapter-end PySCF
      iteration-count table (scalars in tables, curves as figures).
    status: pending
  - id: ac-008
    summary: Doxygen and DOI references on all new symbols
    type: docs
    pass_when: |
      Every new public/internal symbol carries @brief/@param(units)/@return;
      hueckel_guess.h and sap_guess.h cite DOI 10.1021/acs.jctc.8b01089;
      ediis_adiis.h cites DOI 10.1063/1.1470195 and DOI 10.1063/1.3304922.
    status: pending
---

# Acceptance criteria

ac-001/002 为结构闸(无工厂、成员即参数、枚举归位 scf_enums.h);ac-003/004 为
单次调用数学闸(秒级单元);ac-005 守护默认行为零变化与 tests/ 无 E2E;ac-006 是
唯一的端到端科学闸(regressions,PySCF 钉定值);ac-007/008 为教程与文档闸。
