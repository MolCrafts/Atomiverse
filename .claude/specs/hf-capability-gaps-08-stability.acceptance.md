---
slug: hf-capability-gaps-08-stability
criteria:
  - id: ac-001
    summary: OrbitalHessian sigma action matches explicit dense (A±B) build
    type: code
    pass_when: |
      tests/cpu/scf/test_orbital_hessian.cpp passes: for H2/STO-3G and random
      vectors v, |dense_plus(ch)·v − sigma_plus(v,ch)|_max ≤ 1e-12 (idem minus),
      for both singlet and triplet channels.
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_orbital_hessian green: dense_±·v vs sigma_±(v) max-abs
      diff ≤ 1e-12 on H2/STO-3G (n_ov=1) and a synthetic n_ov=15 system,
      3 random LCG vectors, both channels, plus symmetric-matrix and
      diagonal_plus parity checks.
  - id: ac-002
    summary: Singlet A diagonal equals eps_a − eps_i from the SlaterDeterminant
    type: scientific
    pass_when: |
      Unit test asserts dense ¹A diagonal == ε_a − ε_i [Ha] taken from the
      converged SlaterDeterminant (H2/STO-3G) to 1e-12.
    status: verified
    last_checked: 2026-07-21
    note: |
      Element-wise formula pinned to 1e-12: A(0,0) == Δε + 2(ai|ai) − (ii|aa)
      with MO couplings transformed independently in-test, and
      A(0,0) − couplings == ε_a − ε_i (the δδΔε term isolated). Exact
      wording deviation (raw diagonal ≠ bare Δε) audited in the spec Design
      注记. Triplet A/B elements pinned the same way.
  - id: ac-003
    summary: Equilibrium H2 RHF is internally and externally stable
    type: scientific
    pass_when: |
      tests/cpu/scf/test_stability.cpp: at R = 0.735 Å (STO-3G) both
      internal_eigenvalue and external_eigenvalue ≥ 0 and both *_stable flags true.
    status: verified
    last_checked: 2026-07-21
    note: |
      R = 0.735 Å: internal = +1.1351770407 Ha, external = +0.4114522416 Ha
      (both ≥ 0, PySCF 2.14.0 pins matched to 1e-6 in unit test AND in
      reg_h2_rhf_stability with *_stable flags true; engine agrees with the
      pins to all 10 printed digits).
  - id: ac-004
    summary: Stretched H2 shows Seeger-Pople split - internal stable, external not
    type: scientific
    pass_when: |
      At R = 1.5 Å (STO-3G): internal_eigenvalue ≥ 0 AND external_eigenvalue < 0;
      regression reg_h2_rhf_stability matches the PySCF mf.stability() lowest
      external eigenvalue pinned in REFERENCE.md to 1e-6 Ha.
    status: verified
    last_checked: 2026-07-21
    note: |
      R = 1.5 Å: internal = +0.7088965814 Ha (≥ 0), external =
      −0.2092471629 Ha (< 0) — Seeger-Pople trap encoded as its own unit
      test and re-checked in reg_h2_rhf_stability against the REFERENCE.md
      PySCF pin to 1e-6 Ha (observed agreement < 1e-9 Ha).
  - id: ac-005
    summary: Broken-symmetry mode lowers UHF below RHF at stretched geometry
    type: scientific
    pass_when: |
      In regressions/scf/h2_rhf_stability: UHF seeded from StabilityResult
      rotated_C converges with E_UHF < E_RHF by ≥ 1e-4 Ha at R = 1.5 Å,
      consistent with Szabo-Ostlund §3.8.7; regression runs ≤ 60 s.
    status: verified
    last_checked: 2026-07-21
    note: |
      reg_h2_rhf_stability: UHF seeded from rotated_C (α density from the
      rotated occupied column, β from the unrotated determinant, via public
      SCFInitialGuess) converges to E_UHF = −0.9577067934 Ha vs E_RHF =
      −0.9108735546 Ha: ΔE = −0.0468332388 Ha ≫ 1e-4 gate, matches the
      PySCF broken-symmetry pin to 1e-6. Regression wall time ≈ 0.3 s.
  - id: ac-006
    summary: Kernel is a class - no build_* free functions in cpu/scf
    type: code
    pass_when: |
      OrbitalHessian/StabilityAnalysis are constructor-wired classes; grep gate
      tests/architecture/no_factory_scrub.sh passes with the new files present.
    status: verified
    last_checked: 2026-07-21
    note: |
      OrbitalHessian / StabilityAnalysis are constructor-wired classes
      (transform in ctor, no free functions); eri materialization is the
      IntegralPack member eri_working(). no_factory_scrub.sh PASS with all
      new files present (incl. the cpu/scf build_* wildcard).
  - id: ac-007
    summary: Tutorial ch03 gains the SCF stability section with PySCF table
    type: docs
    pass_when: |
      docs/tutorial/ch03-open-shell.md contains a stability-analysis section
      (internal vs external narrative, stretched-H2 hands-on, RHF-vs-UHF
      energy-vs-R curve, chapter-end PySCF lowest-eigenvalue table).
    status: pending
  - id: ac-008
    summary: Full build and unit suite pass with the new tests registered
    type: runtime
    pass_when: |
      cmake --build build succeeds and ctest --test-dir build passes, including
      test_cpu_scf_orbital_hessian and test_cpu_scf_stability.
    status: verified
    last_checked: 2026-07-21
    note: |
      cmake --build build clean; ctest 100% pass except the 5 pre-existing
      MPI baseline failures (test_mpi_halo_exchange_n{1,2,8},
      test_mpi_coul_energy_bruteforce_n{1,8}).
      test_cpu_scf_orbital_hessian (9 tests) and test_cpu_scf_stability
      (8 tests) both green; all 7 pre-existing SCF regressions re-run PASS.
---

# Acceptance criteria

ac-001/002 钉 kernel 的数学正确性(σ-向量与稠密一致、对角结构);ac-003/004 钉
Seeger-Pople 分类本身(含"internal 稳定但 external 不稳定"陷阱与 PySCF 外部
oracle);ac-005 钉破缺对称初猜到 UHF 的桥;ac-006 钉 OOP/无工厂架构规则;
ac-007/008 钉教程交付与全套件绿。
