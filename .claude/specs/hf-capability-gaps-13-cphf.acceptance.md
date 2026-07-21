---
slug: hf-capability-gaps-13-cphf
criteria:
  - id: ac-001
    summary: Second/skeleton derivative integrals match finite differences
    type: scientific
    pass_when: |
      tests/cpu/scf/test_cphf.cpp passes: on HeH+/STO-3G, d2S/d2T/d2V agree
      with central FD of dS/dT/dV to 1e-7, MO-basis S^(x)/F^(x) agree with FD
      of S/F to 1e-6 [Hartree/Bohr], and B^x sums to zero under rigid
      translation.
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_cphf green (HeH+/STO-3G): d2S/d2T/d2V vs 5-point FD
      of dS/dT/dV max dev < 1e-7 (all 324 slices); d2V_nuc vs FD < 1e-9; MO
      S^(x)/F^(x) vs 5-point FD of S / skeleton F at frozen C,P < 1e-6; B^x
      rigid-translation sum |B(0,d)+B(1,d)| < 1e-10 per element.
  - id: ac-002
    summary: CPHF iterative solve equals dense solve using the shared kernel
    type: code
    pass_when: |
      Unit test asserts iterative U^x equals the dense ¹(A+B) direct solve to
      1e-10, and the CPHF LHS action is bit-identical to OrbitalHessian
      sigma_plus(v, singlet) — the link-08 kernel, no reimplementation.
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_cphf: IterativeSolveMatchesDenseDirectSolve — CG
      (diag eps_a−eps_i preconditioner, dense_threshold=0) vs xt::linalg
      dense solve of OrbitalHessian::dense_plus(singlet), all U^x elements
      within 1e-10 (HeH+/6-31G, n_ov=3). LhsApplicationIsTheLink08SigmaPlus
      Singlet: CphfPack::apply_lhs literally calls oh.sigma_plus(v, singlet)
      on the SAME OrbitalHessian instance — identical arithmetic, asserted
      <= 1e-14 (bit-identical path, no reimplementation).
  - id: ac-003
    summary: Analytic Hessian is symmetric, translation-clean, matches FD of forces
    type: scientific
    pass_when: |
      tests/cpu/scf/test_analytic_hessian.cpp: H2/STO-3G analytic Hessian
      matches central FD of analytic forces elementwise to 1e-6 eV/A^2,
      H == H^T, and row translational sums vanish (|sum_j H_ij| ≤ 1e-6).
    status: verified
    last_checked: 2026-07-21
    note: |
      test_cpu_scf_analytic_hessian green: H2/STO-3G (skew axis)
      rhf.hessian() vs 5-point central FD of rhf.forces() elementwise
      <= 1e-6 eV/A^2 (36 elements); H == H^T <= 1e-10 (symmetrized by
      construction); row translational sums <= 1e-6 eV/A^2. He/STO-3G
      hessian() throws std::invalid_argument. Bonus convention-free gate:
      DensityResponseMatchesFdOfConvergedDensity (dP/dx analytic vs 5-point
      FD of the converged density) <= 1e-6.
  - id: ac-004
    summary: FIRST GATE - analytic vs in-house FD frequencies within 1 cm^-1
    type: scientific
    pass_when: |
      regressions/scf/h2o_rhf_freq: H2O HF/6-31G* harmonic frequencies from
      rhf.hessian() and from the in-house FD Hessian (both through the
      existing vib mass-weighting/diagonalizer) agree mode-by-mode to
      ≤ 1 cm^-1.
    status: verified
    last_checked: 2026-07-21
    note: |
      regressions/scf/h2o_rhf_freq PASS: analytic frequencies
      [1843.7086, 4036.1734, 4146.3779] cm^-1 vs the in-house FD production
      line (cpu::vib::Hessian driver + the same RHF at vib::Phase::force):
      max mode deviation 0.0092 cm^-1 (tol 1.0) — both producers meet at
      vib::MassWeighting + vib::HessianDiagonalizer.
  - id: ac-005
    summary: Frequencies match externally pinned PySCF hessian.RHF
    type: scientific
    pass_when: |
      Same regression compares against PySCF hessian.RHF frequencies pinned in
      REFERENCE.md (external run, values committed): every mode ≤ 1 cm^-1;
      regression registered via atv_add_regression(scf h2o_rhf_freq) and runs
      ≤ 60 s.
    status: verified
    last_checked: 2026-07-21
    note: |
      Same regression vs PySCF 2.14.0 hessian.RHF pinned in
      REFERENCE.md (cart=True, conv_tol=1e-12, E=-76.0107068004 Ha,
      [1843.7084, 4036.1737, 4146.3782] cm^-1): max mode deviation
      0.0003 cm^-1 (tol 1.0). Registered atv_add_regression(scf
      h2o_rhf_freq) inside the ATV_BUILD_HF gate; runtime ~3.2 s
      (<= 60 s). All 13 pre-existing SCF regressions re-run green.
  - id: ac-006
    summary: Surface is a method output, no new driver, no factories
    type: code
    pass_when: |
      RHF::hessian(Frame&) returns vib::HessianResult beside forces(); no new
      driver or public result type is introduced; vib::Hessian is unmodified;
      CphfPack is constructor-wired under internal/ and
      tests/architecture/no_factory_scrub.sh passes.
    status: verified
    last_checked: 2026-07-21
    note: |
      RHF::hessian(Frame&) -> vib::HessianResult declared beside
      stability() in rhf.h; no new driver, no new public result type;
      cpu/vib untouched (git diff clean there); internal::CphfPack is
      constructor-wired under internal/ (cphf_pack.h) with borrowed const
      refs; tests/architecture/no_factory_scrub.sh PASS (also as ctest
      test_no_factory_scrub, 4.3 s).
  - id: ac-007
    summary: Tutorial ch20 gains the analytic-Hessian section with PySCF table
    type: docs
    pass_when: |
      docs/tutorial/ch20-hf-properties-environment.md contains the
      "解析 Hessian 与振动频率" section: CPHF orbital-response narrative and
      third reuse of the stability kernel, H2O hands-on, molvis normal-mode
      visualization, chapter-end PySCF frequency table (scalars in a table).
    status: pending
  - id: ac-008
    summary: Full build and unit suite pass with new tests registered
    type: runtime
    pass_when: |
      cmake --build build succeeds and ctest --test-dir build passes,
      including test_cpu_scf_cphf and test_cpu_scf_analytic_hessian; existing
      SCF/gradient tests stay green.
    status: verified
    last_checked: 2026-07-21
    note: |
      cmake --build build succeeds; ctest 200/205 passed — the 5
      failures are exactly the pre-existing MPI baseline
      (test_mpi_halo_exchange_n1/n2/n8, test_mpi_coul_energy_bruteforce_
      n1/n8). test_cpu_scf_cphf (4.2 s) and test_cpu_scf_analytic_hessian
      (1.1 s) green; existing SCF/gradient tests green; test_stage_typing_
      scrub + test_no_factory_scrub green.
---

# Acceptance criteria

ac-001 钉两大经典 bug 源的逐项 FD 门禁(skeleton 导数与 B^x);ac-002 钉共享
kernel 的第三配置(不重写 σ-向量);ac-003/004 钉"第一道闸"——解析对自家
FD 的自洽(微型与 H₂O 两级);ac-005 钉外部 PySCF oracle;ac-006 钉方法输出
/无新 driver/无工厂的架构裁定;ac-007/008 钉教程交付与全套件绿。
