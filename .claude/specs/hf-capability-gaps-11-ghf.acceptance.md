---
slug: hf-capability-gaps-11-ghf
criteria:
  - id: ac-001
    summary: Spinor SlaterDeterminant invariants validated
    type: code
    pass_when: |
      tests/cpu/scf/test_ghf.cpp spinor cases pass: generalized constructor
      stores C(2n x N_mo); density() is 2n x 2n; spin-block accessors return
      the four blocks; validate(S) enforces Tr(P·S⊗1₂)=N and idempotency;
      dimension/occupation violations throw std::invalid_argument.
    status: verified
    last_checked: 2026-07-21
    note: GhfSpinorDeterminant.{GeneralizedFormAndSpinBlocks,
      ValidateSpinorMetric,ConstructionGuards} pass (analytic 4x4 spinor
      determinant; blocks match cos/sin closed forms to 1e-14; validate
      passes at tol 1e-10 with S = I2 and throws std::invalid_argument on
      3x3 S, odd rows, eps-size mismatch, f>1, sum(f)!=N).
  - id: ac-002
    summary: GHF collapses to UHF on unfrustrated H2/STO-3G
    type: code
    pass_when: |
      In tests/cpu/scf/test_ghf.cpp, converged GHF on H2/STO-3G satisfies
      |E_GHF - E_UHF| <= 1e-8 Ha against a UHF run on the same geometry.
    status: verified
    last_checked: 2026-07-21
    note: GHF.CollapsesToUhfOnH2 passes — measured |E_GHF - E_UHF| =
      4.4e-16 Ha (both -1.116759307396 Ha, matching the PySCF pin
      -1.11675930739643 to <1e-10), with the default nonzero
      guess_spin_rotation = 0.2.
  - id: ac-003
    summary: Non-collinear GHF below UHF on triangular H3, matches PySCF
    type: scientific
    pass_when: |
      In tests/cpu/scf/test_ghf.cpp, equilateral H3/STO-3G GHF with
      collinearity-breaking guess gives E_GHF <= E_UHF - 1e-6 Ha and
      |E_GHF - E_pinned(PySCF scf.GHF)| <= 1e-8 Ha (pinned value recorded
      in regressions/scf/h3_ghf_sto3g/REFERENCE.md).
    status: verified
    last_checked: 2026-07-21
    note: GHF.NoncollinearBelowUhfOnTriangularH3 +
      GHF.CollinearGuessStaysOnUhfSaddle pass. Measured E_GHF =
      -1.340440343541 Ha vs PySCF 2.14.0 scf.GHF pin -1.34044034354119
      (|dE| = 1.5e-14); below the engine UHF (-1.328175755381, symmetric
      doublet) and below the best PySCF collinear UHF (-1.33598005401819)
      by 4.46 mHa; ||P_ab||_F = 0.9780588 (PySCF witness identical).
      Noncollinear minimum certified externally: PySCF mf_g.stability()
      internal-stable, collinear UHF stability(external=True) reports
      UHF->GHF instability. Trap: theta = 0 converges to
      -1.335980054018 (the collinear saddle), 4.5 mHa above the minimum.
  - id: ac-004
    summary: GHF registered as concrete CRTP method, no factory/config debt
    type: code
    pass_when: |
      methods.h includes ghf.h; src/cpu/scf/mean_field_method.cpp contains
      `template class MeanFieldMethod<GHF>;`; parameters are members
      (n_electrons, guess_spin_rotation); tests/architecture/
      no_factory_scrub.sh and test_stage_typing_scrub pass.
    status: verified
    last_checked: 2026-07-21
    note: ghf.h included from methods.h; explicit instantiation added
      beside the other five; no Config bag / factory (members only).
      ctest: test_no_factory_scrub Passed (3.52 s),
      test_stage_typing_scrub Passed (0.18 s).
  - id: ac-005
    summary: Regression h3_ghf_sto3g passes against pinned oracle in <=60 s
    type: scientific
    pass_when: |
      `regressions` target runs reg_h3_ghf_sto3g as an out-of-library
      consumer (public headers + link atomiverse only), checks the
      REFERENCE.md pinned PySCF values, and completes in <= 60 s; it is
      not registered in default ctest.
    status: verified
    last_checked: 2026-07-21
    note: run_reg_h3_ghf_sto3g PASS in 1.0 s (H2 collapse |dE| = 4.4e-16;
      H3 E_GHF dE vs pin = -1.5e-14; ||P_ab|| = 0.9780588; theta=0 trap on
      -1.335980054018). Registered via atv_add_regression inside the
      ATV_BUILD_HF gate; not in ctest (custom target only). All 11
      pre-existing SCF regressions re-run green (1.0-4.8 s each).
  - id: ac-006
    summary: Tutorial ch03 gains the GHF spin-symmetry-breaking section
    type: docs
    pass_when: |
      docs/tutorial/ch03-open-shell.md contains a "GHF 与自旋对称破缺"
      section with the RHF→UHF→ROHF→GHF constraint-removal ladder, an
      honest Fukutome-classes mention with the 1981 DOI, the triangular-H3
      hands-on with molvis spin-vector rendering, and chapter-end PySCF
      comparison table rows (scalars in tables).
    status: pending
  - id: ac-007
    summary: New public symbols carry Doxygen with units
    type: docs
    pass_when: |
      Every public symbol added in ghf.h, the new SlaterDeterminant
      accessors, and SpinType::Generalized carries @brief, @param with
      units, and @return.
    status: pending
  - id: ac-008
    summary: Full unit suite stays green and fast
    type: runtime
    pass_when: |
      ctest --test-dir build passes with ATV_BUILD_HF=ON including
      test_cpu_scf_ghf; the whole tests/ suite still completes in seconds
      (no e2e added to ctest).
    status: verified
    last_checked: 2026-07-21
    note: full ctest 2026-07-21 — only the exactly-5 pre-existing MPI
      failures (test_mpi_halo_exchange_n{1,2,8},
      test_mpi_coul_energy_bruteforce_n{1,8}); baseline unchanged.
      test_cpu_scf_ghf: 8 tests, 0.36 s standalone / 0.02 s under ctest;
      test_slater_determinant and the whole SCF family stay green. No
      bitwise assertions (1e-8 pins; OpenMP-safe).
---

# Acceptance criteria

ac-001–ac-004、ac-008 由 /mol:impl 在实现回路中验证(code/runtime);ac-003、
ac-005 为领域验证,判据是外部钉定的 PySCF `scf.GHF`/`scf.UHF` 数值(oracle
依赖仅为数值,非构建依赖);ac-006、ac-007 为文档判据。ac-003 的塌缩/非共线
双测试共同编码 GHF 的核心物理陷阱:无自旋轨道项时 GHF 必须塌缩回 UHF,仅在
自旋阻挫体系上凭非共线性下降。SlaterDeterminant 旋量推广是本环架构决策,
实现前须与 scf-slater-determinant 链定案(扩展而非分叉)。
