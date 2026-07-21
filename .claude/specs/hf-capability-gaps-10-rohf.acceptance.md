---
slug: hf-capability-gaps-10-rohf
criteria:
  - id: ac-001
    summary: ROHF is a concrete CRTP method registered in methods.h, no factories
    type: code
    pass_when: |
      include/atomiverse/cpu/scf/rohf.h declares `class ROHF : public
      MeanFieldMethod<ROHF>` with member `n_alpha`/`n_beta` occupation and no
      `build_*`/`make_*` free function anywhere in the diff; methods.h includes
      rohf.h; tests/architecture/no_factory_scrub.sh passes.
    status: verified
    last_checked: 2026-07-21
    note: |
      rohf.h declares exactly the UHF shape (CRTP MeanFieldMethod<ROHF>,
      n_alpha/n_beta members, default + explicit-count ctors,
      name()=="rohf"); methods.h includes rohf.h; the diff adds zero free
      functions (RohfCoupleStage is a Component class, ctor-wired).
      no_factory_scrub + test_stage_typing_scrub pass in ctest.
  - id: ac-002
    summary: O2 triplet ROHF/STO-3G total energy matches pinned PySCF to 1e-8 Ha
    type: scientific
    pass_when: |
      test_cpu_scf_rohf_solver asserts |E(ROHF, O2 R=1.2075 Å, (9,7), STO-3G) −
      E_pinned| ≤ 1e-8 Ha, where E_pinned is a PySCF 2.13.1 scf.ROHF value
      recorded verbatim (with geometry + command) in the test file header.
    status: verified
    last_checked: 2026-07-21
    note: |
      Pinned with PySCF 2.14.0 (the oracle venv's actual version; spec said
      2.13.1 — audit-noted in the spec). E_pinned = -147.632166990682 Ha
      (cart=True, conv_tol=1e-12); measured |dE| = 8.5e-14 Ha (regression
      print), far under the 1e-8 gate. Geometry + command verbatim in the
      test file header and REFERENCE.md.
  - id: ac-003
    summary: CH3 doublet ROHF/STO-3G total energy matches pinned PySCF to 1e-8 Ha
    type: scientific
    pass_when: |
      test_cpu_scf_rohf_solver asserts |E(ROHF, planar CH3 R(C–H)=1.079 Å,
      (5,4), STO-3G) − E_pinned(PySCF 2.13.1)| ≤ 1e-8 Ha, provenance pinned in
      the test file header.
    status: verified
    last_checked: 2026-07-21
    note: |
      E_pinned = -39.072132718153 Ha (PySCF 2.14.0, planar D3h, H at 0/120/
      240°, cart=True, conv_tol=1e-12). ROHF.CH3DoubletMatchesPyscf passes
      the 1e-8 Ha EXPECT_NEAR (105 ms).
  - id: ac-004
    summary: ROHF is spin-pure; UHF contamination contrast holds
    type: scientific
    pass_when: |
      For converged O2 and CH3 ROHF runs, |spin contamination| ≤ 1e-10
      (⟨S²⟩ = 2.0 resp. 0.75 exactly); a same-geometry O2 UHF run in the same
      test file shows spin contamination > 1e-3.
    status: verified
    last_checked: 2026-07-21
    note: |
      Measured ROHF contamination: O2 0.0 / 2.7e-15 (run-to-run), CH3 ≤1e-10
      gate passes — the single-MO-set analytic identity (C_beta = C_alpha in
      compute_spin_contamination). UHF contrast: warm-started from the ROHF
      density it lands on PySCF's global solution, E = -147.633946785450 Ha,
      contamination 3.410858e-3 (PySCF pin 3.410858e-3) > 1e-3; the engine's
      cold Hcore/MINAO guesses find a higher local UHF solution
      (-147.3786 Ha, contamination 1.27e-2 — also > 1e-3).
  - id: ac-005
    summary: Orbital energies match PySCF only under Guest–Saunders convention
    type: scientific
    pass_when: |
      Occupied orbital energies of O2/CH3 match pinned PySCF canonical-ROHF
      values to 1e-6 Ha, and neither tests nor acceptance gate on any
      non-PySCF-convention orbital energies (comment in test cites the
      convention).
    status: verified
    last_checked: 2026-07-21
    note: |
      All 9 occupied O2 ε and all 5 occupied CH3 ε compared to the PySCF
      2.14.0 mo_energy pins at 1e-6 Ha — pass (RohfCoupleStage replicates
      get_roothaan_fock term by term). The convention-bound-ε pitfall is
      documented in the test header, rohf.h and rohf_couple_stage.h class
      docs, and the ch03 tutorial; no non-PySCF-convention ε is gated
      anywhere.
  - id: ac-006
    summary: Closed-shell degeneracy and guarded unsupported paths
    type: code
    pass_when: |
      Unit tests pass asserting: ROHF(1,1) on H2/STO-3G equals RHF energy to
      1e-8 Ha; ROHF with n_alpha < n_beta throws std::invalid_argument;
      forces()/mp2() on ROHF throw rather than silently taking a
      Restricted/Unrestricted branch.
    status: verified
    last_checked: 2026-07-21
    note: |
      H2: E(ROHF(1,1)) = E(RHF) = -1.116759307396 Ha within 1e-8 (both also
      match the PySCF pin). ROHF(0,2) throws std::invalid_argument from
      resolve_occupation. forces()/mp2()/fci() throw std::invalid_argument;
      the SpinType dispatch sites (gradient.cpp, integral_derivatives.cpp)
      additionally throw explicitly on RestrictedOpen, and soscf/level_shift
      are refused for ROHF.
  - id: ac-007
    summary: Regression o2_rohf_sto3g passes as out-of-library consumer ≤60 s
    type: runtime
    pass_when: |
      regressions/scf/o2_rohf_sto3g builds behind ATV_BUILD_REGRESSIONS using
      only public headers + link atomiverse, passes via the `regressions`
      target in ≤60 s, and REFERENCE.md cites Roothaan 1960 (DOI
      10.1103/RevModPhys.32.179) plus the pinned PySCF value and command.
    status: verified
    last_checked: 2026-07-21
    note: |
      reg_o2_rohf_sto3g: PASS in 0.24 s wall (public headers + link
      atomiverse only). Prints E dE = 5.7e-14 Ha vs pin, ROHF contamination
      0.0, warm-started UHF dE = 2.0e-13 vs the UHF pin with contamination
      3.410858e-3 > 1e-3, and checks E_UHF < E_ROHF variational ordering.
      REFERENCE.md cites Roothaan 1960 (DOI 10.1103/RevModPhys.32.179),
      Guest–Saunders 1974, Tsuchimochi–Scuseria 2010 (DOI 10.1063/1.3503173)
      and records the PySCF 2.14.0 commands verbatim.
  - id: ac-008
    summary: Tutorial ch03 ROHF section and Doxygen convention note shipped
    type: docs
    pass_when: |
      docs/tutorial/ch03-open-shell.md gains a ROHF section (spin-purity
      narrative, coupling-coefficient ambiguity stated honestly, O2/CH3
      hands-on, chapter-end scalar table rows E_total and ⟨S²⟩ vs PySCF, no
      bar charts) and every new public symbol in rohf.h /
      rohf_couple_stage.h carries @brief/@param(units)/@return, with the
      Guest–Saunders convention recorded in the ROHF class doc.
    status: pending
    note: |
      Written (ch03 ROHF section in Chinese, coupling-coefficient ambiguity
      told honestly, O2/CH3 hands-on, chapter-end scalar table E_total/⟨S²⟩,
      no charts; full Doxygen with units on rohf.h + rohf_couple_stage.h;
      stale "ROHF does not exist" roadmap statements updated) — awaiting the
      docs-flow review before flipping to verified.
---

# Acceptance criteria

ac-002/003/005 的钉定值在实现期由外部 PySCF **2.14.0**(oracle venv 实际版
本;spec 原文 2.13.1,差异见 spec 审计注记)一次性生成并连同几何、命令行写
入测试文件头与 REFERENCE.md;总能量是可移植观测量(1e-8 Ha 门槛),轨道能
仅在 PySCF canonical ROHF(Guest–Saunders)约定下比较(1e-6 Ha)。
ac-004 的自旋纯性是解析恒等式(Roothaan 1960),不依赖任何外部引擎。
