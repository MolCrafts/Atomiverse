---
slug: dftb-data-00-honesty
criteria:
  - id: ac-001
    summary: Fabricated U^d and W constants no longer exist anywhere in the tree
    type: code
    pass_when: |
      basis_data/dftb_mio_tables.h no longer declares DftbElementRow::hubbard_du
      or ::spin_w and the four element rows carry 8 fields, not 10;
      tools/gen_dftb_mio_tables.py contains neither SKETCH_DU nor SKETCH_SPIN_W;
      a repo-wide search for those two identifiers returns zero hits.
    status: verified
    last_checked: 2026-09-04
  - id: ac-002
    summary: DFTB2::spin_w_ha is a public member in Hartree, unset by default
    type: code
    pass_when: |
      dftb2.h declares `std::vector<basis_data::DftbSpinConstant> spin_w_ha{};`
      immediately after u_eff, with a Doxygen @brief naming the unit as
      atomic units [Ha] and stating that empty means unset; DFTB2 has no new
      constructor argument, no *Config type, and no set_* for it.
    status: verified
    last_checked: 2026-09-04
  - id: ac-003
    summary: DFTB3::hubbard_du_ha is a public member in Hartree, unset by default
    type: code
    pass_when: |
      dftb3.h declares `std::array<double, 96> hubbard_du_ha{};` after
      with_forces, with a Doxygen @brief naming [Ha] and "zero means unset";
      DFTB3 gains no constructor argument and no config bag.
    status: verified
    last_checked: 2026-09-04
  - id: ac-004
    summary: Missing U^d or W refuses the run with a copy-pasteable how-to
    type: runtime
    pass_when: |
      test_dftb_published_constants passes: with sk_set already declared as
      DftbSkSet::MioSketch, DFTB3::launch throws std::invalid_argument naming
      hubbard_du_ha when it is unset for a Z present in the frame; DFTB2::launch and DFTB2::td_casida throw naming spin_w_ha when
      the spin/triplet path is taken with an empty table; each message contains
      the literal assignment line from basis_data; and closed-shell
      DFTB2::launch with an empty spin_w_ha does NOT throw.
    status: verified
    last_checked: 2026-09-04
  - id: ac-005
    summary: Published tables ship as citable headers with a provenance prologue
    type: docs
    pass_when: |
      basis_data/dftb_3ob_hubbard_derivs.h and basis_data/dftb_spin_constants.h
      each open with a /// Provenance: block naming the source artifact (not a
      paper table number), the DOI, the licence (CC-BY-SA-4.0 with attribution
      for 3ob), the unit [Ha], the "nothing is parsed at build/test/run time"
      statement, the zeta=4.00-not-implemented warning, the finding that no
      published mio-1-1 U^d set exists, and the two unverified-source flags
      (ct300849w and jp068802p not opened).
    status: pending
  - id: ac-006
    summary: Table values equal the published constants exactly
    type: scientific
    evaluator_hint: "marker: dftb-published-constants"
    pass_when: |
      k3obHubbardDerivsHa is exactly {H -0.1857, C -0.1492, N -0.1535,
      O -0.1575} Ha with every other Z zero, and kDftbSpinConstantsPbeHa
      reproduces the DFTB+ manual PBE column (H W_ss -0.072; C -0.031/-0.025/
      -0.025/-0.023; N -0.033/-0.027/-0.027/-0.026; O -0.035/-0.030/-0.030/
      -0.028) Ha, verified by hard-coded exact-equality unit tests.
    status: pending
  - id: ac-007
    summary: HOAO collapse converts Hartree to eV exactly once
    type: runtime
    pass_when: |
      internal::collapse_spin_w_ev returns -1.959219809711 eV for Z=1,
      -0.625861883658 for Z=6, -0.707496042396 for Z=7 and -0.761918814888
      for Z=8 (tol 1e-12), i.e. W_ss for H and W_pp for C/N/O times
      units::kHartreeInEv.
    status: verified
    last_checked: 2026-09-04
  - id: ac-008
    summary: Nothing is applied silently — published values require an explicit line
    type: code
    pass_when: |
      No constructor, build(), launch() or header initialiser assigns
      k3obHubbardDerivsHa or kDftbSpinConstantsPbeHa to a member; every
      in-tree consumer (regressions, unit tests) performs the assignment
      explicitly in its own source.
    status: verified
    last_checked: 2026-09-04
  - id: ac-009
    summary: tools/gen_dftb_3ob_tables.py regenerates the committed header byte-for-byte
    type: code
    pass_when: |
      `python3 tools/gen_dftb_3ob_tables.py --check` exits 0 against the
      committed basis_data/dftb_3ob_tables.h and exits non-zero with a unified
      diff when the header is perturbed; the script is referenced by zero
      CMakeLists.txt / *.cmake files and never registered with add_test.
    status: verified
    last_checked: 2026-09-04
  - id: ac-010
    summary: 3ob SK points are double and re-emitted at full precision
    type: code
    pass_when: |
      Dftb3obSkPoint has eight double fields, the emitted values carry 17
      significant digits with no `f` suffix, the header's /// Regenerate:
      line states the exact command, and the deferred point-major vs
      channel-major layout decision is recorded in the prologue.
    status: verified
    last_checked: 2026-09-04
  - id: ac-011
    summary: The 3ob double widening is reproducibility, not accuracy
    type: scientific
    evaluator_hint: "marker: dftb2-3ob-organic"
    pass_when: |
      reg_dftb2-3ob-organic still passes and its REFERENCE.md records the
      measured energy movement caused by the double re-emission as < 1e-4 eV,
      together with the statement that float storage bounds at <= 2.03e-06 Ha
      cannot explain the 9.4e-04 Ha residual (~470x) and that the residual
      lives in the electronic assembly. No text in the tree claims float
      truncation explains the 2.6e-2 eV residual.
    status: pending
  - id: ac-012
    summary: CLAUDE.md iron law 1 legalises in-tree parameter data without loosening the ban
    type: docs
    pass_when: |
      CLAUDE.md iron law 1 states that in-tree scientific parameter data is
      first-party code carrying provenance and that tests and regressions
      reading it is not a "third-party read", while still forbidding any read
      of anything outside the repo at build or test time; the wording covers
      both basis_data/ headers and a future data/ directory.
    status: pending
  - id: ac-013
    summary: No REFERENCE.md claims an oracle whose binary never ran
    type: docs
    pass_when: |
      No REFERENCE.md under regressions/semiempirical/ carries `oracle: dftb+`
      with an oracle_version admitting "formula-level" or "no DFTB+ binary";
      `oracle: formula-level` appears nowhere (dftb3-periodic-forces now reads
      `oracle: literature`); every downgraded case carries a separate
      `family_oracle: dftb+` line and a paragraph disclaiming absolute identity.
    status: pending
  - id: ac-014
    summary: The three spin pins are re-taken against a real offline DFTB+ 24.1 run
    type: scientific
    evaluator_hint: "marker: dftb-spin"
    pass_when: |
      tools/tb_oracle/inputs/dftbplus/{spin-dftb2-o2,dftb2-spin-analytic-forces,
      dftb2-spin-analytic-stress}/ each hold dftb_in.hsd + geo.gen + the
      verbatim dftb.out.txt from DFTB+ 24.1 with a SpinConstants block and
      ShellResolvedSpin = No; each REFERENCE.md records the measured ATV-DFTB+
      residual and states `oracle: dftb+` iff that residual <= 1e-3 eV,
      otherwise `oracle: literature` + `family_oracle: dftb+` with the residual
      named as blocking debt; td-dftb-h2 and dftb2-multi-k-forces carry
      re-taken values under their existing honest labels; and
      tools/tb_oracle/README.md lists the new twins.
    status: pending
  - id: ac-015
    summary: DFTB3 pins state the two blockers instead of promising DFTB+ identity
    type: docs
    pass_when: |
      dftb3-water-mio, dftb3-analytic-forces and dftb3-periodic-forces
      REFERENCE.md each name (i) that the run declares sk_set =
      DftbSkSet::MioSketch and therefore pairs mio-1-1 SK/repulsive
      (src/cpu/semiempirical/dftb3.cpp:22-23,38,318-319) with 3ob-class U^d,
      a cross-parameter-set pairing the engine warns about and which is not a
      DFTB+-reproducible Hamiltonian, and (ii) the absent gamma^h X-H damping
      (zeta = 4.00); each names the follow-on slug, restates its absolute
      number as an ATV twin snapshot rather than an oracle pin, and drops the
      stale "no DFTB+ binary on this host" claim.
    status: pending
  - id: ac-016
    summary: spin-dftb2-o2 reproduces the hard-coded published-W goldens
    type: runtime
    evaluator_hint: "marker: spin-dftb2-o2"
    pass_when: |
      reg_spin-dftb2-o2 sets the constants through the public one-liner
      `d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;`, uses only public
      headers, launches no third party, and passes with hard-coded
      E_spin = -0.761918814888 eV (tol 1e-9) and a total-energy shift of
      +0.238081185112 eV (tol 1e-4) relative to the superseded pin
      -142.1382039484 eV. The 1e-4 tolerance is noise margin on an EXACT
      identity, and REFERENCE.md records why: dftb2.cpp:1205-1207 assembles
      E = Tr(rho H0) + E2 + E_spin + E_rep from Tr(rho H0), not the band sum;
      O2 is homonuclear so Delta q == 0 and the spin shift is the same
      constant c on both atoms; apply_shift (dftb2.cpp:490-509) therefore
      yields H^sigma = H0 +/- c S; and (H0 + cS)C = eps S C has the same
      eigenvectors as H0 C = eps S C, so rho, m_A, Tr(rho H0), E2 and E_rep
      are all unchanged and the whole shift is Delta E_spin. A measured
      deviation beyond the tolerance is an iron-law-9 stop-and-report with
      path:line, never a tolerance widening.
    status: verified
    last_checked: 2026-09-04
  - id: ac-017
    summary: SE-only and default builds are green
    type: runtime
    pass_when: |
      The SE-only single-domain configuration (SEMIEMPIRICAL + EVALUATE ON,
      HF/PW/GPW/KSPACE/CUDA OFF, TESTS + REGRESSIONS ON) builds and
      `ctest -R dftb` passes, and the repository's default CPU configuration
      builds with the full ctest suite passing.
    status: verified
    last_checked: 2026-09-04
  - id: ac-018
    summary: DFTB3 never picks an SK set for the user; a user-declared mix only warns
    type: runtime
    evaluator_hint: "marker: dftb3-sk-set"
    pass_when: |
      test_dftb_published_constants passes three branches on a DFTB3 whose
      hubbard_du_ha is filled: (a) sk_set left at its default
      DftbSkSet::Unset makes launch() throw std::invalid_argument whose
      message names sk_set and carries the pasteable line
      `d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;` — and it is
      still the sk_set message when hubbard_du_ha is also unset, proving
      sk_set is checked first; (b) sk_set = DftbSkSet::ThreeOb makes launch()
      throw std::runtime_error naming the follow-on slug
      dftb-gamma-h-damping; (c) sk_set = DftbSkSet::MioSketch does NOT throw,
      returns an energy, and emits exactly one ATV_WARN containing
      "supersedes the MIO parameter set" across two consecutive launch()
      calls (captured via LogConfig::logfile + log_init as in
      tests/cpu/dynamics/test_timing_report.cpp:122-144). A default-
      constructed DFTB2 still has sk_set == DftbSkSet::MioSketch and runs
      unchanged, while an explicit DftbSkSet::Unset on DFTB2 throws.
    status: verified
    last_checked: 2026-09-04
  - id: ac-019
    summary: shell_resolved_spin turns the HOAO collapse into a declared approximation
    type: runtime
    evaluator_hint: "marker: dftb2-shell-resolved-spin"
    pass_when: |
      dftb2.h declares `bool shell_resolved_spin = false;` immediately after
      spin_w_ha with a Doxygen @brief stating that the default path collapses
      the shell-resolved W table to its HOAO diagonal (equivalent to DFTB+
      ShellResolvedSpin = No); setting it true makes both DFTB2::launch and
      DFTB2::td_casida throw std::runtime_error naming the follow-on slug
      dftb-shell-resolved-spin; the default false path runs and reproduces the
      collapse values of ac-007; and a closed-shell run with
      shell_resolved_spin = true does not throw (the spin channel is untaken).
    status: verified
    last_checked: 2026-09-04
---

# Acceptance criteria

- **ac-001 – ac-003** are the honesty core: the fabricated numbers are gone and the replacement is a plain public member, not a config bag.
- **ac-004** encodes DFTB+'s rule — missing parameters refuse the run — and, equally important, that closed-shell DFTB2 is *not* affected, which is what makes lazy validation correct rather than merely convenient.
- **ac-005 – ac-008** keep "no invented defaults" true while making the common case one line: the values ship, cited, but only the user applies them.
- **ac-009 – ac-011** make the 3ob table reproducible and force the honest motivation (reproducibility, not accuracy) into the record.
- **ac-012** amends the harness rule without weakening the intent that ctest never depends on external software or network.
- **ac-013 – ac-016** clear the provenance debt: labels that contradict their own bodies are fixed, the pins whose *values* move are re-taken, and the DFTB3 blockers are named rather than papered over with a pin the spec cannot deliver.
- **ac-017** is the delivery gate.
- **ac-018 – ac-019** are the second half of the honesty thesis, added by the grilling audit: an approximation the *program* chose must throw, an approximation the *user* declared may warn, and an approximation nobody declared at all must stop existing. `sk_set` removes the program-made parameter-set choice; `shell_resolved_spin` removes the silent HOAO fallback.
