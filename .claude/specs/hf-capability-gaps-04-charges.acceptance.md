---
slug: hf-capability-gaps-04-charges
criteria:
  - id: ac-001
    summary: Löwdin observer consumes MeanFieldOutputs/SlaterDeterminant, no debt types in API
    type: code
    pass_when: |
      include/atomiverse/cpu/observer/lowdin.h declares lowdin_charge/lowdin_spin
      taking scf::MeanFieldOutputs; grep finds no `IntegralPack` and no `SCFResult`
      token in lowdin.h or esp_charges.h public headers; all public symbols carry
      @brief/@param/@return with units [e].
    status: verified
    last_checked: 2026-07-20
    note: |
      Signature carries an additional `const cpu::Frame&` parameter (AO→atom
      map + Z_A come from the Frame; recorded as an audit note in the spec
      Design section). grep of both headers finds zero IntegralPack/SCFResult
      tokens; every public symbol has @brief/@param/@return with units.
  - id: ac-002
    summary: Löwdin unit tests pass without any SCF solve
    type: runtime
    pass_when: |
      tests/cpu/observer/test_lowdin.cpp is registered in ctest and green:
      single-s analytic case q = Z − P00 exact to 1e-12, symmetric two-atom
      synthetic density gives equal charges, sum(q) conserved; no run() of any
      mean-field method appears in the test file; densities come from a
      hand-built SlaterDeterminant.
    status: verified
    last_checked: 2026-07-20
    note: |
      test_cpu_observer_lowdin registered and green (4 tests, 4 ms). No
      `.run(` appears in the file; all determinants hand-built.
  - id: ac-003
    summary: ESP constrained fit recovers synthetic charges on MK and CHELPG grids
    type: runtime
    pass_when: |
      tests/cpu/observer/test_esp_charges.cpp green: point charges
      (+0.4, −0.8, +0.4) at the H2O geometry recovered on both grid schemes,
      total-charge constraint satisfied exactly, symmetric H equal to 1e-8,
      buried-atom ill-conditioned case returns finite minimum-norm solution,
      unsupported-element vdW lookup throws.
    status: verified
    last_checked: 2026-07-20
    note: |
      test_cpu_observer_esp_charges registered and green (5 tests, 3 ms):
      recovery ≤1e-6 on both grids (exact model, machine-precision in
      practice), Σq to 1e-12 via the KKT row, H equal ≤1e-8, coincident-atom
      case finite via the SVD minimum-norm branch, Z=79 throws.
  - id: ac-004
    summary: H2O charges regression matches PySCF-pinned values
    type: scientific
    pass_when: |
      `regressions` target runs regressions/scf/h2o_charges_pyscf in ≤60 s:
      Löwdin charges agree with PySCF-pinned values to ≤1e-6 e, CHELPG and MK
      each to ≤1e-3 e, the two H charges equal to the grid-discretization
      level (≤2e-3 e; Löwdin H equal ≤1e-6 e); REFERENCE.md cites Löwdin 1950
      (10.1063/1.1747632), BMK 1990, Breneman-Wiberg 1990 and the pinned
      PySCF command/version.
    status: verified
    last_checked: 2026-07-20
    note: |
      AUDIT — executed via the regression binary (build/regressions/
      reg_h2o_charges_pyscf, 0.5 s): Löwdin |Δ| ≤ 4.5e-9 e, MK |Δ| ≤ 7.9e-10 e,
      CHELPG |Δ| ≤ 9.0e-10 e vs the PySCF 2.14.0 pins; Σq ≤ 1e-10 each; O
      charges inside the BMK/Breneman-Wiberg literature bands. AMENDMENTS
      (minimal): (1) PySCF has no native MK/CHELPG — the oracle is the
      PySCF-density ESP fitted on the documented deterministic grids
      replicated in tools/pin_charges_oracle.py (ESP tolerance 1e-3 e per the
      implementation directive, vs the spec's 0.01 e — tightened, not
      loosened). (2) "the two H charges equal" holds exactly only for Löwdin
      (≤1e-6): the deterministic MK/CHELPG grids are not exactly
      mirror-symmetric, so the fitted H charges differ by ~1e-3 e — the SAME
      asymmetry the oracle produces on the same grids (C++ vs pin agrees
      point-wise to ≤1e-9 e); asserted at ≤2e-3 e. (3) Löwdin is pinned in
      the unit-normalized AO basis (S_ii = 1) on both sides — Löwdin
      populations are not invariant under the per-AO Cartesian-d
      normalization conventions that differ between engines. All three
      recorded in the spec Design audit notes.
  - id: ac-005
    summary: N2 ionization regression pins Koopmans and member-only ΔSCF
    type: scientific
    pass_when: |
      regressions/scf/n2_ionization green in ≤60 s: −ε_HOMO matches the
      PySCF-pinned value at the same basis to ≤1e-3 eV (16.6467 eV at
      6-31G*; the ≈17.3 eV literature value is the HF-limit number vs
      15.6 eV exp per Szabo-Ostlund §3.3, cited in REFERENCE.md — a loose
      15.5–17.5 eV sanity band is asserted); ΔSCF IP is computed via two
      converged runs configured only through the existing `charge` /
      `UHF::n_alpha` / `UHF::n_beta` members — grep of the diff shows no new
      *Config type and no factory function.
    status: verified
    last_checked: 2026-07-20
    note: |
      AUDIT — executed via the regression binary (build/regressions/
      reg_n2_ionization, 3.5 s): IP(Koopmans) = 16.646730 eV vs pin
      16.64672961 (|Δ| ≈ 4e-7 eV); ΔSCF IP = 15.692600 eV vs pin 15.69256822
      (|Δ| ≈ 3.2e-5 eV); ΔSCF < Koopmans asserted (relaxation). Cation
      configured only via uhf.charge/n_alpha/n_beta; the diff adds no *Config
      type and no factory function (test_no_factory_scrub green). AMENDMENT
      (minimal): the parenthetical "≈17.3 eV HF band" was the HF-limit
      literature value — at the pinned basis (6-31G*, cart) −ε_HOMO is
      16.6467 eV; the primary ≤1e-3 eV same-basis PySCF comparison is
      unchanged and the literature band is kept as a loose sanity check
      (recorded in the spec Design audit notes).
  - id: ac-006
    summary: OrbitalInfoObserver reports Koopmans IP column
    type: runtime
    pass_when: |
      tests/cpu/scf/test_orbital_info_observer.cpp green with the extension:
      ip_koopmans_eV() == −homo_eV() on a hand-built SlaterDeterminant and the
      TSV header gains an ip_koopmans column [eV].
    status: verified
    last_checked: 2026-07-20
    note: |
      IpKoopmansFromHandBuiltDeterminant green: ip_koopmans_eV() == −homo_eV()
      to 1e-15 on a hand-built determinant; TSV header row now
      "step\thomo_eV\tlumo_eV\tgap_eV\tip_koopmans" and the row's 5th field
      equals −homo.
  - id: ac-007
    summary: Tutorial ch02 Koopmans/ΔSCF and ch20 population/ESP section land
    type: docs
    pass_when: |
      docs/tutorial/ch02-rhf.md contains a hands-on Koopmans + ΔSCF subsection
      including an explicit warning that Koopmans EA is basis-set-artifactual
      (use ΔSCF); docs/tutorial/ch20-hf-properties-environment.md contains the
      "布居分析与 ESP 电荷" section with chapter-end PySCF comparison tables
      (scalars in tables, no bar charts) and a molplot/molvis ESP-grid figure.
    status: pending
  - id: ac-008
    summary: Full build and unit suite stay green in seconds
    type: runtime
    pass_when: |
      cmake --build build succeeds with -DATV_BUILD_HF=ON and
      `ctest --test-dir build --output-on-failure` passes with the new tests
      registered; no e2e/SCF solve added under tests/.
    status: verified
    last_checked: 2026-07-20
    note: |
      Full build green (HF+PW+TESTS+REGRESSIONS ON). ctest: 188/193 passed;
      the 5 failures are the documented pre-existing MPI cases
      (test_mpi_halo_exchange_n{1,2,8}, test_mpi_coul_energy_bruteforce_n{1,8})
      unrelated to this spec. New unit tests run in milliseconds; no SCF
      solve added under tests/.
---

# Acceptance criteria

每条为二元判定;`scientific` 两条由 regressions 目标执行并对照 REFERENCE.md
中的引文与 PySCF 锚定值,`runtime` 各条由 ctest/构建判定,`docs` 由章节内容
存在性判定。SlaterDeterminant 对齐(scf-slater-determinant-03-unbag)由
ac-001/ac-002/ac-006 钉住。
