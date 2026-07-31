---
spec: gpw-d3-atm-stress
created: 2026-07-31
criteria:
  - id: ac-001
    summary: Offline CP2K numerical-stress ATM pin exists and is provenanced
    type: docs
    pass_when: |
      regressions/gpw/gpw-d3-atm-stress/REFERENCE.md records two CP2K 2026.1
      STRESS_TENSOR NUMERICAL runs (C9 .TRUE./.FALSE., identical otherwise)
      with engine string, inputs under tools/cp2k_oracle/inputs/
      gpw-d3-atm-stress/, printed dispersion energies, both stress tables,
      and the derived delta-sigma diagonal in Ha/Bohr^3. Pins are numbers
      CP2K printed — never seeded from ATV.
    status: met
  - id: ac-002
    summary: ATV strain-FD of the ATM-only energy matches the CP2K delta
    type: runtime
    pass_when: |
      reg_gpw-d3-atm-stress computes central strain-FD (fixed fractional
      coordinates) of dftd3_pbe(C9 on) - dftd3_pbe(C9 off) on the per-6 A
      cell and matches the CP2K delta-sigma diagonal within the REFERENCE
      tolerance (target ~1e-12 Ha/Bohr^3, justified from the measured
      residual with stated separation from the signal).
    status: met
  - id: ac-003
    summary: E_disp pins (ATM on and off) reproduce the CP2K printed values
    type: runtime
    pass_when: |
      Standalone dftd3_pbe energies (C9 on/off, periodic 6 A cell, R_CUTOFF
      10.0 doubled convention) match both runs' printed "Dispersion energy"
      to <= 1e-9 Ha, and the product e_disp (GPWEvaluator d3+d3_atm) matches
      the ATM-on pin.
    status: met
  - id: ac-004
    summary: Product analytic fixed-P stress with d3_atm passes the dual gate
    type: runtime
    pass_when: |
      GPWEvaluator{d3, d3_atm, compute_stress, stress_use_fd=false} on the
      pinned cell converges; off-diagonals ~0; kStress eV/A^3 conversion
      consistent; |sigma_an - sigma_fd| on the same method (gpw_stress_fd)
      within the REFERENCE gate (<= ~1e-6 .. 2e-5 Ha/Bohr^3, d3bj-stress
      scale); regression wall <= 60 s.
    status: met
  - id: ac-005
    summary: Cheap periodic unit twin covers d3_atm under strain
    type: runtime
    pass_when: |
      tests/cpu/gpw/test_cpu_gpw_stress.cpp gains
      GpwStress.AnalyticMatchesFdWithD3Atm on a small periodic cell (image
      triples engage); analytic vs gpw_stress_fd diagonal within the unit
      gate; suite stays fast (single SCF at low cutoff).
    status: met
  - id: ac-006
    summary: Perf within 10x of the oracle
    type: runtime
    pass_when: |
      REFERENCE.md records CP2K wall (numerical-stress ATM run) and the ATV
      product analytic pass wall on the same host class; T_atv/T_cp2k <= 10
      (or perf_exempt with reason).
    status: met
  - id: ac-007
    summary: Queue row closes on merge
    type: docs
    pass_when: |
      docs/developer/cp2k-method-queue.md moves gpw-d3-atm-stress to done
      when merged to dev.
    status: met
out_of_scope:
  - D3Result::virial three-body term (fixed-P path rebuilds energy instead)
  - D4 ATM stress; D3(BJ)+ATM pin; k-point dispersion-stress pin
---
