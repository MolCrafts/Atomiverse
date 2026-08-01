---
spec: gpw-ts-kpoint-forces
created: 2026-08-01
criteria:
  - id: ac-001
    summary: Water-dimer free-atom TS E_disp matches libMBD pin
    type: runtime
    pass_when: |
      regressions/gpw/gpw-ts-kpoint-forces/reg_gpw-ts-kpoint-forces.cpp gate A
      |E_disp − (−5.9764872295816794e-04 Ha)| ≤ 1e-8 Ha (isolated free-atom
      ts_pbe; same pin as gpw-ts-forces / libMBD cmbd_ts_energy).
    status: passed
  - id: ac-002
    summary: Analytic TS forces dual-gate vs central FD
    type: runtime
    pass_when: |
      max |F_product − F_fd(h=1e-5 Bohr)| ≤ 1e-6 Ha/Bohr and |sum F| ≤ 1e-8
      Ha/Bohr on the isolated water-dimer free-atom TS energy.
    status: passed
  - id: ac-003
    summary: Multi-k force assembly folds free-atom TS
    type: runtime
    pass_when: |
      On H2 cell 12³ Bohr, k_mesh={1,1,2}, PADE + DZV-GTH-PADE: after one
      multi-k SCF, |F(ts) − F(plain) − F_geom| ≤ 1e-8 Ha/Bohr componentwise
      where F_geom is standalone periodic ts_pbe (rcut 50 Å, free-atom
      volumes). SCF n_kpts ≥ 1 and C_occ_k non-empty.
    status: passed
  - id: ac-004
    summary: Unit test covers multi-k TS product fold
    type: runtime
    pass_when: |
      tests/cpu/gpw/test_cpu_gpw_ts.cpp KpointProductForcesFoldTs passes
      (scoped ctest -R test_cpu_gpw_ts).
    status: passed
  - id: ac-005
    summary: Oracle is libMBD / dual-gate — not fake CP2K
    type: doc
    pass_when: |
      REFERENCE.md and tools/cp2k_oracle/inputs/gpw-ts-kpoint-forces/ note
      state CP2K has no stock TYPE TS; pins cite libMBD + dual-gate;
      perf_exempt with that reason; never oracle:cp2k.
    status: passed
  - id: ac-006
    summary: Public API is flags only (no Config / factory)
    type: code
    pass_when: |
      Product surface remains GpwDft::ts + k_mesh + gpw_forces_*; no public
      *Config and no make_* factory for TS k-forces.
    status: passed
---

# Acceptance — `gpw-ts-kpoint-forces`

**acceptance_met = true** (2026-08-01). All criteria passed: regression
`reg_gpw-ts-kpoint-forces` dual-gate + multi-k wiring green; unit
`test_cpu_gpw_ts` KpointProductForcesFoldTs green under ATV_TEST_HEAVY=ON.
Live spec deleted at merge onto `dev`.
