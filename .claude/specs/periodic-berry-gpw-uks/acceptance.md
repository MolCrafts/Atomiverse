---
spec: periodic-berry-gpw-uks
created: 2026-08-01
criteria:
  - id: ac-001
    summary: Collinear UKS dual-channel KSV path is public and product
    type: code
    pass_when: |
      BerryPolarization::compute(GpwDft&) runs when is_uks() with per-spin
      string phases and quantum-1 wrap; no *Config / make_* surface.
    status: met
  - id: ac-002
    summary: Unit gates cover UKS empty manifold
    type: runtime
    pass_when: |
      test_cpu_gpw_ksv rejects n_alpha+n_beta==0 (and metallic smear) without
      full e2e SCF.
    status: met
  - id: ac-003
    summary: Regression dual-gates symmetry pins on UKS Si
    type: runtime
    pass_when: |
      reg_periodic-berry-gpw-uks: SCF converges; ionic bookkeeping exact;
      centrosymmetric |p_i| distance to {0,1/2} lattice < 0.08; |Z*|<0.5.
    status: met
  - id: ac-004
    summary: Offline CP2K UKS twin + fair wall within 10x
    type: docs
    pass_when: |
      tools/cp2k_oracle/inputs/si_periodic_berry_gpw_uks.inp actually run with
      cp2k.psmp 2026.1; REFERENCE records binary+version+input+output line and
      T_atv/T_oracle ≤ 10 on the single ENERGY twin vs one string-mesh direction.
    status: met
out_of_scope:
  - Metallic Berry / smearing
  - Multiplicity ≠ 1 Born physics
  - CUDA / PW-only Berry
  - Live CP2K in ctest
---

# Acceptance — `periodic-berry-gpw-uks`

**Done** means collinear UKS GPW Berry-phase polarization is dual-gated on
diamond Si (symmetry pins) with a real CP2K Quickstep `&MOMENTS PERIODIC`
UKS twin and fair wall ≤ 10×.
