---
spec: py-vib
created: 2026-08-14
criteria:
  - id: ac-001
    summary: FD Hessian + frequencies from Python
    type: runtime
    pass_when: |
      pytest computes the LJ-dimer stretch frequency from Python and matches
      the closed-form well-curvature value within a stated tolerance.
    status: pending
  - id: ac-002
    summary: RRHO thermochemistry readable
    type: runtime
    pass_when: |
      ZPE and thermal corrections are readable from the bound result and
      finite for the test system.
    status: pending
  - id: ac-003
    summary: Capability gating honest
    type: code
    pass_when: |
      Extension built with ATV_BUILD_VIB=OFF degrades with the
      install-oriented error; surface bookkeeping updated together.
    status: pending
out_of_scope:
  - analytic Hessians
  - periodic phonons
---

# Acceptance — `py-vib`

**Done** means harmonic analysis + RRHO on any bound force source is a
Python workflow with an analytic frequency gate.
