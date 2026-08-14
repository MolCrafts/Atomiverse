---
spec: py-pair-styles
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Every target style bound end-to-end
    type: runtime
    pass_when: |
      Buckingham, Born, Yukawa, WCA, DPD, BondFENE, DihedralCharmm,
      ImproperHarmonic (and Morse iff production-safe) construct and run
      from Python via set_evaluator; codec/materializer parity tests pass.
    status: pending
  - id: ac-002
    summary: Analytic energy checks per style
    type: runtime
    pass_when: |
      Each bound style has a pytest checking a two-body (or minimal-topology)
      energy against a closed-form value computed in the test.
    status: pending
  - id: ac-003
    summary: Surface bookkeeping consistent
    type: code
    pass_when: |
      __all__, _CANONICAL_ALL, _EVALUATOR_DISPATCH, and getting_started rows
      all updated in the same change.
    status: pending
out_of_scope:
  - EAM / SW / tabulated
  - CUDA
---

# Acceptance — `py-pair-styles`

**Done** means the Python pair/bonded catalogue covers the production styles
of the claimed force-field domain with analytic per-style checks.
