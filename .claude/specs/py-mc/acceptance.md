---
spec: py-mc
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Three MC presets run from Python
    type: runtime
    pass_when: |
      NVT, NPT-MC, and GCMC each construct, run, and write results back to
      the molrs Frame from Python; extension built without ATV_BUILD_MC
      degrades with the install-oriented error.
    status: pending
  - id: ac-002
    summary: Behavioral invariants gated
    type: runtime
    pass_when: |
      pytest checks acceptance ratio in (0,1) for NVT and a low-activity
      ideal-gas bound for GCMC <N>.
    status: pending
  - id: ac-003
    summary: Surface bookkeeping consistent
    type: code
    pass_when: |
      __all__/_CANONICAL_ALL/getting_started updated together; MC bindings
      compiled only under ATV_BUILD_MC.
    status: pending
out_of_scope:
  - launcher mirror
  - CBMC / molecular MC
---

# Acceptance — `py-mc`

**Done** means the claimed atomic-fluid MC domain is drivable from Python
with honest capability gating and behavioral invariants.
