---
spec: py-scf-single-point
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Four methods run from Python with ctor-arg parameters
    type: runtime
    pass_when: |
      RHF/UHF/RKS/UKS construct with basis/charge/multiplicity/tol as
      constructor arguments and run(frame)/forces(frame) work on molrs
      Frames; no Config bags, no factories.
    status: pending
  - id: ac-002
    summary: Energies match the existing PySCF pins
    type: runtime
    pass_when: |
      pytest H2O/STO-3G RHF energy equals the regressions/scf/h2o_rhf_sto3g
      pinned value within the stated tolerance; failure modes (unknown
      basis/functional, bad multiplicity) throw with how-tos.
    status: pending
  - id: ac-003
    summary: CLI hf rejection lifted coherently
    type: runtime
    pass_when: |
      python -m atomiverse config hf configures ATV_BUILD_HF like other
      modules; CLI tests updated; the 13-module CLI list no longer contains
      a hard-rejected entry whose symbols exist.
    status: pending
out_of_scope:
  - post-HF / excited states
  - CUDA
---

# Acceptance — `py-scf-single-point`

**Done** means the flagship SCF domain is a Python single-point + forces
surface anchored to its existing PySCF pins.
