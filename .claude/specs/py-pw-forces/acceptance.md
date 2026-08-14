---
spec: py-pw-forces
created: 2026-08-14
criteria:
  - id: ac-001
    summary: PW forces and stress readable from Python
    type: runtime
    pass_when: |
      forces/stress accessors return finite values equal to the CPU path on
      the same fixture; sum-zero and units documented.
    status: pending
  - id: ac-002
    summary: PW drives MD from Python
    type: runtime
    pass_when: |
      A short plane-wave BOMD smoke runs via set_evaluator(PW); energy
      finite, positions move.
    status: pending
  - id: ac-003
    summary: PW __all__ bug fixed both ways
    type: runtime
    pass_when: |
      test_surface passes with a PW=ON extension and a PW=OFF extension;
      __all__ contains PW/PWResult exactly when built.
    status: pending
out_of_scope:
  - USPP/PAW
  - bands/DOS/response bindings
---

# Acceptance — `py-pw-forces`

**Done** means the PW binding matches the claimed CPU PW capability tier
(energy + forces + stress + BOMD) and the PW surface bug is dead in both
build flavors.
