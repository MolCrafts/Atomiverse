---
spec: py-am1-forces
created: 2026-08-14
criteria:
  - id: ac-001
    summary: AM1 forces from Python match the C++ path
    type: runtime
    pass_when: |
      forces(frame) returns finite forces with sum-zero translation
      invariance and writes fx/fy/fz back; values equal the C++ force path
      (same fixture as the C++ unit gate).
    status: pending
  - id: ac-002
    summary: AM1 BOMD runs under the Python MD driver
    type: runtime
    pass_when: |
      set_evaluator(AM1Evaluator) registers on MD; 5-step H2 NVE smoke
      passes; task codec/materializer parity tests cover AM1-in-MD and the
      cross-kind table is consistent.
    status: pending
  - id: ac-003
    summary: Future-spec markers removed honestly
    type: docs
    pass_when: |
      semiempirical/__init__.py and driver.py no longer advertise gradients
      as future; getting_started documents AM1 MD.
    status: pending
out_of_scope:
  - closed-form NDDO forces
  - RM1 / DFTB bindings
---

# Acceptance — `py-am1-forces`

**Done** means semiempirical BOMD is a Python workflow and the AM1 force
story matches the C++ engine exactly.
