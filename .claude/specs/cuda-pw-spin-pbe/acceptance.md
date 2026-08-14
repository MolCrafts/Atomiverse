---
spec: cuda-pw-spin-pbe
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Spin-PBE forces/stress compute on device
    type: runtime
    pass_when: |
      The spin-PBE force and stress rejections are lifted; device results
      match CPU within stated tolerances on the parity units.
    status: pending
  - id: ac-002
    summary: Spin×smear joint follows
    type: runtime
    pass_when: |
      The spin+smear combination computes (or the spec close records it
      re-scoped with the residual named).
    status: pending
  - id: ac-003
    summary: External anchor on the CUDA leg
    type: runtime
    pass_when: |
      Existing spin-PBE PW regressions pass with --device=cuda against the
      same anchors as CPU on GH200; roadmap CUDA column updated honestly.
    status: pending
out_of_scope:
  - USPP/PAW
  - device bands/DOS
---

# Acceptance — `cuda-pw-spin-pbe`

**Done** means the device PW envelope no longer rejects spin-GGA forces and
stress, dual-anchored.
