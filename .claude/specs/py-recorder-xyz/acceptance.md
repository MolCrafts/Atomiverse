---
spec: py-recorder-xyz
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Gating decision recorded and enforced
    type: code
    pass_when: |
      The spec records which gating shape shipped; an IO-less build (SE-only
      configuration) still compiles and either refuses XyzRecorder loudly at
      materialize time or never exposes it — no undefined references, no
      silent no-op.
    status: pending
  - id: ac-002
    summary: XYZ trajectory written from Python
    type: runtime
    pass_when: |
      pytest runs 10 MD steps with XYZ(path, freq=5) and parses the expected
      frame count and per-frame atom count from the file.
    status: pending
  - id: ac-003
    summary: Component unit-tested at the C++ level
    type: runtime
    pass_when: |
      tests/cpu/record/test_xyz_recorder.cpp covers cadence, truncate-then-
      append, and content sanity.
    status: pending
out_of_scope:
  - cube/Zarr recorders
  - CUDA
---

# Acceptance — `py-recorder-xyz`

**Done** means Python MD can produce a trajectory file through a real
Recorder component whose build gating is explicit and loud.
