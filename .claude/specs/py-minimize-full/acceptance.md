---
spec: py-minimize-full
created: 2026-08-14
criteria:
  - id: ac-001
    summary: CG and FIRE reachable and correct from Python
    type: runtime
    pass_when: |
      pytest minimizes the LJ dimer with CG and with FIRE to
      r_min = 2^(1/6) sigma within tolerance; codec/materializer parity
      tests pass for the new optimizer params.
    status: pending
  - id: ac-002
    summary: Variable-cell relax bound
    type: runtime
    pass_when: |
      A small LJ cell relaxes from Python with |pressure| below the stated
      tolerance; PressureConverged is importable and effective.
    status: pending
  - id: ac-003
    summary: One evaluator dispatch table
    type: code
    pass_when: |
      md and minimizer wrappers share a single dispatch table; the Minimize
      wrapper accepts every Potential the MD wrapper accepts; MD suite
      unchanged-green across the refactor.
    status: pending
out_of_scope:
  - TS search / internal coordinates
  - CUDA
---

# Acceptance — `py-minimize-full`

**Done** means Python geometry optimization matches the claimed CPU
minimize domain: three optimizers, cell relax, and any bound Potential as
the force source.
