---
spec: py-coulomb-pme
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Charged MD runs from Python
    type: runtime
    pass_when: |
      A rock-salt mini cell with charge column runs CoulLong + KSpacePME MD
      from Python; the electrostatic energy matches the analytic/pinned
      value of regressions/md/npt_pme_nacl within a stated tolerance.
    status: pending
  - id: ac-002
    summary: Task mirror + wheel gating are complete
    type: runtime
    pass_when: |
      Codec roundtrip and materializer parity tests pass for the new params;
      the wheel builds with ATV_BUILD_KSPACE=ON and the binding degrades
      cleanly (capability-gated) when KSPACE is OFF.
    status: pending
  - id: ac-003
    summary: Failure modes are loud
    type: runtime
    pass_when: |
      Missing charge column and PME-without-CoulLong each raise with a
      how-to from Python (tested).
    status: pending
out_of_scope:
  - CUDA PME
  - PPPM / Ewald-direct / FMM styles
---

# Acceptance — `py-coulomb-pme`

**Done** means a charged periodic system is a first-class Python workload:
real+reciprocal electrostatics bound as a matched pair, anchored to the
existing analytic pin, failing loud on misuse.
