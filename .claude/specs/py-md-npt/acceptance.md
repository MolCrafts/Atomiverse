---
spec: py-md-npt
created: 2026-08-14
criteria:
  - id: ac-001
    summary: PR + Berendsen travel the full task mirror
    type: runtime
    pass_when: |
      task codec roundtrip is field-exact for ParrinelloRahmanNptParams and
      BerendsenBarostatParams; materializer bitwise-parity tests pass for
      both components.
    status: pending
  - id: ac-002
    summary: Python NPT surface is complete
    type: runtime
    pass_when: |
      python suite green with new cases: ParrinelloRahmanNPT via
      set_integrator, BerendsenBarostat via set_barostat, NoseHooverNPT knob
      pass-through (couple_axis/drag/chains/T_stop) observable in behavior.
    status: pending
  - id: ac-003
    summary: Surface bookkeeping stays honest
    type: code
    pass_when: |
      __all__ and test_surface._CANONICAL_ALL updated together; getting_started
      setter table lists every bound integrator/thermostat/barostat.
    status: pending
out_of_scope:
  - CUDA
  - new barostat physics
---

# Acceptance — `py-md-npt`

**Done** means every CPU barostat/NPT integrator the C++ engine claims is
reachable from Python through the launcher-safe task mirror, with parity
tests and honest surface bookkeeping.
