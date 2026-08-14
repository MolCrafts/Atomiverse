---
slug: py-md-npt
status: approved
created: 2026-08-14
domain: python-bindings / classical MD
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-md-npt]
grilled: false
---

# Spec: `py-md-npt` — remaining barostat/thermostat bindings

**Queue row:** `docs/developer/claim-completion-queue.md` Track P, `py-md-npt`.
**Landed slice (2026-08-14):** `Langevin`, `CSVR`, MTK `NoseHooverNPT`
(wire IDs 11–13). This spec covers the residual.

## Summary

Complete the NPT/thermostat Python surface for the claimed classical-MD
domain: bind `ParrinelloRahmanNPT` (full-cell NPT) and the Berendsen
isotropic barostat, and expose the NoseHooverNPT tail knobs that stayed at
C++ defaults in the first slice (per-axis `couple_axis`, `drag`,
`mtchain`/`mpchain`, `T_stop` ramp).

## Design

Same shape as the landed slice — every new component goes through the full
mirror stack, no shortcuts:

| Piece | Location |
| ----- | -------- |
| `ParrinelloRahmanNptParams{T_target, P_target, tau_T, tau_P, dt}` | `cpu/task/task_spec.h`, wire ID 14 |
| `BerendsenBarostatParams` (mirror `cpu::md::Barostat` ctor exactly) | `cpu/task/task_spec.h`, wire ID 15 |
| extended `NoseHooverNptParams` (`drag`, `mtchain`, `mpchain`, `couple_axis[3]`, `T_stop`) | **wire-compatible growth is NOT required** — no released consumers; extend the struct and codec in place |
| codec write/read + `decode_component_id` cases | `src/cpu/task/task_codec.cpp` |
| materializer branches (PR → `set_integrator`, Berendsen → `set_barostat`) | `src/cpu/task/task_materializer.cpp` |
| nanobind params + setter overloads | `python/src/bindings/bind_md.cpp` |
| `ParrinelloRahmanNPT` in `md/integrator.py`, `BerendsenBarostat` in new `md/barostat.py` | python modules + `__init__` lazy map + `__all__` |

`ParrinelloRahmanNPT` is an Integrator (replaces the plain integrator, like
NoseHooverNPT); Berendsen is a `Barostat` kind installed via `set_barostat`
alongside `VelocityVerlet` + a thermostat.

## Files

- `include/atomiverse/cpu/task/task_spec.h`, `src/cpu/task/task_codec.cpp`,
  `src/cpu/task/task_materializer.cpp`
- `python/src/bindings/bind_md.cpp`
- `python/src/atomiverse/md/integrator.py`, new `md/barostat.py`,
  `python/src/atomiverse/__init__.py`
- `tests/cpu/test_task_codec.cpp`, `tests/cpu/test_task_materializer.cpp`
  (bitwise-parity cases per new component)
- `python/tests/test_md.py`, `python/tests/test_surface.py` (`__all__` count)
- `python/docs/getting_started.md` setter table rows

## Tasks

- [ ] Params + wire IDs 14–15 + extended NPT knobs; codec roundtrip RED→GREEN
- [ ] Materializer branches + bitwise-parity unit tests
- [ ] nanobind bindings + python modules + `__all__`/surface-count update
- [ ] pytest: PR-NPT smoke (box volume responds), Berendsen smoke, knob pass-through
- [ ] getting_started rows; full unit gate + python suite green

## Testing

- Unit: codec field-exact roundtrip; materializer bitwise parity vs
  hand-built driver (house pattern).
- Python: NPT smoke on the Ar fixture — energy finite, `frame.box` volume
  changes under a nonzero pressure offset.

## Out of scope

- CUDA barostats (Track C `cuda-md-npt`)
- New C++ barostat physics — binding only
- MC barostats, anisotropic NPH (engine gaps, not binding gaps)
