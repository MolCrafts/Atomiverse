---
slug: py-coulomb-pme
status: approved
created: 2026-08-14
domain: python-bindings / electrostatics
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-coulomb-pme]
grilled: false
---

# Spec: `py-coulomb-pme` — charged systems from Python

## Summary

Charged systems are unreachable from Python today: no `coul_*` pair style
and no reciprocal-space solver is bound. Bind `PairCoulCut`, `CoulLong`
(real-space Ewald part) and the PME `KSpace` component, and make the wheel
build carry `ATV_BUILD_KSPACE`. Gate: the LAMMPS/analytic-pinned NaCl PME
energy of `regressions/md/npt_pme_nacl` reproduced through the Python
surface.

## Design

| Piece | Location |
| ----- | -------- |
| `PairCoulCutParams`, `CoulLongParams{alpha, cutoff}`, `KSpacePmeParams` (mirror the C++ ctors exactly; read them before writing) | `cpu/task/task_spec.h`, next free wire IDs |
| codec + materializer branches (`set_evaluator` for pair styles; PME via its categorical slot — follow how C++ examples register CoulLong+KSpace as a matched `LongRangeParticipant` pair) | `src/cpu/task/*.cpp` |
| nanobind params + setters | `python/src/bindings/bind_md.cpp` |
| `potentials/pair.py` (`CoulCut`), new `longrange.py` (`CoulLong`, `KSpacePME`) | python modules + lazy map + `__all__` |
| wheel define `ATV_BUILD_KSPACE = {env=..., default="ON"}` | `python/pyproject.toml`; binding sources gated `#ifdef ATV_HAS_KSPACE` like PW |

Charge input: per-atom `charge` column on the molrs Frame (already in the
C++ allow-list); a missing charge column with a Coulomb evaluator must fail
loud with a how-to (verify the C++ path already throws; if it silently
zeros, fix it — iron law 4).

**Constraint:** exactly one matched real/reciprocal pair
(`LongRangeParticipant`) — binding must not allow PME without CoulLong;
mirror whatever validation the C++ driver performs and test the failure
mode from Python.

## Files

- task spec/codec/materializer + unit tests (house pattern)
- `python/src/bindings/bind_md.cpp` (+ `#ifdef ATV_HAS_KSPACE` block)
- `python/src/atomiverse/potentials/pair.py`, new
  `python/src/atomiverse/longrange.py`, `__init__.py`
- `python/pyproject.toml` (KSPACE default ON for the wheel)
- `python/tests/test_md.py` (NaCl mini-cell energy vs the pinned value),
  `test_surface.py`
- `python/docs/getting_started.md`

## Tasks

- [ ] Read C++ ctors + the KSpace/CoulLong registration example; write params
- [ ] Codec/materializer + parity tests RED→GREEN
- [ ] nanobind + python modules (KSPACE capability-gated like PW)
- [ ] pytest: rock-salt mini cell — Python PME energy equals the value the
      C++ regression pins (state tolerance); unmatched-pair failure test
- [ ] Wheel define + surface bookkeeping + docs; gates green

## Testing

- Python energy gate against the number `regressions/md/npt_pme_nacl`
  REFERENCE.md pins (analytic Madelung anchor) — not a new self-freeze.
- Failure-mode tests: missing charge column; PME without CoulLong.

## Out of scope

- CUDA PME binding
- Ewald/PPPM/FMM styles beyond PME (follow-on rows)
- Polarizable electrostatics (PIM), electrodes
