---
slug: py-am1-forces
status: code-complete
created: 2026-08-14
domain: python-bindings / semiempirical
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-am1-forces]
grilled: false
---

# Spec: `py-am1-forces` — AM1 gradients + AM1 MD from Python

## Summary

Python AM1 was energy + Mulliken only (shipped by this spec: `forces()` +
MD dispatch; the stale "future spec" markers are gone). Expose the C++ AM1
force path
(fixed-density FD per the roadmap; closed-form derivatives are a separate
engine queue `nddo-closed-form-forces`) and let `AM1Evaluator` serve as an
MD force source — semiempirical BOMD from Python.

## Design

| Piece | Location |
| ----- | -------- |
| `forces(frame)` on the bound AM1 evaluator (write-back `fx/fy/fz` + numpy return, matching the C++ convention) | `python/src/bindings/bind_semiempirical.cpp` |
| `AM1EvaluatorParams` already exists in the task layer — extend the MD variant so `set_evaluator(AM1Evaluator)` registers on `cpu::md::MD` (the C++ engine already runs AM1 BOMD; read how the C++ example binds it before wiring) | `cpu/task/task_spec.h` (add AM1EvaluatorParams to `MDComponentParams` + codec/materializer MD branches; wire ID stays 9) |
| MD wrapper dispatch accepts `AM1Evaluator` | `python/src/atomiverse/md/__init__.py` (+ shared dispatch if `py-minimize-full` landed) |
| docstring cleanup: drop the "future spec" markers | `semiempirical/__init__.py`, `driver.py` |

## Files

- `python/src/bindings/bind_semiempirical.cpp`
- `include/atomiverse/cpu/task/task_spec.h`, `src/cpu/task/task_codec.cpp`,
  `src/cpu/task/task_materializer.cpp` + tests (AM1-in-MD branch)
- `python/src/atomiverse/md/__init__.py`, `semiempirical/__init__.py`,
  `driver.py`
- `python/tests/test_semiempirical.py` (forces vs the CPU value the C++
  unit suite gates; finite + reaction to displacement), `test_md.py`
  (5-step AM1 NVE on H₂: energy drift bounded loosely, positions move)
- `python/docs/getting_started.md`

## Tasks

- [x] Bind forces(frame) on AM1; write-back parity with the C++ path
- [x] AM1EvaluatorParams into the MD task variant + codec/materializer +
      parity tests (cross-kind rejection table in task_codec must be
      updated: AM1 is now MD-legal AND SinglePoint-legal)
- [x] MD dispatch + pytest AM1-BOMD smoke
- [x] Docstring/docs cleanup; surface bookkeeping; gates green

## Testing

- Forces: |F| finite, ΣF ≈ 0 (translation invariance), central-atom force
  responds with the right sign to a small displacement.
- BOMD smoke: 5 NVE steps on H₂, energy finite, no NaN.

## Out of scope

- Closed-form NDDO derivative forces (engine queue)
- RM1 binding, AM1-BCC (follow-on rows)
- DFTB/xTB bindings
