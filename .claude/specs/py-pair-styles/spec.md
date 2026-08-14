---
slug: py-pair-styles
status: approved
created: 2026-08-14
domain: python-bindings / potentials
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-pair-styles]
grilled: false
---

# Spec: `py-pair-styles` — widen the bonded/pair catalogue

## Summary

Python binds 4 of ~21 real-space styles. Bind the production styles of the
claimed force-field domain: pair `Morse`* (*demo-style O(N²) — bind only if
its header does not forbid production use; otherwise skip and note),
`Buckingham`, `Born`, `Yukawa`, `WCA`, `DPD`; bonded `BondFENE`,
`DihedralCharmm`, `ImproperHarmonic`. Mirror each C++ ctor exactly
(read headers first — per-type vector params follow the BondHarmonic
pattern).

## Design

House pattern per style (identical to landed slices):
task-spec `*Params` + wire ID → codec write/read → materializer
(`set_evaluator` / phase escape hatch) → nanobind class + `set_evaluator_*`
lambda + `_EVALUATOR_DISPATCH` entry in `python/src/atomiverse/md/__init__.py`
→ `potentials/{pair,bond,dihedral,improper}.py` re-export → `__all__` +
surface count.

DPD note: its thermostat forces need `dt`/seed — mirror the C++ ctor, and
carry the conservative-only-virial caveat into the docstring (documented
design, `tests/cpu/potentials/test_pair_virial.cpp`).

## Files

- `include/atomiverse/cpu/task/task_spec.h`, `src/cpu/task/task_codec.cpp`,
  `src/cpu/task/task_materializer.cpp` + their unit tests
- `python/src/bindings/bind_md.cpp`
- `python/src/atomiverse/potentials/*.py`, `md/__init__.py`, `__init__.py`
- `python/tests/test_md.py` (ctor + one-energy smoke per style),
  `test_surface.py`
- `python/docs/getting_started.md` table rows

## Tasks

- [ ] Read every target ctor; write params + wire IDs; codec RED→GREEN
- [ ] Materializer + parity tests
- [ ] nanobind + dispatch + python modules
- [ ] pytest per style: two-atom energy against the closed-form value at one
      separation (hand-computed in the test, not a self-freeze)
- [ ] Surface bookkeeping + docs; gates green

## Testing

Per-style closed-form two-body energy checks (e.g. Buckingham
A·exp(−r/ρ) − C/r⁶ at r = 3 Å) — analytic, not pinned to ATV output.

## Out of scope

- EAM/SW (per-element data tables need a data-path design from Python)
- Tabulated styles, class-II cross terms (engine ○)
- CUDA styles
