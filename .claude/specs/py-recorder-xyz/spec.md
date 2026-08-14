---
slug: py-recorder-xyz
status: approved
created: 2026-08-14
domain: python-bindings / record
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-recorder-xyz]
grilled: false
---

# Spec: `py-recorder-xyz` — trajectory output from Python

## Summary

Python MD produces no trajectory file today (`Thermo` is the only bound
recorder). Deliver an XYZ trajectory Recorder component and bind it. A
**design decision gates the implementation** (resolve in step 1, do not
skip): `cpu::XYZ` is a molrs-backed bare writer compiled only under
`ATV_BUILD_IO`, while the task layer (`task_spec.h` / `task_codec.cpp` /
`task_materializer.cpp`) compiles unconditionally — a task-mirrored
XyzRecorder must not create an undefined reference in IO-less builds
(the SE-only working configuration is binding).

## Design

**Step 1 — gating decision (pick one, record in this spec on close):**

- (a) Move the task layer's XyzRecorder support behind `ATV_BUILD_IO`
  cleanly (params struct + wire ID stay unconditional; materializer branch
  throws "built without ATV_BUILD_IO" when OFF — symbol-free via `#ifdef`
  around only the component construction, wire format unchanged), or
- (b) Local-only component: bind without a task mirror; `MdHandle` refuses
  it under the MPI launcher gate with a clear error.

Default recommendation: (a) — keeps one registration story; the throw is a
loud runtime refusal, not silent absence.

**Step 2 — component:** `cpu::XyzRecorder : cpu::Recorder` in
`cpu/record/xyz_recorder.{h,cpp}` (IO block), ctor
`(std::string filename, int freq)`, truncate on first write then append,
delegating to `cpu::XYZ`. Roles via Recorder base (ExternalIO). MPI: rank-0
writes only (mirror `Thermo`).

**Step 3 — binding:** params `{filename, freq}`, `set_record` overload,
`recorder.py` gains `XYZ` (capability check: raise the install-oriented
error if the extension was built without IO — probe a sentinel attr).

## Files

- `include/atomiverse/cpu/record/xyz_recorder.h`,
  `src/cpu/record/xyz_recorder.cpp` (+ CMake in the ATV_BUILD_IO block)
- `tests/cpu/record/test_xyz_recorder.cpp` (freq cadence, truncate/append,
  rank-0 discipline testable single-rank)
- task spec/codec/materializer per the step-1 decision + tests
- `python/src/bindings/bind_md.cpp`, `python/src/atomiverse/recorder.py`,
  `__init__.py`, `python/tests/test_md.py`, `test_surface.py`
- `python/docs/getting_started.md`

## Tasks

- [ ] Decide + record the gating shape (step 1)
- [ ] `cpu::XyzRecorder` + unit tests RED→GREEN
- [ ] Task mirror per decision + parity/refusal tests
- [ ] nanobind + python + pytest (file exists, frame count = steps/freq + 1,
      atom count per frame correct)
- [ ] Surface bookkeeping + docs; gates green

## Testing

- C++ unit: cadence and append semantics against a tmp file.
- Python: run 10 steps at freq 5 → 3 frames; parse atom count per frame.

## Out of scope

- Cube writers, Zarr trajectories (separate rows)
- CUDA
- New trajectory formats
