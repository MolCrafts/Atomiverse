---
slug: py-pw-forces
status: approved
created: 2026-08-14
domain: python-bindings / planewave
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-pw-forces]
grilled: false
---

# Spec: `py-pw-forces` — PW forces/stress accessors + PW as MD evaluator

## Summary

The PW binding stops at single-point SCF energy. Expose the analytic
forces (local + Ewald + KB) and Nielsen–Martin stress the CPU engine
already claims, and let PW serve as an MD force source (plane-wave BOMD
from Python). Also fix the latent surface bug: `__all__` never includes
`PW`/`PWResult` even when built (`__init__.py` static list vs
`test_surface` expectation).

## Design

| Piece | Location |
| ----- | -------- |
| `forces(frame)` / `stress(frame)` accessors on the PW handle (numpy (N,3) + 6-vector; write-back per C++ convention) | `python/src/bindings/bind_pw.cpp` |
| PW-as-evaluator on MD (read how C++ AIMD binds PW before wiring; same capability gate) | `bind_pw.cpp` + MD dispatch |
| `__all__` PW fix: append `PW`/`PWResult` when `hasattr(_cpp, "PW")` | `python/src/atomiverse/__init__.py` (also delete the dead `__dir__` no-op branch) |
| PW-ON test lane: at least one CI-parity run builds the extension with `ATV_BUILD_PW=ON` so the skipped PW tests actually execute | note in spec close; local gate command documented |

## Files

- `python/src/bindings/bind_pw.cpp`
- `python/src/atomiverse/pw.py`, `__init__.py`, `md/__init__.py`
- `python/tests/test_pw.py` (new; skipif not PW), `test_surface.py`
- `python/docs/getting_started.md`

## Tasks

- [ ] Bind forces/stress accessors (mirror `PWResult` conventions; Ry↔eV
      units stated in docstrings)
- [ ] PW evaluator dispatch on MD + BOMD smoke
- [ ] Fix `__all__`/`__dir__` PW bug; test_surface passes under BOTH
      PW=ON and PW=OFF extensions
- [ ] pytest: Si₂ force sanity (ΣF ≈ 0; equilibrium |F| small vs displaced),
      stress finite; values cross-checked against the CPU numbers the
      pw regressions pin
- [ ] Surface bookkeeping + docs; gates green

## Testing

Invariant-first (ΣF = 0, symmetry), plus equality to the C++ path on the
same fixture. No new oracle work — the PW physics pins live in
`regressions/pw/`.

## Out of scope

- USPP/PAW, k-distribution MPI (engine gaps)
- bands/DOS/Berry/MLWF/DFPT bindings (later rows)
- CUDA PW binding
