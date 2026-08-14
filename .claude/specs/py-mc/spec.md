---
slug: py-mc
status: approved
created: 2026-08-14
domain: python-bindings / montecarlo
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-mc]
grilled: false
---

# Spec: `py-mc` — Monte Carlo from Python

## Summary

The claimed atomic-fluid MC domain (Metropolis NVT, NPT-MC, μVT GCMC over
one `McBase` trial loop) has zero Python surface. Bind the three presets
and their movers with the same handle pattern as MD. MC is single-rank and
CPU-only by design — no launcher TaskSpec mirror is required for v1
(document the refusal under the MPI launcher gate like the recorder path).

## Design

| Piece | Location |
| ----- | -------- |
| `McHandle` (owns `cpu::mc::MC`; movers via categorical `set_mover` — read `cpu/montecarlo/mc.h` + movers for the real setter names) | new `python/src/bindings/bind_mc.cpp`, gated `ATV_HAS_MC` |
| params mirrors for the mover ctors (displacement / volume / insertion) | binding-local structs (no task mirror v1) |
| `montecarlo.py` (`MC`, movers, result read-back) | python module + lazy map + `__all__` (capability-gated like PW) |
| CMake: `bind_mc.cpp` under `if(ATV_BUILD_MC)` | root CMakeLists nanobind block |

Frame contract identical to MD: molrs Frame in/out; acceptance statistics
read from typed meta (whatever the C++ preset already publishes — bind, do
not invent new keys).

## Files

- `python/src/bindings/bind_mc.cpp` (new), `bind.cpp` dispatch line,
  CMakeLists nanobind sources
- `python/src/atomiverse/montecarlo.py` (new), `__init__.py`
- `python/tests/test_mc.py` (new), `test_surface.py`
- `python/docs/getting_started.md` MC section

## Tasks

- [ ] Read the MC preset + mover public surface; design the handle setters
- [ ] bind_mc.cpp + capability gating + python module
- [ ] pytest: NVT LJ MC on the Ar fixture — acceptance ratio in (0,1),
      energy finite, positions changed; GCMC ⟨N⟩ sanity on the analytic
      low-density limit (loose bound, cite the mc/gcmc_lj invariant)
- [ ] Surface bookkeeping + docs; gates green

## Testing

Behavioral invariants only (acceptance ∈ (0,1), ⟨N⟩ ballpark vs ideal-gas
zμV/Λ³ at low activity) — the tight ensemble physics stays in
`regressions/mc/` where it already lives.

## Out of scope

- Launcher TaskSpec mirror for MC
- CBMC / molecular insertion (engine gap, queued elsewhere)
- Hybrid MC, replica-exchange MC
