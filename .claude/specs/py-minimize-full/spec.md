---
slug: py-minimize-full
status: approved
created: 2026-08-14
domain: python-bindings / minimize
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-minimize-full]
grilled: false
---

# Spec: `py-minimize-full` — CG/FIRE/cell-relax + general evaluator dispatch

## Summary

Python minimization is LBFGS-only and its evaluator dispatch accepts only
LJ (`minimizer/__init__.py:26-33`). Bind `CG` (Polak–Ribière⁺) and `FIRE`,
the variable-cell relax variants + `PressureConverged`, and generalize the
Minimize wrapper's evaluator dispatch to every Potential the MD binding
already accepts (shared dispatch table, not a second copy).

## Design

| Piece | Location |
| ----- | -------- |
| `CGParams` / `FIREParams` / cell-relax params (mirror ctors; read `cpu/minimize/{cg,fire}.h` + cell variants first) | `cpu/task/task_spec.h` (extend `MinimizeComponentParams`), wire IDs |
| codec + materializer (`set_optimizer`) | `src/cpu/task/*.cpp` |
| nanobind + `set_optimizer` overloads on `MinimizeHandle` | `python/src/bindings/bind_minimizer.cpp` |
| shared `_EVALUATOR_DISPATCH` (one table, imported by both `md/__init__.py` and `minimizer/__init__.py`) | `python/src/atomiverse/_dispatch.py` (new; internal) |
| `minimizer/__init__.py` re-exports `CG`, `FIRE`, cell variants, `PressureConverged` | python modules + `__all__` |

Cell relax: expose the same public shape C++ has (variable-cell variants
are Optimizer/preset members, not new drivers) — read the C++ examples
before binding; do not invent a `CellOpt` driver.

## Files

- task spec/codec/materializer + tests
- `python/src/bindings/bind_minimizer.cpp`
- `python/src/atomiverse/minimizer/__init__.py`, new `_dispatch.py`,
  `md/__init__.py` (switch to shared table), `__init__.py`
- `python/tests/test_minimizer.py` (grow beyond 5 tests: CG + FIRE reach
  the LJ dimer r_min invariant; cell relax reduces |P|), `test_surface.py`
- `python/docs/getting_started.md`

## Tasks

- [ ] Read optimizer/cell ctors; params + codec + materializer + parity tests
- [ ] nanobind + shared dispatch refactor (no behavior change for MD)
- [ ] pytest: CG and FIRE minimize LJ dimer to r_min = 2^(1/6)σ (invariant);
      cell relax drops |pressure| below tolerance on a small LJ cell
- [ ] Surface bookkeeping + docs; gates green

## Testing

Exact invariants (r_min, |P|→0), no self-freezes. Dispatch refactor gated by
the existing MD python suite staying green.

## Out of scope

- SD (CUDA-only component), TS search, internal coordinates (engine ○)
- CUDA minimize binding
