# Reference — arch-e2e-tests-wave2 harness

## Purpose

Out-of-library acceptance twin for **e2e-in-tests wave 2**: residual multi-iter
SCF / MPI PW·GPW campaigns under `tests/` are deleted or slimmed; the unit gate
stays seconds-scale; literature / industrial pins remain in `regressions/**`.

This binary is a **link + public API smoke** (construct `RHF`, no solve). It is
not a physics pin. Evidence of the wave is the tree delta + campaign note.

## Oracle

`oracle: harness` — architecture campaign twin; no external engine.

## Gate

- Links `atomiverse` as an out-of-library consumer (public headers only).
- Exit 0 after constructing a default `cpu::scf::RHF`.
