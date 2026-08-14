---
slug: py-vib
status: approved
created: 2026-08-14
domain: python-bindings / vibration
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-vib]
grilled: false
---

# Spec: `py-vib` — harmonic frequencies + RRHO from Python

## Summary

Bind the claimed vibration domain: `cpu::vib::Hessian` (central-FD 6N over
any force provider) plus normal modes and RRHO thermochemistry read-back.
CPU-only domain; no CUDA column.

## Design

| Piece | Location |
| ----- | -------- |
| `VibHandle` owning the Hessian preset; force source via the shared evaluator dispatch (any bound Potential; AM1 once `py-am1-forces` lands) | new `python/src/bindings/bind_vib.cpp`, gated `ATV_HAS_VIB` |
| result read-back: frequencies [cm⁻¹], modes, ZPE/thermal corrections — bind whatever the C++ preset publishes (Frame meta / result object); do not invent keys | same file |
| `vibration.py` module | python + lazy map + `__all__` (capability-gated) |
| CMake under `if(ATV_BUILD_VIB)` | nanobind block |

## Files

- `python/src/bindings/bind_vib.cpp` (new), `bind.cpp`, CMakeLists
- `python/src/atomiverse/vibration.py` (new), `__init__.py`
- `python/tests/test_vib.py` (new), `test_surface.py`
- `python/docs/getting_started.md`

## Tasks

- [ ] Read `cpu/vibration/` public surface (hessian, thermochem outputs)
- [ ] bind_vib.cpp + python module + capability gating
- [ ] pytest: LJ dimer at r_min → one positive stretch frequency matching
      the analytic ω = sqrt(k/μ) of the LJ well curvature (closed form in
      the test); 5 near-zero trans/rot modes filtered as the C++ suite does
- [ ] Surface bookkeeping + docs; gates green

## Testing

Analytic LJ-dimer frequency — exact invariant, no self-freeze. Tight
literature pins stay in `regressions/scf` freq cases.

## Out of scope

- Analytic Hessians (engine ○ outside RHF/UHF CPHF)
- Periodic phonons (PW/DFPT surface, separate)
- CUDA
