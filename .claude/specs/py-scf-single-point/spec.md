---
slug: py-scf-single-point
status: approved
created: 2026-08-14
domain: python-bindings / scf
queue: claim-completion
deferred: false
depends_on: []
unblocks: [py-scf-single-point]
grilled: false
---

# Spec: `py-scf-single-point` — Gaussian SCF energies + forces from Python

## Summary

The strongest claimed domain (100 PySCF-pinned regressions) has zero Python
surface; `driver.py` names it a future spec and `cli/config.py` hard-rejects
`--hf`. Bind the concrete methods `RHF`, `UHF`, `RKS`, `UKS` as single-point
evaluators — `run(frame)` energy + charges, `forces(frame)` — with basis /
charge / multiplicity / tol as constructor arguments (iron law 3). Lift the
CLI `--hf` rejection.

## Design

| Piece | Location |
| ----- | -------- |
| `bind_scf.cpp`: one params mirror per method (`basis: str → BasisSet` lookup that throws on unknown, charge, multiplicity, tol, max_iter — mirror the C++ ctors exactly), method handles calling `run`/`forces` via the molrs bridge | new `python/src/bindings/bind_scf.cpp`, gated `ATV_HAS_HF` |
| results: energy [eV] to Frame meta (existing C++ behavior), plus a bound last-result view (converged, iterations, E in Ha) mirroring the AM1Result pattern | same file |
| `scf.py` module (`RHF`, `UHF`, `RKS`, `UKS`, functional arg for KS: string → enum, throw on unsupported) | python + lazy map + `__all__` (capability-gated like PW) |
| `single_point(frame, evaluator)` accepts the new evaluators | `driver.py` (drop the "AM1 only" caveat) |
| CLI: remove `HF_REJECT_MSG`; `--hf` configures like other modules | `cli/config.py`, `cli/main.py`, `test_install_cli.py` |
| wheel already builds `ATV_BUILD_HF=ON` — no pyproject change | — |

Forces: return an (N,3) numpy array **and** write `fx/fy/fz` back to the
frame — match whatever the C++ `forces(frame)` path already does; do not
fork conventions.

## Files

- `python/src/bindings/bind_scf.cpp` (new), `bind.cpp`, CMakeLists
- `python/src/atomiverse/scf.py` (new), `driver.py`, `__init__.py`
- `python/src/atomiverse/cli/config.py` (+ CLI tests)
- `python/tests/test_scf.py` (new), `test_surface.py`
- `python/docs/getting_started.md` SCF section

## Tasks

- [ ] Read `cpu/scf/` public ctor surface (basis enum, KS functional enum)
- [ ] bind_scf.cpp: RHF/UHF first (energy + forces + result view), then RKS/UKS
- [ ] python module + driver.single_point extension + CLI unblock
- [ ] pytest: H₂O/STO-3G RHF energy equals the value
      `regressions/scf/h2o_rhf_sto3g` pins (state tol); UHF on a doublet;
      RKS B3LYP smoke; unknown basis/functional throws with a how-to
- [ ] Surface bookkeeping + docs; gates green

## Testing

Python energy gates borrow the existing PySCF-pinned numbers (cite the
regression REFERENCE in the test comment) — no new oracles, no self-freeze.

## Out of scope

- post-HF (MP2/CC) bindings, excited states, Hessians
- periodic HF, GPW, PW (PW has its own row)
- CUDA SCF binding
