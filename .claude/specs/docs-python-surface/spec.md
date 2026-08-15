---
slug: docs-python-surface
status: code-complete
created: 2026-08-14
domain: docs / python
queue: claim-completion
deferred: false
depends_on: []
unblocks: [docs-python-examples, docs-python-nav]
grilled: false
---

# Spec: `docs-python-surface` — examples + published Python docs

Covers both Track D rows `docs-python-examples` and `docs-python-nav`.

## Summary

`python/examples/` holds exactly one script — for the backend that is OFF
by default — and `python/docs/getting_started.md` sits outside the zensical
`docs_dir`, so the package's only user documentation is unpublished. Ship
runnable examples for the always-available surfaces and put the Python
guide into the published site nav.

## Design

- `python/examples/md_lj_argon.py` — build Ar frame (molrs), NVT via
  Langevin, thermo output, read energy/kinetic back. Mirrors
  `examples/07_md_lj.cpp`.
- `python/examples/minimize_lj_dimer.py` — LBFGS to r_min, print distance
  vs 2^(1/6)σ.
- `python/examples/am1_ch4.py` — AM1 single point, print ΔHf-style energy
  and Mulliken charges.
- Each script: MolCrafts stack only (no ASE/matplotlib per pyproject
  comment), runnable headless, < 30 s, exits nonzero on failed sanity
  check so it can be smoke-run.
- Nav: register the Python guide in `zensical.toml` (either move
  `python/docs/getting_started.md` under `docs/user-guide/python.md` or add
  the path — pick the layout the theme supports; one canonical location,
  no duplicated content).
- `docs/user-guide/index.md` card for the Python guide.

## Files

- `python/examples/{md_lj_argon,minimize_lj_dimer,am1_ch4}.py`
- `zensical.toml`, `docs/user-guide/index.md`
  (+ moved/linked python guide page)
- `python/docs/getting_started.md` "Examples" section updated to name the
  real scripts

## Tasks

- [x] Three example scripts, each self-checking
- [x] Publish the Python guide in the site nav (one canonical location)
- [x] Smoke-run all examples against the dev build; fix drift
- [x] getting_started Examples section names them

## Testing

Manual smoke-run recorded at close (scripts exit 0); no ctest registration
(examples are not tests — iron law 1).

## Out of scope

- Narrative tutorials (tier C, /mol:docs Mode B)
- Sphinx/API-doc generation
