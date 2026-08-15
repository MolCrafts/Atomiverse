---
spec: docs-python-surface
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Three runnable examples for always-available surfaces
    type: runtime
    pass_when: |
      md_lj_argon.py, minimize_lj_dimer.py, am1_ch4.py each run to exit 0
      against the dev build and self-check their result.
    status: verified
    last_checked: 2026-08-15
  - id: ac-002
    summary: Python guide is published
    type: docs
    pass_when: |
      The Python getting-started guide appears in the zensical nav from one
      canonical location; user-guide index links it.
    status: pending
  - id: ac-003
    summary: Docs name reality
    type: docs
    pass_when: |
      The Examples section lists exactly the scripts that exist; no plural
      claims over a single file.
    status: pending
out_of_scope:
  - narrative tutorials
---

# Acceptance — `docs-python-surface`

**Done** means a new user can find the published Python guide and run three
working examples of the claimed surfaces.
