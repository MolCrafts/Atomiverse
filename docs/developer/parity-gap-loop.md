# parity-gap-loop

Wave loop closing the gaps between what the harness claims about CPU/CUDA
parity with reference software and what is true on `dev`. Instantiated from
`_templates/oracle-capability-loop-template.rhai` on 2026-08-09, seeded by the
four-track parity assessment (`dev` @ `3f29cc7c`).

Workflow: `.grok/workflows/parity-gap-loop.rhai` ·
Queue: `docs/developer/parity-gap-queue.md` ·
Handoff: `docs/developer/parity-gap-handoff.md`

## What makes this loop different

Unlike the method loops, the oracle is **mixed and per-item** (`oracle:` field
on each slug): `harness` (docs-only truth repair, no pin), `ctest`
(timeout triage), `cpu-parity` (CUDA vs same-settings CPU), `pyscf` / `cp2k` /
`qe` / `lammps` / `literature` (external pins under the provenance law).
Perf gates apply only to slugs whose oracle is a wall twin.

## Per-oracle pointers

| oracle | binding source |
| --- | --- |
| cp2k walls | `.claude/notes/gpw-perf-baseline.md` (only authority; REFERENCE.md perf sections are case-local snapshots) |
| pyscf | `tools/pyscf_oracle/` + `docs/developer/pyscf-method-perf.md` |
| qe | `tools/qe_oracle/` (scaffolded by `pw-perf-qe-install`) |
| lammps | `regressions/md/*/REFERENCE.md` conventions (`lmp` on PATH) |
| cpu-parity | same-settings CPU run in the same binary, tolerance stated in the case |
| provenance law | `.claude/notes/oracle-pin-provenance.md` — a pin is a number the other code printed |

## Binding constraints (also in the workflow law)

- **Freezes honored:** no full CUDA GPW product before the GPW CPU
  architecture freeze; no hybrid/GAPW/PAW/meta-GGA; no periodic exact
  exchange/RSH; no CUDA speed claims from small molecules (ES order step 4).
- **In-flight tree:** the parent working tree carries ~105 uncommitted
  variable-cell/stress files; Merge defers any branch overlapping a dirty
  path; nothing from the in-flight set is ever staged by this loop.
- **Supersede in place:** harness-truth slugs rewrite stale claims at the
  cited lines; append-only corrections are a red.

## Run log

| Date | Wave(s) | Staffed | Merged | Notes |
| --- | --- | --- | --- | --- |
| 2026-08-09 | 1 | 6 | 6 | P0 docs-only harness-truth slugs all merged (825eaba3…156c5f91), 0 deferred; P0 residual is `test-gpw-timeout-triage` (ctest), then P1. |
| 2026-08-09 | 2 | 5 | 1 | `pw-perf-qe-install` merged (6d13a436, `tools/qe_oracle/` live). 3 GH200 slugs deferred with jobs: `reg-cuda-grad-honesty` 1035228 (nvcc compile fail), `cuda-pw-pbe-device` 1035270 (x86_64 libxtb link fail), `cuda-pw-large-cell-measure` 1035278 (running at close). `cuda-md-first-regression` unmerged, parked at `wf/cuda-md-first-regression` (ea17e11a). |
