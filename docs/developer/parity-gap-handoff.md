# parity-gap-loop handoff

Current understanding for the next wave's agents. Rewrite, never append.

- Wave 1 (2026-08-09) merged all six P0 docs-only harness-truth slugs
  (`825eaba3`…`156c5f91`, 0 deferred). The harness docs — HF roadmap CUDA
  rows, `notes.md` perf fossils, `gpw-perf-baseline.md` OT section, DFT
  roadmap §10/§12, CLAUDE.md + architecture.md cuda/gpw truth, four fair-twin
  REFERENCE.md perf sections — are now supersede-in-place current; do not
  re-verify them, verify against them.
- `.claude/notes/gpw-perf-baseline.md` is now internally consistent and the
  single binding wall source (OT ~2.99 s / 1.20×, worst_ratio 2.33 MG MAIN).
- Remaining P0: `test-gpw-timeout-triage` only — a ctest slug, not docs-only.
  The 5-timeout evidence (`test_cpu_gpw_{mbd,hirshfeld,forces,stress}`,
  `test_mpi_pw_kpoint_n4`) came from a loaded login node during a full
  parallel build; load-flake stays the default hypothesis until the serial
  re-run with raised `--timeout` says otherwise.
- After that, P1 in queue order: `reg-cuda-grad-honesty`,
  `cuda-md-first-regression`, `cuda-pw-large-cell-measure`,
  `cuda-pw-pbe-device`. All four need a GH200 batch run — job scripts start
  `#!/bin/bash -l`, load `GPU/buildenv-gcccuda/2026.03-cu13.0`, and a job not
  started within ~20 min means defer with the job id recorded.
- P2 stays ordered: `pw-perf-qe-install` before `pw-perf-fair-walls-batch1`.
- In-flight tree law still holds: the parent working tree carries the
  ~105 uncommitted variable-cell/stress files; never stage or touch them.
