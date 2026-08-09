# parity-gap-loop handoff

Current understanding for the next wave's agents. Rewrite, never append.

- Seeded 2026-08-09 from the parity assessment at `dev` @ `3f29cc7c`.
- No wave has run yet. `## pending` in `parity-gap-queue.md` is priority-true:
  P0 harness-truth (7 docs/triage slugs) → P1 regression honesty + sanctioned
  CUDA (4) → P2 PW perf debt (2, ordered).
- The 5-unit-test-timeout evidence (`test_cpu_gpw_{mbd,hirshfeld,forces,
  stress}`, `test_mpi_pw_kpoint_n4`) came from a loaded login node during a
  full parallel build; treat load as the default hypothesis until the serial
  re-run says otherwise.
- Binding wall numbers live only in `.claude/notes/gpw-perf-baseline.md`; its
  own OT section is one of the stale targets (`doc-gpw-baseline-ot-sync`).
