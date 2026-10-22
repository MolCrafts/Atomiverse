# Specs index

Live specs under `.claude/specs/`. A spec is deleted when done; history is the
git log. What is implemented where: `docs/roadmap/`. Unscheduled debt and open
tickets: `.claude/notes/notes.md`.

- [test-consolidation.md](test-consolidation.md) — one executable for unit
  tests, one for regressions, one regression case per method (census gates,
  CPU/CUDA folding, names, REFERENCE.md template, roadmap links).
- [es-unification.md](es-unification.md) — electronic-structure front/back
  split: nested mount, constructor parameters, loops as Drivers, surface,
  composed propagators, CUDA fronts.
- [gaff-m1-open-work.md](gaff-m1-open-work.md) — GAFF milestone 1 (CPU vs
  LAMMPS, CUDA vs OpenMM) remaining phases and the open GPW reds.
- [cpu-orthogonality.md](cpu-orthogonality.md) — CPU component/execution
  contracts; open: electronic execution communicator, PW/GPW borrowed nesting,
  FixMove origin migration, wider perf coverage.
- [md-unification.md](md-unification.md) — one MD execution model across CPU and
  CUDA: Frame edit/commit, typed model parameters, driver-lifetime
  trajectories, device-controlled CUDA loops, retiring `MultiGpu`.
