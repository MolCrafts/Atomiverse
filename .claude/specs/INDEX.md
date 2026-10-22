# Specs index

Live specs under `.claude/specs/`. A spec is deleted when done; history is the
git log. What is implemented where: `docs/roadmap/`. Unscheduled debt and open
tickets: `.claude/notes/notes.md`.

- [regression-normalization.md](regression-normalization.md) — migration of
  `regressions/` onto the iron-law-1a contract (census gates, CPU/CUDA folding,
  capability names, REFERENCE.md template, internal includes, roadmap links).
- [gaff-m1-open-work.md](gaff-m1-open-work.md) — GAFF milestone 1 (CPU vs
  LAMMPS, CUDA vs OpenMM) remaining phases and the open GPW reds.
- [cpu-orthogonality.md](cpu-orthogonality.md) — CPU component/execution
  contracts; open: electronic execution communicator, PW/GPW borrowed nesting,
  FixMove origin migration, wider perf coverage.
- [md-unification.md](md-unification.md) — one MD execution model across CPU and
  CUDA: Frame edit/commit, typed model parameters, driver-lifetime
  trajectories, device-controlled CUDA loops, retiring `MultiGpu`.
