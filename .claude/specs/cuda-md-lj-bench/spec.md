---
slug: cuda-md-lj-bench
status: approved
created: 2026-08-14
domain: cuda / classical MD perf
queue: claim-completion
deferred: true
depends_on: []
unblocks: [cuda-md-lj-bench, docs-roadmap-legend]
grilled: false
---

# Spec: `cuda-md-lj-bench` — restore the LJ performance evidence

**Deferred:** needs the GH200 node (sbatch). Classical-MD CUDA track — not
blocked by ES policy.

## Summary

The roadmap's only ⚡ mark (LJ 12-6, CPU xsimd + CUDA) cites a removed
suite (`/* benchmarks removed; re-add later */ lj/performance/`). A perf
claim without a runnable benchmark is void (claim bar 5). Restore a
standing LJ benchmark with the A/B methodology of
`.claude/notes/cuda-openmm-md-perf.md` (precision-mode map vs LAMMPS
Kokkos) and re-point the roadmap evidence at it.

## Design

- Home: `benchmarks/lj/` (ATV_BUILD_BENCHMARKS-gated; never in tests/).
- Cases: CPU xsimd single-node + CUDA tile engine, sizes per the A/B note
  (large systems only for CUDA claims), fixed seeds, JSON/CSV output with
  atoms/steps/walltime and the precision mode.
- LAMMPS twin decks under `tools/` per the oracle-pin law (a perf pin is a
  number the other binary printed: record LAMMPS version + input + output
  line; never reimplement).
- Roadmap: `force-fields.md` ⚡ evidence path updated; keep ⚡ only if the
  measured ratio still supports it — otherwise downgrade honestly to ✓.

## Files

- `benchmarks/lj/…` (+ CMake), `tools/gpu_job_lj_bench.sh` (sbatch)
- `tools/lammps_oracle/` twin decks (if not already present for these sizes)
- `docs/roadmap/force-fields.md`, `docs/roadmap/index.md` evidence pointers

## Tasks

- [ ] Rebuild the benchmark targets (CPU + CUDA) with stable output format
- [ ] GH200 batch run + LAMMPS twin run; record both raw outputs
- [ ] Update roadmap evidence; keep or downgrade ⚡ per the numbers
- [ ] Wire a repeatable `sbatch` script; document the invocation

## Testing

The benchmark itself is the deliverable; correctness stays gated by the
existing LJ unit/regression suite (no perf assertions in ctest).

## Out of scope

- New force-field physics; PME benchmarks (separate)
- CI-hosted GPU runs
