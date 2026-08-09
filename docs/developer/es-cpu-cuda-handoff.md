# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev`
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai` (template-shaped GapCensus loop)

## How to run

```text
workflow name=es-cpu-cuda-opt
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 6, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. Re-scan
must be empty before the next GapCensus.

## Open residual (seed — GapCensus owns discovery)

See queue `## pending`. CUDA full GPW product remains blocked. CPU MG MAIN residual
is non-kernel class per baseline.
