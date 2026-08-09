# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` (queue GapCensus wave 2 + Recover)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai` (template-shaped GapCensus loop)
- **Last action:** dual-track GapCensus wave 2 (2026-08-10) refilled pending from
  baseline; Recover moved P0–P2 into `## in_progress` (resume incomplete PW
  tips + fresh PW stream); `builds_wiped=true`

## How to run

```text
workflow name=es-cpu-cuda-opt
args: { "mode": "exhaust", "staff_count": 3, "max_waves": 6, "target_branch": "dev" }
agent_budget: 256+
```

Staff order from `## in_progress`: `cuda-pw-pbe-device` (P0 resume) →
`cuda-pw-large-cell-measure` (P1 resume) → `cuda-pw-nondefault-stream` (P2 fresh).
Still pending: `cuda-rhf-nondefault-stream` (P3), optional
`cpu-simd-2center-setup` (P3 slab). Re-base incomplete tips onto
`wf/escuda-<slug>` off current `dev` before merge.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. Re-scan
must be empty before the next GapCensus. Current parent scan: no `build*` under
`.claude/worktrees` maxdepth 3.

## Open residual (GapCensus owns discovery)

| Track | Residual | Action |
| --- | --- | --- |
| **cpu** | MG MAIN **2.33×** | **blocked** pin-safe surface exhausted (`mg-main-structural`) |
| **cpu** | slab 2-center S/T | optional P3 `cpu-simd-2center-setup` |
| **cuda** | PW still LSDA-only on `dev` | P0 `cuda-pw-pbe-device` (re-base incomplete tip) |
| **cuda** | no large-cell wall yet | P1 `cuda-pw-large-cell-measure` (ES order step 4) |
| **cuda** | PW default-stream residual | P2 `cuda-pw-nondefault-stream` |
| **cuda** | RHF default-stream polish | P3 `cuda-rhf-nondefault-stream` |
| **cuda** | full GPW product | **blocked** |

Foreign worktrees (`wf/cuda-md-first-regression`, stale CUDA-PW tips) belong to
other / incomplete sessions — re-base onto `wf/escuda-<slug>` before reuse;
never `reset --hard`.
