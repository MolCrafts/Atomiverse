# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `948d0d86` (post `cuda-pw-pbe-device` land + large-cell merge)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai` (template-shaped GapCensus loop)
- **Last action:** harness wave 1 (2026-08-10) after `cuda-pw-pbe-device` —
  marked done [x]; closed spec; keep other `[~]`; clearup already ran;
  `builds_wiped=true` (re-scan empty)

## How to run

```text
workflow name=es-cpu-cuda-opt
args: { "mode": "exhaust", "staff_count": 3, "max_waves": 6, "target_branch": "dev" }
agent_budget: 256+
```

Staff order from `## in_progress`: `cuda-pw-large-cell-measure` (P1 resume) →
`cuda-pw-nondefault-stream` (P2 fresh). Still pending: `cuda-pw-forces` (P0),
`cuda-gpw-ks-chain` (P1), `cuda-rhf-nondefault-stream` (P3), optional
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
| **cuda** | large-cell wall harness close | P1 `cuda-pw-large-cell-measure` still `[~]` |
| **cuda** | PW default-stream residual | P2 `cuda-pw-nondefault-stream` |
| **cuda** | energy-only PW gap | P0 `cuda-pw-forces` (pending) |
| **cuda** | GPW footholds uncomposed | P1 `cuda-gpw-ks-chain` (pending) |
| **cuda** | RHF default-stream polish | P3 `cuda-rhf-nondefault-stream` |
| **cuda** | full GPW product | **blocked** |

**Closed this harness:** `cuda-pw-pbe-device` — device closed-shell PBE on
`cuda::pw::PW` (GH200 **1048320**; dual-gate \|ΔE\|=1.421e-14 Ry). Roadmap
PBE CUDA ✓ already on branch (PW closed-shell only). **Queue not exhausted.**

Foreign worktrees (`wf/cuda-md-first-regression`, stale tips) belong to other /
incomplete sessions — re-base onto `wf/escuda-<slug>` before reuse; never
`reset --hard`.
