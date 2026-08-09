# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `e22325b7` (post PBE + large-cell merge; GapCensus wave 4)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai` (template-shaped GapCensus loop)
- **Last action:** GapCensus wave 4 (2026-08-10) — re-diff tree/notes/roadmap;
  large-cell → done; opened `cuda-pw-collinear-spin` + `cuda-pw-metallic-smear`;
  roadmap Periodic GGA CUDA ✓; no code land this step

## How to run

```text
workflow name=es-cpu-cuda-opt
args: { "mode": "exhaust", "staff_count": 3, "max_waves": 6, "target_branch": "dev" }
agent_budget: 256+
```

Staff order: finish `cuda-pw-nondefault-stream` (P2 in_progress) →
`cuda-pw-forces` (P0) → `cuda-gpw-ks-chain` (P1) → `cuda-pw-collinear-spin` /
`cuda-pw-metallic-smear` (P2) → polish P3. Re-base incomplete tips onto
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
| **cuda** | PW default-stream residual | P2 `cuda-pw-nondefault-stream` **in_progress** |
| **cuda** | energy-only PW gap | P0 `cuda-pw-forces` (pending) |
| **cuda** | GPW footholds uncomposed | P1 `cuda-gpw-ks-chain` (pending) |
| **cuda** | collinear spin throw | P2 `cuda-pw-collinear-spin` (pending) |
| **cuda** | metallic smear throw | P2 `cuda-pw-metallic-smear` (pending) |
| **cuda** | RHF default-stream polish | P3 `cuda-rhf-nondefault-stream` |
| **cuda** | full GPW product | **blocked** |

**Closed:** `cuda-pw-pbe-device` (GH200 **1048320**); `cuda-pw-large-cell-measure`
(merge `948d0d86`; 64-at \(T_{\mathrm{cu}}/T_{\mathrm{cp}} < 0.0018\)).
**Queue not exhausted.**

Foreign worktrees (`wf/cuda-md-first-regression`) belong to other loops —
leave alone; never `reset --hard`.
