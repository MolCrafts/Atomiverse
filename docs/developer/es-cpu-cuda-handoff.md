# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `8280ed89` (post nondefault-stream merge; harness wave 2)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 2 — land `cuda-pw-nondefault-stream` (GH200
  **1049681** green); large-cell + PBE already done; CPU walls unchanged
  (worst_ot_ratio / worst fair **2.33×** MG MAIN); no CPU fair-wall land

## How to run

```text
workflow name=es-cpu-cuda-opt
args: { "mode": "exhaust", "staff_count": 3, "max_waves": 6, "target_branch": "dev" }
agent_budget: 256+
```

Staff order: `cuda-pw-forces` (P0) → `cuda-gpw-ks-chain` (P1) →
`cuda-pw-collinear-spin` / `cuda-pw-metallic-smear` (P2) → polish P3
(`cpu-simd-2center-setup` tip exists; `cuda-rhf-nondefault-stream`).
Workers only `wf/escuda-<slug>` off current `dev`.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names (`_pw_stream_work`, `_pw_cell_work`) — never
`build-*`. Re-scan empty before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cpu** | MG MAIN **2.33×** | **blocked** `mg-main-structural` |
| **cpu** | slab 2-center S/T | P3 tip `wf/escuda-cpu-simd-2center-setup` — validate + land |
| **cuda** | energy-only PW | P0 `cuda-pw-forces` |
| **cuda** | GPW footholds uncomposed | P1 `cuda-gpw-ks-chain` |
| **cuda** | collinear spin throw | P2 `cuda-pw-collinear-spin` |
| **cuda** | metallic smear throw | P2 `cuda-pw-metallic-smear` |
| **cuda** | RHF stream polish | P3 `cuda-rhf-nondefault-stream` |
| **cuda** | full GPW product | **blocked** |

**Closed this wave:** `cuda-pw-nondefault-stream` (merge `8280ed89`, GH200
**1049681**). Prior: `cuda-pw-pbe-device` (**1048320**); `cuda-pw-large-cell-measure`
(`948d0d86`). **Queue not exhausted.**

Foreign worktrees (`wf/cuda-md-first-regression`) — leave alone; never
`reset --hard`.
