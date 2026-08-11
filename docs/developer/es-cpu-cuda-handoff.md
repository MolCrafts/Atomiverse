# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `689f772b` (post opt-4 tip merges + GapCensus wave 14)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** GapCensus wave 14 — re-diff after opt-4 land of
  **`cuda-pw-smear-stress`** (merge `b46875bf` / GH200 **1133111**) +
  **`cuda-gpw-kpoint-h-assemble`** (merge `0ef79db9` / GH200 **1132528**).
  Cleared all `[~]`; **new_slugs_added=1** (`kpoint-device-cold-start`).
  **cpu_pending=0**; **pending_count=134**; **in_progress=0**;
  **gaps_remaining=135**. No live worktrees / `wf/*`.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: kpoint-h-residency / kpoint-device-energy / kpoint-payload-device / kpoint-device-cold-start / kpoint-p-residency / kpoint-device-stvnl / device-h-assemble / smear-stress-al2-gate / spin-stress / kmesh-smear-forces; prefer product lift; NO dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | multi-k H residency / device energy / payload / cold-start / P residency / STVnl; Si/water production gates; Γ device-h-assemble + UKS dual; device-stress; MG/UKS F/σ |
| **cuda PW** | smear-stress-al2 production gate; spin-stress; smear-PBE; multi-k F/σ + multi-k PBE; spin×kmesh×smear |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
