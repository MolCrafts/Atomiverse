# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ post opt-4 tip merges (smear-stress + kpoint-h-assemble)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** `es-cpu-cuda-opt-4` auto-merged **6** (uks-device-diis,
  weight-pack, smear-forces, kpoint collocate/integrate, kpoint-device-diis).
  Operator merged residual green tips:
  - `cuda-pw-smear-stress` (GH200 **1133111**)
  - `cuda-gpw-kpoint-h-assemble` (GH200 **1132528**; kept with device DIIS)

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: kpoint-h-residency / kpoint-device-stvnl / spin-stress / multigrid-si-gate / device-stress; prefer product lift; NO dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | multi-k H residency / payload / STVnl / P residency; Si production gates; device-stress; MG/UKS F/σ |
| **cuda PW** | spin-stress; smear-stress multi-atom gate; spin+smear; multi-k PBE F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
