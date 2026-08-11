# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ post opt-3 tip merges (h2o + kpoints + spin-forces)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** `es-cpu-cuda-opt-3` auto-landed stress + stream + device-diis-eigen
  (fail_streak on later waves). Operator merged residual green tips:
  - `cuda-gpw-product-h2o-gate` (GH200 **1130195**)
  - `cuda-gpw-kpoints` (GH200 **1130141**)
  - `cuda-pw-spin-forces` (GH200 **1130196**)

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: PW smear-forces; GPW uks-device-diis, device-weight-pack, kpoint-device-integrate, multigrid-si-gate; no dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | multi-k device depth (integrate/collocate/DIIS/STVnl); uks-device-diis; weight-pack; MG/UKS F/σ; device-stress; Si production gates |
| **cuda PW** | smear F/σ; spin-stress; spin+smear; multi-k PBE F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
