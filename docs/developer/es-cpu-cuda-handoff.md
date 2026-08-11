# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `557a1921` (post opt-3 tip land h2o + kpoints + spin-forces)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** GapCensus+Assign wave 6 — re-diff after tip land; open
  `kpoint-h2o-gate`; staff **3** NEW P2 product lifts:
  - `cuda-gpw-uks-device-diis` → `wf/cuda-gpw-uks-device-diis`
  - `cuda-gpw-device-weight-pack` → `wf/cuda-gpw-device-weight-pack`
  - `cuda-pw-smear-forces` → `wf/cuda-pw-smear-forces`
- **Counts:** pending=90; in_progress=3; cpu_pending=0; cuda_pending=93;
  gaps_remaining=94 (incl. blocked `mg-main-structural`). Queue **not** exhausted.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: finish staffed uks-device-diis / device-weight-pack / smear-forces; next free-slot multi-k device collocate/integrate or spin-stress / product-h2o-forces; no dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW (staffed)** | UKS device DIIS; device weight pack mid-SCF |
| **cuda GPW (next)** | multi-k device spine (collocate/integrate/DIIS/STVnl); Si/water multi-k gates; water F/σ; device-stress; MG/UKS F/σ |
| **cuda PW (staffed)** | smear forces |
| **cuda PW (next)** | spin-stress; smear-stress; multi-k PBE F/σ; spin+smear |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
