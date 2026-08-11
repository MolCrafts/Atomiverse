# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `629aca3b` (post harness wave 3: `cuda-pw-smear-forces`)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 3 — land Assign wave 6 residual:
  - [x] `cuda-pw-smear-forces` — restricted metallic-smear analytic ionic
    forces dual-gate; feat `891baefc` / tip `629aca3b`; GH200 **1130427**
    max\|ΔF\|=**1.36e-11** eV/Å (prior **1130401** / **1130359**); Al₂ FD;
    **perf_exempt**
  - [~] none remaining from Assign wave 6
- **Counts:** pending=118; in_progress=0; cpu_pending=0; cuda_pending=118;
  gaps_remaining=119 (incl. blocked `mg-main-structural`). Queue **not**
  exhausted.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: multi-k device collocate/integrate or device-product-residency / smear-stress / spin-stress; no dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW (staffed done)** | UKS device DIIS + device weight pack **landed** |
| **cuda GPW (next)** | multi-k device spine (collocate/integrate/DIIS/STVnl); Si/water multi-k gates; device-product-residency; device-stvnl; water σ; device-stress; MG/UKS F/σ |
| **cuda PW (staffed done)** | restricted metallic-smear **forces** **landed** |
| **cuda PW (next)** | smear-stress; smear-PBE; spin-stress; multi-k PBE F/σ; spin+smear |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
