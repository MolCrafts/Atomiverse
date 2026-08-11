# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `ce0a0c6f` (post harness wave 2: uks-device-diis + device-weight-pack)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 2 — land 2 of 3 Assign wave 6 staff:
  - [x] `cuda-gpw-uks-device-diis` — dual `AoDiisEigenDevice` on Γ UKS; feat
    `6df76cf7` / merge `4ad9a172`; GH200 **1130279** / re-verify **1130402**
    \|dE\|=**4.760e-06** Ha
  - [x] `cuda-gpw-device-weight-pack` — `ProductWeightPackDevice` mid-SCF; merge
    `ce0a0c6f` / feat `94256cf0`; GH200 **1130344** \|dE\|=**1.616e-07** Ha
  - [~] keep `cuda-pw-smear-forces` → `wf/cuda-pw-smear-forces` (`bafa9c3f` /
    GH200 **1130359**)
- **Counts:** pending=105; in_progress=1; cpu_pending=0; cuda_pending=106;
  gaps_remaining=107 (incl. blocked `mg-main-structural`). Queue **not** exhausted.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: finish staffed smear-forces; next free-slot multi-k device collocate/integrate or device-product-residency / spin-stress; no dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW (staffed done)** | UKS device DIIS + device weight pack **landed** |
| **cuda GPW (next)** | multi-k device spine (collocate/integrate/DIIS/STVnl); Si/water multi-k gates; device-product-residency; device-stvnl; water σ; device-stress; MG/UKS F/σ |
| **cuda PW (staffed)** | smear forces (only remaining `[~]`) |
| **cuda PW (next)** | spin-stress; smear-stress; multi-k PBE F/σ; spin+smear |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
