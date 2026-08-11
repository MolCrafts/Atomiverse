# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `eec0b266` (post GapCensus wave 13; Assign wave 5 tip-green)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** GapCensus wave 13 — re-diff + deepen pending (+4):
  - [~] `cuda-pw-smear-stress` tip `aa84caba` / GH200 **1132266** (tip-green)
  - [~] `cuda-gpw-kpoint-h-assemble` tip `5b64229b` / **1132528** (tip-green)
  - [~] `cuda-gpw-kpoint-device-diis` tip `6fa3bba5` / **1132761** (tip-green)
  - **new:** `kpoint-h-residency` / `kpoint-device-energy` /
    `kpoint-payload-device` / `smear-stress-al2-gate`
- **Counts:** pending=133; in_progress=3; cpu_pending=0; cuda_pending=136;
  gaps_remaining=137 (incl. blocked `mg-main-structural`). Queue **not**
  exhausted.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: multi-k device DIIS/STVnl or device-product-residency / device-h-assemble / smear-stress / spin-stress; no dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW (staffed now)** | multi-k **`kpoint-h-assemble`** + **`kpoint-device-diis`** tip-green (Merge) |
| **cuda GPW (next after)** | `kpoint-h-residency` / `kpoint-device-energy` / `kpoint-payload-device` / `kpoint-p-residency` / `kpoint-device-stvnl`; Si/water multi-k gates; device-h-assemble; uks-h-assemble; water σ; device-stress; MG/UKS F/σ |
| **cuda PW (staffed now)** | **`smear-stress`** tip-green GH200 **1132266** (Merge) |
| **cuda PW (next after)** | `smear-stress-al2-gate`; smear-PBE; spin-stress; multi-k PBE F/σ; spin+smear; kmesh-smear F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
