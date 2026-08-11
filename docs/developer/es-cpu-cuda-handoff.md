# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `e2bcb05e` (land `cuda-gpw-kpoint-h-residency`)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** **Harness wave 4** — 1 of 3 staffed → done [x]:
  `cuda-gpw-kpoint-h-residency` (merge `e2bcb05e` / feat `068d7784` / re-verify
  `ed832fce`; GH200 **1142864** \|dE\|=**3.331e-14** Ha H₂ multi-k RKS PADE
  device H residual residency via `ingest_H_k_device_rm` +
  `residual_from_device`; **perf_exempt**). Keep other `[~]`:
  **`cuda-gpw-multigrid-si-energy-path`** (Assign wave 4 NEW P2) +
  `cuda-gpw-multigrid-si-gate` (tip `fe048ad7`; GH200 **1137743**/**1139046**
  product path BLOCKED until energy-path). Spec closed (deleted
  `.claude/specs/cuda-gpw-kpoint-h-residency/`). Roadmap prose: multi-k device
  H residual residency landed (table CUDA GPW remains **◐** — energy /
  payload / cold-start / P / χ/orth / vnl / h0 + Si/water multi-k F/σ + Si MG
  path open — no ○→✓).
  **pending_count=137**; **in_progress=2**; **cpu_pending=0**;
  **cuda_pending=139**; **gaps_remaining=140**; **new_slugs_added=0** this
  harness. Free staff held: device-payload / uks-payload / device-cold-start /
  multi-k energy/payload/cold-start/P / **chi/orth/vnl/h0** / device-h-assemble /
  smear-stress-al2 / kmesh-smear-forces. Clearup already ran. Queue **not**
  exhausted. No push.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual: cuda-gpw-multigrid-si-energy-path OR cuda-gpw-multigrid-si-gate OR kpoint-device-chi/orth/vnl/h0 OR device-payload; max 3 NEW; forbid dual-gate matrix explosion; no soft tols; GH200 green"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW [~]** | `multigrid-si-energy-path` (staffed NEW P2) + `multigrid-si-gate` (staffed; dual-gate after energy-path) |
| **cuda GPW free** | device-payload / uks-payload / device-cold-start; multi-k energy/payload/cold-start/P; **kpoint-device-chi** / **orth** / **vnl** / **h0-residency**; device-h-assemble |
| **cuda PW free** | smear-stress-al2-gate; kmesh-smear-forces; smear-PBE; spin-PBE F/σ; multi-k F/σ; spin×kmesh×smear |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
