# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `b88b2fd2` (land `cuda-gpw-kpoint-device-stvnl`)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** **Harness wave 2** — 1 of 3 staffed → done [x]:
  `cuda-gpw-kpoint-device-stvnl` (merge `b88b2fd2` / feat `ec578d7b`; GH200
  **1133851** \|dE\|=**3.331e-14** Ha H₂ multi-k RKS PADE device phased STVnl;
  **perf_exempt**). Keep other `[~]`: `cuda-pw-spin-stress` (`6617710e`) +
  `cuda-gpw-kpoint-h-residency` (`181a81c3`, GH200 **1133843**
  \|dE\|=**3.442e-14**). Spec closed (deleted
  `.claude/specs/cuda-gpw-kpoint-device-stvnl/`). Roadmap prose: multi-k device
  STVnl landed (table CUDA GPW remains **◐** — χ/orth + H residency +
  payload/cold-start + Si/water multi-k F/σ depth open — no ○→✓).
  **pending_count=136**; **in_progress=2**; **cpu_pending=0**;
  **cuda_pending=138**; **gaps_remaining=139**; **new_slugs_added=0** this
  harness. Free staff held: device-payload / uks-payload / device-cold-start /
  multi-k energy/payload/cold-start/P / **chi/orth** / device-h-assemble /
  smear-stress-al2 / kmesh-smear-forces / multigrid-si-gate. Clearup already
  ran. Queue **not** exhausted. No push.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual: cuda-pw-spin-stress OR cuda-gpw-kpoint-h-residency OR kpoint-device-chi/orth OR device-payload; max 3 NEW; forbid dual-gate matrix explosion; no soft tols; GH200 green"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW [~]** | `kpoint-h-residency` (staffed; unmerged) |
| **cuda PW [~]** | `spin-stress` (staffed; unmerged) |
| **cuda GPW free** | device-payload / uks-payload / device-cold-start; multi-k energy/payload/cold-start/P; **kpoint-device-chi** / **kpoint-device-orth**; device-h-assemble; multigrid-si-gate held |
| **cuda PW free** | smear-stress-al2-gate; kmesh-smear-forces; smear-PBE; multi-k F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
