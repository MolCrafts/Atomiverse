# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `8b517bb1` (wave-8 landed; wave-9 staff in flight)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** GapCensus wave 9 + Assign 3 P2 product residuals on worktrees:
  - `cuda-gpw-product-h2o-gate` — GH200 **1128194** — `wf/…` @ `9bf5b2ac`
  - `cuda-gpw-stress` — GH200 **1128195** — `wf/…` @ `d58e9815`
  - `cuda-gpw-nondefault-stream` — GH200 **1128196** — `wf/…` @ `5018ed7c`
  Wave-8 on base: device-forces / pbe-product-gate / uks-o2-gate.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "GPW P2 residual: device-diis-eigen, multi-k energy; PW spin-forces + smear-forces; no soft tols"
}
agent_budget: 384+
```

**Staff order:** merge wave-9 tips when GH200 green; then device-diis-eigen /
kpoints + PW spin/smear F/σ. Prefer compose into `GpwDft`, not new footholds.

## Disk

Wipe `build` / `build-*` every wave. Drop ephemeral `job_*.sh` at clearup if
not under `tools/`.

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | (staffed) stress / stream / H₂O gate; residual device DIIS/eigen; multi-k; MG/UKS forces; UKS-PBE |
| **cuda PW** | spin forces/stress; smear forces/stress; spin+smear; multi-k PBE F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
