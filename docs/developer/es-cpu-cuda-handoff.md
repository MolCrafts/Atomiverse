# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ wave-9 merges (stress + nondefault-stream on base)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Wave 9 GapCensus + staff 3; land **2**:
  - `cuda-gpw-stress` GH200 **1128195** (merge on `dev`)
  - `cuda-gpw-nondefault-stream` GH200 **1128196** (merge on `dev`)
  - `cuda-gpw-product-h2o-gate` **no-land** — CUDA water RKS SCF non-convergence
    (device KS residual; retries 1128194/1128217/1128269/1128295)

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "GPW P2 residual: device-diis-eigen (may unblock water), multi-k energy; product-h2o after DIIS; PW spin-forces + smear-forces; no soft tols"
}
agent_budget: 384+
```

**Staff order:** `device-diis-eigen` first (host residual + may fix multi-center
SCF stall), then `kpoints` + PW spin/smear F/σ. Re-staff `product-h2o-gate`
only after device SCF residual improved. Prefer compose into `GpwDft`.

## Disk

Wipe `build` / `build-*` every wave. Drop ephemeral logs at clearup.

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | device DIIS/eigen; multi-k; water production dual-gate; MG/UKS F/σ; UKS-PBE |
| **cuda PW** | spin forces/stress; smear forces/stress; spin+smear; multi-k PBE F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
