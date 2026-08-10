# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `4f78d716` (wave-8 tips landed)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** After `es-cpu-cuda-opt-2` (8 waves, 5 auto-merges), operator
  FF/merged residual green tips:
  - `cuda-gpw-device-forces` (GH200 **1127167**)
  - `cuda-gpw-pbe-product-gate` (GH200 **1127166**)
  - `cuda-gpw-uks-o2-gate` (GH200 **1127173**)
  Prior auto lands: full-product, device-collocate, multigrid-run, uks-gamma,
  pw-pbe-stress, pw-spin-pbe.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "GPW P1 residual: stress, multi-k energy, nondefault-stream, device-diis-eigen; PW spin-forces + smear-forces; no soft tols; no dual-gate spam for already-gated paths"
}
agent_budget: 384+
```

**Staff order:** real product residual first (stress / multi-k / stream / DIIS),
then PW spin/smear F/σ. Prefer compose into `GpwDft`, not new footholds.
GapCensus must **not** explode dual-gate matrix without a stated production twin.

## Disk

Wipe `build` / `build-*` every wave. Drop ephemeral `job_*.sh` at clearup if
not under `tools/`.

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | stress API; multi-k; non-default stream; device DIIS/eigen; STVnl; MG/UKS forces; UKS-PBE |
| **cuda PW** | spin forces/stress; smear forces/stress; spin+smear; multi-k PBE F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
