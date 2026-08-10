# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `e164b767` (land `cuda-gpw-device-diis-eigen`; prior
  GapCensus wave 11 at `bcb873e0`; wave-9 stress **1128195** + stream
  **1128196** still on base)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** **Harness wave 2** — **1 of 3** Assign-wave-2 tips → done:
  - [x] `cuda-gpw-device-diis-eigen` merge `e164b767` GH200 **1129719**
    \|dE\|=**1.616e-07** Ha (prior **1129285**); spec closed
  - [~] `cuda-gpw-product-h2o-gate` tip `a34ff0df` GH200 **1129720**
    \|dE\|=**7.526e-07** max\|dF\|=**3.213e-06** — keep for Merge
  - [~] `cuda-gpw-kpoints` tip `b61779f9` GH200 **1129721**
    \|dE\|=**3.331e-14** — keep for Merge
- **Next:** Merge remaining `[~]` (rebase onto `e164b767` if needed). When
  slots free: PW P2 spin/smear/kmesh(+PBE E) or GPW residual (force-stream /
  device-stress / device-stvnl). Do **not** re-staff landed collocate / MG /
  UKS / PBE / forces / stress / stream / DIIS-eigen.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "Merge product-h2o + kpoints; then PW spin-forces + smear-forces or GPW force-stream/device-stress; no soft tols"
}
agent_budget: 384+
```

**Staff order (when slots free after merge):** prefer one real product lift
over joint-path matrix explosion. Next NEW theme: PW `spin-forces` /
`smear-forces` (P2, independent) or deeper GPW residual (force-stream /
device-stress / device-stvnl) — not hybrid/GAPW.

## Disk

Wipe `build` / `build-*` every wave. Drop ephemeral logs at clearup.

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | **in_progress:** water gate + multi-k; then MG/UKS F/σ, force-stream, device-stress, STVnl/weight-pack, joints |
| **cuda PW** | spin forces/stress; smear forces/stress; spin+smear; multi-k PBE energy/F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
