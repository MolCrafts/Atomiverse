# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `3bfe1d14` (GapCensus wave 4 — multi-k tip residual
  spine + Si gate on post–diis harness `c2fd4ae8` / land `e164b767`)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** **Assign wave 4** — capacity **3/3** tip-green (no NEW):
  - [~] `cuda-gpw-product-h2o-gate` tip `5731b00c` GH200 **1130066**
    \|dE\|=**7.526e-07** max\|dF\|=**3.213e-06** — **resume** (Merge)
  - [~] `cuda-gpw-kpoints` tip `14368bf3` GH200 **1130067**
    \|dE\|=**3.286e-14** — **resume** (Merge; residual → `kpoint-device-*`)
  - [~] `cuda-pw-spin-forces` tip `b6b01efe` GH200 **1130077**
    max\|ΔF\|=**8.51e-13** eV/Å — **resume** (Merge)
- **Next:** Merge the three tip-green branches (rebase onto `3bfe1d14` if
  behind). When slots free: `smear-forces` (theme P2) or GPW
  `uks-device-diis` / `device-weight-pack` / force-stream / device-stress /
  multi-k depth (`kpoint-device-collocate` after kpoints land). Do **not**
  re-staff landed collocate / MG / UKS / PBE / forces / stress / stream /
  DIIS-eigen. No hybrid/GAPW. No soft tols.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "Merge h2o + kpoints + spin-forces; next NEW smear-forces or uks-device-diis/weight-pack; no soft tols"
}
agent_budget: 384+
```

**Staff order (when slots free):** prefer one real product lift over
joint-path matrix explosion. Next NEW: PW `smear-forces` (P2) or GPW
`uks-device-diis` / `device-weight-pack` — not hybrid/GAPW, not dual-gate
matrix explosion.

## Disk

Wipe `build` / `build-*` every wave. Drop ephemeral logs at clearup.

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW** | **in_progress:** water gate + multi-k (Merge); then uks-device-diis, weight-pack, kpoint-device-*, force-stream, device-stress, STVnl, joints |
| **cuda PW** | **in_progress:** spin-forces (Merge); then smear F/σ, spin-stress, spin+smear, multi-k PBE energy/F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
