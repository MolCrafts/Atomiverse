# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `e2198619` (GapCensus wave 5 — multi-k integrate +
  production-depth residual; Assign wave 5 on same tip set)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** **Assign wave 5** — capacity **3/3** tip-green (no NEW):
  - [~] `cuda-gpw-product-h2o-gate` tip `8fe0edec` GH200 **1130140**
    \|dE\|=**7.526e-07** max\|dF\|=**3.213e-06** — **resume** (Merge)
  - [~] `cuda-gpw-kpoints` tip `18ba6fc5` GH200 **1130141**
    \|dE\|=**3.375e-14** — **resume** (Merge; residual → `kpoint-device-*`
    incl. **integrate** / `kpoint-si-gate`)
  - [~] `cuda-pw-spin-forces` tip `a5e941f2` GH200 **1130145**
    max\|ΔF\|=**9.25e-13** eV/Å — **resume** (Merge)
- **Next:** Merge the three tip-green branches (rebase onto `e2198619` if
  behind — parent delta is queue chore only). When slots free: prefer one
  real product lift — `smear-forces` (theme P2) or GPW `uks-device-diis` /
  `device-weight-pack` / force-stream / device-stress / multi-k depth
  (`kpoint-device-collocate` after kpoints land). Do **not** re-staff landed
  collocate / MG / UKS / PBE / forces / stress / stream / DIIS-eigen. No
  hybrid/GAPW. No soft tols. Prefer product lift over dual-gate matrix
  explosion.

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
| **cuda GPW** | **in_progress:** water gate + multi-k (Merge); then uks-device-diis, weight-pack, kpoint-device-* (+ integrate), multigrid-si-gate, force-stream, device-stress, STVnl, joints |
| **cuda PW** | **in_progress:** spin-forces (Merge); then smear F/σ, spin-stress, spin+smear, multi-k PBE energy/F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
