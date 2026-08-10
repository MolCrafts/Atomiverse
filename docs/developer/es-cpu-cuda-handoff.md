# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `f679807c` (landed `cuda-gpw-multigrid-run` +
  `cuda-gpw-uks-gamma`; queue harness bookkeeping)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 7 2026-08-10 — mark **2/3** staffed P1 tips
  done [x]:
  - **`cuda-gpw-multigrid-run`** (feat `2d780cff` / `e201a6cb`; GH200
    **1126433** \|dE\|=**1.587e-7** Ha H₂ Γ RKS PADE ngrids=4; prior
    **1125894**)
  - **`cuda-gpw-uks-gamma`** (feat `f679807c` / `254ff67c`; GH200 **1126430**
    \|ΔE\|=**4.760e-6** Ha high-spin H₂ UKS PADE ⟨S²⟩=2.0; prior **1125905**)
  Keep **[~]** `cuda-gpw-device-forces` tip `wf/cuda-gpw-device-forces` @
  `9ec138dc` (GH200 **1126434** max\|ΔF\|=**2.321e-6** Ha/Bohr — not on base).
  Specs closed (deleted): `cuda-gpw-multigrid-run/`, `cuda-gpw-uks-gamma/`.
  **cpu_pending=0**; **cuda_pending=51**; **in_progress=1**;
  **gaps_remaining≈53**. Fair wall MG MAIN **2.33×** still blocked pin-unsafe.
  Clearup already ran; no open `wf/*` tips except unmerged
  `wf/cuda-gpw-device-forces`.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 8,
  "target_branch": "dev",
  "theme": "GPW P1 depth (device-forces merge, pbe-product-gate) + MG/UKS child F/σ; PW spin F/σ; GH200 green; no soft tols"
}
agent_budget: 384+
```

**Staff order (P1 first):**
1. Merge `cuda-gpw-device-forces` then `cuda-gpw-pbe-product-gate`
2. P2: kpoints / stress / smear-forces / spin-forces / spin-stress / spin-smear / nondefault-stream / diis-eigen / h2o-gate / spin-kmesh / kmesh-F/σ
3. P3 after parents: multigrid-forces / uks-forces / joint MG·UKS·k F/σ + PBE path gates + spin-pbe F/σ / device-stvnl

Workers only `wf/escuda-<slug>` or `wf/<slug>` off current `dev`.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names — never leave multi-GB trees. Re-scan empty
before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cuda GPW** | device collocate + **MG energy** + **UKS energy** **landed**; host DIIS/eigen + weight pack; CPU-backed forces; no stress | merge device-forces; staff P1–P2 depth |
| **cuda GPW** | multi-k reject; MG/UKS F/σ + PBE product gate open | staff P2–P3 after parents |
| **cuda PW** | spin-PBE energy **landed**; spin/smear F/σ + spin-smear energy open | staff P2–P3 depth |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
