# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `69296751` (landed `cuda-gpw-device-collocate`; queue harness bookkeeping)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 5 2026-08-10 — mark **`cuda-gpw-device-collocate`**
  done [x] (feat `430a2268` / land `69296751`; GH200 **1125709**
  \|ΔE\|=**1.617e-7** Ha max\|ΔF\|=**2.137e-6** Ha/Bohr H₂ Γ RKS PADE).
  No other [~]. Spec closed (deleted). **cpu_pending=0**; **cuda_pending=29**;
  **gaps_remaining≈30**. Fair wall MG MAIN **2.33×** still blocked pin-unsafe.
  Clearup already ran; worktrees empty; no open `wf/*` tips.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 8,
  "target_branch": "dev",
  "theme": "GPW P1 depth (device-forces, multigrid-run, uks-gamma, pbe-product-gate) + PW spin F/σ; GH200 green; no soft tols"
}
agent_budget: 384+
```

**Staff order (P1 first):**
1. `cuda-gpw-device-forces` + `cuda-gpw-multigrid-run` + `cuda-gpw-uks-gamma` + `cuda-gpw-pbe-product-gate`
2. P2: kpoints / stress / smear-forces / spin-forces / spin-stress / spin-smear / nondefault-stream / diis-eigen / h2o-gate
3. P3 after parents: spin-pbe-forces / spin-pbe-stress / uks-forces / uks-pbe-gate / multigrid-forces / kpoint-forces / spin-smear F/σ / joint-path rows

Workers only `wf/escuda-<slug>` or `wf/<slug>` off current `dev`.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names — never leave multi-GB trees. Re-scan empty
before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cuda GPW** | device collocate **landed**; host DIIS/eigen + weight pack; CPU-backed forces; no stress | staff P1 depth |
| **cuda GPW** | multigrid / UKS / multi-k reject | staff P1–P2 |
| **cuda PW** | spin-PBE energy **landed**; spin/smear F/σ + spin-smear energy open | staff P2–P3 depth |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
