# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `18663d13` (landed `cuda-pw-pbe-stress`; queue harness bookkeeping)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 3 2026-08-10 — mark **`cuda-pw-pbe-stress`**
  done [x] (feat `b0523de0` / land `18663d13`; GH200 **1123959**
  max‖Δσ‖=**5.585e-12** eV/Å³). Keep other [~]:
  `cuda-gpw-device-collocate`, `cuda-pw-spin-pbe`. Spec closed (deleted).
  **cpu_pending=0**; **cuda_pending=20**; **gaps_remaining≈21**. Fair wall MG
  MAIN **2.33×** still blocked pin-unsafe. Clearup already ran;
  worktrees empty for landed slug; `wf/*` remain for two open tips.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 8,
  "target_branch": "dev",
  "theme": "merge device-collocate + spin-pbe; then GPW product depth (forces, multigrid-run, uks-gamma, pbe-product-gate); GH200 green; no soft tols"
}
agent_budget: 384+
```

**Staff order (P1 first):**
1. **merge** `cuda-gpw-device-collocate` + `cuda-pw-spin-pbe` (tips GH200-green)
2. `cuda-gpw-device-forces` + `cuda-gpw-multigrid-run` + `cuda-gpw-uks-gamma` + `cuda-gpw-pbe-product-gate`
3. P2: kpoints / stress / smear-forces / spin-forces / spin-stress / spin-smear / nondefault-stream / diis-eigen / h2o-gate

Workers only `wf/escuda-<slug>` or `wf/<slug>` off current `dev`.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names — never leave multi-GB trees. Re-scan empty
before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cuda GPW** | host collocate + host integrate; CPU-backed forces; no stress | staff/merge P1 depth |
| **cuda GPW** | multigrid / UKS / multi-k reject | staff P1–P2 |
| **cuda PW** | spin-PBE energy (tip); spin/smear forces/stress | merge spin-pbe then P2 depth |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
