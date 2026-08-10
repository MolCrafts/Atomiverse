# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `fb2c2951` (product FF-merged)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Landed **`cuda-gpw-full-product`** first slice (GH200
  **1120817** dual-gate green). Open = 13 CUDA depth rows (GPW compose +
  device forces/stress + PW spin/PBE/smear depth). **cpu_pending=0**.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 8,
  "target_branch": "dev",
  "theme": "GPW product depth (device-collocate, device-forces, multigrid-run, uks-gamma) + PW pbe-stress + spin-pbe; GH200 green; no soft tols"
}
agent_budget: 384+
```

**Staff order (P1 first):**
1. `cuda-gpw-device-collocate` + `cuda-gpw-device-forces`
2. `cuda-gpw-multigrid-run` + `cuda-gpw-uks-gamma`
3. `cuda-pw-pbe-stress` + `cuda-pw-spin-pbe`
4. P2: kpoints / stress / smear-forces / spin-forces / spin-stress / spin-smear

Workers only `wf/escuda-<slug>` or `wf/<slug>` off current `dev`.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names — never leave multi-GB trees. Re-scan empty
before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cuda GPW** | host collocate; CPU-backed forces; no stress | staff P1 depth |
| **cuda GPW** | multigrid / UKS / multi-k reject | staff P1–P2 |
| **cuda PW** | PBE stress; spin-PBE; spin/smear forces/stress | staff P1–P2 |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
