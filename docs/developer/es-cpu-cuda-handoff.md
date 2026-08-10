# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `a6049f4f` (product FF-merged; queue land commit)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** GapCensus re-diff 2026-08-10 — product first slice **held**
  on tree (`cuda::gpw::GpwDft` + hybrid KS); **cpu_pending=0**; open = **13**
  CUDA depth rows (GPW compose/forces/stress + PW spin/PBE/smear). Fair wall
  MG MAIN **2.33×** still blocked pin-unsafe. Worktrees / `wf/*` / `build*`
  empty.

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
| **cuda GPW** | host collocate + host integrate; CPU-backed forces; no stress | staff P1 depth |
| **cuda GPW** | multigrid / UKS / multi-k reject | staff P1–P2 |
| **cuda PW** | PBE stress dual-gate; spin-PBE; spin/smear forces/stress | staff P1–P2 |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
