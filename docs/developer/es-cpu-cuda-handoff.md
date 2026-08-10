# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `8122eb0e` (post prior GapCensus + arch residual close)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** GapCensus re-diff 2026-08-10 — tree/notes/baseline/roadmap vs
  `dev`. Confirmed **no** `cuda::gpw::GpwDft` (internal stages only). Fair walls
  authority **2.33×** MG MAIN; **cpu_pending=0**. Pending **8**: product P0 +
  closed-shell PW depth (PBE stress, smear forces/stress) + collinear spin
  depth (spin-PBE, spin forces/stress/smear). Worktrees/`build*`/`wf/*` clean.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 1,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "cuda-gpw-full-product under freeze contract (GpwDft public shape)"
}
agent_budget: 256+
```

**Staff order:** (1) **`cuda-gpw-full-product` (P0)** only until landed;
(2) `cuda-pw-pbe-stress` + `cuda-pw-spin-pbe` (P1); (3) remaining PW depth P2.
Workers only `wf/escuda-<slug>` off current `dev`. First product slice = Γ
closed-shell DIIS RKS energy+forces; compose `cuda/gpw/internal/*`; GH200
required for green.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names — never leave multi-GB trees. Re-scan empty
before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cuda** | no public `GpwDft` | **staff** `cuda-gpw-full-product` |
| **cuda** | PBE stress dual-gate; smear forces/stress | pending P1–P2 |
| **cuda** | PW spin-PBE / spin forces / spin stress / spin+smear | pending P1–P2 |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |
| **cuda** | higher-L RHF product | out of loop product scope |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
