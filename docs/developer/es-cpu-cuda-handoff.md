# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `0316e7d0` (post arch residual close + MD reg merge)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** GapCensus 2026-08-10 — re-diff tree/notes/roadmap vs `dev`.
  Confirmed **no** `cuda::gpw::GpwDft` public type (internal stage units only).
  Reconciled queue: **`cuda-gpw-full-product` P0** + four PW spin-depth rows
  discovered from device reject paths. **cpu_pending=0**. Worktrees/`build*`/
  `wf/*` clean.

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
(2) then `cuda-pw-spin-pbe` → forces/stress/smear depth. Workers only
`wf/escuda-<slug>` off current `dev`. First product slice = Γ closed-shell DIIS
RKS energy+forces; compose `cuda/gpw/internal/*`; GH200 required for green.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names — never leave multi-GB trees. Re-scan empty
before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cuda** | no public `GpwDft` | **staff** `cuda-gpw-full-product` |
| **cuda** | PW spin-PBE / spin forces / spin stress / spin+smear | pending P1–P2 after product |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |
| **cuda** | higher-L RHF product | out of loop product scope |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
