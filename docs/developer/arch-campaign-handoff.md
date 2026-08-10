# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `19280211` (post `arch-gpw-driver-node` / `bf5897fb`)
- **Wave-1 done:** e2e-in-tests delete (78 sources)
- **Wave-2 done:** `arch-gpw-driver-node` — GpwDft is `cpu::Driver<gpw::Phase>`;
  Γ RKS DIIS driven by stages; residual multi-k/UKS/OT → `arch-gpw-driver-paths`
- **GapCensus wave 1 (2026-08-10):** refilled **pending=6** (P0 paths + PW;
  P1 CUDA SCF fold + shape freeze; P2 e2e-wave2 + mg-main). **in_progress
  empty**. Blocked: `cuda-gpw-full-product` (until paths + freeze). No open
  arch specs; no `wf/arch-*` worktrees. Foreign leftover `wf/escuda-*` /
  `wf/cuda-*` / MD branches not stolen.
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed
  `drive_gamma_rks_diis_` / `gpw/stages/*` — do not invent a second SCF shape.
- **gaps_remaining=7** (6 pending + blocked full CUDA GPW product).

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
