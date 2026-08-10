# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `350644ca` (post `arch-cuda-scf-solver-fold` +
  `arch-e2e-tests-wave2` merges + harness wave-2 chore)
- **Wave-1 done:** e2e-in-tests delete (78 sources); `arch-gpw-driver-node`
- **Wave-2 done:**
  - `arch-cuda-scf-solver-fold` — no free-loop `SCFSolver`; RHF owns
    `internal::RhfScfCycle` in `scf::Phase` order; nested `cuda::Driver`
    residual
  - `arch-e2e-tests-wave2` — residual multi-iter out of `tests/`;
    `ATV_TEST_HEAVY_REGEX=^$`
- **GapCensus wave 3 (2026-08-10):** re-diff harness + notes + queues + tree
  vs `dev`. **pending=4** (P0 paths + PW; P1 shape freeze; P2 mg-main).
  **in_progress empty**. **new_slugs_added=0**. Blocked:
  `cuda-gpw-full-product` (until paths + freeze); scope-out soft-tol /
  hybrid-GAPW stay blocked never-pending. Specs: no open arch specs. No
  `wf/arch-*` worktrees. Orphan tips `wf/arch-gpw-driver-paths` /
  `wf/arch-pw-driver-node` **not** on `dev` — do not mark done. Foreign
  leftover `wf/escuda-*` / `wf/cuda-*` / MD branches not stolen.
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed
  `drive_gamma_rks_diis_` / `gpw/stages/*` — do not invent a second SCF shape.
- **gaps_remaining=5** (4 pending + blocked full CUDA GPW product).

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
