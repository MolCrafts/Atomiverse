# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `f016215e` (post `arch-pw-driver-node` merge + this
  harness chore)
- **Wave-1 done:** e2e-in-tests delete (78 sources); `arch-gpw-driver-node`
- **Wave-2 done:**
  - `arch-cuda-scf-solver-fold` — no free-loop `SCFSolver`; RHF owns
    `internal::RhfScfCycle` in `scf::Phase` order; nested `cuda::Driver`
    residual
  - `arch-e2e-tests-wave2` — residual multi-iter out of `tests/`;
    `ATV_TEST_HEAVY_REGEX=^$`
- **Wave-3 done (this harness):**
  - `arch-pw-driver-node` — `PW` is `cpu::Driver<pw::Phase>`; restricted
    fixed-occ stages; spin/smear free-loop residual; merge `f016215e` /
    impl `1bad527a`; unit `test_cpu_pw_driver_node` +
    `reg_arch-pw-driver-node`
- **Keep incomplete `[~]`:** `arch-gpw-driver-paths` @ `cca084b6` (ahead=1
  behind=2 vs `dev`; rebase onto `f016215e` before gate). **Hold** P1
  shape-freeze + P2 mg-main until paths merge.
- **GapCensus this harness:** re-diff harness + notes + queues + tree vs
  `dev` @ `f016215e`. **pending held=2**, **in_progress=1**,
  **new_slugs_added=0**. Blocked: `cuda-gpw-full-product` (until paths +
  freeze). Spec closed: `arch-pw-driver-node`. Clearup already ran (no
  isolation worktrees; build dirs empty). Foreign leftover branches not
  stolen.
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed
  `drive_gamma_rks_diis_` / `gpw/stages/*` + landed PW fixed-occ stages —
  do not invent a second SCF shape.
- **gaps_remaining=4** (1 in_progress + 2 pending held + blocked full CUDA
  GPW product).

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
