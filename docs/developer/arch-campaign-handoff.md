# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `d24283d1` (post harness wave-3 chore marking
  `arch-pw-driver-node` done; GapCensus wave-4 on same tip)
- **Wave-1 done:** e2e-in-tests delete (78 sources); `arch-gpw-driver-node`
- **Wave-2 done:**
  - `arch-cuda-scf-solver-fold` — no free-loop `SCFSolver`; RHF owns
    `internal::RhfScfCycle` in `scf::Phase` order; nested `cuda::Driver`
    residual
  - `arch-e2e-tests-wave2` — residual multi-iter out of `tests/`;
    `ATV_TEST_HEAVY_REGEX=^$`
- **Wave-3 done:**
  - `arch-pw-driver-node` — `PW` is `cpu::Driver<pw::Phase>`; restricted
    fixed-occ stages; spin/smear free-loop residual; merge `f016215e` /
    impl `1bad527a`; unit `test_cpu_pw_driver_node` +
    `reg_arch-pw-driver-node`
- **GapCensus wave-4:** re-diff harness + notes + queues + tree vs
  `dev` @ `d24283d1`. **Staffed** residual `arch-pw-driver-paths` (new).
  **pending=3** (pw-paths + held freeze + held mg-main), **in_progress=1**,
  **new_slugs_added=1**. Blocked: `cuda-gpw-full-product` (until paths +
  freeze). Roadmap arch prose updated (PW is a Driver node). Clearup not
  required this census (no worktrees opened).
- **Keep incomplete `[~]`:** `arch-gpw-driver-paths` @ `cca084b6` (ahead=1
  behind=3 vs `dev`; rebase onto `d24283d1` before gate). **Hold** P1
  shape-freeze + P2 mg-main until GPW paths merge.
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed
  `drive_gamma_rks_diis_` / `gpw/stages/*` + landed PW fixed-occ stages —
  do not invent a second SCF shape.
- **gaps_remaining=5** (1 in_progress + 3 pending + blocked full CUDA
  GPW product).

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
