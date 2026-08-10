# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `1c2d74d6` (GapCensus wave-5 chore; no arch product
  merge since PW node `f016215e` / harness wave-3 `d24283d1`)
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
- **GapCensus wave-5:** re-diff harness + notes + queues + tree vs
  `dev` @ `26fb8328`. **Tree on `dev` unchanged** (Γ RKS DIIS only / PW
  fixed-occ only / folded CUDA RHF / CUDA GPW stages≠product / HEAVY
  `^$`). **No new slug.** Tips advanced onto `26fb8328` then parent moved
  to `1c2d74d6` ⇒ tips **ahead=1 behind=1** — **unmerged tip ≠ done**.
- **Assign wave-5:** staff **2** resumes only (`[~]`); **new_assigned=0**,
  **incomplete_remaining=2**. Skipped held freeze + mg-main. Tips:
  - `wf/arch-gpw-driver-paths` @ `06bc94fb` — claim multi-path DIIS;
    OT residual; **rebase onto `1c2d74d6`** before Worktrees/gate; tip
    spec `.claude/specs/arch-gpw-driver-paths/SPEC.md`
  - `wf/arch-pw-driver-paths` @ `98e98ed5` — claim all product SCF paths
    on stages; **rebase onto `1c2d74d6`**; tip spec
    `.claude/specs/arch-pw-driver-paths/SPEC.md`
  **pending held=2** (shape-freeze + mg-main), **in_progress=2**,
  **gaps_remaining=5**. Blocked: `cuda-gpw-full-product` (until paths +
  freeze).
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed
  `drive_gamma_rks_diis_` / `gpw/stages/*` + landed PW fixed-occ stages
  (+ tip multi-path Ops) — do not invent a second SCF shape.
- **gaps_remaining=5** (2 in_progress + 2 pending held + blocked full CUDA
  GPW product).

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
