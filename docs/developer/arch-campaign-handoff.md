# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `307eaaa9` (post GapCensus wave-8; Assign wave-8 re-staffs 3 resumes; harness wave-5
  merge `e34ccf92` still latest product merge for this campaign)
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
- **Wave-5 done:**
  - `arch-gpw-driver-paths` — multi-path **DIIS** (Γ UKS + multi-k RKS/UKS)
    on `gpw::Phase` via path-local Ops + `drive_path_diis_` /
    `drive_stage_loop_`; OT free-loop residual documented (not a new slug);
    merge `e34ccf92` / impl `42454014` + gate docs `2ba03049`;
    `test_cpu_gpw_driver_node` + `reg_arch-gpw-driver-paths` +
    twin regs `gpw-uks-o2-gamma` / `gpw-kpoints-si`; spec closed
- **GapCensus wave 8:** reconfirmed open set; **no new slugs**. Tree facts
  match queue. Tips advanced product claims then census chore on `dev`.
- **Assign wave 8:** staff **3** resumes → `[~]`:
  - resume `arch-pw-driver-paths` @ `e04ba781` (**rebase onto current `dev`**
    before gate)
  - resume `arch-cuda
  - resume `arch-cuda-gpw-public-shape-freeze` @ `a625c232` (rebase)
  - resume `arch-mg
  - resume `arch-mg-main-structural` @ `3d9a64ba` (rebase; prefer freeze-first
    if contention)

- **new_assigned=0**, **incomplete_remaining=3**, pending unheld=0, blocked
  staffable product=1 (`cuda-gpw-full-product` still needs freeze).
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed multi-path
  GPW DIIS + landed PW fixed-occ stages (+ tip PW multi-path Ops / freeze
  note / MG structural) — do not invent a second SCF shape. Freeze is
  **doc/harness** (public shape = CPU `GpwDft`); MG is structural MAIN
  path, not soft pins.
- **gaps_remaining=4** (3 in_progress + blocked full CUDA GPW product).
  **new_slugs_added=0**. Queue **not** exhausted.

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
