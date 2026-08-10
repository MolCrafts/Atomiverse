# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `0147fe02` (merge `wf/arch-cuda-gpw-public-shape-freeze`
  / docs `aea603bb`)
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
- **Wave-8 done (this harness):**
  - `arch-cuda-gpw-public-shape-freeze` — freeze note
    `.claude/notes/cuda-gpw-public-shape.md` + shape-only harness
    `reg_arch-cuda-gpw-public-shape-freeze`; CUDA full GPW product public
    shape **equals** CPU `GpwDft`; OT residual does not block; unlocks
    staffing `cuda-gpw-full-product` under that contract (not auto-landed);
    merge `0147fe02` / docs `aea603bb`; spec closed
- **Still `[~]`:**
  - `arch-pw-driver-paths` @ `c20bb04f` (ahead=1 behind=2; **rebase onto
    `0147fe02`** before gate)
  - `arch-mg-main-structural` @ `9ee9f49e` (feat `1f62a117`; ahead=2
    behind=2; **rebase onto `0147fe02`** before gate)
  Unmerged tip ≠ done.
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed multi-path
  GPW DIIS + landed PW fixed-occ stages + freeze note
  ([cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md))
  (+ tip PW multi-path Ops / MG structural) — do not invent a second SCF
  shape. MG is structural MAIN path, not soft pins.
- **incomplete_remaining=2**, pending unheld=0, blocked staffable product=1
  (`cuda-gpw-full-product` under freeze contract; GH200 + CUDA↔CPU tols;
  not staffed this campaign wave).
- **gaps_remaining=3** (2 in_progress + blocked full CUDA GPW product).
  **new_slugs_added=0**. Queue **not** exhausted.

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
