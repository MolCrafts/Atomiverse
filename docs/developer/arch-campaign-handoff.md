# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` @ `9cbdc47b` (post harness wave-5 merge `e34ccf92`)
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
- **GapCensus wave 6:** reconfirmed open set; **no new slugs**. Tree facts
  match queue. Roadmap arch prose updated (multi-path DIIS landed; freeze
  remains CUDA-GPW product gate).
- **Still `[~]`:** `arch-pw-driver-paths` @ `9edaa157` (ahead=1 behind=4;
  **rebase onto `9cbdc47b`** before gate). Unmerged tip ≠ done.
- **Pending staffable (unheld):** `arch-cuda-gpw-public-shape-freeze` (P1)
  + `arch-mg-main-structural` (P2) — dep on GPW multi-path DIIS **satisfied**.
- **Lever:** copy `MeanFieldMethod` + `scf/stages/` + landed multi-path
  GPW DIIS + landed PW fixed-occ stages (+ tip PW multi-path Ops) — do not
  invent a second SCF shape.
- **gaps_remaining=4** (1 in_progress + 2 pending + blocked full CUDA GPW
  product). **new_slugs_added=0**. Queue **not** exhausted.

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
