# Architecture campaign loop

Instance of the [oracle capability loop](oracle-capability-loop.md) for
**architecture / refactor** work on Atomiverse.

Script: `.grok/workflows/arch-campaign.rhai`  
Queue: [arch-campaign-queue.md](arch-campaign-queue.md)  
Handoff: [arch-campaign-handoff.md](arch-campaign-handoff.md)

## Intent

Autonomous waves:

```text
GapCensus → Assign → Worktrees → Merge? → Clearup → Harness
```

Each wave **re-diffs** CLAUDE iron laws + notes + tree vs `dev`, refills
pending architecture debt, implements **one shape change at a time**, then
**always** clearups worktrees and wipes builds.

## Scope in

- Driver/stage reshape (GPW, PW, CUDA SCF fold)
- Test iron-law cleanup (e2e out of `tests/`)
- Documented freezes that unlock later product work
- Structural MG path rewrite **after** Driver node (no soft pins)

## Scope out

- New electronic methods (hybrid/GAPW/…)
- Pin-safe collint micro-opts (exhausted)
- Classical MD CUDA (OpenMM track)
- Leaving multi-GB `build*` trees

## Disk

Wipe `build` / `build-*` under parent and every worktree every wave.

## Run log

(autonomous harness appends lines here)

- **2026-08-10 harness wave 2 (parent `dev` @ `42b8982c`):** 2 of staffed →
  done [x]: `arch-cuda-scf-solver-fold` (merge `3a20af0a` / impl `772497c1` —
  free-loop `SCFSolver` folded under `internal::RhfScfCycle`; public
  `RHF{basis}.run`/`forces` stable; nested `cuda::Driver<scf::Phase>` residual)
  and `arch-e2e-tests-wave2` (merge `42b8982c` / impl `49d03074` + polish
  `c6b623be` — residual multi-iter out of `tests/`; `ATV_TEST_HEAVY_REGEX=^$`;
  `reg_arch-e2e-tests-wave2`). Specs closed (deleted
  `.claude/specs/arch-cuda-scf-solver-fold/` +
  `.claude/specs/arch-e2e-tests-wave2/`). Keep other pending `[ ]`:
  `arch-gpw-driver-paths`, `arch-pw-driver-node`,
  `arch-cuda-gpw-public-shape-freeze`, `arch-mg-main-structural`. Pending **4**;
  blocked **1** staffable product (`cuda-gpw-full-product`);
  **gaps_remaining=5**. Orphan tips `wf/arch-gpw-driver-paths` /
  `wf/arch-pw-driver-node` not on `dev`. Clearup already ran (no isolation
  worktrees; build dirs empty). Roadmap: no ○ rows for these slugs — no ○→✓.
  Queue **not** exhausted (GapCensus owns discovery). No push.
- **2026-08-10 GapCensus wave 3 (parent `dev` @ `350644ca`):** re-diff only —
  tree reconfirmed Γ-RKS-only GPW Driver, free-loop PW, folded CUDA RHF,
  CUDA GPW stages≠product, HEAVY `^$`. **pending=4**, **gaps_remaining=5**,
  **new_slugs_added=0**. No roadmap ○→✓ (capability tables are product
  physics; arch prose already accurate). Optional `chore(queue)` only. No
  push.
- **2026-08-10 Assign wave 3 (parent `dev` @ `1eb6fca2`):** staff **2**
  resumes → `[~]`: `arch-gpw-driver-paths` (`6553a036`, ahead=1 behind=7) +
  `arch-pw-driver-node` (`42842e25`, ahead=1 behind=7). Skipped P1
  `arch-cuda-gpw-public-shape-freeze` + P2 `arch-mg-main-structural` (dep on
  paths). **new_assigned=0**, **incomplete_remaining=2**, pending held=2,
  blocked=1 staffable product. No push.
- **2026-08-10 harness wave 3 (parent `dev` @ `f016215e`):** 1 of staffed →
  done [x]: `arch-pw-driver-node` (merge `f016215e` / impl `1bad527a` —
  `PW` → `cpu::Driver<pw::Phase>` restricted fixed-occ
  `veff→diagonalize→density→mix→check` + `PwConverged`; public
  `build`/`run`/`forces`/`stress` stable; spin/smear free-loop residual;
  `test_cpu_pw_driver_node` + `reg_arch-pw-driver-node`). Spec closed
  (deleted `.claude/specs/arch-pw-driver-node/`). Keep other `[~]`:
  `arch-gpw-driver-paths` @ `cca084b6` (ahead=1 behind=2; rebase onto
  `f016215e` before gate). Pending held **2** (shape-freeze + mg-main);
  blocked **1** staffable product (`cuda-gpw-full-product`);
  **gaps_remaining=4**; **new_slugs_added=0**. CLAUDE remaining-debt:
  drop "PW not yet a Driver node"; residual is spin/smear free loops +
  GPW multi-path. Clearup already ran (no isolation worktrees; build dirs
  empty). Roadmap: no ○ rows for these arch slugs — no ○→✓. Queue **not**
  exhausted (GapCensus owns discovery). No push.
- **2026-08-10 GapCensus wave 4 (parent `dev` @ `d24283d1`):** re-diff only —
  tree reconfirmed Γ-RKS-only GPW Driver, PW restricted fixed-occ Driver +
  spin/smear free loops, folded CUDA RHF, CUDA GPW stages≠product, HEAVY
  `^$`. **Staffed residual** `arch-pw-driver-paths` (parent node landed
  wave-3; foothold ≠ closed). **pending=3**, **gaps_remaining=5**,
  **new_slugs_added=1**. Roadmap arch prose: drop "PW not-yet-Driver"; name
  both path residuals. Keep `[~]` `wf/arch-gpw-driver-paths` @ `cca084b6`
  (ahead=1 behind=3; rebase). Optional `chore(queue)` only. No push.
- **2026-08-10 Assign wave 4 (parent `dev` @ `d24283d1` / queue `26fb8328`):**
  staff **2** → `[~]`: resume `arch-gpw-driver-paths` (`cca084b6`, ahead=1
  behind=3; rebase onto `d24283d1` before gate) + new
  `arch-pw-driver-paths` (branch `wf/arch-pw-driver-paths` off `dev`; no tip
  yet). Skipped P1 `arch-cuda-gpw-public-shape-freeze` + P2
  `arch-mg-main-structural` (dep on GPW paths). **new_assigned=1**,
  **incomplete_remaining=2**, pending held=2, blocked=1 staffable product.
  No open arch specs. Lever: MeanFieldMethod + scf/stages + landed Γ-RKS
  GPW + PW fixed-occ (+ tip multi-path Ops). No push.
- **2026-08-10 GapCensus wave 5 (parent `dev` @ `26fb8328`):** re-diff only —
  tree reconfirmed Γ-RKS-only GPW Driver, PW restricted fixed-occ Driver +
  spin/smear free loops, folded CUDA RHF, CUDA GPW stages≠product, HEAVY
  `^$`. Tips rebased/advanced onto `26fb8328` (ahead=1 behind=0):
  `wf/arch-gpw-driver-paths` @ `06bc94fb` (DIIS multi-path claim; OT residual);
  `wf/arch-pw-driver-paths` @ `98e98ed5` (all product SCF claim). **Unmerged
  tip ≠ done.** **pending=2** (held freeze + mg-main), **in_progress=2**,
  **gaps_remaining=5**, **new_slugs_added=0**. Roadmap: no ○→✓ (arch prose
  already accurate for `dev`). Optional `chore(queue)` only. No push.
- **2026-08-10 Assign wave 5 (parent `dev` @ `1c2d74d6`):** staff **2**
  resumes → `[~]`: `arch-gpw-driver-paths` (`06bc94fb`, ahead=1 behind=1;
  rebase onto `1c2d74d6` before gate) + `arch-pw-driver-paths` (`98e98ed5`,
  ahead=1 behind=1; rebase). Specs on tips only
  (`.claude/specs/arch-{gpw,pw}-driver-paths/SPEC.md`). Skipped P1
  `arch-cuda-gpw-public-shape-freeze` + P2 `arch-mg-main-structural` (dep on
  GPW paths merge). **new_assigned=0**, **incomplete_remaining=2**, pending
  held=2, blocked=1 staffable product (`cuda-gpw-full-product`). Lever:
  MeanFieldMethod + scf/stages + landed Γ-RKS GPW + PW fixed-occ (+ tip
  multi-path Ops). No push.
- **2026-08-10 harness wave 5 (parent `dev` @ `e34ccf92`):** 1 of staffed →
  done [x]: `arch-gpw-driver-paths` (merge `e34ccf92` / impl `42454014` +
  gate docs `2ba03049` — multi-path **DIIS** Γ UKS + multi-k RKS/UKS via
  path-local Ops + `drive_path_diis_` / `drive_stage_loop_`; public
  `run`/`build`/`forces` stable; OT free-loop residual documented, not a
  new slug; `test_cpu_gpw_driver_node` + `reg_arch-gpw-driver-paths` +
  twins `gpw-uks-o2-gamma` / `gpw-kpoints-si`). Spec closed (deleted
  `.claude/specs/arch-gpw-driver-paths/`). Keep other `[~]`:
  `arch-pw-driver-paths` @ `9edaa157` (ahead=1 behind=3; rebase onto
  `e34ccf92` before gate). Pending unheld **2** (shape-freeze + mg-main;
  dep on GPW multi-path DIIS satisfied); blocked **1** staffable product
  (`cuda-gpw-full-product` still needs freeze); **gaps_remaining=4**;
  **new_slugs_added=0**. CLAUDE remaining-debt: GPW multi-path DIIS landed;
  residual is OT free loops + PW spin/smear paths. Clearup already ran
  (no isolation worktrees; build dirs empty). Roadmap: no ○ rows for these
  arch slugs — no ○→✓. Queue **not** exhausted (GapCensus owns discovery).
  No push.
- **2026-08-10 GapCensus wave 6 (parent `dev` @ `9cbdc47b`):** re-diff only —
  tree reconfirmed multi-path GPW DIIS Driver (OT free-loop residual), PW
  restricted fixed-occ Driver + spin/smear free loops, folded CUDA RHF,
  CUDA GPW stages≠product, HEAVY `^$`. **Unmerged tip ≠ done** —
  `wf/arch-pw-driver-paths` @ `9edaa157` (ahead=1 behind=4; rebase onto
  `9cbdc47b`). **pending=2** (unheld freeze + mg-main), **in_progress=1**,
  **gaps_remaining=4**, **new_slugs_added=0**. Roadmap arch prose: GPW
  multi-path DIIS landed; freeze remains the CUDA-GPW product gate. No
  product-table ○→✓. Optional `chore(queue)` only. No push.
- **2026-08-10 Assign wave 6 (parent `dev` @ `9b98b54f`):** staff **3** →
  `[~]`: resume `arch-pw-driver-paths` (`9edaa157`, ahead=1 behind=5;
  rebase onto `9b98b54f` before gate) + new
  `arch-cuda-gpw-public-shape-freeze` (P1) + new `arch-mg-main-structural`
  (P2; prefer freeze-first if contention). Dep on GPW multi-path DIIS
  satisfied. **new_assigned=2**, **incomplete_remaining=3**, pending
  unheld=0, blocked=1 staffable product (`cuda-gpw-full-product`). Lever:
  MeanFieldMethod + scf/stages + multi-path GPW DIIS + PW fixed-occ (+ tip
  multi-path PW Ops). No push.
- **2026-08-10 GapCensus wave 7 (parent `dev` after Assign `52d555b7` + this chore):** re-diff only — tree reconfirmed multi-path GPW DIIS Driver (OT free-loop residual), PW restricted fixed-occ Driver + spin/smear free loops, folded CUDA RHF, CUDA GPW stages≠product, HEAVY `^$`. **Unmerged tip ≠ done** — all three incomplete tips claim product (ahead=1 behind=1 vs post-chore `dev`; behind is docs-only census): `wf/arch-pw-driver-paths` @ `0035b0f4`, `wf/arch-cuda-gpw-public-shape-freeze` @ `d466f5a8`, `wf/arch-mg-main-structural` @ `2ed7a836`. **pending=0** (unheld), **in_progress=3**, **gaps_remaining=4**, **new_slugs_added=0**. Roadmap arch prose already accurate for `dev` — no product-table ○→✓. Optional `chore(queue)` only. No push.
