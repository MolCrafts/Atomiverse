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
