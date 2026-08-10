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
