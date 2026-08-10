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
