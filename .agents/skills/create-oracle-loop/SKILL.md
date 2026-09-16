---
name: create-oracle-loop
description: >
  Scaffold or update an Atomiverse oracle capability loop (PySCF/CP2K/MetalWalls-style):
  continuous gap-census → assign → worktrees → merge → clearup → harness.
  Copy the template, fill CONFIG, append loop-specific rules via extra_law.
argument-hint: "<oracle-name e.g. lammps-md | openmm | …>"
user-invocable: true
---

# Create / update an oracle capability loop

## Read first

1. `docs/developer/oracle-capability-loop.md` — binding design
2. `.grok/workflows/_templates/oracle-capability-loop-template.rhai` — **canonical skeleton**
3. Live twins only as CONFIG examples: `pyscf` / `cp2k` / `metalwalls-method-loop.rhai`

## Scaffold

1. **Copy** template → `.grok/workflows/<loop_id>-method-loop.rhai`  
   Rhai has **no include** — full script required. Do not invent a second control flow.
2. **Fill CONFIG** (paths + cmake + scope). See § CONFIG below.
3. **Add loop-specific rules** in `extra_law` (optional but usual). See § Extra requirements.
4. **Set meta** `name` / `description` / `when_to_use` to the real loop (not TEMPLATE).
5. **Seed docs** if missing: loop doc, queue, handoff, `tools/<oracle>_oracle/README.md`.
6. **Smoke:** `workflow` `validate_only: true` +  
   `{ "mode":"burst", "staff_count":2, "max_waves":1, "skip_recover":true }`
7. **Run:** `agent_budget: 1024`, `staff_count` from CONFIG default, `mode: exhaust`.

## CONFIG (required)

| Field | Purpose |
| ----- | ------- |
| `loop_id` | Short id (`pyscf`, `cp2k`, `metalwalls`) |
| `oracle_name` | Pin / fair-bench label |
| `queue_path` / `handoff_path` / `loop_doc` | Harness docs under `docs/developer/` |
| `map_path` / `gap_sources` | GapCensus inputs |
| `oracle_readme` | Offline pin contract |
| `regression_hint` | Path pattern e.g. `regressions/scf/<slug>/` |
| `cmake_flags` | **Only** flags this loop may use (forbid wrong backends) |
| `test_filter` | Anchored `ctest -R` regex bounding this loop's domain suite — merge gate only; workers build/run only their touched `test_*` + `reg_<slug>` |
| `scope_in` / `scope_out` | What GapCensus may / must not queue |
| `extra_law` | **Loop-specific binding text** (see below); `""` if none |
| `default_staff_count` etc. | Defaults still overridable by `args` |

Everything **below** CONFIG in the template (args clamp, schemas, law core, wave loop)
is shared machinery. Prefer not to edit it. If you must (e.g. extra worker JSON field),
keep phase order and Clearup semantics identical.

## Extra requirements (`extra_law`)

Users **may and should** put oracle-specific constraints here. Appended **after** the
shared template law so agents always see both.

**Put in `extra_law`:**

- Oracle priority (e.g. LAMMPS-first, else MetalWalls)
- Forbidden product surfaces (e.g. “no HF/PW in this MD loop”)
- Architecture rules (e.g. electrode = Component on `md::MD`, not a new Driver)
- Pin / bench tool paths (`tools/pyscf_oracle/pin_scf.py`, fair_bench, …)
- Units / citation quirks (Ha/Ry internal OK when cited)
- Temp wipe globs (`/tmp/pyscf_oracle_*`)
- Worker acceptance extras (e.g. `md_only_build=true`)

**Do not put in `extra_law` (already in template law):**

- GapCensus every wave / foothold ≠ closed
- Spec-first + `acceptance_met`
- DISK wipe after build
- Scoped build/test (`--target atomiverse <test_*> reg_<slug>`, `ctest -R`; no bare `cmake --build build`, no full ctest)
- CLEARUP same-wave worktree/subagent rules
- Parent `target_branch` / `wf/<slug>` / no push

**Shape:**

```rhai
let extra_law = "";
extra_law += "ORACLE: if feature exists in LAMMPS, pin+bench vs LAMMPS; MW only for MW-only physics. ";
extra_law += "FORBIDDEN cmake: ATV_BUILD_HF=ON, ATV_BUILD_CUDA=ON. ";
extra_law += "Arch: const-pot = Component on cpu::md::MD; never charge solve inside KSpace*. ";
```

Optional **narrow** prompt hooks (only if `extra_law` is not enough):

- Append one line to the worker prompt for pin tool invocation
- Append merge conflict hotspots (`gpw/`, `tools/cp2k_oracle/`, …)
- Extend `worker_schema` properties (keep required fields)

Never fork a fourth phase order or skip Clearup.

## Binding phase order (per wave)

```text
GapCensus → Assign → Worktrees → Merge? → Clearup → Harness? → Tutorial?
```

| Rule | Binding |
| ---- | ------- |
| GapCensus | Every wave; foothold ≠ capability closed |
| Clearup | **Always** after Worktrees (greens or reds) |
| Merge / Harness / Tutorial | Only when mergeable work landed |
| `queue_empty` | `gaps_remaining=0` + empty assign × `stall_no_work` |
| Workers | Spec-first, offline pin, dual gate; no nested long-lived subagents |
| Subagents | `parallel` return ⇒ dead; remove worktrees same wave |

## Checklist

- [ ] Copied from **current** template (not an old live dialect)
- [ ] CONFIG has no `REPLACE`; cmake cannot enable wrong product
- [ ] `extra_law` holds only loop-specific rules (or `""`)
- [ ] Queue / handoff / oracle README exist
- [ ] Dual gate + offline pin only
- [ ] Phase order + Clearup unchanged
- [ ] `validate_only` passes
- [ ] Listed under Instances in abstract doc

## Updating PySCF / CP2K / MetalWalls

Rebase instances onto the template: replace machinery wholesale, re-apply only CONFIG +
`extra_law` (+ tiny prompt hooks). Do not maintain parallel long `law +=` dialects.
