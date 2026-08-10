# Oracle capability loop — abstract pattern

Shared design for **PySCF / CP2K / MetalWalls / TB (tight-binding)** (and future
oracle-backed) unattended method loops. Concrete scripts live under
`.grok/workflows/*-method-loop.rhai`.
**Instantiate new loops with skill** `.agents/skills/create-oracle-loop/`.

## Intent (binding)

```text
repeat until true capability gaps are gone (or budget / max_waves):
  1. Gap-census: oracle/map/roadmap  vs  what ATV has on target_branch
  2. Update queue ## pending + roadmap ○/✓  (+ optional draft specs)
  3. Assign up to staff_count (incomplete ≫ pending)
  4. Parallel worktrees: spec → offline pin → impl → dual gate → commit wf/<slug>
  5. Merge greens → target_branch (skip if no greens)
  6. Clearup (ALWAYS this wave): drop finished worktrees + prune merged tips + wipe builds
  7. Harness: mark done, close specs, handoff, run log (if merged)
  8. Tutorial pages (optional, if merged)
  9. → back to (1)   // NOT "stop because seed queue empty"
```

**Wrong (old behaviour):** one Census at shift start → burn a seed pending list →
`queue_empty` while the oracle still has features we lack.

**Also wrong:** leave `staff_count` isolation worktrees + full `build/` trees until
the shift ends (disk full; Recover confuses zombie trees with real incomplete work).

**Right:** after each implement wave, **re-diff the oracle surface** (and project
capability map / roadmap), then **reopen pending**. A foothold regression
**does not** close a capability row if production depth is still missing —
open a **deeper slug**. And **the same wave** that ran workers must **clear** their
subagents / worktrees before the next GapCensus.

## Roles

| Role | Worktree | Writes |
| ---- | -------- | ------ |
| Bootstrap / Recover / **GapCensus** / Assign / Merge / **Clearup** / Harness / Index | parent | `target_branch` only |
| Feature worker | isolation | `wf/<slug>` until merge |

## Dual gate (every worker)

1. **Correctness:** offline oracle pin in `regressions/.../REFERENCE.md` (never live oracle in ctest).
2. **Speed (optional):** \(T_{\mathrm{atv}}/T_{\mathrm{oracle}} \le\) `max_ratio`, or `perf_exempt` + reason.

`status=green` requires `acceptance_met=true` when the loop is spec-first.

## Scoped build & test (binding)

Every tree compiles and tests **only the part under development** — never the
whole product. Unit tests are one `test_*` executable per topic and regressions
are standalone `reg_<slug>` / `run_reg_<slug>` targets, so scoping is exact:

```bash
cmake -B build <cmake_flags>                 # configure once per tree
cmake --build build -j$(nproc) \
      --target atomiverse <touched test_*> reg_<slug>   # NEVER bare `cmake --build build`
ctest --test-dir build -R '^(<touched test_*>)$'        # only your tests
cmake --build build --target run_reg_<slug>             # your regression
```

| Rule | Binding behaviour |
| ---- | ----------------- |
| Worker build | `--target atomiverse` + the `test_*` binaries the slug touched/added + `reg_<slug>` |
| Worker test | `ctest -R` on those test names + `run_reg_<slug>`; nothing else |
| Merge gate | Scoped rebuild + the loop-domain suite `ctest -R '<test_filter>'` on `target_branch` |
| Forbidden | Bare `cmake --build build` (compiles every test binary); full unfiltered `ctest` anywhere in the loop |

`test_filter` is a per-loop CONFIG knob: an anchored regex bounding the loop's
domain suite (e.g. cp2k `^test_cpu_(gpw|pw|scf|xc)_`). Cross-domain breakage is
CI's / the operator's job, not the loop's.

## Continuous discovery rules

1. **GapCensus every wave start** (and after merge before next assign if you split phases).
2. **Sources:** oracle docs/examples/API surface + `.claude/notes/*map*` + `docs/roadmap/*.md` ○ rows + tree scan (`include/`, `regressions/`).
3. **Foothold ≠ done:** if only a toy pin exists, keep a deeper pending slug.
4. **Update artifacts each census:**
   - `docs/developer/<loop>-method-queue.md` — rewrite `## pending` (preserve done/blocked/in_progress)
   - `docs/roadmap/*` — ○→✓ only when truly on branch; add ○ for newly discovered gaps
   - optional: open `.claude/specs/<slug>/` stubs for large gaps (workers still own full specs)
5. **Stop `queue_empty` only when** GapCensus reports `gaps_remaining=0` **and**
   Assign finds no work for `stall_no_work` consecutive waves.
6. **Never** stop solely because the original seed list is exhausted.

## Subagent / worktree clearup (binding — TIMELY)

`agent()` / `parallel()` **block until the child finishes**. When a worker panel
returns, those subagents are **done** — they must not be treated as live sessions
for the next wave. Isolation worktrees and builds, however, **persist on disk**
until explicitly removed.

| Rule | Binding behaviour |
| ---- | ----------------- |
| Barrier | `parallel(workers)` return ⇒ workers finished; log that clearup is owed **this wave** |
| Always Clearup | Run **Clearup after Worktrees every wave**, including all-red waves (no greens) |
| Order | Merge (if greens) **then** Clearup, so branch tips still exist for `git merge` |
| Worktrees | `git worktree list` → wipe `build*` under this-wave trees → `git worktree remove` |
| Branches | `git branch -d wf/<slug>` only if fully merged into `target_branch`; never `-D` unmerged incomplete |
| Nested agents | Workers **must not** spawn long-lived nested subagents, monitors, or child workflows |
| Parent agents | Census / assign / merge / clearup / harness are single-shot |
| Cross-wave | Do **not** resume prior-wave worker `agent_id`s; Recover reads git + queue + specs |
| Forbidden | Holding N full worktree builds until `max_waves` ends |

Clearup reports `worktrees_removed`, `branches_pruned`, `builds_wiped` (see template
`clearup_schema`).

## Per-oracle configuration (fill when creating a loop)

| Knob | PySCF | CP2K | MetalWalls | SE / tight-binding |
| ---- | ----- | ---- | ---------- | ------------------ |
| Oracle | offline PySCF | offline CP2K | LAMMPS-first, else MetalWalls | **Family-locked:** OpenMOPAC (NDDO), DFTB+ (DFTB), xtb (GFN); Sparrow/CP2K arch-only |
| Queue | `pyscf-method-queue.md` | `cp2k-method-queue.md` | `metalwalls-method-queue.md` | `tb-method-queue.md` |
| Regression root | `regressions/scf/` | `regressions/{gpw,pw,scf}/` | `regressions/{md,forcefield}/` | `regressions/semiempirical/` |
| Build flags | HF ON; CUDA OFF | HF+PW+GPW ON; CUDA OFF | MD+KSPACE ON; HF/PW/GPW/CUDA OFF | **SE-only**: SEMIEMPIRICAL+EVALUATE ON; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/CUDA OFF |
| Test filter | `^test_cpu_(scf\|linalg)_` + molecular extras | `^test_cpu_(gpw\|pw\|scf\|xc)_` | `^test_cpu_(md\|mc\|pair\|fix\|observer\|…)` | `^test_cpu_(am1\|rm1\|dftb2\|xtb_gfn0\|nddo_)` |
| Scope IN | molecular Gaussian / post-HF / continuum | GPW / PW / periodic | classical electrochem MD | DFTB0→3 ladder, NDDO PM depth, late GFN-xTB |
| Scope OUT | classical MD, GPW/PW | molecular post-HF, pure PySCF | SCF, GPW/PW, CUDA, PLUMED, 2D product, cross-lang product | SCF product, GPW/PW product, classical MD, CUDA; cross-family oracle misuse |
| Map note | roadmap chapters | `docs/roadmap/dft.md` | `.claude/notes/metalwalls-capability-map.md` | `docs/roadmap/semiempirical.md` + `tools/tb_oracle/` |

## Template & skill

| Asset | Path |
| ---- | ---- |
| Rhai skeleton | `.grok/workflows/_templates/oracle-capability-loop-template.rhai` |
| Skill | `.agents/skills/create-oracle-loop/SKILL.md` |
| Instances | `pyscf-method-loop.rhai`, `cp2k-method-loop.rhai`, `metalwalls-method-loop.rhai`, `tb-method-loop.rhai`, `es-cpu-cuda-opt.rhai` (dual-track opt), `arch-campaign.rhai` (architecture/refactor; same phase order) |

Rhai has **no include** — instances are full scripts copied from the template.
Specialize only:

1. **CONFIG** — paths, cmake, scope, defaults  
2. **`extra_law`** — loop-specific binding rules (oracle priority, forbidden
   backends, arch constraints, pin tools). Shared DISK/CLEARUP/GapCensus stay
   in template law — do not restate them.  
3. Optional tiny hooks (worker pin tool line, merge conflict hotspots, one extra
   worker_schema field).

Phase order and Clearup rules must not drift. Skill:
`.agents/skills/create-oracle-loop/SKILL.md`.

## Args (common)

| Arg | Default | Meaning |
| --- | ------- | ------- |
| `staff_count` | 10 | Parallel worktrees |
| `max_waves` | 20 | Max implement waves this shift |
| `target_branch` | `dev` | Merge destination |
| `do_perf` / `require_perf` | loop-specific | Fair wall gate |
| `write_tutorials` | true | Handbook pages |
| `mode` | `exhaust` | Until gaps gone / budget |
| `stall_no_work` | 2 | Empty assign waves after gap-census before stop |

## Disk / worktree hygiene (binding)

After **every** cmake/build + test/regression cycle, and in **Clearup** the same
wave, delete build products and temps so the next wave does not fill the disk:

```bash
# from tree root that was built (parent or worktree)
rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
# oracle pin temps as appropriate, e.g. /tmp/atv-* /tmp/*_oracle_*
```

| When | Who |
| ---- | --- |
| After worker build+test(+perf) | Worker last step, before exit |
| After merge-time rebuild on parent | Merge agent |
| After Worktrees (+ Merge) same wave | **Clearup** agent — remove worktrees + prune merged tips + wipe builds |
| Harness (if merged) | Re-check parent only; Clearup already dropped worker trees |

**Do not** delete source, `regressions/`, `REFERENCE.md`, specs, or `.git`.
