# GPW performance loop

Continuous optimization of **existing** CPU GPW (and tightly coupled grid
helpers). Not a capability census — that is `cp2k-method-loop`.

| | |
| --- | --- |
| Workflow | `.grok/workflows/gpw-perf-loop.rhai` |
| Queue | `docs/developer/gpw-perf-queue.md` |
| Architecture | `.claude/notes/gpw-perf-architecture.md` |
| Target branch | `dev` (default) |

## Wave shape

```
Recover → Census (re-rank remaining hotspots) → Assign → Worktrees
  → Merge? → Clearup (always) → Harness?
```

Clearup **always** runs after Worktrees (greens or reds): remove isolation
worktrees, prune fully-merged `wf/perf-*` tips, **wipe every leftover
`build*`** on parent and staffed trees.

## Green criteria

1. **Correctness:** scoped unit tests + listed twin regressions still pass
   within existing tols (no pin softening).
2. **Speed:** fair wall on a matched single-SCF twin improves, **or** the
   task is pure extraction/refactor with **no** wall regression on the four
   baseline twins (document both before/after).
3. **Disk:** worker exits only after wiping its `build*`; merge wipes parent
   if it rebuilt; clearup double-checks.

## Disk law (shared nobackup)

Wipe after **every** build+test and at clearup:

```text
rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache
rm -rf /tmp/atv-* /tmp/cp2k_oracle_*   # only temps you created
```

Never delete: sources, `regressions/`, `.claude/specs/`, `.git`, oracle inputs.

## Branch naming

`wf/perf-<slug>` (e.g. `wf/perf-p0-fft-plan-cache`) so capability `wf/<slug>`
tips stay distinct.

## CMake (scoped)

```bash
module load buildenv-gcc/2026.03-mpich
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target atomiverse <tests> <regs>
```

Then **wipe build** before leaving the worktree.

## Run log

(autonomous harness appends lines here)

- **2026-08-01 harness wave 1 (parent `dev` @ `924cf858`):** 1 of staffed
  → done [x]: `p0-profile-baseline` (fair-wall + `ATV_GPW_PROF` table in
  `.claude/notes/gpw-perf-baseline.md`; thin k/uks prof marks; four twins
  PASS; unit `test_cpu_(gpw|xc)_` 12/12). One staff stays `[~]`:
  `p0-fft-plan-cache` on unmerged `wf/perf-p0-fft-plan-cache` @ `af9e8c82`
  (FFTW plan/buffer cache tip; do not demote). Residual pending **10**
  (P0×3 + P1×4 + P2×3). Blocked **1** (`global-xtensor-xsimd`). Clearup
  already ran. Queue **not** exhausted (census pending still open). No push.
- **2026-08-01 recover (parent `dev` @ `3de007ef`):** incomplete_count=**1**.
  Open `wf/perf-*` unmerged: only `wf/perf-p0-fft-plan-cache` @ `af9e8c82`
  (1 commit ahead of merge-base `1054b08e`; dev +3 post-base including
  profile-baseline — resume needs merge/rebase onto current `dev`). Queue
  `## in_progress` already normalized; no demote; no pending promoted;
  no invented slugs. Staffed pick ≤3 → **1**: `p0-fft-plan-cache`. Pending
  **10**, blocked **1**, done **1**. No worktrees; no leftover `build*`.
  No push.
- **2026-08-01 PERF-CENSUS wave 2 (parent `dev` @ `1498080d`):** re-diff
  `cpu/gpw` hotspots (solver/collocate/multigrid/kpoints/poisson/OT/forces)
  + baseline table. Architecture ranks + measured wall order **unchanged**.
  Rewrote `## pending` only (still P0×3 + P1×4 + P2×3 = **10**); no new
  capability slugs; `new_slugs_added=0`. In_progress: `p0-fft-plan-cache`
  still `[~]` — tip rebased to `7cb7cc51` on current parent (queue
  `in_progress` line may still show older SHA; do not demote). Blocked **1**.
  No builds left. No push.
- **2026-08-01 ASSIGN wave 2 (parent `dev` @ `c59dc18b`):** staff **3**
  P0 real-wall items → `[~]`: (1) resume `p0-fft-plan-cache` @ `7cb7cc51`
  (rebase onto `dev`, measure, merge-ready); (2) `p0-collocate-buffers`
  (#1 MG integrate wall); (3) `p0-ot-double-energy` (multi-k OT double
  ρ→H; sole `gpw_solver.cpp` worker). Deferred `p0-k-zgemm` to next wave
  (file-disjoint but lower measured rank). Soft multigrid overlap
  fft∩collocate — fft finishes first. Pending **8** (P0×1 + P1×4 + P2×3),
  blocked **1**, done **1**. No worktrees this step; no builds; no push.
