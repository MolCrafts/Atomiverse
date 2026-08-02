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
- **2026-08-02 cp2k-perf-loop trial wave (parent `dev` @ `1bee3462`):** merged
  `wf/perf-p0-k-zgemm` @ `980157ed` → dev `c2fb3324` (BLAS `zgemm_` in
  `linalg::c_mul`/`c_dagger_mul`; `kpoints.cpp` duplicate matmuls deleted —
  one zgemm pair serves multi-k SCF + every OT step). Merge gate: unit
  `^test_cpu_(gpw|pw|scf|xc)_|^test_xc_` **60/60** green post-merge; worker
  regs kpoints-si / ot-water / multigrid-kpoints PASS, tolerances unchanged.
  Walls (OMP=4, min of 2): kpoints-si reg 19.17→19.00 s (ratio 5.81→5.76 vs
  CP2K MAIN pin 3.3 s; reg wall inflated vs MAIN-only ~1.1), ot-water
  14.00→13.97 s, MG MAIN 65.1→65.1 s — parity within noise on small-nb
  twins; scalar O(n³) ceiling removed. P0 now empty; pending **7**
  (P1×4 + P2×3), blocked **1**, done **5**. Builds wiped. No push.
- **2026-08-02 cp2k-perf wave 2 bench+harness (parent `dev` @ `97de6c19`):**
  merged `wf/perf-p1-ot-convergence` @ `a2facc8e` → dev `97de6c19` (Γ RKS OT
  line-search economics: carry accepted KS build as next E(0), quadratic
  retry α; KS builds/OT iter 2.32→1.11). Official post-merge walls (OMP=4,
  min of 3, contended loadavg ~10–16, four twins PASS unchanged tols):
  **ot-water 13.97→8.87 s (−36.5%)**; kpoints-si reg 19.07 s (prints no MAIN
  wall; MAIN-only est ~3.7 s → ratio ~1.1, reg-wall ratio 5.78 vs pin 3.3 s);
  MG MAIN 65.1 s (reg 91.30; ratio 2.76 vs 23.6 s); uks `wall=` 19.11 s
  (ratio 0.90 vs 21.3 s; 17.6 s on quieter node — load drift, path
  untouched). worst_ratio (MAIN walls where defined) **5.78** (kpoints-si
  reg-wall; MAIN-only class ~1.1). Baseline note fair-wall table re-dated
  2026-08-02 wave2. Pending **6** (P1×3 + P2×3), blocked **1**, done **6**.
  Builds wiped. No push.
