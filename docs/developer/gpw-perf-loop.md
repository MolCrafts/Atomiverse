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
