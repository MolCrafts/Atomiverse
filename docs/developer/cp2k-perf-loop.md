# CP2K performance loop

CP2K-anchored optimization of the **landed** GPW/PW CPU surface. Rewrite of
`gpw-perf-loop` where the CP2K wall is the hard metric, not an optional
reference: every landing reports absolute `T_atv/T_cp2k` on a matched twin
against a **pinned real-CP2K run**. Not a capability census — that is
`cp2k-method-loop`.

| | |
| --- | --- |
| Workflow | `.grok/workflows/cp2k-perf-loop.rhai` |
| Queue | `docs/developer/gpw-perf-queue.md` (shared with `gpw-perf-loop`) |
| Architecture | `.claude/notes/gpw-perf-architecture.md` |
| Baseline walls | `.claude/notes/gpw-perf-baseline.md` |
| CP2K install | `source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh` → `cp2k.psmp` |
| Target branch | `dev` (default) |
| Stop bonus | `cp2k_matched` when every twin reaches `target_ratio` (default 1.0) |

## Wave shape

```
Recover → Baseline (once: pin CP2K + parent ATV walls per twin)
  → Census (rank by T_atv/T_cp2k contribution) → Assign → Worktrees
  → Merge? → Clearup (always) → Harness (+ target-ratio check)
```

Clearup **always** runs after Worktrees (greens or reds): remove isolation
worktrees, prune fully-merged `wf/perf-*` tips, **wipe every leftover
`build*`** on parent and staffed trees. Branch prefix stays `wf/perf-` so
Recover picks up in-flight `gpw-perf-loop` tips seamlessly.

## Fair twins

`gpw-uks-o2-gamma`, `gpw-kpoints-si`, `gpw-multigrid-kpoints`,
`ot-minimizer-water` — wall = the **same single twin SCF job**,
1 MPI × OMP=4. `T_cp2k` comes only from a real `cp2k.psmp` run pinned with
full provenance (binary + version + input + output line, per
`.claude/notes/oracle-pin-provenance.md`); never estimated, never
self-goldened from ATV.

## Green criteria

1. **Correctness:** scoped unit tests (`^test_cpu_(gpw|pw|scf|xc)_|^test_xc_`)
   + claimed twin regressions pass within existing tols (no pin softening).
2. **CP2K anchor:** a measured absolute `ratio_after = T_atv/T_cp2k` on the
   row's twin (before **and** after), **or** an honest `extraction_only`
   with no wall regression.
3. **FD-force law:** any change touching the energy path
   (collocate/integrate/Poisson/XC/V_eff/PP) re-runs the FD force oracle —
   energy pins cannot see a force/energy functional mismatch.
4. **Bit-exact law:** default gate is bit-identical vs current pins. A change
   that reorders FP summation (vectorized reductions) lands only with a
   stated tolerance vs the CP2K pins plus a fresh FD-force check, documented
   in the commit and queue row.
5. **SIMD law (first-class since 2026-08-03):** portable xsimd is a primary
   optimization lane at any priority (`native_integrals.cpp` /
   `multigrid.cpp` `axis_dot` idiom); arch intrinsics / per-arch dispatch are
   forbidden (x86_64 + aarch64 must both build — CP2K's AVX2-only kernel
   falls back to scalar on aarch64, single-source xsimd beats that trade);
   never global `XTENSOR_USE_XSIMD`.
6. **Disk:** worker exits only after wiping its `build*`; merge wipes parent
   if it rebuilt; clearup double-checks.

## Worker runtime law

Two workers were killed by these before they became law:

- **Verify the worktree base first** — auto-created worktrees can sit at a
  stale snapshot; branch `wf/perf-<slug>` from the target branch explicitly
  and confirm the target files exist before working.
- **Never end a turn waiting on a background event.** A worker that ends its
  turn is terminated — background builds and monitors never re-invoke it.
  Long builds run **foreground** with a timeout and are re-run to completion
  (incremental resume).

## Disk law (shared nobackup)

Wipe after **every** build+test and at clearup:

```text
rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache
rm -rf /tmp/atv-* /tmp/cp2k_oracle_*   # only temps you created
```

Never delete: sources, `regressions/`, `.claude/specs/`, `.git`, oracle inputs.

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

- 2026-08-06 harness wave 2: merged `p0-stvnl-setup-residual` (`e1eaf084`, worker `fc6fe3d5`); queue done [x] with worker ratio_after **2.762**; parent harness re-measured `gpw-multigrid-kpoints` MAIN quiet **1.6 s** / CP2K pin **0.688 s** → **2.33×** (E pin bit-identical −15.8023961242 Ha; ST_k~340 + Vnl_k~60 ms); other three twin ATV walls not re-run (path untouched) — binding worst_ratio still **3.02** (`ot-minimizer-water`); twins_at_target=false (target 1.05); pending **6** (P0×3 + P1×1 + P2×2); builds wiped.
