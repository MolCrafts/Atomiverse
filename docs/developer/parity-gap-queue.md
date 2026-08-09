# Reference-parity gap queue

Worklist for workflow `parity-gap-loop`. Seeded from the **2026-08-09 parity
assessment** (`dev` @ `3f29cc7c`; four-track survey: roadmap census, CUDA↔CPU
completeness, ES perf benchmarking, MD line). One goal:

> Every claim the harness makes about CPU/CUDA parity with reference software
> is either **true on the branch** or **queued here with a stated gate**.

| Track | Goal | Gate |
| --- | --- | --- |
| **harness-truth** | Stale roadmap/notes/REFERENCE claims rewritten in place | cited lines superseded; no append-only corrections |
| **regression-honesty** | Every `--device=cuda` leg provably exercises the device path | fail-closed assertion (`last_*_on_device()`-class) or documented host-fallback |
| **pw-perf-debt** | The 22 `perf_exempt` PW cases get real fair walls | QE `pw.x` installed offline; same-settings NC-UPF twin; pin provenance law |
| **cuda-sanctioned** | Only CUDA waves already sanctioned by `.claude/notes/cuda-rhf-device-only.md:145-147` | CPU same-settings oracle; tolerances stated; no small-molecule speed claims |

**Not this queue:** full CUDA GPW method product (blocked on GPW CPU
architecture freeze, `docs/developer/es-cpu-cuda-queue.md:97`); hybrid / GAPW /
PAW / meta-GGA; periodic exact exchange / RSH (owner: `pyscf-method-loop`
blocked rows); restaffing `cp2k-perf-loop` fair twins; pin-unsafe MG windows;
MD feature width (rigid bodies / SHAKE / RESPA / NEMD — MD roadmap loops own
those); new electronic methods (`cp2k-method-loop` / `pyscf-method-loop`).

**In-flight tree law:** the parent working tree carries ~105 uncommitted
variable-cell/stress files. Workers branch `wf/<slug>` off clean HEAD in
isolation worktrees; Merge defers any branch whose diff overlaps a dirty path.

**Disk law:** wipe `build` / `build-*` after every cmake/ctest on every tree.

**CUDA runtime:** login nodes have **no** CUDA toolkit. CUDA build/test only
via batch on GH200 (`sbatch -p gpu -A naiss2026-4-715-gpu`, module
`GPU/buildenv-gcccuda/2026.03-cu13.0`).

Assessment provenance: binding pins `.claude/notes/gpw-perf-baseline.md`;
CUDA ES state `.claude/notes/cuda-rhf-device-only.md`; order + freezes
`.claude/notes/es-optimization-order.md`; pin law
`.claude/notes/oracle-pin-provenance.md`.

## pending

Priority order. `oracle:` names the code that produced (or will produce) the
pin — `harness` marks docs-only truth repairs (no build, no pin).

### P0 — harness truth

- [ ] `test-gpw-timeout-triage` — oracle: ctest.
  2026-08-09 full suite: 5/377 Timeout — `test_cpu_gpw_{mbd,hirshfeld,forces,
  stress}` + `test_mpi_pw_kpoint_n4` (234 s total suite, no assertion
  failures). Re-run the five serially on a quiet shell with `--timeout`
  raised, record wall each; classify load-flake vs genuinely heavy. If heavy:
  shrink the test system (unit suite is seconds-scale by iron law) on a
  branch; if flake: report evidence, no code change. Read-only on parent
  build; any fix goes through a worktree.

### P1 — regression honesty + sanctioned CUDA

- [ ] `cuda-md-first-regression` — oracle: lammps or literature.
  CUDA MD has zero regression cases (`grep -rl cuda regressions/{md,
  forcefield,mc}` = 0) while the OpenMM-engine track is explicitly unfrozen.
  Add one: LJ argon NVE (existing CPU case's geometry), `--device=cuda`,
  energy-conservation envelope + cited pin, ≤60 s, public headers only.
  Validate via one GH200 batch run. Wave-2 attempt is parked unmerged at
  branch `wf/cuda-md-first-regression` (`ea17e11a`; GH200 jobs 1035232 /
  1035271 failed) — resume from that branch, do not restart from scratch.

### P2 — PW perf debt

- [ ] `pw-perf-fair-walls-batch1` — oracle: qe. Depends: `pw-perf-qe-install`
  (merged `6d13a436`; `tools/qe_oracle/` is live).
  Repin the first batch of exempt PW cases with real same-settings NC-UPF
  walls (`pw-pbe-gga-si`, `pw-analytic-stress-si`, `pw-uks-kmesh-fe`,
  `bands-dos-si`): T_atv/T_qe with full provenance, REFERENCE.md perf sections
  rewritten from `perf_exempt` to measured ratios. Remaining cases become
  `pw-perf-fair-walls-batch2+` via GapCensus.

## in_progress

- [~] `reg-cuda-grad-honesty` — deferred 2026-08-09, GH200 job 1035228
  (FAILED: nvcc syntax errors in the new
  `regressions/scf/h2o_rhf_sto3g_grad/reg_h2o_rhf_sto3g_grad.cu`). Worktree
  `.claude/worktrees/wf_f384f427-0b3-11` @ `wf/reg-cuda-grad-honesty`
  (`9fa88e9e`). Next: fix the `.cu` compile, resubmit one batch run.
- [~] `cuda-pw-large-cell-measure` — deferred 2026-08-09, GH200 job 1035278
  (RUNNING at wave close; earlier attempts 1035136 / 1035238 failed). Worktree
  `.claude/worktrees/wf_f384f427-0b3-13` @ `wf/cuda-pw-large-cell-measure`
  (`d574d65d`), build kept for the pending job. Next: harvest
  `.claude/cuda-pw-large-cell-1035278.{out,err}`, write the measurement note.
- [~] `cuda-pw-pbe-device` — deferred 2026-08-09, GH200 job 1035270
  (FAILED: aarch64 link pulled x86_64 `libxtb.so` — "file in wrong format" —
  while linking `test_cuda_pw_xc_pbe`). Worktree
  `.claude/worktrees/wf_f384f427-0b3-14` @ `wf/cuda-pw-pbe-device`
  (`e1147831`). Next: scrub the xtb lib path from the GH200 link line,
  resubmit.

## blocked

- `cuda-gpw-full-product` — gate: GPW CPU public-shape architecture freeze
  (`docs/developer/es-cpu-cuda-queue.md:97`). Owner: `es-cpu-cuda-opt`. The
  four footholds are green; the product is out of this loop's scope until the
  gate lifts.
- `mg-main-structural` — gate: MG collocate/integrate 2.33× residual needs
  pin-unsafe sphere windows or a task-list rewrite; micro-SIMD exhausted and
  P0 closed (`docs/developer/gpw-perf-queue.md:61,83,85`). Owner:
  `cp2k-perf-loop`.

## done

- [x] `doc-hf-cuda-truth` — 2026-08-09, merge `825eaba3`. HF roadmap CUDA rows
  rewritten to es-cpu-cuda waves 1–10 truth (device s/p ERI + forces, CDIIS,
  level shift).
- [x] `doc-notes-perf-fossils` — 2026-08-09, merge `958b4940`. Three
  `notes.md` fossils superseded in place (MG wall, OT-plateau lever, reshape
  wave count).
- [x] `doc-gpw-baseline-ot-sync` — 2026-08-09, merge `647ecbe2`.
  `gpw-perf-baseline.md` OT section synced to post-OT-match walls
  (~2.99 s / 1.20×); worst_ratio now 2.33 (MG MAIN).
- [x] `doc-dft-roadmap-consistency` — 2026-08-09, merge `814119c2`.
  `docs/roadmap/dft.md` §10 residual list, stress-case `[~]` rows, and the
  CUDA-PW line reconciled with wave-7/8 truth.
- [x] `doc-arch-cuda-gpw-truth` — 2026-08-09, merge `862c40f5`. `CLAUDE.md`
  backend table + `architecture.md` now state the truth: 4 cuda/gpw kernel
  footholds exist, no method product.
- [x] `doc-fair-twin-reference-supersede` — 2026-08-09, merge `156c5f91`.
  Four stale REFERENCE.md perf sections rewritten to case-local snapshots
  pointing at `gpw-perf-baseline.md` as the single wall source.
- [x] `pw-perf-qe-install` — 2026-08-09, merge `6d13a436`. QE offline oracle
  scaffolded at `tools/qe_oracle/` (README + provenance rules + smoke-si
  input + pin_notes with version/build line); unblocks
  `pw-perf-fair-walls-batch1`.
