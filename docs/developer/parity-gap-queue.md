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

### P2 — PW perf debt

- [ ] `pw-perf-fair-walls-batch2` — oracle: qe. Depends: `tools/qe_oracle/`
  (live; proven end-to-end by batch 1, merge `2a7bd70b`).
  2026-08-09 post-batch-1 census: **18** PW `REFERENCE.md` files still carry
  `perf_exempt`. Repin the next slice with real same-settings NC-UPF walls
  (`pw-lda-si`, `pw-spin-gga`, `pw-spin-polarized-o2`, `pw-uks-fe-smearing`),
  same shape as batch 1 (T_atv/T_qe + QE inputs under
  `tools/qe_oracle/inputs/`). `pw-variable-cell-si` overlaps the in-flight
  variable-cell tree — leave it for a later batch. Remaining cases become
  `pw-perf-fair-walls-batch3+` via GapCensus.

## in_progress

- [~] `test-gpw-unit-shrink` — assigned 2026-08-09 run 2 round 1 (oracle:
  ctest). From `test-gpw-timeout-triage` (done): shrink `test_cpu_gpw_mbd`
  (34.07 s), `test_cpu_gpw_hirshfeld` (98.27 s, 4.8 GB RSS) and
  `test_cpu_gpw_stress` (13.57 s) to seconds-scale per iron law 1 (registered
  per-test TIMEOUT is 10 s), and restore load headroom in
  `test_cpu_gpw_forces` (8.2–8.4 s, ~18% margin). Size change only — keep
  every assertion. Worktree + scoped build required.
- [~] `cuda-md-first-regression` — assigned 2026-08-09 run 2 round 1 (oracle:
  lammps or literature). Resume the parked wave-2 attempt at branch
  `wf/cuda-md-first-regression` (`ea17e11a`, LJ argon NVE multi-device case;
  GH200 jobs 1035232 / 1035271 failed) — re-anchor onto current dev tip,
  diagnose the job failures, validate via one GH200 batch run. Do not restart
  from scratch.
- [~] `cuda-pw-large-cell-measure` — deferred 2026-08-09, GH200 job 1035278
  (still RUNNING at wave-3 close, ~48 min into the 1 h limit; earlier attempts
  1035136 / 1035238 failed). Worktree
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

- [x] `reg-cuda-grad-honesty` — 2026-08-09, merge `8e716f5d`. Fail-closed CUDA
  gradient honesty e2e: device s/p STO-3G analytic-gradient regression
  asserting on-device SCF+force residency, plus the documented def2-SVP
  host-fallback counterpart. GH200 job 1039997 green (ALL CUDA LEGS PASS);
  `wf/reg-cuda-grad-honesty` pruned.
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
- [x] `pw-perf-fair-walls-batch1` — 2026-08-09, merge `2a7bd70b`. Four exempt
  PW cases repinned with measured same-settings QE 7.6 NC-UPF walls
  (`pw-pbe-gga-si`, `pw-analytic-stress-si`, `pw-uks-kmesh-fe`,
  `bands-dos-si`): REFERENCE.md perf sections rewritten from `perf_exempt` to
  T_atv/T_qe ratios, QE inputs pinned under `tools/qe_oracle/inputs/`.
- [x] `test-gpw-timeout-triage` — 2026-08-09, no code change (read-only
  verdict; no branch). Quiet-shell walls with the registered env:
  `mbd` 34.07 s, `hirshfeld` 98.27 s @ 4.8 GB RSS, `stress` 13.57 s —
  genuinely heavy vs the 10 s per-test registration; `forces` 8.2–8.4 s and
  `test_mpi_pw_kpoint_n4` 2.3–2.5 s — load-flakes (original suite ran at
  load ~51 on the 48-core node). No assertion failures anywhere. Follow-up
  filed as `test-gpw-unit-shrink` (pending P0).
