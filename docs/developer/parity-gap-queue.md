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

- [ ] `reg-cuda-grad-honesty` — oracle: pyscf (existing pin) + cpu-parity.
  `regressions/scf/h2o_rhf_def2svp_grad --device=cuda` silently grades the
  host path (Def2SVP fails `device_sp_sto3g_ok`; `src/cuda/scf/rhf.cu:417-420`
  calls `cpu::scf::RHF::forces`) with no `last_force_on_device()` assertion —
  a false-positive "CUDA gradient regression". Fix: REFERENCE.md documents the
  host-fallback leg honestly, and the CUDA leg asserts the fallback flags so
  the case fails loudly if the envelope ever changes; add an STO-3G s/p
  device-path gradient leg (device forces are a delivered feature with only
  unit coverage). Validate via one GH200 batch run.
- [ ] `cuda-md-first-regression` — oracle: lammps or literature.
  CUDA MD has zero regression cases (`grep -rl cuda regressions/{md,
  forcefield,mc}` = 0) while the OpenMM-engine track is explicitly unfrozen.
  Add one: LJ argon NVE (existing CPU case's geometry), `--device=cuda`,
  energy-conservation envelope + cited pin, ≤60 s, public headers only.
  Validate via one GH200 batch run.
- [ ] `cuda-pw-large-cell-measure` — oracle: cpu-parity (wall).
  ES order step 4, sanctioned by `.claude/notes/cuda-rhf-device-only.md:
  145-147`: first large-cell `cuda::pw::PW` vs same-settings CPU wall
  measurement on GH200 (LSDA envelope; scale Si supercell until >60 s CPU).
  Output = measurement note in `.claude/notes/` with job ids + methodology;
  **no perf claim enters docs/roadmap from small cells**.
- [ ] `cuda-pw-pbe-device` — oracle: cpu-parity.
  Sanctioned next wave (`cuda-rhf-device-only.md:145-147`): PBE GGA on device
  for `cuda::pw::PW` (today throws at `src/cuda/planewave/pw_solver.cu:63`).
  CPU same-settings oracle; stated tolerances vs `tests/cuda/planewave/`
  stage-parity pattern; energy-path change re-runs the CPU force regressions.

### P2 — PW perf debt (ordered; second depends on first)

- [ ] `pw-perf-qe-install` — oracle: qe.
  22/27 `regressions/pw/*/REFERENCE.md` are `perf_exempt` solely because no
  QE/ABINIT/CP2K binary existed on the pin host (e.g. `pw-pbe-gga-si/
  REFERENCE.md:30-31`). Per the strict-loop law, "no binary" is not a green
  path: build QE `pw.x` offline on the x86_64 login env, scaffold
  `tools/qe_oracle/` (README + provenance rules mirroring `tools/cp2k_oracle/`),
  record version + build line. No ATV code changes.
- [ ] `pw-perf-fair-walls-batch1` — oracle: qe. Depends: `pw-perf-qe-install`.
  Repin the first batch of exempt PW cases with real same-settings NC-UPF
  walls (`pw-pbe-gga-si`, `pw-analytic-stress-si`, `pw-uks-kmesh-fe`,
  `bands-dos-si`): T_atv/T_qe with full provenance, REFERENCE.md perf sections
  rewritten from `perf_exempt` to measured ratios. Remaining cases become
  `pw-perf-fair-walls-batch2+` via GapCensus.

## in_progress

(empty)

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
