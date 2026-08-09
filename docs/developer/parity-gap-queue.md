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

### P0 — harness truth (docs-only, one wave)

- [ ] `doc-hf-cuda-truth` — oracle: harness.
  `docs/roadmap/hartree-fock.md:92,104-108` still claims CUDA has no DIIS, no
  gradients, device ERI s-type only. Superseded by es-cpu-cuda waves 1–5:
  device s/p integrals + analytic s/p forces, Saunders–Hillier level shift,
  device CDIIS (`docs/developer/es-cpu-cuda-queue.md:103,113,126,137`,
  `.claude/notes/cuda-rhf-device-only.md:44-56`). Rewrite the CUDA rows.
- [ ] `doc-notes-perf-fossils` — oracle: harness.
  `.claude/notes/notes.md:9-13` ("MG MAIN ~2.1 s ≈ 1.6×") and `:30-31` ("OT
  plateaus ~60 iters — the single biggest lever") are fossils:
  `es-optimization-order.md:50-55` marks the walls superseded, and OT now
  matches CP2K at 10 iterations (`docs/developer/gpw-perf-queue.md:82-84`).
  Also `:42` says cuda-rhf-reshape is at waves 0–5; the topic note says 0–10.
  Rewrite all three in place.
- [ ] `doc-gpw-baseline-ot-sync` — oracle: harness.
  `.claude/notes/gpw-perf-baseline.md:114-125,142` still carries the
  pre-OT-match `ot-minimizer-water` section (7.55 s, ratio 3.02, worst_ratio
  3.02). Post `p0-ot-cp2k-match` + `p0-ot-match-residual` the OT wall is
  ~2.99 s / 1.20× with SCF body ≤1.0× (`gpw-perf-queue.md:36,82-84`);
  worst_ratio is 2.33 (MG MAIN). Re-measure OT wall if cheap, else carry the
  queue numbers with provenance; sync the section and the worst_ratio line.
- [ ] `doc-dft-roadmap-consistency` — oracle: harness.
  `docs/roadmap/dft.md:528-533` (§10 "pending residual" list) contradicts the
  same file's intro `:85-88` and §12 `:752-761` (wave-8 landed those rows);
  `gpw-ts-kpoint-stress` / `gpw-metallic-stress` still `[~]` though the queue
  has them `[x]` (`docs/developer/cp2k-method-queue.md:1686-1687`); `:438`
  should name `cuda::pw::PW` (LSDA, restricted, energy-only) instead of the
  bare "CUDA path remains LDA". Fix all three spots.
- [ ] `doc-arch-cuda-gpw-truth` — oracle: harness.
  `CLAUDE.md` backend table marks `gpw/` CPU-only and
  `.claude/notes/architecture.md:28,34` says 16 cuda dirs / no gpw. The tree
  has `include/atomiverse/cuda/gpw/internal/` = 4 unit-tested kernel footholds
  (Poisson FFT, Padé LDA XC, ρ collocate, V·shape integrate) with **no method
  product** — `cuda-gpw-full-product` stays blocked. Minimal truth edit in
  both files: footholds exist, method product does not.
- [ ] `doc-fair-twin-reference-supersede` — oracle: harness.
  Four fair-twin REFERENCE.md perf sections are stale by 3–8× vs the binding
  pins (`gpw-multigrid-kpoints/REFERENCE.md:121` "1.6", `gpw-uks-o2-gamma
  :64` "0.98", `ot-uks-o2:91` "7.1", `ot-minimizer-water:105` "9.0"). Rewrite
  each perf section to carry only the case-local measurement date + a pointer
  to `.claude/notes/gpw-perf-baseline.md` as the single binding wall source.
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

(empty — seeded 2026-08-09)
