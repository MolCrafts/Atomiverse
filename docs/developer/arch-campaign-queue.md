# Architecture campaign queue

Worklist for workflow `arch-campaign`. **Architecture and refactor first** —
not pin-safe micro-opt, not new electronic methods, not classical MD CUDA.

Binding order (CLAUDE.md → Electronic structure optimization order):

1. **CPU architecture** (Driver/stage shape, kill forked SCF copies)
2. CPU performance (only after structure)
3. CUDA ES method slices (same-settings CPU oracle)
4. Large-system CUDA perf

## pending

Staff only `wf/arch-<slug>` off current `dev`. GapCensus every wave may deepen
rows; foothold ≠ closed capability. **Unmerged tip ≠ done** — only green merge
onto `dev` closes a slug.

**GapCensus wave 8 (2026-08-10)** on `dev` @ `9271cae3` (wave-7 census
chore). Re-diff CLAUDE iron laws + ES order, notes, arch/e2e/es queues, tree
scan `cpu/{gpw,planewave,scf}` + `cuda/scf` + `tests/` —
**no new staffable residual slug** (`new_slugs_added=0`). Tree facts on
`dev` (product source of truth; unmerged tips are claims only):

| Surface | On `dev` |
| --- | --- |
| `GpwDft` | `cpu::Driver<gpw::Phase>` + path-local Ops + `drive_path_diis_` / `drive_stage_loop_` for **multi-path DIIS** (Γ RKS/UKS + multi-k RKS/UKS) |
| GPW residual free loops | all **OT** (`use_ot`) — Γ + multi-k; same-capability follow-on, **not** a new slug |
| `cpu::pw::PW` | `cpu::Driver<pw::Phase>` + stages for **restricted fixed-occ**; collinear spin + metallic smearing free-loop residual |
| `cuda::scf::RHF` | Component+HF + private `internal::RhfScfCycle` (`scf::Phase` order); **no** public/free-loop `SCFSolver` type |
| Nested CUDA Driver | residual — full `cuda::Driver<scf::Phase>` needs nested CUDA DriverNode (**not** a separate staffable product slug yet) |
| CUDA GPW | stage footholds only (`ks_chain` / MG / GGA); **no** public method product; freeze note **not** on `dev` |
| e2e iron law | wave-1 + wave-2 **merged**; `ATV_TEST_HEAVY_REGEX` empty (`^$`); optional wave-3 only if unique pin gaps appear (none staffed this wave) |

**Unmerged tips** (no isolation worktree; **not** on `dev` — keep `[~]`; do
not mark done). All three rebased onto `9271cae3` and are **ahead=1
behind=0** vs current `dev` (product claims advanced since wave-7 tip SHAs):

| Tip | Claim (unverified until merge) |
| --- | --- |
| `wf/arch-pw-driver-paths` | all product SCF paths on same `pw::Phase` stages via path-local StepFns + shared `drive_stage_loop_` (`e04ba781`) |
| `wf/arch-cuda-gpw-public-shape-freeze` | freeze doc + harness: CUDA GPW product shape equals CPU `GpwDft` (`a625c232`) |
| `wf/arch-mg-main-structural` | single-owner multi-level MAIN product path (`3d9a64ba`) |

Open arch specs on `dev`: **none** (specs live only on tips under
`.claude/specs/arch-{pw-driver-paths,cuda-gpw-public-shape-freeze,mg-main-structural}/`).
Lever: `MeanFieldMethod` + `cpu/scf/stages/` + landed multi-path GPW DIIS
stages + landed PW fixed-occ stages (+ tip multi-path PW Ops / freeze note /
MG structural).

### P0 — ES loop residual paths (CPU)

(none staffable unassigned — `arch-pw-driver-paths` remains `[~]`; tip
advanced to `e04ba781`)

### P1 — freeze product shape (CUDA GPW)

(none staffable unassigned — `arch-cuda-gpw-public-shape-freeze` remains
`[~]`; tip `a625c232`)

### P2 — MG structure (after Driver paths)

(none staffable unassigned — `arch-mg-main-structural` remains `[~]`; tip
`3d9a64ba`; prefer freeze-first among freeze/mg if contention at merge)

## in_progress

**Assign wave 7** (parent `dev` @ `9271cae3`) — **3** incomplete resumes
(`[~]`); **new_assigned=0** (pending unheld=0; no open arch specs on `dev`;
`new_slugs_added=0`). Tips rebased onto `9271cae3` and advanced product
claims (**ahead=1 behind=0**). Unmerged tip ≠ done.
**incomplete_remaining=3**. Pending unheld: 0. Blocked product: 1.
**gaps_remaining=4**.

- [~] `arch-pw-driver-paths` | **P0** Migrate collinear spin + metallic
  smearing free loops onto same `pw::Phase` stages
  (`veff → diagonalize → density → mix → check`) as restricted fixed-occ;
  keep public `build`/`run`/`forces`/`stress` stable | method=cpu-arch |
  twin: `regressions/pw/*` (pw-spin-gga, pw-uks-kmesh-fe, pw-uks-fe-smearing,
  pw-smearing-al, metallic DOS/bands) + `test_cpu_pw_driver_node` +
  `reg_arch-pw-driver-paths` (tip) | files: `pw_solver.cpp` free-loop arms,
  `planewave/stages/`, `drive_*` | gate: pins green; public surface stable;
  wipe builds | **tip** `wf/arch-pw-driver-paths` @ `e04ba781` (ahead=1
  behind=0 vs `dev` @ `9271cae3`; claim all product SCF paths on stages) |
  spec (tip): `.claude/specs/arch-pw-driver-paths/SPEC.md` |
- [~] `arch-cuda-gpw-public-shape-freeze` | **P1** Document + enforce: full
  CUDA GPW method product **equals** CPU `GpwDft` public shape after
  multi-path DIIS Driver (`arch-gpw-driver-paths` **merged**); compose existing
  stage footholds only; OT free-loop residual is documented on CPU and does
  not block the freeze doc | method=arch-doc | twin: n/a (harness) + tip
  `reg_arch-cuda-gpw-public-shape-freeze` | files: notes + queue unlock
  condition for `cuda-gpw-full-product` | **tip**
  `wf/arch-cuda-gpw-public-shape-freeze` @ `a625c232` (ahead=1 behind=0 vs
  `dev` @ `9271cae3`) | gate: freeze doc lands; unlock path for full product
  stated | spec (tip): `.claude/specs/arch-cuda-gpw-public-shape-freeze/` |
- [~] `arch-mg-main-structural` | **P2** Task-list / products-default path
  rewrite toward MAIN ~1× (explicitly **not** pin-softening) — only after
  GPW Driver paths stable | method=cpu-arch | twin: `gpw-multigrid-kpoints` +
  tip `reg_arch-mg-main-structural` | files: multigrid / products | gate:
  pins green; baseline note rewrite | **tip** `wf/arch-mg-main-structural`
  @ `3d9a64ba` (ahead=1 behind=0 vs `dev` @ `9271cae3`; prefer after freeze
  if both race the same surface) | spec (tip):
  `.claude/specs/arch-mg-main-structural/` |

## blocked

- [!] `cuda-gpw-full-product` | blocked until public-shape freeze green —
  multi-path DIIS Driver is on `dev`; CUDA stage units ≠ method product;
  freeze still required before full CUDA GPW product
- [!] `mg-main-soft-tol` | never staff as soften tols / self-golden
- [!] `hybrid-gapw-paw-metagga` | method loops, not this campaign

## done

- [x] `arch-gpw-driver-node` | **P0** `GpwDft` → `cpu::Driver<gpw::Phase>`;
  Γ RKS DIIS via stages; public `run`/`build` stable | merge: `bf5897fb` |
  evidence: `test_cpu_gpw_driver_node` + `reg_gpw-water-pbe-mt`; residual paths
  staffed as `arch-gpw-driver-paths`
- [x] `arch-e2e-tests-wave1` | **P0** Delete 78 heavy e2e sources duplicated by
  regressions; ungate kernel unit regex tokens | merge: wave-1 | evidence:
  `docs/developer/e2e-in-tests-campaign.md`
- [x] `arch-cuda-scf-solver-fold` | **P1** Fold free-loop `SCFSolver` under
  `cuda::scf::RHF` private `internal::RhfScfCycle` (`scf::Phase` order); no
  public SCFSolver; RHF stays Component+HF (nested CUDA Driver residual) |
  merge: `3a20af0a` / impl `772497c1` | evidence:
  `regressions/architecture/arch-cuda-scf-solver-fold` + `test_cuda_scf*`
  (GH200); spec closed
- [x] `arch-e2e-tests-wave2` | **P2** Residual multi-iter out of `tests/`
  (TDA/NAC/ALDA/hessian/delta-SCF/MPI kpoint + GPW vol product SCF +
  multiscale/qm_engine multi-iter); slim mixed units; empty
  `ATV_TEST_HEAVY_REGEX`; harness `reg_arch-e2e-tests-wave2` | merge:
  `42b8982c` / impl `49d03074` + polish `c6b623be` | evidence:
  `docs/developer/e2e-in-tests-campaign.md` +
  `regressions/architecture/arch-e2e-tests-wave2`; spec closed
- [x] `arch-pw-driver-node` | **P0** `PW` → `cpu::Driver<pw::Phase>`;
  restricted fixed-occ via stages
  (`veff → diagonalize → density → mix → check`) + `PwConverged`; public
  `build`/`run`/`forces`/`stress` stable; collinear spin + metallic smear
  free-loop residual | merge: `f016215e` / impl `1bad527a` | evidence:
  `test_cpu_pw_driver_node` + `reg_arch-pw-driver-node` + `regressions/pw/*`;
  spec closed; residual paths staffed as `arch-pw-driver-paths` (GapCensus
  wave-4)
- [x] `arch-gpw-driver-paths` | **P0** Migrate Γ UKS + multi-k RKS/UKS **DIIS**
  onto same `gpw::Phase` stages via path-local Ops + `drive_path_diis_` /
  `drive_stage_loop_`; OT free-loop residual remains (same-capability
  follow-on, not a new slug); public `run`/`build`/`forces` stable | merge:
  `e34ccf92` / impl `42454014` + gate docs `2ba03049` | evidence:
  `test_cpu_gpw_driver_node` + `reg_arch-gpw-driver-paths` +
  `reg_gpw-uks-o2-gamma` + `reg_gpw-kpoints-si`; spec closed
