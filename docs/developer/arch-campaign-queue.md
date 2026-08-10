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

**GapCensus harness wave 5 (2026-08-10)** on `dev` @ `26fb8328` (post
GapCensus wave-4 chore; PW merge still `f016215e` / impl `1bad527a`; no new
arch merge onto `dev` since wave-4). Re-diff CLAUDE iron laws + ES order,
notes, arch/e2e/es queues, tree scan `cpu/{gpw,planewave,scf}` + `cuda/scf` +
`tests/` — **no new staffable residual slug** (`new_slugs_added=0`). Tree
facts on `dev` (product source of truth; tips are claims only):

| Surface | On `dev` |
| --- | --- |
| `GpwDft` | `cpu::Driver<gpw::Phase>` + `drive_gamma_rks_diis_` + `gpw/stages/*` for **Γ RKS DIIS only** |
| GPW residual free loops | `run_gpw_scf_kpoints*` (RKS/UKS DIIS), Γ UKS DIIS, all **OT** (`use_ot`) |
| `cpu::pw::PW` | `cpu::Driver<pw::Phase>` + stages for **restricted fixed-occ**; collinear spin + metallic smearing free-loop residual |
| `cuda::scf::RHF` | Component+HF + private `internal::RhfScfCycle` (`scf::Phase` order); **no** public/free-loop `SCFSolver` type |
| Nested CUDA Driver | residual — full `cuda::Driver<scf::Phase>` needs nested CUDA DriverNode (**not** a separate staffable product slug yet) |
| CUDA GPW | stage footholds only (`ks_chain` / MG / GGA); **no** public method product |
| e2e iron law | wave-1 + wave-2 **merged**; `ATV_TEST_HEAVY_REGEX` empty (`^$`); optional wave-3 only if unique pin gaps appear (none staffed this census) |

**Unmerged tips** (no isolation worktree; **not** on `dev` — keep `[~]`; do
not mark done). Both tips base `26fb8328` + 1 feat; parent moved to
`1c2d74d6` (GapCensus wave-5 chore) ⇒ **ahead=1 behind=1** each — rebase
onto current `dev` before gate / Worktrees:

| Tip | Claim (unverified until merge) |
| --- | --- |
| `wf/arch-gpw-driver-paths` | multi-path **DIIS** on stages (`drive_path_diis_` / `drive_gamma_uks_diis_`); **OT free-loop residual** (`06bc94fb`; ahead=1 behind=1 vs `dev` @ `1c2d74d6`) |
| `wf/arch-pw-driver-paths` | collinear + metallic smear (+ collinear×smear) on same `pw::Phase` stages via path-local StepFns (`98e98ed5`; ahead=1 behind=1 vs `dev` @ `1c2d74d6`) |

Open arch specs live **only on the tips** (not on `dev` tree
`.claude/specs/` — still only unrelated `periodic-rhf-gamma` /
`rsh-eri-cam-b3lyp`). Spec paths on tips:
`.claude/specs/arch-gpw-driver-paths/SPEC.md`,
`.claude/specs/arch-pw-driver-paths/SPEC.md` (tip claims
`acceptance_met=true`; **unmerged tip ≠ done**). Lever still:
`MeanFieldMethod` + `cpu/scf/stages/` + landed Γ RKS GPW stages + landed PW
fixed-occ stages (+ tip multi-path Ops).

### P0 — ES loop residual paths (CPU)

(none staffable unassigned — both path residuals remain `[~]` from Assign
wave 5; tips unmerged; rebase then Worktrees / merge gate)

### P1 — freeze product shape (CUDA GPW)

- [ ] `arch-cuda-gpw-public-shape-freeze` | **P1** Document + enforce: full
  CUDA GPW method product **equals** CPU `GpwDft` public shape after
  multi-path Driver (`arch-gpw-driver-paths`) lands; compose existing stage
  footholds only | method=arch-doc | twin: n/a (harness) | files: notes +
  queue unlock condition for `cuda-gpw-full-product` | **hold** until
  `arch-gpw-driver-paths` merges |

### P2 — MG structure (after Driver paths)

- [ ] `arch-mg-main-structural` | **P2** Task-list / products-default path
  rewrite toward MAIN ~1× (explicitly **not** pin-softening) — only after
  GPW Driver paths stable | method=cpu-arch | twin: `gpw-multigrid-kpoints` |
  files: multigrid / products | gate: pins green; baseline note rewrite |
  **hold** until `arch-gpw-driver-paths` green on `dev` |

## in_progress

**Assign wave 5** (parent `dev` @ `1c2d74d6`) — **2** incomplete resumes
(`[~]`); **new_assigned=0**. Skipped held P1 shape-freeze + P2 mg-main
(dep on GPW paths merge). Unmerged tip ≠ done. Staff ≤4; only 2 staffable.

- [~] `arch-gpw-driver-paths` | **P0** Migrate Γ UKS + multi-k RKS/UKS **DIIS**
  onto same `gpw::Phase` stages (`ks_build → mix → diagonalize → density →
  check`); OT free-loop residual may trail as same-capability follow-on (not a
  separate method / not a new slug) | method=cpu-arch | twin:
  `regressions/gpw/*` (uks-o2-gamma, multigrid-kpoints, kpoints-si) +
  `reg_arch-gpw-driver-paths` (tip) | files: `gpw_solver.cpp`
  `run_gpw_scf_kpoints*`, `gpw/stages/`, `drive_*` | gate: pins green; public
  `run`/`forces` stable; wipe builds | **tip** `wf/arch-gpw-driver-paths` @
  `06bc94fb` (ahead=1 behind=1 vs `dev`; **rebase onto `1c2d74d6`** before
  gate; claim multi-path DIIS; OT residual) | spec (tip):
  `.claude/specs/arch-gpw-driver-paths/SPEC.md` |
- [~] `arch-pw-driver-paths` | **P0** Migrate collinear spin + metallic
  smearing free loops onto same `pw::Phase` stages
  (`veff → diagonalize → density → mix → check`) as restricted fixed-occ;
  keep public `build`/`run`/`forces`/`stress` stable | method=cpu-arch |
  twin: `regressions/pw/*` (pw-spin-gga, pw-uks-kmesh-fe, pw-uks-fe-smearing,
  pw-smearing-al, metallic DOS/bands) + `test_cpu_pw_driver_node` +
  `reg_arch-pw-driver-paths` (tip) | files: `pw_solver.cpp` free-loop arms,
  `planewave/stages/`, `drive_*` | gate: pins green; public surface stable;
  wipe builds | **tip** `wf/arch-pw-driver-paths` @ `98e98ed5` (ahead=1
  behind=1 vs `dev`; **rebase onto `1c2d74d6`** before gate; claim all
  product SCF paths on stages) | spec (tip):
  `.claude/specs/arch-pw-driver-paths/SPEC.md` |

## blocked

- [!] `cuda-gpw-full-product` | blocked until `arch-gpw-driver-paths` (full
  multi-path Driver) + public-shape freeze green — Γ RKS Driver foothold alone
  is not full product shape freeze; CUDA stage units ≠ method product
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
