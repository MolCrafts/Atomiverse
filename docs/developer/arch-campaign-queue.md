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

**GapCensus wave 2 (2026-08-10)** on `dev` @ `c77f8145` (post wave-1 queue
rewrite; no arch land since `arch-gpw-driver-node` @ `bf5897fb`). Tree facts
reconfirmed:

| Surface | On `dev` |
| --- | --- |
| `GpwDft` | `cpu::Driver<gpw::Phase>` + `drive_gamma_rks_diis_` + `gpw/stages/*` for **Γ RKS DIIS only** |
| GPW residual free loops | `run_gpw_scf_kpoints*` (RKS/UKS DIIS), Γ UKS DIIS, all **OT** (`use_ot`) |
| `cpu::pw::PW` | plain class — **no** `pw::Phase`, no stages, free SCF in `pw_solver.cpp` |
| `cuda::scf::SCFSolver` | free-loop internal under `RHF` (`unique_ptr<SCFSolver> solver_`); **not** Driver-like fold |
| CUDA GPW | stage footholds only (`ks_chain` / MG / GGA); **no** public method product |
| e2e iron law | wave-1 + wave-2 residual multi-iter out of `tests/`; `ATV_TEST_HEAVY_REGEX` empty (`^$`) on tip |

Orphan **branch tips** (no isolation worktrees; **not** on `dev` — do not mark
done; Assign may rebase/re-staff or discard):

| Tip | Claim (unverified until merge) |
| --- | --- |
| `wf/arch-gpw-driver-paths` | multi-path DIIS on stages; OT residual |
| `wf/arch-pw-driver-node` | `PW` → `Driver<pw::Phase>` restricted fixed-occ |
| `wf/arch-cuda-scf-solver-fold` | fold free-loop under RHF private cycle |

No open arch specs under `.claude/specs/` (only unrelated
`periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`). Lever still:
`MeanFieldMethod` + `cpu/scf/stages/` + landed Γ RKS GPW stages.

### P0 — ES loop nodes (CPU)

- [ ] `arch-gpw-driver-paths` | **P0** Migrate Γ UKS + multi-k RKS/UKS **DIIS**
  onto same `gpw::Phase` stages (`ks_build → mix → diagonalize → density →
  check`); OT free-loop residual may trail as same-capability follow-on (not a
  separate method / not a new slug) | method=cpu-arch | twin:
  `regressions/gpw/*` (uks-o2-gamma, multigrid-kpoints, kpoints-si) | files:
  `gpw_solver.cpp` `run_gpw_scf_kpoints*`, `gpw/stages/`, `drive_*` | gate:
  pins green; public `run`/`forces` stable; wipe builds |
- [ ] `arch-pw-driver-node` | **P0** Reshape `cpu::pw::PW` →
  `cpu::Driver<pw::Phase>` after (or parallel once) GPW stage pattern is clear
  | method=cpu-arch | twin: `regressions/pw/*` (lda-si baseline) | files:
  `pw_solver.*`, new `pw/stages/`, `core/stage.h` `pw::Phase` | gate: public
  `run`/`forces`/`stress` stable; spin/smear may remain free-loop residual if
  restricted fixed-occ lands first |

### P1 — CUDA SCF debt + freeze product shape

- [ ] `arch-cuda-scf-solver-fold` | **P1** Fold `cuda::scf::SCFSolver` fully
  private under `cuda::scf::RHF` Driver-like shape (no dual free-loop
  architecture; kill synthetic dual path as public surface) | method=cuda-arch
  | twin: `tests/cuda/scf/test_rhf.cu` + CPU RHF oracle | files: `cuda/scf/*`
  | gate: GH200 green; device-only; no push |
- [ ] `arch-cuda-gpw-public-shape-freeze` | **P1** Document + enforce: full
  CUDA GPW method product **equals** CPU `GpwDft` public shape after
  multi-path Driver (`arch-gpw-driver-paths`) lands; compose existing stage
  footholds only | method=arch-doc | twin: n/a (harness) | files: notes +
  queue unlock condition for `cuda-gpw-full-product` |

### P2 — MG structure (after Driver paths)

- [ ] `arch-mg-main-structural` | **P2** Task-list / products-default path
  rewrite toward MAIN ~1× (explicitly **not** pin-softening) — only after
  GPW Driver paths stable | method=cpu-arch | twin: `gpw-multigrid-kpoints` |
  files: multigrid / products | gate: pins green; baseline note rewrite |

## in_progress

*(none — `wf/arch-e2e-tests-wave2` tip ready for merge gate)*

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
- [x] `arch-e2e-tests-wave2` | **P2** Residual multi-iter out of `tests/`
  (TDA/NAC/ALDA/hessian/delta-SCF/MPI kpoint + GPW vol product SCF +
  multiscale/qm_engine multi-iter); slim mixed units; empty
  `ATV_TEST_HEAVY_REGEX`; harness `reg_arch-e2e-tests-wave2` | branch:
  `wf/arch-e2e-tests-wave2` | evidence: `docs/developer/e2e-in-tests-campaign.md`
  (merge still required onto `dev`)
