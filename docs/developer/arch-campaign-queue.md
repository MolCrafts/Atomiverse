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

**GapCensus wave 3 (2026-08-10)** on `dev` @ `350644ca` (post harness wave-2
chore after merges of `arch-cuda-scf-solver-fold` + `arch-e2e-tests-wave2`).
Re-diff of CLAUDE iron laws + ES order, notes, arch/e2e/es queues, and tree
scan of `cpu/{gpw,planewave,scf}` + `cuda/scf` + `tests/` — **no new staffable
arch slugs**. Tree facts reconfirmed:

| Surface | On `dev` |
| --- | --- |
| `GpwDft` | `cpu::Driver<gpw::Phase>` + `drive_gamma_rks_diis_` + `gpw/stages/*` for **Γ RKS DIIS only** |
| GPW residual free loops | `run_gpw_scf_kpoints*` (RKS/UKS DIIS), Γ UKS DIIS, all **OT** (`use_ot`) |
| `cpu::pw::PW` | plain class — **no** `pw::Phase`, no stages; free SCF in `internal/scf_loop` + `pw_solver` |
| `cuda::scf::RHF` | Component+HF + private `internal::RhfScfCycle` (`scf::Phase` order); **no** public/free-loop `SCFSolver` type |
| Nested CUDA Driver | residual — full `cuda::Driver<scf::Phase>` needs nested CUDA DriverNode (**not** a separate staffable product slug yet) |
| CUDA GPW | stage footholds only (`ks_chain` / MG / GGA); **no** public method product |
| e2e iron law | wave-1 + wave-2 **merged**; `ATV_TEST_HEAVY_REGEX` empty (`^$`); optional wave-3 only if unique pin gaps appear (none staffed this census) |

Orphan **branch tips** (no isolation worktrees; **not** on `dev` — do not mark
done; Assign may rebase/re-staff or discard):

| Tip | Claim (unverified until merge) |
| --- | --- |
| `wf/arch-gpw-driver-paths` | multi-path DIIS on stages; OT residual (`6553a036`, base `9c421cc0`) |
| `wf/arch-pw-driver-node` | `PW` → `Driver<pw::Phase>` restricted fixed-occ (`42842e25`, base `9c421cc0`) |

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

### P1 — freeze product shape (CUDA GPW)

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

*(none)*

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
