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
rows; foothold ≠ closed capability.

**GapCensus wave 1 (2026-08-10)** on `dev` @ `19280211` (post
`arch-gpw-driver-node` merge `bf5897fb`). Tree facts: `GpwDft :
cpu::Driver<gpw::Phase>` + `drive_gamma_rks_diis_` + `gpw/stages/*` for **Γ
RKS DIIS only**; `run_gpw_scf_kpoints*` / UKS / OT still free loops; `cpu::pw::PW`
plain class (no `pw::Phase` / no stages); `cuda::scf::SCFSolver` dual under
`RHF`; CUDA GPW = stage footholds only (`ks_chain` / MG, no public product);
e2e wave-1 done, residual multi-iter under `tests/` + HEAVY_REGEX safety net.
No open `wf/arch-*` worktrees; no open arch specs. Lever:
`MeanFieldMethod` + `cpu/scf/stages/` + landed Γ RKS GPW stages.

### P0 — ES loop nodes (CPU)

- [ ] `arch-gpw-driver-paths` | **P0** Migrate Γ UKS + multi-k RKS/UKS DIIS
  onto same `gpw::Phase` stages; OT free-loop residual may trail as
  same-capability follow-on (not a separate method) | method=cpu-arch |
  twin: `regressions/gpw/*` (uks-o2-gamma, multigrid-kpoints, kpoints-si) |
  files: `gpw_solver.cpp` `run_gpw_scf_kpoints*`, `gpw/stages/` | gate: pins
  green; public `run`/`forces` stable; wipe builds |
- [ ] `arch-pw-driver-node` | **P0** Reshape `cpu::pw::PW` →
  `cpu::Driver<pw::Phase>` after (or parallel once) GPW stage pattern is
  clear | method=cpu-arch | twin: `regressions/pw/*` | files: `pw_solver.*`,
  new `pw/stages/`, `core/stage.h` `pw::Phase` | gate: public `run`/`forces`
  stable |

### P1 — CUDA SCF debt + freeze product shape

- [ ] `arch-cuda-scf-solver-fold` | **P1** Fold `cuda::scf::SCFSolver` fully
  private under `cuda::scf::RHF` Driver-like shape (no dual architecture) |
  method=cuda-arch | twin: `tests/cuda/scf/test_rhf.cu` + CPU RHF oracle |
  files: `cuda/scf/*` | gate: GH200 green; device-only |
- [ ] `arch-cuda-gpw-public-shape-freeze` | **P1** Document + enforce: full
  CUDA GPW method product **equals** CPU `GpwDft` public shape after
  multi-path Driver (`arch-gpw-driver-paths`) lands; compose existing stage
  footholds only | method=arch-doc | twin: n/a (harness) | files: notes +
  queue unlock condition for `cuda-gpw-full-product` |

### P2 — test iron law residual + MG structure

- [ ] `arch-e2e-tests-wave2` | **P2** Residual audit of `tests/` for
  multi-iter solves after wave-1 (e.g. `test_soscf` RHF.run loops,
  `test_full_grid_alda`, MPI PW/GPW multi-iter, residual HEAVY_REGEX safety
  net); migrate unique only | method=harness | twin: ctest default gate
  seconds-scale | files: `tests/**` |
- [ ] `arch-mg-main-structural` | **P2** Task-list / products-default path
  rewrite toward MAIN ~1× (explicitly **not** pin-softening) — only after
  GPW Driver paths stable | method=cpu-arch | twin: `gpw-multigrid-kpoints` |
  files: multigrid / products | gate: pins green; baseline note rewrite |

## in_progress

*(empty — GapCensus only; Assign staffs ≤4 from pending)*

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
