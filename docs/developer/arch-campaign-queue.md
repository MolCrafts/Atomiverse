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

### P0 — ES loop nodes (CPU)

- [ ] `arch-gpw-driver-node` | **P0** Reshape `cpu::gpw::GpwDft` into
  `cpu::Driver<gpw::Phase>` (or equivalent typed phase enum) with stage
  components; public surface stays members + `run`/`forces`/`build`; no
  collint numerical churn | method=cpu-arch | twin: existing `regressions/gpw/*`
  pins green; unit workspace/ks_grid unchanged | files:
  `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp`,
  stages under `gpw/stages/` if extracted | gate: no public API break; scoped
  unit + reg green; wipe builds |
- [ ] `arch-pw-driver-node` | **P0** Same reshape for `cpu::pw::PW` after or
  parallel to GPW once phase pattern lands | method=cpu-arch | twin:
  `regressions/pw/*` | files: `pw_solver.*`, PW stages | gate: public
  `run`/`forces` stable |

### P1 — CUDA SCF debt + freeze product shape

- [ ] `arch-cuda-scf-solver-fold` | **P1** Fold `cuda::scf::SCFSolver` fully
  private under `cuda::scf::RHF` Driver-like shape (no dual architecture) |
  method=cuda-arch | twin: `tests/cuda/scf/test_rhf.cu` + CPU RHF oracle |
  files: `cuda/scf/*` | gate: GH200 green; device-only |
- [ ] `arch-cuda-gpw-public-shape-freeze` | **P1** Document + enforce:
  full CUDA GPW method product **equals** CPU `GpwDft` public shape after
  Driver node lands; compose existing stage footholds only | method=arch-doc |
  twin: n/a (harness) | files: notes + queue unlock condition for
  `cuda-gpw-full-product` |

### P2 — test iron law residual + MG structure

- [ ] `arch-e2e-tests-wave2` | **P2** Residual audit of `tests/` for multi-iter
  solves after wave-1 delete; migrate unique only | method=harness | twin:
  ctest default gate seconds-scale | files: `tests/**` |
- [ ] `arch-mg-main-structural` | **P2** Task-list / products-default path
  rewrite toward MAIN ~1× (explicitly **not** pin-softening) — only after
  GPW Driver node stable | method=cpu-arch | twin: `gpw-multigrid-kpoints` |
  files: multigrid / products | gate: pins green; baseline note rewrite |

## in_progress

*(empty)*

## blocked

- [!] `cuda-gpw-full-product` | blocked until `arch-gpw-driver-node` +
  public-shape freeze green
- [!] `mg-main-soft-tol` | never staff as soften tols / self-golden
- [!] `hybrid-gapw-paw-metagga` | method loops, not this campaign

## done

- [x] `arch-e2e-tests-wave1` | **P0** Delete 78 heavy e2e sources duplicated by
  regressions; ungate kernel unit regex tokens | merge: *(this commit)* |
  evidence: `docs/developer/e2e-in-tests-campaign.md`
