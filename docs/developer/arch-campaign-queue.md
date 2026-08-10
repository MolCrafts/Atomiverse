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

**Post-land 2026-08-10** (`dev` after `arch-pw-driver-paths` +
`arch-mg-main-structural` merges): residual staffable **empty** for CPU arch
paths. Open product under freeze: `cuda-gpw-full-product` (blocked → implement
under contract only).

| Surface | On `dev` |
| --- | --- |
| `GpwDft` | `cpu::Driver<gpw::Phase>` + multi-path DIIS + **OT free-loop residual** |
| `cpu::pw::PW` | `cpu::Driver<pw::Phase>` + **all product SCF paths** on stages (fixed-occ + spin + smear) |
| MG MAIN products | single-owner multi-level path (`arch-mg-main-structural`) |
| `cuda::scf::RHF` | private `RhfScfCycle`; no public free-loop `SCFSolver` |
| CUDA GPW | stage footholds; public shape **frozen** to CPU `GpwDft` |
| e2e iron law | wave-1 + wave-2 done; `ATV_TEST_HEAVY_REGEX` empty |

### P0 / P1 / P2 staffable

*(empty — path residuals closed)*

## in_progress

*(empty)*

## blocked

- [!] `cuda-gpw-full-product` | implement under freeze contract
  ([cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md));
  compose `internal/` stages → public method; GH200 + CUDA↔CPU tols
- [!] `mg-main-soft-tol` | never staff as soften tols / self-golden
- [!] `hybrid-gapw-paw-metagga` | method loops, not this campaign

## done

- [x] `arch-gpw-driver-node` | **P0** `GpwDft` → `cpu::Driver<gpw::Phase>` |
  merge: `bf5897fb`
- [x] `arch-gpw-driver-paths` | **P0** multi-path DIIS on stages | merge:
  `e34ccf92`
- [x] `arch-pw-driver-node` | **P0** `PW` → `cpu::Driver<pw::Phase>` fixed-occ |
  merge: `f016215e`
- [x] `arch-pw-driver-paths` | **P0** spin + metallic smear on same stages |
  merge: `8dcb5de0` / feat `1d0c0e13` | evidence: `test_cpu_pw_driver_node` +
  `reg_arch-pw-driver-paths` PASS post-rebase
- [x] `arch-mg-main-structural` | **P2** single-owner multi-level MAIN products |
  merge: `aef1cd3b` / feat `8ea75fa6` | evidence: `test_cpu_gpw_multigrid` +
  `reg_arch-mg-main-structural` PASS post-rebase
- [x] `arch-cuda-scf-solver-fold` | **P1** fold `SCFSolver` | merge: `3a20af0a`
- [x] `arch-cuda-gpw-public-shape-freeze` | **P1** freeze note + reg | merge:
  `0147fe02`
- [x] `arch-e2e-tests-wave1` / `arch-e2e-tests-wave2` | iron law | see
  [e2e-in-tests-campaign.md](e2e-in-tests-campaign.md)
