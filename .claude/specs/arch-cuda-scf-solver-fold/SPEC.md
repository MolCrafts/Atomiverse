# arch-cuda-scf-solver-fold

## Intent

Fold free-loop `cuda::scf::SCFSolver` dual architecture into the concrete
method: RHF owns a private RHF SCF cycle; synthetic tests own a private
synthetic cycle. Loop steps follow `scf::Phase` order
(`fock → mix → diagonalize → density → check`), matching MeanFieldMethod /
GpwDft stage order. **No generic SCFSolver type.**

## Lever

MeanFieldMethod `drive_scf_` + `scf/stages/` + GpwDft `drive_gamma_rks_diis_`.
CUDA has **no nested DriverNode** (Plan slots are Component-only), so RHF
**stays** `cuda::Component` + `cuda::HF` for Evaluate/MD nesting. Full
`cuda::Driver<scf::Phase>` inheritance is residual until nested CUDA drivers
exist. This slug kills the dual free-loop bag, not invent nested-driver infra.

## Shape change

| Before | After |
| --- | --- |
| `SCFSolver` free loop bag used by RHF + SCF | `internal::RhfScfCycle` (RHF only) + `internal::SyntheticScfCycle` (test SCF only) |
| Dual architecture under RHF | RHF private `drive_scf_` via RhfScfCycle in `scf::Phase` order |
| Tests construct `SCFSolver` | Public `RHF{basis}.run` / occupation via RHF |

## Public surface (frozen)

- `cuda::scf::RHF{basis, charge, tol, max_iter, level_shift}`
- `run(cpu::Frame&)` / `forces(cpu::Frame&)` / Component lifecycle
- Nested `set_evaluator(RHF{…})` / Evaluate `on(evaluate).add_ref(rhf)`
- No `*Config`, no `make_*` factories, no public SCFSolver

## Acceptance

- [x] No `cuda::scf::SCFSolver` type
- [x] RHF still Component+HF; public run/forces stable
- [x] Private cycle drives scf::Phase order (fock/mix/diag/density/check)
- [x] Unit: structural + occupation via RHF; scoped `test_cuda_scf*` green on GH200
- [x] Regression `architecture/arch-cuda-scf-solver-fold` literature pin green on GH200
- [x] `acceptance_met=true`

## Out

CUDA nested DriverNode; full `RHF : cuda::Driver<scf::Phase>`; new ES methods;
pin softening; cuda-gpw-full-product.
