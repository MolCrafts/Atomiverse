# arch-gpw-driver-node

## Intent

Reshape `cpu::gpw::GpwDft` into a real **Driver loop node**
(`cpu::Driver<gpw::Phase>`) with typed stage components for the KS SCF
iteration, matching the MeanFieldMethod pattern (`fock → mix → diagonalize →
density → check`).

## Lever

**Architecture (ES order step 1):** ownership of the self-consistency loop moves
from free `for (iter)` bodies inside `run_gpw_scf_impl` onto `Driver` stages +
a Criterion. No collint numerical churn; public `run` / `build` / members stay.

## Pattern (research)

`MeanFieldMethod : cpu::Driver<scf::Phase>` wires heap stages by reference in
`ensure_driver_built_`, then `setup_pass → step_once × N → poll_criteria →
teardown`. Algorithms never declare convergence; `ScfConverged` measures and
`verdict()` judges.

GPW reuses that skeleton with domain stages under `cpu/gpw/stages/` and
`gpw::Phase` (kind = `PhaseKind::SCF`).

## Public surface (frozen)

- Members + `build(atoms)` + `run(atoms, …)` + `last_result()` + post-SCF APIs
- No `*Config`, no `make_*` / `build_*` factories
- Frame I/O stays on `GPWEvaluator`; GpwDft remains the method object

## Stages

| Phase | Component | Work |
| --- | --- | --- |
| `ks_build` | `KsBuildStage` | `build_H(P) → H, e_total`; DIIS error from H,P,S,X |
| `mix` | `MixStage` | Pulay DIIS on H (or linear-mix bookkeeping) |
| `diagonalize` | `DiagonalizeStage` | eigenproblem (+ level shift) |
| `density` | `DensityStage` | `P ← dens(C)` |
| `check` | `GpwConverged` | Criterion: `‖ΔE‖ < tol` ∧ `max‖err‖ < err_tol` |

## Scope this slug

1. `gpw::Phase` + `PhaseTraits` + Driver inheritance
2. Stage components + Γ **RKS DIIS** (and linear-mix) path driven by Driver
3. Unit: inheritance + stage bindings only (no full SCF under `tests/`)
4. Existing `regressions/gpw/*` green (offline pins)

### Residual (deeper pending if needed)

- Γ UKS / multi-k RKS/UKS DIIS through the same stages
- OT paths (different control flow; not soft-migrated)

## Acceptance

- [x] `GpwDft` is `cpu::Driver<gpw::Phase>`
- [x] Γ RKS DIIS loop uses `step_once` + `GpwConverged` (no free `break` on residual)
- [x] Public API unchanged; no collint diffs
- [x] Scoped unit green; at least one GPW regression green
- [x] `acceptance_met=true`

## Out

collint micro-opt; soft tols; full CUDA GPW product; MG MAIN rewrite
