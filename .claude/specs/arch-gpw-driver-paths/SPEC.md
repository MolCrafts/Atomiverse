# arch-gpw-driver-paths

## Intent

Migrate residual **DIIS** self-consistency free loops on `GpwDft` (Γ UKS,
multi-k RKS, multi-k UKS) onto the same `cpu::Driver<gpw::Phase>` stage
skeleton landed by `arch-gpw-driver-node` (Γ RKS DIIS). OT free loops stay
free (different control flow; not soft-migrated this wave).

## Lever

**Architecture (ES order step 1):** path-local stage `Op` callbacks + shared
`drive_stage_loop_` (`setup_pass → step_once × N → poll_criteria`). Physics
builders stay path-local; the Driver owns iteration and Criterion judgment.
Public `run` / `build` / members / `forces` stay stable. No collint churn.

## Pattern (research)

`MeanFieldMethod` + `cpu/scf/stages/` + landed `drive_gamma_rks_diis_` /
`gpw/stages/*`. Stages become path-local `std::function<void()>` launchers
bound at `ks_build → mix → diagonalize → density → check`; residual gate
still sets `skip_update` in the ks_build Op; `GpwConverged` only judges.

## Shape change (named)

| Before | After |
| --- | --- |
| Γ RKS DIIS: typed RealMatrix stages | Same phases; Ops capture RealMatrix state (bit-identical gate) |
| Γ UKS DIIS: free `for (iter)` | Ops + `drive_stage_loop_` |
| multi-k RKS/UKS DIIS: free `for (iter)` | Ops + `StageDrive` callback into `GpwDft` private drive |
| OT (Γ / multi-k) | free loop residual (follow-on, not a separate method) |

## Public surface (frozen)

- Members + `build(atoms)` + `run(atoms, …)` + `last_result()` + post-SCF APIs
- No `*Config`, no `make_*` / `build_*` factories
- Frame I/O stays on `GPWEvaluator`

## Acceptance

- [x] Stages bind `gpw::Phase` slots; unit tests green (no full SCF in tests/)
- [x] Γ UKS DIIS uses `step_once` + `GpwConverged` (no free residual break)
- [x] multi-k RKS + multi-k UKS DIIS use the same Driver loop
- [x] OT remains free loop (documented residual)
- [x] Twin regs green: `gpw-uks-o2-gamma`, `gpw-kpoints-si` (and arch reg)
- [x] Public API unchanged; wipe builds
- [x] `acceptance_met=true`

## Out

OT stage migration; collint micro-opt; soft tols; CUDA GPW product; MG MAIN rewrite
