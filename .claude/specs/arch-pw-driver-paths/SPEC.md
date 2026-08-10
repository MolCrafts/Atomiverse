# arch-pw-driver-paths

## Intent

Migrate residual plane-wave SCF free loops (collinear fixed-occupation,
restricted metallic smearing, collinear + metallic smearing) onto the same
`cpu::Driver<pw::Phase>` stage skeleton landed by `arch-pw-driver-node`
(restricted fixed-occupation). Public `build` / `run` / `forces` / `stress`
stay stable.

## Lever

**Architecture (ES order step 1):** path-local stage `StepFn` callbacks +
shared `drive_stage_loop_` (`setup_pass → step_once × N → poll_criteria`).
Physics builders stay path-local; the Driver owns iteration and Criterion
judgment (`PwConverged`). Density-mix order (ρ-space, after density rebuild),
not H-space DIIS. Pattern: `MeanFieldMethod` + `cpu/scf/stages/` + landed
`drive_restricted_fixed_occ_` / `planewave/stages/*`.

## Shape change (named before code)

| Before | After |
| --- | --- |
| Restricted fixed-occ: stages + `drive_restricted_fixed_occ_` | Same phases; rename shared driver to `drive_stage_loop_` |
| Collinear fixed-occ: free `for (iter)` | StepFns + `drive_stage_loop_` |
| Restricted metallic smear: free `for (iter)` | StepFns + `drive_stage_loop_` |
| Collinear + metallic smear: free `for (iter)` | StepFns + `drive_stage_loop_` |

Phase map (all product paths):
`veff → diagonalize → density → mix → check`.

Path-local stage roles:
- **veff** — Poisson (+ XC / spin XC) from current ρ
- **diagonalize** — Davidson per (spin,)k; stash eigenpairs when occupations need the full spectrum
- **density** — occupations (smear/fixed) + real-space accumulate / extract
- **mix** — Pulay or linear mix, residual gate → `loop_converged_`, energy bookkeeping
- **check** — `PwConverged` judges residual flag only

## Public surface (frozen)

- Members + `build(frame)` + `run(frame)` + `last_result()` + `pw_forces_*` / `pw_stress*`
- No `*Config`, no `make_*` / `build_*` factories
- Frame I/O; workflow entry still `PWEvaluator` or direct method

## Acceptance

- [x] Stages bind `pw::Phase` slots; unit `test_cpu_pw_driver_node` green (no full SCF in tests/)
- [x] Collinear fixed-occ, restricted smear, collinear+smear use `step_once` + `PwConverged` (no free residual `break`)
- [x] Arch reg `reg_arch-pw-driver-paths` shape gate green
- [x] Twin product regs green: `pw-smearing-al`, `pw-spin-polarized-o2`, `arch-pw-driver-node`
- [x] Public API unchanged; wipe builds after gate
- [x] `acceptance_met=true`

## Out

CUDA PW product; collint micro-opt; soft tols / self-golden; new electronic methods
