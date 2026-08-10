# arch-pw-driver-node

## Intent

Reshape `cpu::pw::PW` into a real **Driver loop node**
(`cpu::Driver<pw::Phase>`) with typed stage components for the plane-wave
KS SCF iteration, matching the MeanFieldMethod / GpwDft pattern.

## Lever

**Architecture (ES order step 1):** ownership of the self-consistency loop
moves from free `for (iter)` bodies inside `PW::run` onto `Driver` stages +
a Criterion. No collint numerical churn; public members + `build(frame)` /
`run(frame)` / free `pw_forces_*` stay stable.

## Pattern (research)

`MeanFieldMethod : cpu::Driver<scf::Phase>` and
`GpwDft : cpu::Driver<gpw::Phase>` wire heap stages by reference in
`ensure_driver_built_`, then `setup_pass → step_once × N → poll_criteria →
teardown`. Algorithms never declare convergence; a Criterion measures and
`verdict()` judges.

PW reuses that skeleton with domain stages under `cpu/planewave/stages/` and
`pw::Phase` (kind = `PhaseKind::SCF`).

**Physics note:** PW is **density-mixing** (Pulay on ρ), not H-space DIIS.
Phase order is therefore
`veff → diagonalize → density → mix → check`
(potential from ρ → Davidson → accumulate ρ → Pulay mix → residual), not
the Fock/H DIIS order `build → mix → diagonalize → density → check`.

## Public surface (frozen)

- Members + `build(frame)` + `run(frame)` + `last_result()`
- Free functions `pw_forces_analytic` / `pw_forces_fd` / `pw_stress*`
- `PWEvaluator` still owns the method; Frame I/O unchanged
- No `*Config`, no `make_*` / `build_*` factories

## Stages

| Phase | Component | Work (restricted fixed-occ path) |
| --- | --- | --- |
| `veff` | `VeffStage` | Poisson + XC + Veff from current ρ |
| `diagonalize` | `DiagonalizeStage` | per-k Davidson + density accumulate |
| `density` | `DensityStage` | FFT ρ(G) extract (pre-mix bookkeeping) |
| `mix` | `MixStage` | Pulay density mix via `DensityBuilder::finish` |
| `check` | `PwConverged` | Criterion: ‖Δρ‖₂ < tol after warmup |

## Scope this slug

1. `pw::Phase` + `PhaseTraits` + Driver inheritance
2. Stage components + **restricted fixed-occupation** path driven by Driver
3. Unit: inheritance + stage bindings only (no full SCF under `tests/`)
4. Architecture twin `regressions/architecture/arch-pw-driver-node` (offline
   ABINIT-class Si LDA band) + existing `regressions/pw/pw-lda-si` green

### Residual (deeper pending if needed)

- Collinear spin-polarized free-loop path
- Metallic smearing (restricted + UKS) free-loop paths
- Share internal `run_scf_fixed_occ` with the same stages

## Acceptance

- [x] `PW` is `cpu::Driver<pw::Phase>`
- [x] Restricted fixed-occ loop uses `step_once` + `PwConverged`
- [x] Public API unchanged; no collint diffs required beyond architecture
- [x] Scoped unit green; `reg_arch-pw-driver-node` green
- [x] `acceptance_met=true`

## Out

collint micro-opt; soft tols; CUDA PW product; new electronic methods
