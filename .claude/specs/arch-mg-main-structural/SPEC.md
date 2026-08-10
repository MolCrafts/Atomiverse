# arch-mg-main-structural

## Intent

Rewrite the **multigrid MAIN product path** so multi-level collocate /
prolong / restrict / integrate has **one owner** — not four forked lambdas
inside Γ RKS, Γ UKS, multi-k RKS, multi-k UKS. Explicitly **not** pin
softening, not collint micro-opt, not soft tols / self-golden.

Toward MAIN ~1× vs CP2K is the campaign goal; this wave lands the structural
single-source for that path (same extract order as `ks_grid_build_*`).

## Lever

**Architecture (ES order step 1):** extract at the second (and 4th) real call
site — free functions on the multigrid product surface, parallel to
`ks_grid_build_rks/uks` for the fine-grid ρ→V_eff piece. Public `GpwDft`
`run` / `build` / `forces` / members stay stable. No pin-safe collint churn.

## Pattern (research)

`MeanFieldMethod` + `cpu/scf/stages/` + landed multi-path GPW DIIS
(`drive_path_diis_` / path-local Ops) + `ks_grid_build_*` extract. Physics
kernels (`collocate_products*`, `integrate_products*`, spectral transfer)
stay; the **composition** that every SCF runner re-implemented inline moves
to:

| Helper | Role |
| --- | --- |
| `multigrid_collocate_rho` | Γ P → fine ρ (levels + prolong) |
| `multigrid_integrate_veff` | fine V_eff → V_ks (restrict + levels) |
| `multigrid_collocate_rho_cells` | Bloch D(t) → fine ρ |
| `multigrid_integrate_veff_cells` | fine V_eff → V_t stack |

## Shape change (named)

| Before | After |
| --- | --- |
| Multi-level product loops forked ×4 in `gpw_solver.cpp` | Shared free functions in `multigrid.{h,cpp}` |
| MAIN path ownership implicit in each runner | One algorithm; runners call helpers |
| Fine-grid KS already shared (`ks_grid_build_*`) | Product multi-level path matches that extract style |

## Public surface (frozen)

- `GpwDft` members (`ngrids`, `rel_cutoff`, …) + `build` / `run` / `forces`
- No `*Config`, no `make_*` / `build_*` factories
- Frame I/O stays on `GPWEvaluator`

## Acceptance

- [x] Multi-level MAIN path helpers exist and unit-test the forked-loop identity + adjointness
- [x] All four SCF runners (Γ RKS/UKS, multi-k RKS/UKS) call the helpers
- [x] Twin reg `gpw-multigrid-kpoints` green (offline CP2K pin; tolerances unchanged)
      — MAIN E = −15.8024626200 Ha, \|Δ\| = 4.621e-05 Ha vs pin (tol 1e-3)
- [x] Arch reg `reg_arch-mg-main-structural` green (shape)
- [x] Public API unchanged
- [x] Baseline / architecture note rewritten for single-owner MAIN path
- [x] Unit `test_cpu_gpw_multigrid` green (MainPath suite)
- [x] wipe builds after gate (post-commit)

## Gate evidence

Re-gate after rebase onto `dev` @ Assign wave-8 (`f790f485`), 2026-08-10:

| Gate | Result |
| --- | --- |
| `test_cpu_gpw_multigrid` | PASS (0.85 s) |
| `run_reg_arch-mg-main-structural` | PASS |
| `run_reg_gpw-multigrid-kpoints` | PASS — \|dE\|=4.621e-05 Ha vs offline CP2K pin; MAIN E = −15.8024626200 Ha |

Lever: extract multi-level product composition (second+ call site) parallel to
`ks_grid_build_*` / MeanFieldMethod stage extract. Wall residual ~2.33× remains
on the shared MAIN path (not pin-softened; further cut is kernel residual).

## Out

Pin-safe collint micro-opts; soft tols / self-golden; products-default-k A/B
flip (gpw-perf queue after MAIN nearer parity); CUDA GPW full product; OT
stage migration.
