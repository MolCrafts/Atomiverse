---
slug: gpw-vdw-kpoints
status: approved
created: 2026-07-31
revised: 2026-07-31
domain: gpw / nonlocal vdW / k-points
queue: cp2k-method-loop (wave 3, shift)
deferred: false
depends_on: [gpw-vdw-multigrid, gpw-multigrid-kpoints]
unblocks: []
grilled: false
---

# Spec: `gpw-vdw-kpoints` — nonlocal vdW-DF/DF2/rVV10 on the k-point path

## Problem

Self-consistent nonlocal vdW (DRSLL / LMKLL / rVV10) lives only in the Γ
closed-shell RKS runner: `build_H_and_energy` folds `V_nl = δE_c^nl/δρ` into
`veff` each SCF iteration and the post-SCF `E_c^nl` on the converged density is
reported via `e_disp`. The k-point branch rejects the three flags fail-fast at
`GpwDft::build` (`gpw_solver.cpp` k-path validation, the reject introduced by
`gpw-multigrid-kpoints` when the silent drop was found). CP2K supports
`&VDW_POTENTIAL POTENTIAL_TYPE NON_LOCAL` together with `&KPOINTS`.

## Change (minimal, mirrors Γ exactly)

All inside `src/cpu/gpw/gpw_solver.cpp` (`run_gpw_scf_kpoints` — the multi-k
closed-shell RKS runner) plus the `GpwDft::build` gate:

1. **SCF potential**: in `build_k_energy`, after
   `veff = v_hartree + v_xc + v_locsr` on the finest grid, when
   `vdw_df || vdw_df2 || rvv10` evaluate
   `nonlocal_vdw_on_grid(grid, rho, …, potential=true)` and add `nl.v_r` to
   `veff` **before** `last_veff` is captured and before the multigrid
   restriction (`mg_vt_stack`) — identical placement to the Γ runner, so the
   integrated KS matrix (single-grid `integrate_potential_k` and multigrid
   `V_t` stack alike) stays the exact gradient of the collocated
   `E_H + E_xc + E_c^nl`. `rho` is already the k-summed total density on the
   finest grid on both k paths. Like Γ, `E_c^nl` is **not** added to the SCF
   convergence energy — it is reported through `e_disp`.
2. **Post-SCF energy**: in `fold_dispersion_k`, add the vdW branch: recompute
   the final k-summed density via `collocate_rho_k(P_k)` and evaluate
   `nonlocal_vdw_on_grid(…, potential=false)`; set `out.e_disp` and fold into
   `out.e_total`. One lambda covers every RKS-k return site (DIIS, OT, OT+U,
   smearing, fixed-density/CPMD).
3. **Gate**: delete the k-path `reject("nonlocal vdW …")` in `GpwDft::build`.
   UKS + vdW stays rejected by the earlier `is_uks()` branch (Γ and k alike;
   a separate queue pick owns UKS vdW). Update the comment.
4. **Unit test**: flip `GpwSolverClass.BuildRejectsKpointNonlocalVdw` to
   accept; add a k-UKS vdW reject pin and a k+multigrid+vdW accept pin
   (build-time validation only, no SCF in `tests/`).

Out of scope: UKS vdW (separate pick), analytic k-point vdW forces/stress
gates (the k force assembly already folds V_nl into its grid-term `veff`;
gating those is a follow-up), CUDA.

## Oracle

CP2K 2026.1 (`cp2k.psmp`, local Arrhenius install — same binary as the
`gpw-vdw-multigrid` pin), offline only via `tools/cp2k_oracle`. Twin input =
`gpw-vdw-multigrid` water-dimer DRSLL input + `&KPOINTS SCHEME MONKHORST-PACK`
mesh. Pin `ENERGY| Total` and `Dispersion energy` [Ha].

## Regression

`regressions/gpw/gpw-vdw-kpoints/` (out-of-library consumer, ≤ 60 s):

- **G1 Γ-limit identity**: explicit `k_points = {(0,0,0)}, w=1` k-path run vs
  the shipped Γ runner at identical cheap settings, with a nonlocal-vdW flag
  on — `|ΔE_tot|` and `|ΔE_disp|` at solver tolerance (~1e-10 Ha class).
- **G2 CP2K parity**: MP-mesh run vs the offline pin — `E_disp` at the
  `gpw-vdw-multigrid` precedent class (measured, tol ~1e-4 Ha), `E_tot`
  within the documented Γ-dimer twin-family systematic envelope.
- Perf: `T_atv/T_oracle ≤ 10` or documented `perf_exempt`.
