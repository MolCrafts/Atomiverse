# gpw-vdw-uks — Γ collinear-UKS nonlocal vdW-DF/DF2/rVV10

**Status:** in-flight (wave 3, cp2k-method-loop)
**Branch:** `wf/gpw-vdw-uks`
**Chapter:** gpw

## Problem

`GpwDft::build` fail-fasts at `src/cpu/gpw/gpw_solver.cpp:5188` when any of
`vdw_df` / `vdw_df2` / `rvv10` is combined with UKS (`n_alpha`/`n_beta`),
because the self-consistent nonlocal potential V_nl and the post-SCF
E_c^nl fold live only in the Γ closed-shell RKS runner — accepting the flags
would silently drop E_c^nl (`e_disp` stays 0 while the run "succeeds").
This was the found-rot fail-fast the `gpw-vdw-multigrid` merge declared.

## Fix (extend gpw-vdw-scf-potential × gpw-uks-pbe)

Nonlocal vdW is a functional of the **total** density ρ = ρα + ρβ (CP2K
`qs_dispersion_nonloc`: V_nl added identically to both spin potentials).

1. `build_uks_H` (Γ UKS dual-KS builder in `run_gpw_scf_impl`): evaluate
   `nonlocal_vdw_on_grid(grid, rho_total, …, potential=true)` each SCF
   iteration; add `nl.v_r` to **both** `veff_a` and `veff_b` before
   integration; keep the outer `e_d3` in sync (`e_d3 = nl.energy`) so
   `fold_dispersion` reports the SC E_c^nl on `GpwResult::e_disp`.
   (The force side `assemble_gpw_forces` already handles UKS + V_nl.)
2. Remove the Γ-UKS fail-fast in `GpwDft::build`. The **multi-k** reject
   (`gpw_solver.cpp:5326`, k path has no nonlocal fold at all) stays —
   that is the separate `gpw-vdw-kpoints` gate.
3. Found rot, fixed in passing (same function): the UKS OT gnorm-converged
   exit did `res.e_total += e_d3` instead of `fold_dispersion(res, Pa+Pb)`,
   so a converged UKS-OT run left `e_disp = 0` (and skipped the deferred
   Hirshfeld TS/MBD fold entirely). All other UKS exits fold; make this one
   fold too.
4. Flip the unit pin `GpwSolverClass.BuildRejectsUksNonlocalVdw` →
   `BuildAcceptsGammaUksNonlocalVdw`.

## Oracle

CP2K 2026.1 (`~/work/cp2k/install/bin/cp2k.psmp`, offline only, never in
ctest): O₂ triplet, 12 Å cubic **PERIODIC XYZ**, LSD MULTIPLICITY 3,
PBE + `&NON_LOCAL TYPE DRSLL` (vdW_kernel_table.dat), CUTOFF 400 /
REL_CUTOFF 60 / NGRIDS 4, DZVP-MOLOPT-GTH + GTH-PBE-q6.
Input: `tools/cp2k_oracle/inputs/gpw-vdw-uks/o2_uks_pbe_drsll.inp`.
Pinned numbers: total energy [Ha] + `Dispersion energy` [Ha].

## Acceptance

- [x] Regression `regressions/gpw/gpw-vdw-uks/` (REFERENCE.md + reg):
  - O₂ triplet Γ-UKS PBE + DRSLL: `e_disp` vs CP2K LSD pin ≤ 1e-4 Ha
    (measured 1.3e-5), `e_total` vs pin ≤ 1e-3 Ha (measured 6.8e-5).
  - Closed-shell spin-degeneracy identity: water-dimer vdW-DF as UKS
    (nα = nβ = 8, 16 valence e⁻) vs RKS (n_occ = 8 double-occupied):
    |ΔE_total| ≤ 3e-4 eV (measured 2.6e-8 eV) and |ΔE_disp| ≤ 1e-5 Ha
    (measured 1.9e-7 Ha).
- [x] Unit gate flipped (`BuildAcceptsGammaUksNonlocalVdw`), k-path reject
  still pinned (`BuildRejectsKpointNonlocalVdw`).
- [x] Scoped build + unit gates + `reg_gpw-vdw-uks` green (~217 s total,
  family SC-vdW / UKS wall class).
- [x] Perf: ratio ~1.23 vs CP2K 159.5 s (within_10x).
