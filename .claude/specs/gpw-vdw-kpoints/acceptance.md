---
spec: gpw-vdw-kpoints
created: 2026-07-31
criteria:
  - id: ac-001
    summary: k-path SCF folds V_nl and reports post-SCF E_c^nl
    type: code
    pass_when: |
      run_gpw_scf_kpoints adds nonlocal_vdw_on_grid(...).v_r to veff before
      last_veff / mg_vt_stack, and fold_dispersion_k evaluates the final
      E_c^nl on the k-summed density (e_disp folded into e_total). No change
      to the SCF convergence-energy convention (mirrors Γ).
    status: pass
  - id: ac-002
    summary: build() accepts k-RKS vdW; k-UKS vdW still rejects
    type: runtime
    pass_when: |
      test_cpu_gpw_solver: k_mesh + vdw_df builds (single-grid and
      ngrids>1/rel_cutoff); UKS + vdW (Γ and k) still throws
      std::invalid_argument. Unit suite green.
    status: pass
  - id: ac-003
    summary: Gamma-limit identity at solver tolerance
    type: runtime
    pass_when: |
      regressions/gpw/gpw-vdw-kpoints: explicit k=(0,0,0) k-path run with a
      nonlocal vdW flag matches the shipped Γ runner at identical settings on
      E_tot and E_disp to <= 1e-8 Ha (expected ~1e-10 class).
    status: pass
  - id: ac-004
    summary: CP2K MP-mesh parity on E_disp and E_tot
    type: runtime
    pass_when: |
      Same regression: MP-mesh DRSLL run vs the offline CP2K 2026.1 pin in
      REFERENCE.md — |E_disp - pin| <= ~1e-4 Ha (gpw-vdw-multigrid class,
      tightened to measurement) and E_tot within the documented family
      envelope. Pin is a number CP2K printed (input + output line recorded).
    status: pass
  - id: ac-005
    summary: Perf within 10x of CP2K or perf_exempt documented
    type: performance
    pass_when: |
      REFERENCE.md records T_atv / T_cp2k <= 10 for the parity run (or a
      documented perf_exempt rationale).
    status: pass
out_of_scope:
  - UKS nonlocal vdW (separate queue pick)
  - Analytic k-point vdW force/stress gates
  - CUDA
  - Live CP2K in ctest
---

# Acceptance — `gpw-vdw-kpoints`

**Done** means `GpwDft` with `vdw_df` / `vdw_df2` / `rvv10` runs on the
k-point closed-shell RKS path (single-grid and REL_CUTOFF multigrid), with the
self-consistent V_nl in the KS matrix and the post-SCF E_c^nl on `e_disp`,
gated by a Γ-limit identity and an offline CP2K 2026.1 KPOINTS pin.
