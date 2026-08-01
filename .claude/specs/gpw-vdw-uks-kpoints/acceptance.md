# Acceptance — gpw-vdw-uks-kpoints

- [x] `GpwDft::build` accepts multi-k UKS + `vdw_df` / `vdw_df2` / `rvv10`
      (single-grid and REL_CUTOFF multigrid).
- [x] Unit: `BuildAcceptsKpointUksNonlocalVdw`,
      `BuildAcceptsKpointUksMultigridNonlocalVdw` (test_cpu_gpw_solver PASS).
- [x] Regression `reg_gpw-vdw-uks-kpoints` green (2026-08-01 OMP=8):
  - G1 Γ-limit |ΔE_tot| 3e-13 Ha, |ΔE_disp| 5e-15 Ha
  - G2 |E_disp − CP2K| 4.0e-5 Ha; |E_tot − CP2K| 7.93e-3 Ha
  - G3 closed-shell multi-k UKS ≡ RKS |ΔE| 5e-8 eV; |ΔE_disp| 1.9e-7 Ha
- [x] Offline CP2K twin under `tools/cp2k_oracle/inputs/gpw-vdw-uks-kpoints/`
      with REFERENCE.md binary + version + input + output line + fair wall.
- [x] \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.35\) on the G2 twin
      (ATV 21.2 s / CP2K 60.2 s, OMP=8).
