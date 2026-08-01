# Spec — periodic-scs-ump2-si

## Intent

Γ-point **RI-GPW SCS-UMP2** on periodic sc-Si₂: Grimme scales
(`mp2_os_scale=1.2`, `mp2_ss_scale=1/3`) applied to the product **UKS**
UMP2 path (`GpwDft` with `n_alpha`/`n_beta`, `mp2_ri=true`). Same public
member names as molecular `UHF` and closed-shell GPW SCS-MP2.

## Capability (already on tree)

- `GpwDft::mp2` UKS Γ path writes unscaled `e_corr_os` / `e_corr_ss` and
  `e_corr = mp2_os_scale * OS + mp2_ss_scale * SS`
  (`src/cpu/gpw/mp2.cpp`).
- Deps: `periodic-ump2-si` ✓, `periodic-scs-mp2-si` ✓.

## Acceptance

1. Offline CP2K 2026.1 twin `tools/cp2k_oracle/inputs/periodic-scs-ump2-si.inp`
   (UKS + `SCALE_S`/`SCALE_T` + `METHOD MP2_GPW`) **actually run** for pin
   provenance + fair wall.
2. Regression `reg_periodic-scs-ump2-si`:
   - SCF UKS PBE converges; β orbitals present.
   - Plain UMP2 then SCS-UMP2 on the same orbitals.
   - Algebra: `e_corr ≡ 1.2·OS + (1/3)·SS`; unscaled OS/SS invariant under
     scale members; `e_total ≡ e_scf + e_corr`; `ri == true`.
   - Absolute `e_corr` dual-gate vs closed-shell SCS product band when the
     UKS solution stays near closed-shell (nα=nβ); document CP2K UKS
     absolute digits (may spin-contaminate on this deck).
3. Fair wall: \(T_{\mathrm{atv}}(\mathrm{SCF+SCS})/T_{\mathrm{cp2k}} \le 10\).
4. ≤ 60 s single process; no live CP2K in ctest.

## Out of scope

Multi-k SCS-UMP2; molecular PySCF SCS; forces/stress.
