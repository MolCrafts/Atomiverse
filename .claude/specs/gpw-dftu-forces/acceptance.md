# Acceptance — gpw-dftu-forces

- [x] Unit dual gate: Γ-UKS + Dudarev +U analytic forces match multi-SCF FD
      on H₂ HS toy (U−J = 2 eV) within 5e-5 Ha/Bohr (max\|dF\|=6.9e-6).
- [x] Regression `reg_gpw-dftu-forces`:
  - A2 full-6N FD dual gate ≤ 5e-5 Ha/Bohr (6.9e-6)
  - A3 \|ΣF\| ≤ 1e-6; U activation floor
  - B \|E − E_CP2K\| ≤ 5e-2 Ha (1.18e-2; single grid vs multigrid+OT)
  - B transverse force ~ 0 (axial NiO)
- [x] Offline CP2K pin under `tools/cp2k_oracle/inputs/gpw-dftu-forces/`
      with binary + version + key lines in REFERENCE.md
- [x] Fair wall ratio T_atv / T_cp2k ≤ 10 on the NiO ENERGY_FORCE twin
      (32.88 / 13.24 = **2.483**)
- [x] `acceptance_met = true` — dual gate + CP2K energy pin + within_10x
