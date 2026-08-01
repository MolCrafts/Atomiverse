# Acceptance — gpw-d4-atm-kpoint-stress

- [x] Product multi-k analytic stress with `d4 + d4_atm` converges and publishes
      Frame `kStress` [eV/Å³] + `last_stress_ha_bohr3()`.
- [x] Hard E_disp / strained-cell pins vs offline dftd4 3.7.0 (s9=1 and 2b).
- [x] Hard σ_ATM pin vs offline CP2K multi-k NUMERICAL Δσ (ATM − 2b).
- [x] Dual gate analytic vs multi-SCF strain-FD (xx; unit twin all axes).
- [x] Product wiring: σ(ATM) − σ(2b) equals standalone σ_ATM (s9 fold).
- [x] Fair wall \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\) vs real CP2K ATM twin
      (measured **0.077**; `perf_exempt = false`).
- [x] Regression `regressions/gpw/gpw-d4-atm-kpoint-stress` green offline.
- [x] Unit twin `GpwStress.D4AtmMultiKAnalyticMatchesFd` green.

`acceptance_met = true`
