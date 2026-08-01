# gpw-d3bj-kpoint-stress

**Status:** acceptance_met = true (2026-08-01)

## Goal

Dedicated multi-k product pin for GPW PBE-D3(BJ) analytic cell stress
(Monkhorst–Pack + geometry-only D3(BJ) in `energy_fixed_P_k`).

## Acceptance (all met)

1. Offline CP2K 2026.1 NUMERICAL-stress difference pin on water dimer +
   MONKHORST-PACK 1×1×2 (D3BJ − plain) + printed Dispersion energy.
2. Product `GPWEvaluator` `k_mesh={1,1,2}` `d3bj=true` analytic stress
   matches pins (`reg_gpw-d3bj-kpoint-stress`).
3. Dual gate analytic vs multi-SCF FD (xx e2e; unit twin all axes).
4. Fair wall vs real CP2K D3BJ twin (1 MPI × 8 OMP): ratio **0.382** ≤ 10.
5. Unit: `GpwStress.D3BjMultiKAnalyticMatchesFd` + `D3BjMultiKChangesDiagonal`.

## Deliverables

- `regressions/gpw/gpw-d3bj-kpoint-stress/`
- `tools/cp2k_oracle/inputs/gpw-d3bj-kpoint-stress/`
- unit tests in `tests/cpu/gpw/test_cpu_gpw_stress.cpp`
