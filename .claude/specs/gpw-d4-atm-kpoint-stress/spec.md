# Spec — gpw-d4-atm-kpoint-stress

## Capability

Multi-k GPW (PBE) analytic fixed-\(P_k\) cell stress with DFT-D4 three-body ATM
(\(s_9 = 1\)) folded into the strained-cell Lagrangian rebuild.

## Public surface

- `cpu::gpw::GPWEvaluator` / `GpwDft`: `k_mesh = {1,1,2}`, `d4 = true`,
  `d4_atm = true`, `compute_stress = true`, `stress_use_fd = false`.
- Implementation: `src/cpu/gpw/internal/gpw_stress.cpp` multi-k path
  (`energy_fixed_P_k`) rebuilds `dftd4_energy(..., s9=1)`.

## Oracle

- Offline CP2K 2026.1 + libdftd4 NUMERICAL stress twin pair (ATM vs 2b).
- Offline dftd4 3.7.0 strained-cell energy pins (shared with `gpw-d4-atm-stress`).
- Fair wall: same ATM multi-k NUMERICAL job, 1 MPI × 8 OMP.

## Out of scope

Molecular post-HF, classical MW MD, CUDA ports, live oracle in ctest.
