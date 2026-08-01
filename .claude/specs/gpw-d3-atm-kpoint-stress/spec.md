# Spec: gpw-d3-atm-kpoint-stress

## Goal

Dedicated offline pin that the **multi-k** fixed-\(P_k\) analytic cell-stress
Lagrangian folds geometry-only **DFT-D3 zero damping + three-body ATM (C9)**
into \(\sigma\), matching a CP2K 2026.1 NUMERICAL stress-difference twin on a
matched GPW PBE GTH/MOLOPT system with a Monkhorst–Pack k mesh.

## Status of algorithm path

**Product already.** `cpu::gpw::internal::energy_fixed_P_k` rebuilds
`dftd3_pbe(..., calculate_c9 = cfg.d3_atm)` at every strained cell
(`src/cpu/gpw/internal/gpw_stress.cpp`). Public path:
`GPWEvaluator{d3, d3_atm, d3_rcut_bohr, k_mesh, compute_stress}`.

This slug is **pin-first**: no new fold code; close the unpinned multi-k
C9 stress residual left after `gpw-d3-atm-stress` (Γ) and `gpw-kpoint-stress`
(electronic multi-k).

## System

Periodic water dimer, 6 Å cubic cell, DZVP-MOLOPT-SR-GTH / GTH-PBE, PBE,
R_CUTOFF 10.0 Å (doubled → `d3_rcut_bohr`), electronic mesh **1×1×2**,
same geometry as `gpw-d3-atm-stress`.

## Acceptance

1. Offline CP2K 2026.1 twin pair (C9 T/F) with `&KPOINTS MONKHORST-PACK 1 1 2`
   and `STRESS_TENSOR NUMERICAL`; pins in REFERENCE.md with binary+version+
   input+output line.
2. Regression `regressions/gpw/gpw-d3-atm-kpoint-stress/`:
   - E_disp (C9 and 2b) vs CP2K (geometry-only; must match Γ pins).
   - \(\sigma_{\mathrm{ATM}} = \sigma(\mathrm{C9})-\sigma(2b)\) vs CP2K Δ
     (use CP2K `DX` strain step for the standalone FD of E_ATM).
   - Product multi-k analytic stress (`k_mesh={1,1,2}`, `d3+d3_atm`) vs
     multi-SCF `gpw_stress_fd` dual gate on same method.
   - Multi-k payload present (`C_occ_k` / `n_kpts`).
3. Fair wall: ATV product analytic multi-k stress vs same C9 CP2K twin job
   (1 MPI × fixed OMP); \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\).
4. ≤ 60 s wall for the regression product path.

## Out of scope

D3(BJ) multi-k, D4 multi-k, TS/XDM multi-k (sibling slugs); multi-k UKS
dispersion stress; virial-path ATM (energy rebuild only).
