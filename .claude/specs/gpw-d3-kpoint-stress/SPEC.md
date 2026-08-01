# Spec — gpw-d3-kpoint-stress

## Goal

Dedicated offline pin that the **multi-k** GPW analytic fixed-\(P_k\) cell
stress Lagrangian folds **DFT-D3 zero damping** (geometry-only) into
\(\sigma\). Capability path already productized in `gpw_stress.cpp` after
`gpw-d3-stress` + `gpw-kpoint-stress`; this slug closes the multi-k pin gap.

## Acceptance

1. **E_disp** from multi-k SCF equals CP2K `Dispersion energy` (tol 1e-8 Ha).
2. **Δσ = σ(d3) − σ(plain)** on the multi-k analytic product matches the
   CP2K NUMERICAL stress-difference twin (tol ~2e-9 Ha/Bohr³ diagonal).
3. **Internal dual gate:** analytic fixed-\(P_k\) vs multi-SCF strain-FD on
   the same multi-k `d3=true` method (tol 2e-5 Ha/Bohr³).
4. Multi-k payload present (`n_kpts ≥ 1`, `C_occ_k` non-empty); orthorhombic
   off-diagonals ~0; Frame `kStress` unit conversion.
5. **Fair wall:** ATV product (SCF + analytic stress) / CP2K NUMERICAL D3 twin
   ≤ 10.0 (1 MPI × fixed OMP on the same twin job class).

## Twin

- Geometry: periodic water dimer 12 Å cube (same as `gpw-d3-stress`).
- Method: GPW PBE + DFT-D3 zero, ATM off, `R_CUTOFF 20.0` Å.
- Mesh: Monkhorst–Pack **1×1×2**.
- CP2K: `STRESS_TENSOR NUMERICAL`, DIAGONALIZATION + DIRECT_P_MIXING.
- Inputs: `tools/cp2k_oracle/inputs/gpw-d3-kpoint-stress/`.

## Non-goals

D3(BJ), ATM/C9, multi-k UKS, D4/TS/XDM/MBD k-stress (sibling slugs).
