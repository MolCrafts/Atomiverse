# Spec — `gpw-hirshfeld-uks-kpoints`

## Goal

Product-pin **multi-k collinear UKS** Hirshfeld volume ratios for TS/MBD on
the existing `fold_dispersion_k_uks` path (post-SCF
\(\rho=\sum_k w_k(\rho_k^\alpha+\rho_k^\beta)\) → \(V/V_{\mathrm{free}}\) →
scaled MBD@rsSCS / TS). Capability code already ships; this slug freezes the
dual gate and offline provenance.

## Scope

- In: `GpwDft` / `GPWEvaluator` with `n_alpha`/`n_beta`, `k_mesh`, `mbd`|`ts`,
  `hirshfeld=true`; `GpwResult::e_disp`, `volume_ratios`.
- Out: density-chain MBD forces/stress, spin-resolved Hirshfeld, CP2K TS/MBD
  (stock QS has none).

## Acceptance

1. Unit gate: multi-k UKS + mbd + hirshfeld reports volumes + attractive
   `e_disp` (or zero for single atom).
2. Regression dual gate:
   - A1 closed-shell multi-k UKS ≡ multi-k RKS E_disp (≤ 5e-6 Ha).
   - A2 absolute E_disp pin on O₂ multi-k UKS Hirshfeld-MBD (≤ 1e-8 Ha).
3. Offline oracle notes under `tools/cp2k_oracle/inputs/gpw-hirshfeld-uks-kpoints/`
   (libMBD / literature — no stock CP2K TS/MBD).
4. Wall: `perf_exempt` (same policy as `gpw-hirshfeld-uks` /
   `gpw-hirshfeld-kpoints` — E_disp pin; SCF dominates).

## Deps

`gpw-hirshfeld-uks` ✓, `gpw-hirshfeld-kpoints` ✓.
