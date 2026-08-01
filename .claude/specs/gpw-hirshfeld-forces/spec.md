---
slug: gpw-hirshfeld-forces
status: done
created: 2026-08-01
domain: gpw / dispersion
queue: cp2k-method-loop
depends_on: [gpw-mbd-hirshfeld, gpw-ts-forces, gpw-mbd-forces]
---

# Spec: `gpw-hirshfeld-forces` — Hirshfeld-scaled TS/MBD analytic forces

## Summary

Close the rot where **energy** folds post-SCF Hirshfeld volume ratios into
TS/MBD but **forces and stress** still called free-atom kernels. Product path:
fixed-\(v\) analytic forces/stress with \(v_A = V_A/V_A^{\mathrm{free}}\) from
the SCF density (density-chain \(\partial v/\partial R\) out of scope).

## Acceptance

1. `assemble_gpw_forces` / k-path and `gpw_stress` fixed-\(P\) pass
   `scf.volume_ratios` (or recompute from ρ) into `ts_pbe` / `mbd_rsscs_pbe`
   when `GpwDft::hirshfeld`.
2. Unit: fixed-\(v\) MBD/TS force dual-gate vs central FD; H₂ product wiring.
3. Regression `reg_gpw-hirshfeld-forces`: energy pin, force FD dual, product
   wiring, stress fold move; wall ≤ 60 s.
4. Oracle: literature/libMBD (CP2K has no stock TS/MBD) — `perf_exempt`.

## Out of scope

Density-response force chain \(\partial v/\partial R\); multi-k product force
pin (covered by free-atom `gpw-mbd-kpoint-forces` + energy Hirshfeld k-point
slugs).
