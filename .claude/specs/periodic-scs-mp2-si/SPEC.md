# Spec: periodic-scs-mp2-si

## Goal

Pin **Γ-point RI-GPW SCS-MP2** on artificial sc-Si₂ with Grimme scales
`mp2_os_scale = 1.2`, `mp2_ss_scale = 1/3` against a matched offline CP2K
2026.1 `MP2_GPW` twin (`SCALE_S` / `SCALE_T`).

## Capability

Already product-shaped on `GpwDft` (deps: `periodic-ri-mp2-si`):

- `mp2_ri = true` RI-GPW path
- `mp2_os_scale` / `mp2_ss_scale` applied as
  `e_corr = c_OS * e_OS + c_SS * e_SS` with unscaled channels preserved

This slug is the **dedicated scale pin**, not a new algorithm.

## Acceptance

1. Offline CP2K twin input + run (binary + version + input + output line).
2. Regression `regressions/gpw/periodic-scs-mp2-si` dual gate:
   - algebraic: `e_corr == 1.2*OS + (1/3)*SS`; unscaled OS/SS invariant
   - pin: `|E_corr − E_CP2K| ≤ 1e-3 Ha`
3. Fair wall: `T_atv(SCF+SCS) / T_cp2k(twin) ≤ 10` from a real CP2K run.
4. Unit: SCS linear composition documented in `test_cpu_gpw_mp2`.

## Non-goals

Multi-k SCS pin; SOS-MP2 product pin; molecular post-HF (pyscf loop).

## Status

- acceptance_met: **true** (2026-08-01)
- CP2K pin: E_corr = −0.297941206394 Ha (git:5e54ba2)
- ATV: E_corr = −0.2982932565 Ha (|Δ| 3.5e-4)
- ratio_atv_over_oracle: 0.424 (7.55 / 17.81)
