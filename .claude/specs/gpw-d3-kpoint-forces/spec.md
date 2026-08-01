---
slug: gpw-d3-kpoint-forces
status: acceptance_met
created: 2026-08-01
domain: gpw / dispersion forces
queue: cp2k-method-loop
depends_on: [gpw-d3-forces, gpw-d3-kpoint-stress, gpw-kpoint-forces]
---

# Spec: `gpw-d3-kpoint-forces` — multi-k GPW PBE-D3 (zero) analytic forces

## Summary

Dedicated offline CP2K pin for **multi-k** GPW analytic ionic forces with
**DFT-D3 zero damping** (two-body). Capability already lives in
`assemble_gpw_forces_kpoints` (`d3` branch); this slug certifies it with a
matched multi-k ENERGY_FORCE twin (D3 on − plain ΔF) and a dual-gate
regression, analogous to `gpw-d3-kpoint-stress` (stress) and Γ
`gpw-d3-forces`.

## Acceptance (summary)

1. Standalone `dftd3_pbe` \(E_{\mathrm{disp}}\) and \(F_{\mathrm{disp}}\) match
   offline CP2K multi-k pins.
2. Product multi-k SCF + force assembly: `e_disp` pin, `C_occ_k` payload,
   wiring \(F(\mathrm{d3})-F(\mathrm{plain})=F_{\mathrm{standalone}}\).
3. Fair wall \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\) on the same D3
   multi-k ENERGY_FORCE twin (real CP2K run; not perf_exempt).

## Oracle

- **oracle:** cp2k
- **twin:** `tools/cp2k_oracle/inputs/gpw-d3-kpoint-forces/`
- **binary:** CP2K 2026.1 `cp2k.psmp` git:5e54ba2
