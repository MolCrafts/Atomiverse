# Spec — `gpw-optb88-forces`

## Goal

Dedicated offline CP2K ENERGY_FORCE pin for **analytic ionic forces** on
Γ-point GPW **optB88-vdW** (OptB88 exchange + PW92 + DRSLL nonlocal), proving
the product SC \(V_{\mathrm{nl}}\) force path with `XcKind::OptB88VdW`.

## Capability (product already)

- Semi-local: `XcKind::OptB88VdW` → `xc_grid_optb88_vdw` (optB88-x + PW92-c).
- Nonlocal: `GpwDft::vdw_df` / `GPWEvaluator::vdw_df` → SC DRSLL
  \(V_{\mathrm{nl}}\) every SCF + \(E_c^{\mathrm{nl}}\) on `e_disp`.
- Forces: `assemble_gpw_forces` adds \(V_{\mathrm{nl}}\) into `veff` (same as
  `gpw-vdw-scf-potential`); energy product pins exist under
  `gpw-optb88-{vdw,uks,kpoints,uks-kpoints}` — **no ENERGY_FORCE pin**.

## Acceptance

| Gate | Criterion |
| ---- | --------- |
| A1 | Offline CP2K 2026.1 ENERGY_FORCE twin (OptB88 + DRSLL) actually run |
| A2 | \(\lvert E_{\mathrm{disp}}-E_{\mathrm{disp}}^{\mathrm{CP2K}}\rvert \le 10^{-4}\) Ha |
| A3 | \(\lvert E_{\mathrm{tot}}-E_{\mathrm{tot}}^{\mathrm{CP2K}}\rvert \le 2.5\times 10^{-2}\) Ha (OptB88 native vs libxc multigrid residual class) |
| A4 | \(\max\lvert F_{\mathrm{ATV}}-F_{\mathrm{CP2K}}\rvert \le 10^{-2}\) Ha/Bohr (grid residual class; CP2K \(\lvert\sum F\rvert\sim 10^{-2}\)) |
| A5 | Dual: OptB88+DRSLL vs OptB88-plain \(\lvert\Delta E\rvert > 10^{-4}\) Ha **or** \(\max\lvert\Delta F\rvert > 10^{-4}\) Ha/Bohr |
| A6 | Fair wall same twin job, 1 MPI × 8 OMP: \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) |

## Out of scope

- Multi-k / UKS OptB88 forces (separate queue rows).
- Stress with OptB88+DRSLL.
- Geometry-only pairwise dispersion (D3/D4/TS).

## Twin

`tools/cp2k_oracle/inputs/gpw-optb88-forces/water_dimer_optb88_vdw_force.inp`
