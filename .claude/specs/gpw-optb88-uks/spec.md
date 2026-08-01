# gpw-optb88-uks — Γ collinear-UKS optB88-vdW (OptB88 + DRSLL)

**Status:** in-flight (wave 2, cp2k-method-loop)
**Branch:** `wf/gpw-optb88-uks`
**Chapter:** gpw

## Problem

`XcKind::OptB88VdW` (optB88 exchange + PW92 LDA-c) and the UKS spin-grid
kernel `xc_grid_optb88_vdw_spin` are product-shaped on Γ GPW, and Γ-UKS
already folds self-consistent nonlocal DRSLL (`vdw_df`) into both spin
potentials (`gpw-vdw-uks`). What is missing is the **matched product pin**
for the combo: Γ-UKS × OptB88VdW × DRSLL against a real CP2K twin. Only the
RKS pin `reg_gpw-optb88-vdw` (water dimer) exists today — same UKS-GGA expand
pattern as revPBE / RPBE / BLYP.

## Fix (pin product path; no new algorithm surface)

Capability first is already landed:

1. Semi-local spin path: `build_uks_H` routes `XcKind::OptB88VdW` →
   `xc_grid_optb88_vdw_spin` (allow-list with PBE / PBEsol / RevPBE / RPBE /
   BLYP).
2. Nonlocal SC path: `vdw_df` on Γ-UKS adds \(V_{\mathrm{nl}}(\rho_\alpha+\rho_\beta)\)
   identically to both spin potentials; \(E_c^{\mathrm{nl}}\) lands on
   `GpwResult::e_disp` (from `gpw-vdw-uks` / `gpw-vdw-scf-potential`).
3. Public API: `GPWEvaluator` / `GpwDft` — set `xc = XcKind::OptB88VdW`,
   `n_alpha`/`n_beta`, `vdw_df = true`. No Config bags, no factories.

This slug **pins** that product combination:

- Offline CP2K 2026.1 twin: O₂ triplet LSD, LIBXC `XC_GGA_X_OPTB88_VDW` +
  `PW92` + `NON_LOCAL TYPE DRSLL`, PERIODIC 12 Å cell, 400 Ry.
- Regression dual gates: absolute \(E_{\mathrm{tot}}\) + \(E_{\mathrm{disp}}\)
  vs CP2K; OptB88+DRSLL vs PBE+DRSLL activation floor; optional closed-shell
  UKS≡RKS identity on water dimer with OptB88+DRSLL.
- Unit: spin-grid collapse / polarized OptB88 ≠ PBE (mirror BLYP spin test).

## Oracle

CP2K **2026.1** (`cp2k.psmp`, source revision **git:5e54ba2**,
`/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install`, offline
only — never in ctest).

Input: `tools/cp2k_oracle/inputs/gpw-optb88-uks/o2_uks_optb88_vdw.inp`.

Pinned: total energy [Ha], Dispersion energy [Ha], fair wall (1 MPI × OMP=8
on the same twin job).

## Acceptance

- [x] Twin `.inp` + offline CP2K run recorded (binary + version + input +
  output lines + wall).
- [x] `regressions/gpw/gpw-optb88-uks/` (REFERENCE.md + reg) green:
  - O₂ triplet Γ-UKS OptB88VdW + DRSLL: \(|E_{\mathrm{tot}}-E_{\mathrm{CP2K}}|\)
    ≤ 2.5e−2 Ha (measured 1.90e−2); \(|E_{\mathrm{disp}}-E_{\mathrm{CP2K}}|\)
    ≤ 1e−4 Ha (measured 1.5e−5).
  - Dual gate: \(|E_{\mathrm{OptB88+DRSLL}}-E_{\mathrm{PBE+DRSLL}}| > 1e-4\) Ha
    (measured 0.220 Ha).
  - Closed-shell identity (water dimer OptB88 + DRSLL): UKS nα=nβ=8 vs RKS
    n_occ=8 — \(|\Delta E|\) ≤ 3e−4 eV and \(|\Delta E_{\mathrm{disp}}|\) ≤ 1e−5 Ha
    (measured 1.6e−8 eV / 5.8e−10 Ha).
- [x] Unit: `OptB88VdWSpinCollapseAndDiffersFromPBE` green.
- [x] Perf: fair \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 3.35\) on the matched
  twin (220.9 / 66.011 s); **within_10x**.
- [x] Scoped build only (`atomiverse`, unit, `reg_gpw-optb88-uks`).

## Out of scope

- Multi-k RKS/UKS optB88 (`gpw-optb88-kpoints`, future multi-k UKS optB88)
- Other exchange variants (optPBE-vdW, optB86b-vdW)
- Plane-wave path
- Forces / stress dedicated pins for OptB88-UKS
