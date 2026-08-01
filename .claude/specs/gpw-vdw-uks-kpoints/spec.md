# Spec — gpw-vdw-uks-kpoints

## Capability

Self-consistent nonlocal vdW-DF (DRSLL / LMKLL / rVV10) on the **multi-k
collinear UKS** GPW path. \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\)
is a functional of the **total** density \(\rho=\rho_\alpha+\rho_\beta\) and is
added identically to both spin potentials every SCF iteration
(`build_uks_k_energy` + DIIS twin); post-SCF \(E_c^{\mathrm{nl}}\) lands on
`GpwResult::e_disp` via `fold_dispersion_k_uks`.

Closes the former `GpwDft::build` fail-fast
`"nonlocal vdW … with multi-k UKS"`.

## Matched twin

CP2K 2026.1 Quickstep GPW: `LSD` + `&KPOINTS MONKHORST-PACK` +
`&VDW_POTENTIAL NON_LOCAL TYPE DRSLL` (GTH-PBE + MOLOPT).

## Gates

| Gate | Claim |
|------|--------|
| G1 | Multi-k UKS path at explicit \(k=\Gamma\) ≡ shipped Γ UKS (same settings, vdW on) |
| G2 | MP multigrid multi-k UKS vs offline CP2K LSD + DRSLL pin (\(E_{\mathrm{tot}}\), \(E_c^{\mathrm{nl}}\)) |
| G3 | Closed-shell multi-k UKS ≡ multi-k RKS (both vdW) on the same mesh |

## Out of scope

- Analytic multi-k UKS vdW force/stress dedicated pins (V_nl already rides
  force assembly; follow-up).
- Multi-k UKS + vdW + multigrid triple stress combo.
- CUDA.
