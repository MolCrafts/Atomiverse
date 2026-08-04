# REFERENCE — gpw-vdw-stress

## Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-vdw-stress/water_dimer_vdw_df_stress.inp` |
| Class | Quickstep PBE + `NON_LOCAL TYPE DRSLL` + `STRESS_TENSOR NUMERICAL` |
| System | Water dimer periodic box 12 Å, DZVP-MOLOPT-SR-GTH, CUTOFF 200 Ry |

## Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| \(E_{\mathrm{c}}^{\mathrm{nl}}\) | ~0.15194 Ha (DRSLL class) | \(5\times10^{-4}\) Ha |
| Analytic vs multi-SCF FD max\|Δσ_diag\| | product dual-gate | \(5\times10^{-4}\) Ha/Bohr³ |
| Nonlocal contribution | \(\|\sigma_{\mathrm{vdw}}-\sigma_{\mathrm{bare}}\| > 10^{-6}\) | wiring |

## Fair wall

| Side | Wall |
| --- | --- |
| ATV SCF (OMP=8, first green 2026-08-04) | 10.51 s |
| Analytic stress assembly | ~25 s |
| Multi-SCF FD | ~59 s (gate only; not fair twin) |
| CP2K ENERGY/stress twin | run offline; gate \(r\le 10\) on SCF |
