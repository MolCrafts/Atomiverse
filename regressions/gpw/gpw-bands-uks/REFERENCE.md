# REFERENCE — gpw-bands-uks

## Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-bands-uks/si_uks_bands.inp` |
| Class | Quickstep GTH-GPW collinear UKS + `&PRINT &BAND_STRUCTURE` |
| System | Diamond Si BCT 4-atom, DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE, MP 2×2×2 |

## Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| \(E_{\mathrm{tot}}\) | −15.802416412655198 Ha | \(10^{-3}\) Ha |
| Path gap (closed-shell UKS ≡ RKS) | 0.893516 eV class | 80 meV |
| \(n_{\mathrm{spin}}\) | 2 | exact |
| α≡β spectrum (closed shell) | max\|Δ\| < 1e-6 Ha | product |

## Fair wall

| Side | Wall |
| --- | --- |
| ATV (OMP=8, first green 2026-08-04) | 7.56 s (`ATV_WALL_S`) |
| CP2K twin | run offline; gate \(r\le 10\) |
