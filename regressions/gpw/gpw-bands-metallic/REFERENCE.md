# REFERENCE — gpw-bands-metallic

## Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-bands-metallic/na_bcc_smear_bands.inp` |
| Class | Quickstep GTH-GPW RKS + SMEAR FERMI_DIRAC + `&PRINT &BAND_STRUCTURE` |
| System | BCC Na conventional 2-atom, DZVP-MOLOPT-SR-GTH + GTH-PBE-q9, PADE, MP 2×2×2 |

## Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| Free energy \(F\) | −92.126330608378 Ha (ATV freeze class) | \(10^{-3}\) Ha |
| \(E_F\) class | 0.054478 Ha (CP2K dos-metallic twin) | 0.05 Ha |
| Metallic character | path spectrum straddles \(E_F\) | product |

## Fair wall

| Side | Wall |
| --- | --- |
| ATV (OMP=8, first green 2026-08-04) | 1.77 s (`ATV_WALL_S`) |
| CP2K twin | run offline; gate \(r\le 10\) |
