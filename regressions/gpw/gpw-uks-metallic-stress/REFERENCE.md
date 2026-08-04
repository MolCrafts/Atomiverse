# REFERENCE — gpw-uks-metallic-stress

## Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-uks-metallic-stress/na_bcc_uks_smear_stress.inp` |
| Class | Quickstep LSD UKS + SMEAR FERMI_DIRAC + `STRESS_TENSOR NUMERICAL` |
| System | BCC Na, GTH-PBE-q9 / DZVP-MOLOPT-SR-GTH, PADE, MP 2×2×2 |

## Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| Free energy \(F\) | −92.126330608004 Ha | \(10^{-3}\) Ha |
| Analytic vs FD max\|Δσ_diag\| | product | \(2\times10^{-4}\) Ha/Bohr³ |
| Closed-shell UKS ≡ RKS σ | product identity | \(5\times10^{-5}\) Ha/Bohr³ |
| Mean σ_diag | −1.9177e-2 Ha/Bohr³ (sibling metallic-stress) | \(5\times10^{-4}\) |
| CP2K NUMERICAL class | −1.705e-2 Ha/Bohr³ | \(5\times10^{-3}\) soft |

## Fair wall

| Side | Wall |
| --- | --- |
| ATV SCF (OMP=8, first green 2026-08-04) | 0.78 s |
| CP2K NUMERICAL stress twin | class of metallic-stress ENERGY wall (~0.9 s); gate \(r\le 10\) |
