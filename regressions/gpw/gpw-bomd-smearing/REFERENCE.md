# REFERENCE — gpw-bomd-smearing

## Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-bomd-smearing/h2_bomd_smear_kpoints_nve.inp` |
| Class | Quickstep AIMD NVE + SMEAR FERMI_DIRAC + KPOINTS 1×1×2 |
| System | Periodic H₂, DZV-GTH-PADE / GTH-PADE-q1, CUTOFF 300 Ry |

## Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| Step-0 free energy class | ~−1.07 Ha | 0.05 Ha soft |
| Fractional \(f_k\) payload | present each step | product |
| NVE relative drift (4×0.5 fs) | family envelope | 0.10 |

## Measured first green (2026-08-04, OMP=8)

| Quantity | Value |
| --- | --- |
| Step-0 \(F\) | −1.0744567933 Ha |
| \(E_F\) | −0.131510 Ha |
| NVE worst relative drift (4×0.5 fs) | 4.25e-3 |
| Wall (full reg) | ~2.0 s |

## Fair wall

Short traj / step-0 ENERGY twin \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\) (CP2K offline).
