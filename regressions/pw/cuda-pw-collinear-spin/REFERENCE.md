# cuda-pw-collinear-spin

## What

Dual-gate collinear fixed-occupation UKS (O2 triplet, multiplicity 3 → M = 2)
on `cuda::pw::PW` with **`XCType::LSDA`**. Same geometry and occupations as
`pw-spin-polarized-o2`, restricted to the device-supported surface (no PBE
spin-GGA, no smear).

## Oracle

1. **Magnetization identity** — M = multiplicity − 1 = 2 (integer fill).
2. **Same-settings CPU twin** — `cpu::pw::PW` at identical members is the
   correctness oracle for CUDA energy (ES CUDA open-gate law).

Not a freeze of CUDA’s own energy. Not the PBE literature pin from
`pw-spin-polarized-o2` (that functional has no collinear device path yet).

## Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| ecut | 12 Ry |
| k_mesh | 1 (Γ) |
| box | 8 Å cubic |
| R(O–O) | 1.2075 Å |
| spin_polarized | true |
| multiplicity | 3 |
| xc | LSDA |
| mix_beta | 0.3 |
| tol | 5e-4 |

## Tolerances

| Quantity | Tol |
| --- | --- |
| mag | 1e-9 |
| \|E_cuda − E_cpu\| | 2e-3 Ha |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.
