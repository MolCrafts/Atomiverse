# cuda-pw-metallic-smear

## What

Dual-gate restricted Fermi–Dirac metallic smear (FCC Al primitive cell) on
`cuda::pw::PW`. Same geometry and occupations as `pw-smearing-al`.

## Oracle

1. **Electron count** — \(N_{\mathrm{occ}} = 3\) (Al \(Z_{\mathrm{val}}\)).
2. **Same-settings CPU twin** — `cpu::pw::PW` at identical members is the
   correctness oracle for CUDA free energy (ES CUDA open-gate law).

Not a freeze of CUDA’s own energy. Not forces/stress/spin+smear.

## Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/Al_pz_nc.upf` |
| ecut | 20 Ry |
| k_mesh | 2 (MP \(2\times2\times2\)) |
| cell | FCC primitive \(a = 4.05\) Å |
| smearing | Fermi–Dirac |
| degauss | 0.02 Ry |
| extra_bands | 4 |
| mix_beta | 0.4 |
| tol | 1e-5 |

## Tolerances

| Quantity | Tol |
| --- | --- |
| \(N_{\mathrm{occ}}\) | 1e-5 |
| \|E_cuda − E_cpu\| | 5e-3 Ry |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.
