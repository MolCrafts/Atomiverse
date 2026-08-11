# cuda-pw-smear-forces

## What

Dual-gate restricted Fermi–Dirac metallic-smear **analytic ionic forces** on
`cuda::pw::PW` vs same-settings `cpu::pw::PW`. Geometry is a stretched Al
dimer in a cubic box (relative bond stretch so |F| is non-vacuous; a monatomic
1-atom cell has F≡0 by lattice gauge).

## Oracle

1. **Same-settings CPU twin** — `cpu::pw::pw_forces_analytic` under identical
   members (restricted FD smear, fractional occupations). Not a self-golden of
   CUDA.
2. **Non-vacuous** — max|F_cpu| > 0.05 eV/Å.

Not spin∧smear, not stress, not smear×PBE force depth.

## Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/Al_pz_nc.upf` |
| ecut | 16 Ry |
| k_mesh | 1 (Γ) |
| cell | cubic \(a = 6.5\) Å |
| geometry | Al₂ bond \(= 2.55\) Å along z |
| smearing | Fermi–Dirac |
| degauss | 0.02 Ry |
| extra_bands | 4 |
| mix_beta | 0.4 |
| tol | 1e-5 |

## Tolerances

| Quantity | Tol |
| --- | --- |
| max\|F_cuda − F_cpu\| | 5e-3 eV/Å |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.

## Perf

**perf_exempt** — small-cell correctness dual-gate only (no CP2K wall claim).

## Measured (GH200)

| Job | max\|ΔF\| |
| --- | ---: |
| **1130427** (rebased wave-9) | **1.36e-11** eV/Å |
| 1130401 (pre-rebase) | 1.36e-11 eV/Å |
