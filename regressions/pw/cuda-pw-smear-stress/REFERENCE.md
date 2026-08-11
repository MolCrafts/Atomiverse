# cuda-pw-smear-stress

## What

Dual-gate restricted Fermi–Dirac metallic-smear **Nielsen–Martin cell stress**
on `cuda::pw::PW` vs same-settings `cpu::pw::PW`. Geometry is compressed FCC
Al (1 atom primitive) so hydrostatic |σ| is non-vacuous and valence is odd
(requires fractional occupations).

The free energy surface is the **Mermin free energy**
\(F = E_{\mathrm{KS}} - TS\); stress is
\(\sigma = -(1/\Omega)\,\partial F/\partial\varepsilon\).

## Oracle

1. **Same-settings CPU twin** — `cpu::pw::pw_stress` under identical members
   (restricted FD smear, fractional occupations, Mermin F). Not a self-golden
   of CUDA.
2. **Non-vacuous** — mean|diag_cpu| > 1e-3 eV/Å³.
3. **Cubic symmetry** — |σ_xx−σ_yy|, |σ_yy−σ_zz|, |σ_off| within soft bound.

Not spin∧smear, not multi-k smear stress, not smear×PBE stress.

## Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/Al_pz_nc.upf` |
| ecut | 16 Ry |
| k_mesh | 1 (Γ) |
| cell | FCC primitive \(a = 3.90\) Å (compressed) |
| XC | LSDA |
| smearing | Fermi–Dirac |
| degauss | 0.02 Ry |
| extra_bands | 4 |
| mix_beta | 0.4 |
| tol | 1e-5 |

## Tolerances

| Quantity | Tol |
| --- | --- |
| max\|σ_cuda − σ_cpu\| | 5e-2 eV/Å³ |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.

## Perf

**perf_exempt** — small-cell correctness dual-gate only (no CP2K wall claim).

## Measured (GH200)

| Job | max\|Δσ\| | notes |
| --- | ---: | --- |
| **1132266** | **7.252e-11** eV/Å³ | dual-gate both; mean\|diag_cpu\|=3.493; \|ΔE\|=1.63e-13 Ry; units green |
| 1131116 | — | exit 127 (job-script wipe mid-run); not a product fail |
| 1130720 | — | unit suite timeout (pre unit-gate trim); not a product fail |
