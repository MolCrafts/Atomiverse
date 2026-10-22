# cuda-npt-pme-nacl — CUDA PME under a changing cell

`reg_cuda-npt-pme-nacl` (needs `ATV_BUILD_CUDA`, `ATV_BUILD_KSPACE`,
`ATV_BUILD_EVALUATE`; one GPU).

| Gate | System | Check |
|---|---|---|
| A | 8³ rocksalt point charges, d = 2.82 Å, CoulLong + KSpacePME (α = 0.35, r_c = 9 Å, spacing 0.5 Å, order 6) | E = −N M k_e/(2d), M = 1.747564594633182; W_aa = E/3, W_ab = 0 (homogeneous degree −1 ⇒ virial = energy); relative 3e-3, off-diagonal 1e-6·\|E\| |
| B | Same lattice, Joung–Cheatham TIP3P NaCl LJ (LB mixing), PairLJCutCoulLong + KSpacePME (order 5, 24³ mesh), MTK NPT 300 K / 1 bar, τ_T 100 fs, τ_P 1000 fs, dt 1 fs, 100 steps | CPU↔CUDA step-0 virial (3e-5 double / 1e-4 mixed; the CPU twin tabulates erfc), final volume (1e-5 / 1e-4), CUDA conserved-quantity drift < 2e-4 eV/atom |
| C | Perturbed lattice scaled by s ∈ {0.95, 0.98, 1.02, 1.05}, mesh fixed at the build-time 24³ | PME vs converged Ewald (accuracy 1e-10): max force error < 5× the s = 1 error, \|ΔE/E\| < 1e-4 |

References: Madelung constant of rocksalt, M = 1.747564594633…
(Y. Sakamoto, J. Chem. Phys. 28, 164 (1958)); virial of a homogeneous
potential (Clausius); Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008),
doi:10.1021/jp8001614; MTK: Martyna, Tobias and Klein, J. Chem. Phys. 101,
4177 (1994), doi:10.1063/1.467468; smooth PME virial: Essmann et al.,
J. Chem. Phys. 103, 8577 (1995), doi:10.1063/1.470117.

Gate C documents the fixed-mesh contract of `cuda::KSpacePME`: the FFT
dimensions are chosen once at build and are not re-chosen when the cell
changes. The recorded error is not a claim that the mesh error is
maintained automatically.

## Recorded results

GH200, CUDA 13.0.3, 2026-10-21.

| Gate | double | mixed |
|---|---|---|
| A: \|E/E_M − 1\| | 1.40e-5 | 1.39e-5 |
| A: max \|W_aa/(E_M/3) − 1\| | 3.24e-4 | 3.23e-4 |
| A: max \|W_ab/E_M\| | 5.8e-19 | 2.9e-10 |
| B: step-0 \|ΔW\|/\|W\| CPU↔CUDA | 7.31e-6 | 8.44e-6 |
| B: 100-step \|ΔV\|/V CPU↔CUDA | 3.53e-7 | 2.25e-7 |
| B: CUDA \|H(100) − H(1)\|/N [eV] | 4.98e-6 | 4.97e-6 |

Gate C (fixed 24³ mesh, both modes identical to three digits):

| s | \|ΔE/E\| | max\|ΔF\|/max\|F\| |
|---|---|---|
| 0.95 | 7.11e-7 | 9.28e-5 |
| 0.98 | 1.17e-6 | 1.03e-4 |
| 1.00 | 1.60e-6 | 1.07e-4 |
| 1.02 | 2.14e-6 | 1.30e-4 |
| 1.05 | 3.22e-6 | 1.74e-4 |

The force error grows by 1.6× at +5 % linear strain (the effective mesh
spacing grows with the cell), and stays within the s = 1 accuracy class.
