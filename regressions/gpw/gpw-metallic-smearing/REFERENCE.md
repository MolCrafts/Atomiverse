# Reference — metallic Fermi–Dirac smearing (BCC Na, GPW multi-k)

## Method

Spin-unpolarized **GPW** Kohn–Sham DFT with **Fermi–Dirac** electronic
smearing (Mermin free energy \(F = E_{\mathrm{KS}} - TS\)). Public API:
`cpu::gpw::GpwDft` members `smearing`, `degauss` [Ry], `extra_bands`,
`mp_order`. Occupations reuse `cpu::pw::internal` (FD / Methfessel–Paxton).

## System (Atomiverse pin)

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) — in-tree table |
| Pseudopotential | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e = 18\)) |
| XC | PADE (LDA) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| SCF | DIIS 6, `tol = 1e-6` Ha, `EPS_DEFAULT = 1e-7` |

### Offline free-energy pin (frozen ATV)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) | **−92.126330608378 Ha** | \(10^{-4}\) Ha |
| \(N_{\mathrm{occ}}=\sum w f\) | **18.0** | \(10^{-5}\) |
| \(E_F\) | **0.055826 Ha** (finite) | \(\lvert E_F\rvert < 10^3\) Ha |
| Entropy term \(-\sigma S\) | **−2.26×10⁻⁴ Ha** | finite |

Recorded 2026-07-26 on Arrhenius (x86_64); dual-gate re-verify after rebase
onto `dev` (7 SCF iters, wall ≈ 1.0 s single rank, `OMP_NUM_THREADS=4`).

Absolute free energy is **not** compared to CP2K (grid / `EPS_DEFAULT` /
rel-cutoff / Poisson details differ by ~0.6 Ha). Electron count + Fermi
smearing algorithm class are the product twin.

## Offline CP2K twin (algorithm / SMEAR documentation)

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, source 85b8a9b49c) |
| Input | `tools/cp2k_oracle/inputs/gpw-metallic-smearing/na_bcc_fermi_dirac.inp` |
| Method | Quickstep GPW, PADE, `GTH-PBE-q9` + `DZVP-MOLOPT-SR-GTH` |
| Cell | BCC conventional \(a=4.225\) Å, 2 Na |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (≈ 0.02 Ry \(k_B T\)) |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| CUTOFF / REL_CUTOFF | 150 / 40 Ry, NGRIDS 4 |
| Example \(E_{\mathrm{tot}}\) | **−92.76593904 Ha** (8 SCF steps; not ATV pin) |
| Fermi energy | **0.054489 Ha** |
| Al algorithm twin | `tools/cp2k_oracle/inputs/al_fermi_dirac.inp` |

## Perf gate

| Engine | Wall (single rank, `OMP_NUM_THREADS=4`) |
| ------ | -------------------------------------- |
| Atomiverse `reg_gpw-metallic-smearing` | **0.999 s** |
| CP2K twin (`na_bcc_fermi_dirac.inp`) | **89.51 s** |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.011 \ll 10
\]

`within_10x = true`. Fairness note: same system/basis/PP class; ATV uses
in-tree tables and its own multi-grid; wall is indicative for the dual gate.
