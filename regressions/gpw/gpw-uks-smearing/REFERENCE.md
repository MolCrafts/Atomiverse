# Reference — collinear UKS + metallic Fermi–Dirac smearing (BCC Na, GPW multi-k)

## Method

Collinear spin-polarized **GPW** Kohn–Sham DFT with **Fermi–Dirac** electronic
smearing (Mermin free energy \(F = E_{\mathrm{KS}} - TS\)). Shared chemical
potential \(\mu\) over \(\alpha\) and \(\beta\) spin-orbitals; occupations
\(f_{\sigma kb}\in[0,1]\); magnetization is free under smear
(`n_alpha`/`n_beta` seed only). Public API: `cpu::gpw::GpwDft` members
`smearing`, `degauss` [Ry], `extra_bands`, `mp_order`, `n_alpha`/`n_beta`.
Occupations reuse `cpu::pw::internal` with `spin_degen = 1`.

Closes the product gap left by multi-k RKS smear (`gpw-metallic-smearing`)
and fixed-occ multi-k UKS (`gpw-kpoints-uks`).

## System (Atomiverse pin)

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) — in-tree table |
| Pseudopotential | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e = 18\)) |
| XC | PADE (LSD) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Spin seed | \(n_\alpha=10\), \(n_\beta=8\) (ferro; free under smear) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| SCF | DIIS 6, `tol = 1e-6` Ha, `EPS_DEFAULT = 1e-7` |

### Offline free-energy pin (frozen ATV)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) | **−92.126330608004 Ha** | \(10^{-4}\) Ha |
| \(N_{\mathrm{occ}}=\sum_{\sigma k b} w_k f\) | **18.0** | \(10^{-5}\) |
| \(E_F\) | **0.055825 Ha** (finite) | \(\lvert E_F\rvert < 10^3\) Ha |
| Entropy term \(-\sigma S\) | **−2.26×10⁻⁴ Ha** | finite |

Recorded 2026-07-26 on Arrhenius (x86_64); 16 SCF iters, wall ≈ 1.56 s
single rank, `OMP_NUM_THREADS=4`. Matches the RKS Na FD free energy of
`reg_gpw-metallic-smearing` to \(\sim 4\times 10^{-13}\) Ha (non-magnetic
collapse under smear, as expected for BCC Na).

### Identity (same binary)

Closed-shell multi-k H₂: UKS \(n_\alpha=n_\beta=1\) + FD ≡ RKS \(n_{\mathrm{occ}}=1\)
+ FD to \(\lvert\Delta F\rvert \lesssim 10^{-15}\) Ha; both recover \(N_e=2\).

## Offline CP2K twin (algorithm / SMEAR documentation)

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (local install under `…/work/cp2k/install`) |
| Input | `tools/cp2k_oracle/inputs/gpw-uks-smearing/na_bcc_uks_fermi_dirac.inp` |
| Method | Quickstep GPW **LSD** + FERMI_DIRAC, PADE, GTH-PBE-q9 + DZVP-MOLOPT-SR-GTH |
| Cell | BCC conventional \(a=4.225\) Å, 2 Na |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (≈ 0.02 Ry \(k_B T\)) |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| CUTOFF / REL_CUTOFF | 150 / 40 Ry, NGRIDS 4 |
| MULTIPLICITY | 3 (ferro seed) |

Absolute free energy is **not** required to match CP2K (grid / `EPS_DEFAULT` /
rel-cutoff differ by ~0.6 Ha class as RKS twin). Electron count + UKS+smear
algorithm class are the product twin.

RKS algorithm twin wall (same system, closed shell):
`tools/cp2k_oracle/inputs/gpw-metallic-smearing/na_bcc_fermi_dirac.inp`
**≈ 89.5 s** (OMP=4) — used as fair wall class for the dual gate.

## Perf gate

| Engine | Wall (single rank, `OMP_NUM_THREADS=4`) |
| ------ | -------------------------------------- |
| Atomiverse `reg_gpw-uks-smearing` (Na pin portion) | **≈ 1.56 s** |
| CP2K RKS twin class (`na_bcc_fermi_dirac.inp`) | **89.51 s** |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.017 \ll 10
\]

`within_10x = true`. Fairness note: UKS CP2K twin is the same system with
`LSD`; wall class matches the RKS FD twin already timed offline.

## Dual gate

1. Correctness: R1–R5 (converge, Ne, \(E_F\), free-energy pin, closed-shell identity).
2. Perf: within_10x vs CP2K Na FD twin wall.
