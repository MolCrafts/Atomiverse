# Reference — UKS OT + metallic Fermi–Dirac smearing (BCC Na, GPW multi-k)

## Method

Collinear spin-polarized **GPW** Kohn–Sham DFT with **Orbital Transformation
(OT)** SCF **and** **Fermi–Dirac** electronic smearing (Mermin free energy
\(F = E_{\mathrm{KS}} - TS\)). Shared chemical potential \(\mu\) over
\(\alpha\) and \(\beta\) spin-orbitals; occupations \(f_{\sigma kb}\in[0,1]\);
magnetization free under smear (`n_alpha`/`n_beta` seed only). Public API:
`use_ot`, `n_alpha`/`n_beta`, `smearing`, `degauss` [Ry], `extra_bands`,
`mp_order`. Occupations reuse `cpu::pw::internal` with `spin_degen = 1`.
OT manifolds expand to the full independent AO spectrum; projected gradient
uses non-uniform spinor occupations.

Product path co-enables multi-k UKS OT (`ot-uks-kpoints`) and multi-k UKS
DIIS+smear (`gpw-uks-smearing`).

## System (Atomiverse pin)

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) |
| Pseudopotential | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e = 18\)) |
| XC | PADE (LSD) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Spin seed | \(n_\alpha=10\), \(n_\beta=8\) (ferro; free under smear) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| OT | `use_ot = true`, `extra_bands = 4` |
| DIIS twin | same without `use_ot` |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | `build` accepts UKS OT + FD | — |
| A2 | OT+FD SCF converges | — |
| A3 | \(\lvert F_{\mathrm{OT}} - F_{\mathrm{DIIS}}\rvert\) | \(10^{-4}\) Ha |
| A4 | \(N_{\mathrm{occ}} = 18\) | \(10^{-5}\) |
| A5 | frozen free-energy pin (first green) | \(10^{-3}\) Ha |
| A6 | finite \(E_F\), entropy term | — |

## Offline free-energy pin (first green 2026-07-26)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) (OT) | **−92.1263304935 Ha** | \(10^{-3}\) Ha |
| Free energy \(F\) (DIIS twin) | −92.1263306080 Ha | — |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.15×10⁻⁷ Ha** | \(10^{-4}\) Ha |
| \(N_{\mathrm{occ}}\) | **18.0** | \(10^{-5}\) |
| OT iters | 12 | — |

Absolute free energy is **not** compared to CP2K (grid / `EPS_DEFAULT` /
rel-cutoff differ by ~0.6 Ha). Electron count + OT≡DIIS identity are the
product gates. Same system class as `reg_gpw-uks-smearing` /
`reg_ot-smearing` (non-magnetic collapse under smear for BCC Na).

## Offline CP2K twin

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** |
| Input | `tools/cp2k_oracle/inputs/ot-uks-smearing/na_bcc_uks_fermi_dirac.inp` |
| Note | CP2K OT does not combine with k-points; twin is DIIS + LSD + FERMI_DIRAC |
| Wall class | same as `gpw-uks-smearing` / RKS FD (~89.5 s OMP=4) |

## Perf gate

| Engine | Wall (single rank, `OMP_NUM_THREADS=4`) |
| ------ | -------------------------------------- |
| Atomiverse `reg_ot-uks-smearing` (OT+FD only) | **~10.3 s** |
| CP2K twin class (`na_bcc_uks_fermi_dirac.inp`) | **~89.5 s** |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 10.3 / 89.5 \approx 0.115 \ll 10
\]

`within_10x = true`. Fairness: ATV OT wall only (DIIS twin also runs in the
regression binary for the identity gate — not counted in \(T_{\mathrm{atv}}\)).

**ctest never launches CP2K.**
