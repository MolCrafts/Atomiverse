# Reference — OT + metallic Fermi–Dirac smearing (BCC Na, GPW multi-k)

## Method

Spin-unpolarized **GPW** Kohn–Sham DFT with **Orbital Transformation (OT)**
SCF **and** **Fermi–Dirac** electronic smearing (Mermin free energy
\(F = E_{\mathrm{KS}} - TS\)). Public API members: `use_ot`, `smearing`,
`degauss` [Ry], `extra_bands`, `mp_order`. Occupations reuse
`cpu::pw::internal` (FD / Methfessel–Paxton). OT manifold expands to
`n_occ + max(extra_bands, 1)` with fractional band occupations; projected
gradient is \(g = (I - S C C^\dagger)(2 H C f)\).

Product path: multi-k closed-shell RKS OT + smear. Multi-k UKS OT + smear is
productized under `ot-uks-smearing` (build accepts the combo).

## System (Atomiverse pin)

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) |
| Pseudopotential | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e = 18\)) |
| XC | PADE (LDA) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| OT | `use_ot = true`, `extra_bands = 4` |
| DIIS twin | same without `use_ot` |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | `build` accepts RKS OT + FD | — |
| A2 | OT+FD SCF converges | — |
| A3 | \(\lvert F_{\mathrm{OT}} - F_{\mathrm{DIIS}}\rvert\) | \(10^{-4}\) Ha |
| A4 | \(N_{\mathrm{occ}} = 18\) | \(10^{-5}\) |
| A5 | frozen free-energy pin (first green) | \(10^{-3}\) Ha |
| A6 | multi-k UKS OT + FD smear build accepted | — |

## Offline free-energy pin (first green 2026-07-26)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) (OT) | **−92.1263304935 Ha** | \(10^{-3}\) Ha |
| Free energy \(F\) (DIIS twin) | −92.1263306084 Ha | — |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.15×10⁻⁷ Ha** | \(10^{-4}\) Ha |
| \(N_{\mathrm{occ}}\) | **18.0** | \(10^{-5}\) |
| OT iters | 12 | — |

Absolute free energy is **not** compared to CP2K (grid / `EPS_DEFAULT` /
rel-cutoff differ by ~0.6 Ha). Electron count + OT≡DIIS identity are the
product gates. Same system class as `reg_gpw-metallic-smearing`.

## Offline CP2K twin

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (same class as metallic-smearing) |
| Input | `tools/cp2k_oracle/inputs/ot-smearing/na_ot_fermi_dirac.inp` |
| Note | CP2K OT does not combine with SMEAR; twin is DIIS + FERMI_DIRAC |
| Example \(E_{\mathrm{tot}}\) | **−92.76593904 Ha** (not ATV pin) |

## Perf gate

| Engine | Wall (single rank, `OMP_NUM_THREADS=4`) |
| ------ | -------------------------------------- |
| Atomiverse `reg_ot-smearing` (OT+FD only) | **~18.2 s** |
| CP2K twin (`na_ot_fermi_dirac.inp`) | **~89.5 s** (same as metallic-smearing) |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 18.2 / 89.5 \approx 0.20 \ll 10
\]

`within_10x = true`. Fairness: ATV OT wall only (DIIS twin also runs in the
regression binary for the identity gate — not counted in \(T_{\mathrm{atv}}\)).

**ctest never launches CP2K.**
