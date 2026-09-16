# Reference — metallic Fermi–Dirac smearing (FCC Al, plane-wave)

## Method

Spin-unpolarized plane-wave DFT with **Fermi–Dirac** electronic smearing
(Mermin free energy \(F = E_{\mathrm{KS}} - TS\)). Public API:
`cpu::pw::PW` members `smearing`, `degauss` [Ry], `extra_bands`.

Occupation formulas unit-tested against closed form
(Methfessel–Paxton PRB **40**, 3616 (1989); Mermin *Phys. Rev.* **137**, A1441).

## System (Atomiverse pin)

| Quantity | Value |
| -------- | ----- |
| Geometry | FCC Al primitive cell, \(a = 4.05\) Å, 1 atom at origin |
| PP | `tests/data/pw/Al_pz_nc.upf` (NC, LDA/PZ, \(Z_{\mathrm{val}}=3\)) |
| \(E_{\mathrm{cut}}\) | 20 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| Extra bands | 4 (\(\Rightarrow\) 6 bands total for \(N_e=3\)) |
| SCF | `tol=1e-5`, `max_iter=60`, `mix_beta=0.4` |

### Offline free-energy pin (frozen ATV)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) | **−7.54535571 Ry** | 0.05 Ry |
| \(N_{\mathrm{occ}}=\sum w f\) | **3.0** | \(10^{-5}\) |
| \(E_F\) | finite, \(\lvert E_F\rvert < 10^3\) Ry | — |

Recorded 2026-07-23 on Arrhenius (x86_64) with the settings above.

## Offline CP2K twin (algorithm / SMEAR documentation)

**Not an absolute-energy twin** (GTH-GPW ≠ NC-UPF PW). Documents that CP2K
`&SMEAR METHOD FERMI_DIRAC` + `ADDED_MOS` is the product counterpart of
ATV metallic occupations.

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Input | `tools/cp2k_oracle/inputs/al_fermi_dirac.inp` |
| Method | Quickstep GPW, PADE, `GTH-PADE-q3` + `DZVP-MOLOPT-SR-GTH` |
| Cell | same primitive FCC \(a=4.05\) Å |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (\(\approx 0.02\) Ry \(k_B T\)) |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| Example \(E_{\mathrm{tot}}\) | ≈ −2.096 Ha (run-dependent on grid; not compared to ATV Ry pin) |
| Fermi energy | reported by CP2K (Ha) |

## Perf gate

| Engine | Wall (single rank) |
| ------ | ------------------ |
| Atomiverse `reg_pw-smearing-al` | **1.70 s** |
| CP2K twin (`al_fermi_dirac.inp`) | **211 s** |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.008 \ll 10
\]

`within_10x = true`. Fairness note: different basis/PP/grid; wall is
indicative only for the dual gate.
