# Reference — multi-k GPW OT + Dudarev DFT+U + metallic FD smearing

## Method

Closed-shell **GPW** Kohn–Sham DFT with **Orbital Transformation (OT)** SCF on
the **complex multi-k** path (`use_ot = true` + explicit `k_points` /
`k_mesh`), combined with **Dudarev PLUS_U** (`plus_u`) and **Fermi–Dirac**
metallic occupation smearing (Mermin free energy \(F = E_{\mathrm{KS}} - TS\)).

Product path: integer multi-k OT+U keeps OUTER_SCF + fixed-H OT; **U + smear**
uses the free-energy PR+ Armijo OT manifold (full AO spectrum, fractional
\(f\)), with \(V_U\) folded into \(H_k\) via `build_k_energy`. Same fixed point
as multi-k DIIS + U + FD.

Public API: concrete `cpu::gpw::GpwDft` members `use_ot`, `plus_u`, `smearing`,
`degauss`, `extra_bands`, `k_points` / `k_mesh`. No public `*Config`, no
factories.

**Related:** multi-k UKS OT + U + smear is product (`ot-uks-dftu-smearing`).

## System (regression)

| Item | Value |
| ---- | ----- |
| Molecule | H₂ (DZV-GTH-PADE / GTH-PADE-q1) |
| Cell | **8 Bohr** cubic, periodic |
| Bond | ±0.7 Bohr from cell centre along *x* |
| XC | PADE |
| Cutoff | **100 Ry**, single grid |
| n_occ | 1 (closed-shell RKS, \(N_e = 2\)) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| k | explicit κ = (¼,0,0), w=1 |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 2` |
| OT | `use_ot = true` |

## Offline CP2K oracle

- **Engine:** CP2K **2026.1** (`cp2k.psmp`)
- **Input:** `tools/cp2k_oracle/inputs/ot-dftu-smearing/h2_ot_dftu_smear.inp`
- **OT+k:** CP2K 2026.1 **aborts** with
  `OT not possible with kpoint calculations` (`qs_scf_initialization.F`).
  Absolute CP2K OT+k digits **do not exist**.
- **OT+SMEAR:** CP2K product OT uses fixed integer occupations; SMEAR is a
  DIIS twin feature.
- **Twin:** multi-k DIIS + FERMI_DIRAC + PLUS_U (input above). Fixed-point
  twin of ATV OT+U+smear — not a fair OT wall twin.
- **Pin strategy:** internal exactness OT ≡ DIIS free energy is **decisive**.
  Frozen first-green ATV absolute for A6.

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| -------- | ----: |
| \(F_{\mathrm{OT}}\) (U=4, FD) | **−1.081998158198 Ha** |
| \(F_{\mathrm{DIIS}}\) same | −1.081997918765 Ha |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **2.4×10⁻⁷ Ha** |
| \(E_U\) (OT) | ≈ −8.57×10⁻⁴ Ha |
| \(N_e\) | **2.0** |
| \(F(U=0)\) OT+FD | −1.108020857737 Ha |
| \(\lvert F(U)-F(0)\rvert\) | ≈ 0.026 Ha |
| OT iters / DIIS iters | 11 / 42 |
| OT wall (Release, OMP=4) | **0.169 s** |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A0 | `build` accepts RKS OT + U + FD | — |
| A1 | OT+U+FD and DIIS+U+FD converge | — |
| A2 | \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | \(10^{-4}\) Ha |
| A3 | \(E_U\) non-trivial; OT vs DIIS \(E_U\) | \(10^{-3}\) Ha |
| A4 | \(N_e = 2\) | \(10^{-5}\) |
| A5 | \(\lvert F(U)-F(0)\rvert\) activation | \(\ge 10^{-4}\) Ha |
| A6 | absolute vs frozen first-green | \(10^{-3}\) Ha |
| A7 | multi-k UKS OT + U + FD accepted at build | — |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | CP2K 2026.1 has no OT+kpoint path (hard abort) and no OT+SMEAR product combo; DIIS multi-k twin is not a fair OT wall comparison. Decisive gates are internal OT≡DIIS free-energy identities (same pattern as `ot-dftu-kpoints` / `ot-smearing`). |
| r = T_atv/T_cp2k | n/a |
| `within_10x` | n/a (exempt) |
| ATV wall (A0–A7) | ≈ 0.4 s Release |

**ctest never launches CP2K.**

## Citations

- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\).
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) — free-energy DFT.

## Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- A6 frozen free energy: −1.081998158198 → **−1.400026280953 Ha** (OT ≡ DIIS parity 1.7e-7 Ha).
