# Reference — multi-k GPW UKS OT + Dudarev DFT+U + metallic FD smearing

## Method

Collinear **UKS** Kohn–Sham DFT (GPW) with **Orbital Transformation (OT)** SCF
on the **complex multi-k** path (`use_ot = true` + explicit `k_points`),
combined with **Dudarev PLUS_U** (`plus_u`) and **Fermi–Dirac** metallic
occupation smearing (Mermin free energy \(F = E_{\mathrm{KS}} - TS\)).

Product path: integer multi-k UKS OT+U keeps OUTER_SCF + fixed-H OT;
**U + smear** uses the free-energy PR+ Armijo OT manifold (full AO spectrum
per spin, shared \(\mu\) over \(\alpha+\beta\), \(f_\sigma\in[0,1]\)), with
\(V_U^\sigma\) folded into \(H_k^\sigma\) via `build_uks_k_energy`. Same fixed
point as multi-k UKS DIIS + U + FD.

Public API: concrete `cpu::gpw::GpwDft` members `use_ot`, `plus_u`, `smearing`,
`degauss`, `extra_bands`, `n_alpha`/`n_beta`, `k_points` / `k_mesh`. No public
`*Config`, no factories.

Prior product combos: RKS OT+U+smear (`ot-dftu-smearing`), UKS OT+smear
(`ot-uks-smearing`), UKS OT+U integer (`ot-uks-dftu`).

## System (regression)

| Item | Value |
| ---- | ----- |
| Molecule | H₂ (DZV-GTH-PADE / GTH-PADE-q1) |
| Cell | **8 Bohr** cubic, periodic |
| Bond | ±0.7 Bohr from cell centre along *x* |
| XC | PADE (LSD) |
| Cutoff | **100 Ry**, single grid |
| Spin seed | `n_alpha = 1`, `n_beta = 1` (equal-spin; free under smear) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| k | explicit κ = (¼,0,0), w=1 |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 2` |
| OT | `use_ot = true` |

**Seed note:** under FD smear, occupations are free. Equal-spin seed
(`nα=nβ=1`) keeps DIIS and OT on the same Mermin minimum. A high-spin integer
seed (`nα=2`/`nβ=0`) can leave DIIS in a metastable high-spin solution while
free-energy OT finds the lower closed-shell-like minimum — not an OT bug.

## Offline CP2K oracle

- **Engine:** CP2K **2026.1** (`cp2k.psmp`)
- **Input:** `tools/cp2k_oracle/inputs/ot-uks-dftu-smearing/h2_uks_ot_dftu_smear.inp`
- **OT+k:** CP2K 2026.1 **aborts** with
  `OT not possible with kpoint calculations` (`qs_scf_initialization.F`).
  Absolute CP2K OT+k digits **do not exist**.
- **OT+SMEAR:** CP2K product OT uses fixed integer occupations; SMEAR is a
  DIIS twin feature.
- **Twin:** multi-k UKS DIIS + FERMI_DIRAC + PLUS_U (input above). Fixed-point
  twin of ATV UKS OT+U+smear — not a fair OT wall twin.
- **Pin strategy:** internal exactness OT ≡ DIIS free energy is **decisive**.
  Frozen first-green ATV absolute for A6.

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| -------- | ----: |
| \(F_{\mathrm{OT}}\) (U=4, FD) | **−1.081998158198 Ha** |
| \(F_{\mathrm{DIIS}}\) same | −1.081997968639 Ha |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.9×10⁻⁷ Ha** |
| \(E_U\) (OT) | ≈ −8.57×10⁻⁴ Ha |
| \(N_e\) | **2.0** |
| \(F(U=0)\) OT+FD | −1.108020857737 Ha |
| \(\lvert F(U)-F(0)\rvert\) | ≈ 0.026 Ha |
| OT iters / DIIS iters | 11 / 43 |
| OT wall (Release, OMP=4) | **0.325 s** |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A0 | `build` accepts UKS OT + U + FD | — |
| A1 | OT+U+FD and DIIS+U+FD converge | — |
| A2 | \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | \(10^{-4}\) Ha |
| A3 | \(E_U\) non-trivial; OT vs DIIS \(E_U\) | \(10^{-3}\) Ha |
| A4 | \(N_e = 2\) | \(10^{-5}\) |
| A5 | \(\lvert F(U)-F(0)\rvert\) activation | \(\ge 10^{-4}\) Ha |
| A6 | absolute vs frozen first-green | \(10^{-3}\) Ha |

## Performance

| Item | Value |
| ---- | ----- |
| Fair wall twin | **none** (CP2K aborts OT+k / OT+SMEAR) |
| Status | **perf_exempt** — internal OT≡DIIS identity is the product gate |
| Reason | Same class as `ot-dftu-smearing` / `ot-uks-dftu` |

## Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- A6 frozen free energy: −1.081998158198 → **−1.400026280953 Ha** (equal-spin ≡ RKS Mermin minimum).
