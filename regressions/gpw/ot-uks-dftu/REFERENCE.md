# Reference — multi-k UKS GPW OT + Dudarev DFT+U

## Method

Collinear **UKS GPW** with **Orbital Transformation (OT)** on the **complex
multi-k** path (`use_ot = true` + `n_alpha`/`n_beta` + `k_mesh` / explicit
`k_points`) combined with **Dudarev PLUS_U** (`plus_u`).

Implementation (product): multi-k UKS OT+U uses a **CP2K-style OUTER_SCF** —
rebuild \(H_k^\sigma\) (including \(V_U^\sigma\)) from the k-summed spin
densities, then **fixed-H OT** band minimize on each local \((k,\sigma)\)
(`ot_minimize_fixed_k`). Same fixed point as multi-k UKS DIIS+U.

Product API: concrete `cpu::gpw::GpwDft` members `use_ot`, `n_alpha`/`n_beta`,
`k_mesh` / `k_points`, `plus_u`. No public `*Config`, no factories.

**Prior product pins:** multi-k UKS OT (`reg_ot-uks-kpoints`), multi-k RKS
OT+U (`reg_ot-dftu-kpoints`), Γ-UKS OT+U (`reg_ot-dftu`).

## System (regression)

| Item | Value |
| ---- | ----- |
| Molecule | high-spin H₂ (DZV-GTH-PADE / GTH-PADE-q1) |
| Cell | **8 Bohr** cubic, periodic |
| Bond | ±0.7 Bohr from cell centre along *x* |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) |
| XC | PADE LSD |
| Cutoff | **100 Ry**, single grid |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| k (A1) | explicit Γ |
| k (A2) | explicit κ = (¼,0,0), w=1 |
| k (A3) | MP 2×1×1, time-reversal |

## Offline CP2K oracle

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, `/nobackup/.../work/cp2k/install`)
- **Input:** `tools/cp2k_oracle/inputs/ot-uks-dftu/h2_uks_ot_dftu_kpoints.inp`
- **OT+k:** CP2K 2026.1 **aborts** with
  `OT not possible with kpoint calculations` (`qs_scf_initialization.F`).
  Absolute CP2K OT+k+UKS digits **do not exist** on this oracle version.
- **DIIS multi-k UKS PLUS_U twin:** same geometry / U / spin / k-mesh with
  DIAGONALIZATION (input above). SCF path is the fair fixed-point twin of
  ATV multi-k UKS OT+U (not a wall twin for OT).
- **Pin strategy:** internal exactness (OT ≡ DIIS multi-k UKS with U; A1–A2)
  is **decisive**. First-green ATV absolute for A3.

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| -------- | ----: |
| A1 k-Γ OT+U / DIIS+U | −0.7802796 / −0.7802792 Ha (\|Δ\| ≈ 4×10⁻⁷) |
| A2 OT+U (κ=¼) | **−0.802825557 Ha** (32 outer OT) |
| A2 DIIS+U same | −0.802825142 Ha (66 DIIS) |
| \|OT − DIIS\| A2 | ≈ 4.1×10⁻⁷ Ha |
| A3 E(U=0, OT MP2×1×1) | −0.706134617 Ha |
| A3 E(U=4, OT MP2×1×1) | **−0.802825557 Ha** (pin) |
| A3 e_dftu | ≈ −0.1803 Ha |
| A3 \|E(U)−E(0)\| | ≈ 0.0967 Ha |
| ATV wall (A1–A3, OMP=8) | **0.48 s** Release |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | multi-k DIIS+U at Γ ≡ multi-k OT+U at Γ (UKS) | 1e-4 Ha |
| A1 | e_dftu non-trivial; OT vs DIIS e_dftu | 1e-3 Ha |
| A2 | multi-k OT+U ≡ multi-k DIIS+U (κ=¼) | 1e-4 Ha |
| A3 | MP 2×1×1 OT+U converges | — |
| A3 | \|E(U)−E(0)\| activation | ≥ 1e-4 Ha |
| A3 | absolute vs frozen first-green | 1e-3 Ha |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | CP2K 2026.1 has no OT+kpoint path (hard abort); DIIS multi-k twin is not a fair OT wall comparison. Decisive gates are internal OT≡DIIS identities (same pattern as `ot-dftu-kpoints` / `ot-uks-kpoints`). |
| r = T_atv/T_cp2k | n/a |
| `within_10x` | n/a (exempt) |
| ATV wall (A1–A3) | ≈ 0.5 s Release |

**ctest never launches CP2K.**

## Citations

- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\).
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.

## Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- A3 absolute: −0.802825557373 → **−0.986384296380 Ha** (OT ≡ DIIS parity 1.7e-7 Ha).
