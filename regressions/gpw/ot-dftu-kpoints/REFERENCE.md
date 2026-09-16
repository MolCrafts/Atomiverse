# Reference — multi-k GPW OT + Dudarev DFT+U

## Method

Closed-shell **GPW** Kohn–Sham DFT with **Orbital Transformation (OT)** SCF
on the **complex multi-k** path (`use_ot = true` + `k_mesh` / explicit
`k_points`) combined with **Dudarev PLUS_U** (`plus_u`).

Implementation (product): multi-k OT+U uses a **CP2K-style OUTER_SCF** —
rebuild \(H_k\) (including \(V_U\)) from the k-summed density, then
**fixed-H OT** band minimize on each local \(k\) (`ot_minimize_fixed_k`).
This reaches the same fixed point as multi-k DIIS+U (shared
`build_k_energy`). The one-step-per-outer Armijo path remains for multi-k
OT without PLUS_U.

Product API: concrete `cpu::gpw::GpwDft` members `use_ot`, `k_mesh` /
`k_points`, `plus_u`. No public `*Config`, no factories.

**Out of scope:** multi-k UKS OT (+U) — `ot-uks-kpoints`.

## System (regression)

| Item | Value |
| ---- | ----- |
| Molecule | H₂ (DZV-GTH-PADE / GTH-PADE-q1) |
| Cell | **8 Bohr** cubic, periodic |
| Bond | ±0.7 Bohr from cell centre along *x* |
| XC | PADE |
| Cutoff | **100 Ry**, single grid |
| n_occ | 1 (closed-shell RKS) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| k (A1) | explicit Γ |
| k (A2) | explicit κ = (¼,0,0), w=1 |
| k (A3) | MP 2×2×2, time-reversal → 4 k |

## Offline CP2K oracle

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, `/nobackup/.../work/cp2k/install`)
- **Input:** `tools/cp2k_oracle/inputs/ot-dftu-kpoints/h2_ot_dftu_kpoints.inp`
- **OT+k:** CP2K 2026.1 **aborts** with
  `OT not possible with kpoint calculations` (`qs_scf_initialization.F`).
  Absolute CP2K OT+k digits **do not exist** on this oracle version.
- **DIIS multi-k PLUS_U twin:** same geometry / U / MP 2×2×2 with
  DIAGONALIZATION (input above). SCF path is the fair fixed-point twin of
  ATV OT+U (not a wall twin for OT). Recorded when SCF converges offline.
- **Pin strategy:** internal exactness (OT ≡ DIIS multi-k with U; A1–A2)
  is **decisive**. Frozen first-green ATV absolute for A3.

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| -------- | ----: |
| A1 k-Γ OT+U / DIIS+U | −1.0867759 / −1.0867757 Ha (\|Δ\| ≈ 2×10⁻⁷) |
| A2 OT+U (κ=¼) | **−1.081998145 Ha** (13 outer OT) |
| A2 DIIS+U same | −1.081997919 Ha (42 DIIS) |
| \|OT − DIIS\| A2 | ≈ 2.3×10⁻⁷ Ha |
| A3 E(U=0, OT MP2³) | −1.102850837 Ha |
| A3 E(U=4, OT MP2³) | **−1.077477969 Ha** (pin) |
| A3 e_dftu | ≈ −0.0023 Ha |
| A3 \|E(U)−E(0)\| | ≈ 0.0254 Ha |

## Gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | multi-k DIIS+U at Γ ≡ multi-k OT+U at Γ | 1e-4 Ha |
| A1 | e_dftu non-trivial; OT vs DIIS e_dftu | 1e-3 Ha |
| A2 | multi-k OT+U ≡ multi-k DIIS+U (κ=¼) | 1e-4 Ha |
| A3 | MP 2×2×2 OT+U converges; n_kpts=4 | — |
| A3 | \|E(U)−E(0)\| activation | ≥ 1e-4 Ha |
| A3 | absolute vs frozen first-green | 1e-3 Ha |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | CP2K 2026.1 has no OT+kpoint path (hard abort); DIIS multi-k twin is not a fair OT wall comparison. Decisive gates are internal OT≡DIIS identities (same pattern as `ot-kpoints`). |
| r = T_atv/T_cp2k | n/a |
| `within_10x` | n/a (exempt) |
| ATV wall (A1–A3) | ≈ 15–25 s Release |

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

- A3 absolute: −1.077477969428 → **−1.396485808593 Ha** (OT ≡ DIIS parity 1.7e-7 Ha).
