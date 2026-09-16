# Reference — multi-k UKS GPW + Dudarev DFT+U (PLUS_U + Monkhorst–Pack)

## Method

Unrestricted Kohn–Sham GPW with **Monkhorst–Pack k-point sampling** and
**Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken + `&KPOINTS`):

\[
n^\sigma=\sum_k w_k\,\tfrac12\mathrm{Herm}\big[(P_k^\sigma S_k)+(S_k P_k^\sigma)\big],
\quad
E_U^\sigma=\tfrac{U}{2}\big[\mathrm{Tr}\,n^\sigma-\mathrm{Tr}(n^\sigma n^\sigma)\big],
\quad
V^\sigma=U\big(\tfrac12 I-n^\sigma\big)
\]

added to every \(H_k^\sigma\). Builds on Γ DFT+U (`gpw-dftu-nio` /
`gpw-dftu-stress`) and multi-k UKS (`gpw-kpoints-uks`).

Public API: concrete `cpu::gpw::GpwDft` members `k_mesh` / `k_points`,
`n_alpha`/`n_beta`, `plus_u` (`PlusU{element,l,u_ev}`). No public `*Config`,
no factories.

## Regression system (fast dual-gate)

High-spin H₂ toy (DZV-GTH-PADE) — identity suite + absolute ATV pin:

| Item | Value |
|------|--------|
| Cell | 8 × 8 × 8 Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) |
| XC | PADE LSD |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| Cutoff | 100 Ry (A1) / 120 Ry (A2–A3) |
| k mesh (A2–A3) | MONKHORST-PACK 2×2×2, time-reversal → 4 k |

### Identity gates (decisive)

| Gate | Result (2026-07-26) | Tol |
|------|---------------------|-----|
| A1 Γ-limit (k=Γ complex + U vs real Γ-UKS + U) | \|ΔE\| ≈ 4e-7 Ha | 1e-6 Ha |
| A1 e_dftu decomposition | \|Δ\| ≲ 1e-4 Ha | 1e-4 Ha |
| A2 activation \|E(U=4 eV) − E(U=0)\| | ≈ 9.5e-2 Ha | ≥ 1e-4 Ha |
| A3 absolute E(MP+U) | −0.814170782662 Ha | 1e-3 Ha |

## Offline CP2K twin (pathology note)

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, local install under
  `/nobackup/.../work/cp2k/install`).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-kpoints/h2_uks_dftu_kpoints.inp`
- **System:** same H₂ geometry / DZV / PADE / U=4 eV / MP 2×2×2.
- **Date:** 2026-07-26
- **Result:** SCF converged in 42 steps but CP2K reports **unphysical Mulliken
  charges** and a pathologically large negative DFT+U term
  (\(E_U\approx-159\) Ha, \(E_{\mathrm{tot}}\approx-158\) Ha). Not used as an
  absolute energy twin for the regression. Wall ≈ **9.4 s** (OMP=8).
- **Product twin path:** reuse Γ-UKS PLUS_U NiO molecule pin
  (`reg_gpw-dftu-nio`, CP2K E = −184.92466 Ha) as the literature-facing
  PLUS_U number; multi-k capability is proven by A1–A3 identities above.
  A fair multi-k NiO solid twin remains follow-on (SZV solid AFM needs
  k-points + larger basis; see `gpw-dftu-nio` REFERENCE).

```
  *** SCF run converged in    42 steps ***
  DFT+U energy:                                              -158.76078199959053
  Total energy:                                              -157.66686993943307
 *** WARNING … DFT+U energy contribution is negative ***
 *** possibly due to unphysical Mulliken charges! ***
```

## Atomiverse absolute pin (first green)

| Quantity | Value |
|----------|--------|
| **E(MP 2×2×2 TR + U=4 eV, 120 Ry)** | **−0.814170782662 Ha** |
| E(MP, U=0) | −0.719059673672 Ha |
| e_dftu | ≈ −0.177 Ha (multi-function s shell; can be <0) |
| Tol | **1e-3 Ha** |
| Wall (A1–A3, OMP default) | ≈ **49 s** |

## Perf gate

| Side | Wall | Notes |
|------|------|--------|
| ATV regression (A1–A3) | ≈ 49 s | Release, identity suite |
| CP2K H₂ twin | ≈ 9.4 s | **not a fair energy twin** (unphysical Mulliken) |
| `perf_exempt` | **true** | No fair absolute CP2K energy twin for this toy; decisive gates are internal identities (same pattern as `gpw-kpoints-uks`). Ratio n/a. |

## Citations

- V. I. Anisimov et al., *Phys. Rev. B* **44**, 943 (1991) — DFT+U.
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified U.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH.

## Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- E(MP 2×2×2 TR + U=4 eV, 120 Ry): −0.814170782662 → **−0.997491857710 Ha** (A1 Γ-limit identity now 1.9e-8 Ha; DIIS budget 400).
