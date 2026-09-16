# Reference — multi-k collinear UKS GPW

## Method

Unrestricted Kohn–Sham GPW with **Monkhorst–Pack k-point sampling** and
collinear spin (CP2K `&KPOINTS` + `LSD`): dual complex Bloch densities
\(P^\alpha_k\), \(P^\beta_k\), spin-resolved XC, dual \(H_k^\sigma\), fixed
integer fillings. Builds on `gpw-kpoints-si` (RKS k path) and Γ-UKS
(`gpw-uks-pbe` / `gpw-uks-o2-gamma`).

Public API: concrete `cpu::gpw::GpwDft` with members `n_alpha`/`n_beta`,
`k_mesh` / `k_points`+`k_point_weights`, `k_time_reversal`, `XcKind`.
No public `*Config`, no factories.

## Regression system (fast dual-gate)

High-spin H₂ toy (compact DZV-GTH-PADE basis) — same class of internal
exactness identities as `gpw-kpoints-si`, extended to UKS:

| Item | Value |
|------|--------|
| Cell | 8 × 8 × 8 Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) (high-spin toy) |
| XC | PADE LSD |
| Cutoff (main pin) | 120 Ry |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal → 4 k |

### Identity gates (decisive; independent of oracle)

| Gate | Result (2026-07-25) | Tol |
|------|---------------------|-----|
| A1 Γ-limit (k=Γ complex vs real Γ-UKS) | \|Δ\| ≈ 4e-11 Ha | 1e-6 |
| A2 closed-shell UKS ≡ RKS multi-k | \|Δ\| ≈ 0 | 5e-6 |
| A3 time-reversal 8 k vs 4 k | \|Δ\| ≈ 0 | 5e-6 |
| A4 activation \|E_MP − E_Γ\| | 2.79e-2 Ha | ≥ 1e-4 |

### Absolute pin (first-green ATV)

| Quantity | Value |
|----------|--------|
| **E(MP 2×2×2 TR, 120 Ry)** | **−0.719059673672 Ha** |
| E(Γ, 120 Ry) | −0.691163126674 Ha |
| Tol | 1e-3 Ha |

## Offline CP2K product twin (O₂, documentation)

Product-shaped twin for the capability (not the H₂ wall budget):

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, local install under
  `…/work/cp2k/install`; libcp2k 2026.1).
- **Input:** `tools/cp2k_oracle/inputs/gpw-kpoints-uks/o2_uks_kpoints.inp`
- **System:** O₂ R=1.2075 Å, cubic **8 Å** periodic, DZVP-MOLOPT-GTH +
  GTH-PBE-q6, **UKS PBE**, multiplicity 3 (Nα=7, Nβ=5), CUTOFF 200 Ry,
  NGRIDS 1, MONKHORST-PACK 2 2 2, Broyden α=0.3.
- **Date:** 2026-07-25
- **Result:** SCF converged in 22 steps;  
  **E_tot = −31.92842018097085 Ha**  
  Wall (1 rank × OMP=4): **≈ 175 s** (`CP2K` timer MAXIMUM).

```
  *** SCF run converged in    22 steps ***
  Total energy:                                               -31.92842018097085
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928420180970853
```

ATV O₂ MOLOPT multi-k is deliberately **not** in the ctest regression (wall
exceeds 60 s class on dual Γ+MP). The H₂ identities prove the k-UKS path;
the O₂ CP2K number is the offline product pin for future absolute re-twins.

## Perf gate

| Side | Wall | Notes |
|------|------|--------|
| ATV regression (A1–A5, OMP=8) | **~4.4 s** | Release |
| CP2K O₂ twin | ~175 s | product geometry, not H₂ |
| `perf_exempt` | **true** | Fair twin wall would need ATV O₂ MOLOPT multi-k product run (not the H₂ identity suite); H₂ has no CP2K twin. |
| ratio | n/a (exempt) | |

When an ATV O₂ product wall is recorded offline against the same input,
replace with measured \(r \le 10\).

## Citations

- Monkhorst & Pack, *Phys. Rev. B* **13**, 5188 (1976).
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- Spec: `.claude/specs/gpw-kpoints-uks/` (deleted on harness close).
