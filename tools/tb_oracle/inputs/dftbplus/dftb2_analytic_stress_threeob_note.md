# Offline pin note — `dftb2-analytic-stress-threeob`

**Date:** 2026-07-29
**Slug:** dftb2-analytic-stress-threeob
**oracle:** dftb+ (stress/virial **module boundary** / literature)
**oracle_version:** 24.1 — formula-level; **no DFTB+ binary** on this host
(`module spider dftb` → nothing), same situation as the wave-3
`dftb2-analytic-stress` note.

## Why not a bit-identical DFTB+ σ?

1. No `dftb+` executable or Lmod module exists on the pin host, so an absolute
   σ cannot be produced. Family-locked oracle rules forbid substituting
   OpenMOPAC/xtb for a DFTB number.
2. The SK tables *are* production DFTB.org 3ob-3-1 (unlike the MioSketch case,
   where the tables themselves are an in-tree sketch), so an absolute pin is
   **achievable** — it needs only a host with DFTB+. See "How to upgrade".

## What is pinned instead

- **Identity 1 (strain-FD):** the closed-form ThreeOb pair virial equals the
  multi-SCC affine strain-FD of the *same* converged 3ob energy,
  σ = −(1/Ω) ∂E/∂ε, Voigt (xx,yy,zz,xy,xz,yz) eV/Å³, on asymmetric CH₄ —
  **all six components**, residual ≤ 5e-13 eV/Å³ vs a 1e-8 gate.
- **Identity 2 (isotropic trace):** tr(σ)/3 = −dE/dΩ from an isotropic-scaling
  central difference (Ω(ε) = (1+ε)³Ω₀ ⇒ dΩ/dε|₀ = 3Ω₀).
- **Fossil guard:** the restricted-MioSketch virial is byte-stable across this
  change (`tests/cpu/semiempirical/test_dftb2.cpp`
  `DFTB2.MioAnalyticStressFossilGuard`, 1e-12 eV/Å³).

## Formula sources

- Elstner et al., Phys. Rev. B **58**, 7260 (1998) — SCC-DFTB2 energy.
- Gaus, Cui, Elstner, J. Chem. Theory Comput. **9**, 338 (2013) — 3ob-3-1.
- Nielsen & Martin, Phys. Rev. B **32**, 3780 (1985) — quantum stress theorem
  (mechanical convention σ = −(1/Ω) ∂E/∂ε).
- Hourahine et al., J. Chem. Phys. **152**, 124101 (2020) — DFTB+ (24.1) stress
  module boundary.

## Geometry (asymmetric CH₄, Å)

C at origin; H along ±tetrahedral directions with four distinct radii
1.09540 / 1.12714 / 1.06364 / 1.10598 Å. Full coordinate table and the
grid-knot placement argument (3ob SK grid = 0.02 Bohr, linear interpolation) are
in `regressions/semiempirical/dftb2-analytic-stress-threeob/REFERENCE.md`.

Cell: cubic 20 Å. `scc_tol = 1e-10`, `max_scc = 80`.

## How to upgrade this to an absolute DFTB+ pin

Reuse `tools/tb_oracle/inputs/dftbplus/ch4_3ob_dftb2/` (the CH₄/3ob DFTB2 twin
already used for the `dftb2-3ob-organic` **energy** pin), swap `geo.gen` for the
asymmetric geometry above, and add:

```hsd
Analysis   { CalculateForces = Yes }
Options    { WriteResultsTag = Yes; WriteDetailedOut = Yes }
```

DFTB+ reports the stress tensor in `detailed.out` / `results.tag` in Ha/Bohr³;
convert to eV/Å³ with `kHa2eV / kBohr2A³`. A molecular (non-periodic) DFTB+ run
does **not** emit a stress — the twin must be run in the same 20 Å cubic
`Periodic = Yes` cell as ATV, and ATV's ThreeOb path being a molecular cluster
sum (no lattice images) is then itself the discrepancy to quantify (Gap 2 in
REFERENCE.md).
