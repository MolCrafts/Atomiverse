# Pin notes — extended-lagrangian-dipoles

**Date:** 2026-07-27
**Slug:** `extended-lagrangian-dipoles` (wave 5)
**Oracle:** metalwalls (map §2 "XL dipoles"); **no MetalWalls binary on the
agent host** → literature-form pins with full citations, `perf_exempt`.

MetalWalls' own mass-zero dipole route is MaZe (Coretti, Bonella & Ciccotti,
J. Chem. Phys. **149**, 191102 (2018)); the landed component is the adiabatic
extended-Lagrangian iEL/SCF scheme (Albaugh, Demerdash & Head-Gordon,
J. Chem. Phys. **143**, 174104 (2015)) built on the Niklasson dissipative
recursion (Niklasson et al., J. Chem. Phys. **130**, 214109 (2009), Table I;
time-reversible base scheme: Niklasson, Phys. Rev. Lett. **100**, 123004
(2008)) — the same foothold shape as the shipped `ConstPotMaZe` (CG+ASPC) for
electrode charges.

## Procedure

1. **Stationary fixed-point pin (hard numbers).** The XL recursion's fixed
   point for a stationary geometry is the SCF PIM solution, so the two-site
   Thole/Applequist closed form pins the XL component's cold *and* warm
   launches. Geometry and numbers are shared with
   `regressions/md/pim-induced-dipole-bulk/REFERENCE.md`
   (q = ±1 e, α = 1.0/1.5 Å³, r = 3 Å, thole_a = 0.39, L = 50 Å):
   E_pol = **−0.12040946414707571 eV**,
   F_x = **±0.040341072868572643 eV/Å**,
   μ_A = 0.082882545215708481 e·Å, μ_B = 0.12342697961661356 e·Å.
   Re-derived independently (no Atomiverse code) by
   `inputs/metalwalls/extended-lagrangian-dipoles/pin_two_site.py` —
   agreement to ~1e-14 relative (closed-form linear solve vs SCC).
2. **Dynamics gates (defining properties of the scheme, from the iEL/SCF
   paper):** on a bulk polarizable 8-ion toy (velocity-Verlet, 150 × 0.5 fs),
   (a) the XL trajectory's dipoles track a cold-start SCF Born–Oppenheimer
   twin to < 1e-6 e·Å; (b) the warm-started SCC costs strictly fewer total
   iterations than the cold twin; (c) KE + E_pol is conserved
   (|drift| < 5e-6 eV).

## Perf

`perf_exempt` — no MetalWalls binary on PATH for a fair mass-zero/XL dipole
wall twin (LAMMPS has no iEL/SCF induced-dipole twin either; its polarizable
paths are Drude/QEq). When an MW twin appears, record T_atv/T_MW in the
regression REFERENCE and require ≤ 10.
