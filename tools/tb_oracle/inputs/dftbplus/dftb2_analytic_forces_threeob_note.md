# Offline pin note — `dftb2-analytic-forces-threeob`

**Date:** 2026-07-27  
**Slug:** dftb2-analytic-forces-threeob  
**oracle:** dftb+ (force **module boundary** / literature + 3ob-3-1 SK)  
**oracle_version:** formula-level (Hourahine et al. 2020; Elstner 1998; Gaus
  2013 3ob-3-1); no DFTB+ binary required for the FD identity gate

## Why not a bit-identical DFTB+ force vector?

1. Absolute force identity vs DFTB+ + 3ob-3-1 is optional product depth; this
   slug closes the **module** gap (closed-form F on ThreeOb, not re-SCC FD).
2. Related absolute energy pin already exists: `dftb2-3ob-organic` (CH4,
   DFTB+ 24.1, \(E = -87.868568\) eV).
3. No guaranteed `dftb+` Lmod module on every pin host.

## What is pinned

- **Identity:** analytic nuclear gradient of ATV’s SCC-DFTB2 **ThreeOb** energy
  equals central FD of that energy (re-SCC, energy-only) on off-eq H2O within
  regression tolerance (1.0 eV/Å on gated components).
- **Formula source:** DFTB+ / Elstner force structure:
  \(\mathrm{Tr}(P\partial H_0)-\mathrm{Tr}(W\partial S)\) + fixed-\(\Delta q\)
  SCC \(\gamma\) geometry terms + \(\partial E_{\mathrm{rep}}\) (3ob spline).

## Geometry (H2O, Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Box: cubic 20 Å (isolated molecule).

## Citations

- Elstner et al., Phys. Rev. B **58**, 7260 (1998).
- Hourahine et al., J. Chem. Phys. **152**, 124101 (2020) (DFTB+).
- Gaus et al., J. Chem. Theory Comput. **9**, 338 (2013) (3ob-3-1).
