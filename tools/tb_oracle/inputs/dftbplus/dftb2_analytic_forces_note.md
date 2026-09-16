# Offline pin note — `dftb2-analytic-forces`

**Date:** 2026-07-27  
**Slug:** dftb2-analytic-forces  
**oracle:** dftb+ (force **module boundary** / literature)  
**oracle_version:** formula-level (Hourahine et al. 2020; Elstner 1998); no DFTB+ binary on Arrhenius login modules at pin time

## Why not a bit-identical DFTB+ number?

1. ATV `basis_data/dftb_mio_tables.h` is an in-tree **mio-style sketch** (H/C/N/O
   hoppings / Hubbard / exponential repulsive), **not** a DFTB.org mio-1-1
   `.skf` dump. Absolute \(E\)/\(F\) vs DFTB+ + mio-1-1 is a different slug
   (`dftb2-dftbplus-energy-pin` / `dftb2-3ob-organic`).
2. No `dftb+` / DFTB+ Lmod module on this host at wave-1 pin time.

## What is pinned

- **Identity:** analytic nuclear gradient of ATV’s SCC-DFTB2 energy equals
  central FD of that energy (re-SCC, energy-only) on off-eq H2O within
  regression tolerance.
- **Formula source:** DFTB+ / Elstner force structure:
  \(\mathrm{Tr}(P\partial H_0)-\mathrm{Tr}(W\partial S)\) + fixed-\(\Delta q\) SCC
  \(\gamma\) geometry terms + \(\partial E_{\mathrm{rep}}\).

## Geometry (H2O, Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Box: cubic 20 Å (isolated molecule; no PBC images in practice).

## Citations

- Elstner et al., Phys. Rev. B **58**, 7260 (1998).
- Hourahine et al., J. Chem. Phys. **152**, 124101 (2020) (DFTB+).
