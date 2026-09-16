# Offline pin note — `dftb2-spin-analytic-forces`

**Date:** 2026-07-27  
**Slug:** dftb2-spin-analytic-forces  
**oracle:** dftb+ (collinear spin force **module boundary** / literature)  
**oracle_version:** formula-level (Hourahine et al. 2020; Elstner 1998;
Köhler collinear spin-DFTB); no DFTB+ binary on Arrhenius login modules at pin time

## Why not a bit-identical DFTB+ number?

1. ATV `basis_data/dftb_mio_tables.h` is an in-tree **mio-style sketch** (H/C/N/O
   hoppings / Hubbard / spin_W / exponential repulsive), **not** a DFTB.org
   mio-1-1 `.skf` dump with production spin constants.
2. No `dftb+` Lmod module on this host at wave-2 pin time.
3. Absolute E/F identity vs DFTB+ + mio-1-1 is a different hygiene track
   (`dftb2-dftbplus-energy-pin`).

## What is pinned

- **Identity:** analytic nuclear gradient of ATV’s **collinear unrestricted**
  SCC-DFTB2 energy equals central FD of that energy (re-SCC, energy-only) on
  off-eq triplet O₂ within regression tolerance.
- **Formula source:** DFTB+ / Elstner / Köhler force structure for open shell:
  \(\sum_\sigma\big[\mathrm{Tr}(P^\sigma\partial H_0)-\mathrm{Tr}(W^\sigma\partial S)
  +\mathrm{Tr}(P^\sigma\varepsilon^\sigma\cdot\partial S)\big]\)
  + fixed-\(\Delta q\) SCC \(\gamma\) geometry terms + \(\partial E_{\mathrm{rep}}\).
  Spin \(W_A\) is onsite constant → no extra \(\partial_R\) term at fixed \(m\).

## Geometry (O₂ triplet, Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| O | 8 | 1.30 | 0.00 | 0.00 |

Box: cubic 30 Å. Multiplicity 3, charge 0.

## Citations

- Elstner et al., Phys. Rev. B **58**, 7260 (1998).
- Köhler, Frauenheim, et al. — collinear spin-DFTB channel (DFTB+ spin module).
- Hourahine et al., J. Chem. Phys. **152**, 124101 (2020) (DFTB+).
