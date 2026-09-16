# Offline pin note — `dftb2-analytic-stress`

**Date:** 2026-07-27  
**Slug:** dftb2-analytic-stress  
**oracle:** dftb+ (stress **module boundary** / literature)  
**oracle_version:** formula-level (Hourahine et al. 2020; Elstner 1998); no DFTB+ binary on Arrhenius login modules at pin time

## Why not a bit-identical DFTB+ number?

1. ATV `basis_data/dftb_mio_tables.h` is an in-tree **mio-style sketch** (H/C/N/O
   hoppings / Hubbard / exponential repulsive), **not** a DFTB.org mio-1-1
   `.skf` dump. Absolute \(E\)/\(\sigma\) vs DFTB+ + mio-1-1 is a different slug.
2. No `dftb+` / DFTB+ Lmod module on this host at wave-2 pin time.

## What is pinned

- **Identity:** analytic intensive cell stress of ATV’s restricted MioSketch
  SCC-DFTB2 equals multi-SCC **affine strain-FD** of that energy
  (\(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\), Voigt eV/Å³) on off-eq
  H2O within regression tolerance.
- **Formula source:** DFTB+ / Elstner stress structure — pair virial of the
  same HF+Pulay+γ+repulsive terms as nuclear forces under affine nuclear+cell
  strain (same mechanical convention as ATV PW/GPW `kStress`).

## Geometry (H2O, Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Box: cubic 20 Å (isolated molecule; min-image only).

## Citations

- Elstner et al., Phys. Rev. B **58**, 7260 (1998).
- Hourahine et al., J. Chem. Phys. **152**, 124101 (2020) (DFTB+).
