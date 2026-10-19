# Reference — LAMMPS CommBrick periodic halo table

oracle: literature
oracle_version: n/a (source-defined closed-form counts; no oracle binary is run)
perf_exempt: yes — this regression makes no performance claim.

## Invariants under test

All axes use explicit periodic images, including axes with one process. Domain
bounds and halo slab bands use fractional lattice coordinates. On a uniform
partition, LAMMPS `CommBrick::setup` computes

\[
h_d = \left\lfloor r_{\mathrm{shell}} N_d / H_d \right\rfloor + 1,
\]

where \(H_d\) is the perpendicular face height (equal to \(L_d\) for an
orthorhombic box). The extra round includes atoms exactly on a slab edge.

Hard-coded orthorhombic cases:

- \(L=15.78\) Å, \(r=6.6\) Å, \(N_x=2,4,8\): hops **1, 2, 4**.
- Exact slab width \(r=L/4\): **2** hops.
- One rank: all **6** face swaps are active local copies, carrying lattice
  image flags +1 at the low face and −1 at the high face.
- Four processes along x: shells **12 Å** and **20 Å** require **4** and
  **6** hops respectively, including complete periodic rings. Transport
  does not impose a minimum-image cutoff restriction.

`Halo::setup` constructs geometry; it does not perform copies or populate
received-row counts. The communication regression in the neighboring
`lammps-comm-brick-02-brick` case checks actual transport.

## Sources

- LAMMPS `src/comm_brick.cpp`, `CommBrick::setup`, uniform `maxneed` and
  periodic image flags, checkout
  `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.
- S. Plimpton, J. Comput. Phys. **117**, 1–19 (1995),
  DOI [10.1006/jcph.1995.1039](https://doi.org/10.1006/jcph.1995.1039).
- A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
  DOI [10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).
