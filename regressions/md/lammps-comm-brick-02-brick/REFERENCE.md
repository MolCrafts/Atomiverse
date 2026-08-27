# Reference — LAMMPS CommBrick communicator

oracle: literature
oracle_version: n/a (closed-form 8-atom brick counts; no oracle binary is run)
perf_exempt: yes — this slug makes **no** performance claim.

## Invariant under test

8 atoms on a cubic box \(L = 20\) Å, ghost shell \(6\) Å, obtained through
`cpu::Comm*` published by `DomainRuntime` (never `new CommBrick`, never
`MD.run`).

- n1 (`mpi_size == 1`): \(n_{\mathrm{ghost}} = 0\)
- n2 (`ATV_FORCE_DIMS=2,1,1`): \(n_{\mathrm{ghost}} = 8\) per rank;
  reverse \(\sum f_x = 8\) eV/Å per rank; exchange conserves 8 owners.

- S. Plimpton, J. Comput. Phys. **117**, 1–19 (1995),
  DOI 10.1006/jcph.1995.1039
- A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171
- LAMMPS `78a7ddf` `src/comm.h` / `src/comm_brick.cpp`
