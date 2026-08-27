# Reference — LAMMPS CommBrick halo table (Halo → Halo)

oracle: literature
oracle_version: n/a (closed-form hop counts; no oracle binary is run)
perf_exempt: yes — this slug makes **no** performance claim.

## Invariant under test

Orthogonal brick decomposition, periodic axis \(d\), uniform partition
(Å):

\[
h_d = \left\lceil r_{\mathrm{shell}} \cdot N_d / L_d \right\rceil
\]

Hard-coded: \(L = 15.78\) Å, \(r = 6.6\) Å, \(N_x = 2,4,8\) → hops
\(1,2,4\). Equal-slab (\(r = L/N\)) stays one hop. `sendself` on an
undecomposed axis is a skip (`nrecv == 0`), not a local memcpy. The
minimum-image bound \(r \le L_d - w_{\mathrm{self}}\) refuses with
`std::runtime_error`. `comm::Halo` is the MPI-free swap table, not a
`Component`.

- S. Plimpton, J. Comput. Phys. **117**, 1–19 (1995),
  DOI 10.1006/jcph.1995.1039
- A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171
- LAMMPS `78a7ddf` `src/comm_brick.cpp` sendlist
