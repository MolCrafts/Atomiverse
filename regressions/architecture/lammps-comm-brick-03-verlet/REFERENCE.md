# lammps-comm-brick-03-verlet

oracle: literature
oracle_version: n/a (isolated atom; no pair interaction)
perf_exempt: yes — this slug makes **no** performance claim.

## Invariant under test

One atom in a \(10\) Å cubic box has no pair partners inside any cutoff
smaller than the box. An MD run of 10 Verlet steps therefore leaves

\[
E = 0~\mathrm{eV},\qquad n_{\mathrm{local}} = 1.
\]

`Runtime::step_verlet_` dispatches `ctx.md.comm` (LAMMPS `Verlet::run`
nflag XOR + newton-on reverse). No third-party binary.

- S. Plimpton, J. Comput. Phys. **117**, 1–19 (1995),
  DOI 10.1006/jcph.1995.1039
- A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171
- LAMMPS `78a7ddf` `src/verlet.cpp`
