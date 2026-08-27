# lammps-comm-brick-05-presets

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image sum is recomputed inside the regression)
perf_exempt: yes — this slug makes **no** performance claim.

## Invariant under test

`cpu::Evaluate` + `PairLJCut` + `NeighborList` under MPI domain
decomposition reproduces the flat O(N²) minimum-image LJ sum. Halo /
exchange / reverse live on `ctx.md.comm`; the compiled Plan contains
`reduce` and does not contain `halo_send` / `halo_recv` / `migrate` /
`halo_refresh_*`.

\[
\bigl|E_{\mathrm{dd}} - E_{\mathrm{brute}}\bigr| / |E_{\mathrm{brute}}|
\le 10^{-9}.
\]

- S. Plimpton, J. Comput. Phys. **117**, 1–19 (1995),
  DOI 10.1006/jcph.1995.1039
- A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171
- A. Rahman, Phys. Rev. **136**, A405 (1964),
  DOI 10.1103/PhysRev.136.A405 — LJ argon parameters
