# lammps-comm-brick-04-packer

oracle: literature
oracle_version: n/a (hard-coded packer goldens; no third-party process)
perf_exempt: yes — this slug makes **no** performance claim.

## Invariant under test

A cubic box of edge \(L = 10\) Å and a sendlist row at \(x = 9\) Å with
periodic image flag \(p_x = +1\) pack as

\[
x_{\mathrm{XSHAKE}} = x + p_x L = 19.0~\mathrm{Å}.
\]

Velocities and charges are not displacements: \(v_x = 0.4\) Å/fs packs
\(0.4\) Å/fs and \(q = 0.25\,e\) packs \(0.25\,e\) even with the same
\(p_x = +1\).

Public headers + link `atomiverse`. No MD closed loop.

- H. C. Andersen, J. Comput. Phys. **52**, 24–34 (1983),
  DOI 10.1016/0021-9991(83)90014-1
- L. J. V. Ahrens-Iwers et al., J. Chem. Phys. **157**, 084801 (2022),
  DOI 10.1063/5.0100191
- LAMMPS `78a7ddf` `src/RIGID/fix_rattle.cpp:652-678`
