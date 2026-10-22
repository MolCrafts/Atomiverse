# Many-body classical potentials (EAM, Stillinger–Weber)

Classical force styles whose energy does not decompose into independent pairs:
the analytic embedded-atom method (Sutton–Chen, a long-range Finnis–Sinclair
form) and the Stillinger–Weber two- plus three-body potential. Derivations and
exercises live in [Many-Body Potentials](../ch16-many-body-potentials.md);
backend availability is the [force-field roadmap](../../roadmap/force-fields.md).

Both styles are CPU components mounted in a force slot and parameterized by
lookup constructors that throw on an untabulated `(element, name)` pair:

```cpp
cpu::Evaluate eval;
eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
eval.on(evaluate::Phase::evaluate)
    .add(cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"}, cutoff});
// or: .add(cpu::StillingerWeber{cpu::SWPotential{molrs::Element::Si, "stillinger-weber"}});
eval.build();
eval.run(frame);                       // writes kEnergy [eV]
```

Case: `ctest -R '^md/manybody-potentials$'` — sources and oracle record in
[`regressions/md/manybody-potentials/`](../../../regressions/md/manybody-potentials/).
One section of the case per `##` heading below:
`atv_regressions --case md/manybody-potentials --section <name>`. The oracle of
both sections is `literature`: the fit targets of the published
parameterizations. Forces are gated against finite differences of the same
energies by the unit tests, not by the case.

## Scope

- **EAM** is single-element analytic Sutton–Chen. Density contributions
  accumulate onto both endpoints of a half list; reverse communication sums
  ghost densities onto their owners, and forward communication returns the
  embedding derivative to ghosts before the force pass, so the coordinate halo
  needs one cutoff and no ghost-centred rows. The style runs under MPI.
- **Stillinger–Weber** needs every neighbour of a centre for the angular term,
  so it requests a full neighbour list and selects each two-body pair once by
  tag. The style runs under MPI.
- No CUDA counterpart of either style; no Tersoff, MEAM, ReaxFF or AIREBO.
- No alloy cross terms and no tabulated-file readers: parameters come only from
  the in-tree tables.

## eam_cu — EAM copper: lattice constant and cohesive energy {#eam_cu}

Sutton–Chen copper (\(n=9\), \(m=6\), \(\varepsilon = 1.2382\times10^{-2}\) eV,
\(c = 39.432\), \(a = 3.61\) Å; Sutton & Chen, *Philos. Mag. Lett.* **61**, 139
(1990), DOI 10.1080/09500839008206493). A lattice scan of 864-atom fcc Cu at a
10 Å cutoff, with parabolic refinement of the minimum, must reproduce the two
fit targets within 2 %: \(a_0 = 3.61\) Å (measured 3.6121 Å) and
\(E_{\mathrm{coh}} = 3.54\) eV/atom (measured 3.4828 eV). The cohesive energy
is low by construction: the power-law form has no intrinsic range, so a finite
cutoff truncates density and binding. The oracle record gives the cutoff
convergence and the tolerance argument.

## sw_si — Stillinger–Weber silicon {#sw_si}

Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985), DOI
10.1103/PhysRevB.31.5262; both terms vanish identically at
\(a\sigma = 3.7712\) Å, so there is no truncation tail. A 216-atom
diamond-cubic lattice scan must reproduce \(a_0 = 5.431\) Å (measured 5.4312 Å)
and \(E_{\mathrm{coh}} = 4.3363\) eV/atom (measured 4.3364 eV) within 1 %, and
fcc silicon at the same number density must lie higher in energy
(−3.4520 vs −4.3364 eV/atom) — the check that only an angular three-body term
can pass.
