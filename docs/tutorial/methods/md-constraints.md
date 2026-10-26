# Rigid water and virtual sites

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Rigid-water models need massless charge sites (the TIP4P M site, the TIP5P lone
pairs) whose positions follow the parent atoms and whose forces are projected
back onto those parents (Feenstra partition) so that total force and torque on
the molecule are preserved. Atomiverse offers two routes, both run on the
ordinary `cpu::Evaluate` / `cpu::md::MD` force pipeline rather than a new
Driver category:

- **General virtual sites** — `cpu::md::VirtualSites`, a table of site
  constructions (`add_weighted3`, `add_out_of_plane3`) with `place` /
  `redistribute` operations, mounted as the nested `VirtualSites::Place` /
  `VirtualSites::Redistribute` Components. Sites are ordinary Frame atoms and
  the pair styles stay ordinary (`PairLJCut`, `PairCoulCut`).
- **Specialised TIP4P pair** — `cpu::PairLJCutTip4pCut` (LAMMPS
  `lj/cut/tip4p/cut`), which reconstructs M internally from the O/H atoms.

The case also carries the `rigid_water`, `settle_tip3p` and `tip3p_two_water`
sections; this page documents the two TIP4P / TIP5P sections, which pin the same
two-water TIP4P single point (energy and parent forces) through the two routes.
CPU only.

Case: `ctest -R '^md/constraints$'` — sources and oracle record in [`regressions/md/constraints/`](../../../regressions/md/constraints/). One section of the case per `##` heading below: `atv_regressions --case md/constraints --section <name>`.

## Common setup

**Theory.** For the TIP4P M site with O→M distance \(q_{\mathrm{dist}}\),

\[
\mathbf{r}_M=\mathbf{r}_O+\frac{\alpha}{2}(\mathbf{d}_1+\mathbf{d}_2),\qquad
\alpha=\frac{q_{\mathrm{dist}}}{\cos(\theta_{\mathrm{HOH}}/2)\,r_{\mathrm{OH}}},
\]

with \(\mathbf{d}_i=\mathrm{min\_image}(\mathbf{r}_{H_i}-\mathbf{r}_O)\)
(LAMMPS tip4p convention). The oxygen nominal charge sits on M; hydrogens keep
\(q_H\); LJ acts only on O–O. Coulomb forces on M are projected back to O, H1,
H2 (Feenstra et al., *J. Comput. Chem.* **20**, 786 (1999); the same
construction LAMMPS uses in `pair_lj_cut_tip4p_cut` and GROMACS uses for
vsites).

**Pinned system** (classic Jorgensen TIP4P, 1983/85):

| Item | Setting |
| ---- | ------- |
| Geometry | O–O = 2.8 Å along \(x\); second water rotated \(\pi\) about \(z\) |
| \(r_{\mathrm{OH}}\) | 0.9572 Å |
| \(\theta_{\mathrm{HOH}}\) | 104.52° |
| \(q_{\mathrm{dist}}\) (OM) | 0.15 Å |
| \(q_M\) (\(q_O\) on M), \(q_H\) | −1.04 e, +0.52 e |
| \(\varepsilon_{\mathrm{OO}}\) | 0.1550 kcal/mol → eV via `kcal_per_mol_to_ev` |
| \(\sigma_{\mathrm{OO}}\) | 3.15365 Å |
| \(r_{c,\mathrm{LJ}} = r_{c,\mathrm{Coul}}\) | 10.0 Å, cutoff LJ + cutoff Coulomb (no PME/PPPM) |
| LJ shift | yes, \(V(r_c)=0\) (`pair_modify shift yes`) |
| special_bonds | lj/coul 0 0 0 (exclude 1–2, 1–3, 1–4) |
| Box | cubic 20 Å, periodic |

**Oracle.** **LAMMPS 30 Mar 2026 - Development** (`develop` /
`patch_30Mar2026-1074-g7f680de296`), `pair_style lj/cut/tip4p/cut`, metal units
(eV, eV/Å), `run 0`. Twin inputs `in.tip4p_dimer_metal`,
`data.tip4p_dimer_metal`, `log.metal` under the MetalWalls oracle tooling;
offline pin only, ctest never launches LAMMPS. LAMMPS rather than MetalWalls
because four-site TIP4P pair physics exists in LAMMPS; MetalWalls is reserved
for MW-only electrode physics. Regenerate on a host with LAMMPS:

```bash
cd <LAMMPS oracle input dir for this case>
lmp -in in.tip4p_dimer_metal -log log.metal
```

Oracle total energy \(E = \)`0.52924228894955327451` eV
(\(E_{\mathrm{vdwl}}=\)`0.057185016777344513383`,
\(E_{\mathrm{coul}}=\)`0.4720572721722087195`). Parent forces (atoms O1 H1 H2
O2 H3 H4; \(f_y=0\)):

| atom | \(f_x\) [eV/Å] | \(f_z\) [eV/Å] |
| ---- | -------------: | -------------: |
| 1 O | −0.094382238074752045076 | 0.33187666441899288383 |
| 2 H | −1.0593571566097745862 | −0.31522207906906996033 |
| 3 H | −0.057823257196248978307 | −0.016654585349923187176 |
| 4 O | 0.094382238074751600987 | 0.33187666441899305037 |
| 5 H | 1.0593571566097748082 | −0.31522207906906962727 |
| 6 H | 0.057823257196249200351 | −0.01665458534992322534 |

**Fair wall.** Both sections time a process launch of the section against
LAMMPS `in.tip4p_dimer_metal` / `run 0`, serial (`OMP_NUM_THREADS=1`), 50
launches after 3 warmups, x86_64 login node, Release MD-only build
(`buildenv-gcc/2026.03-mpich`, `ATV_BUILD_MD=ON`). Wall includes process start
and build, so it is dominated by startup on this tiny system and says nothing
about bulk MD throughput. Neither section is perf-exempt; gate
\(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} \le 10\).

**Shared limitations.** Rigid two-water footholds, not a production water
library (no parameter database, no automatic molecule templates) and not bulk
aqueous or electrode–water production MD. No long-range TIP4P
(`lj/cut/tip4p/long`, PME/PPPM), TIP4P/2005 or TIP4P/Ice. No rigid
SETTLE/SHAKE/RATTLE constraints coupled to sites here. Parameters are
constructor arguments; no Config bag.

**References.**

- W. L. Jorgensen *et al.*, *J. Chem. Phys.* **79**, 926 (1983); *J. Chem. Phys.* **103**, 10252 (1985 context).
- H. W. Mahoney & W. L. Jorgensen, *J. Chem. Phys.* **112**, 8910 (2000) (TIP5P).
- Feenstra et al., *J. Comput. Chem.* **20**, 786 (1999) (virtual-site force partition).
- LAMMPS `pair_style lj/cut/tip4p/cut` (metal).
- Roadmap: `docs/roadmap/molecular-dynamics.md` (virtual sites / TIP4P–TIP5P);
  capability map `.claude/notes/metalwalls-capability-map.md` → **E9**.

## virtual-site-multi-topology — General virtual sites: TIP4P dimer + TIP5P geometry (E9) {#virtual-site-multi-topology}

### Summary

`cpu::md::VirtualSites` reconstructs the two M sites of the TIP4P dimer as
Frame atoms, ordinary `PairLJCut` + `PairCoulCut` evaluate the pairs, and
`Redistribute` partitions the M forces onto O/H. The section pins the TIP4P
energy and parent forces against the LAMMPS twin shared with
[`four-site-tip4p-foothold`](#four-site-tip4p-foothold) (same physics, general
VirtualSites path instead of the specialised pair), and dual-gates **TIP5P**
equilibrium geometry (\(r_{\mathrm{OL}}\), \(\theta_{\mathrm{LOL}}\)) via
`OutOfPlane3`. Fair wall \(r = 0.0837\) (within 10×).

### Method

#### Weighted3 (TIP4P M-site)

Site position as a linear combination of oxygen and two hydrogens:

\[
\mathbf{r}_s
  =(1-\alpha)\,\mathbf{r}_0
  +\frac{\alpha}{2}\,\mathbf{r}_1
  +\frac{\alpha}{2}\,\mathbf{r}_2
  =\mathbf{r}_0+\frac{\alpha}{2}(\mathbf{d}_1+\mathbf{d}_2)
\]

with \(\alpha\) from `VirtualSites::tip4p_alpha`. Force partition (constant
Jacobian):

\[
\mathbf{F}_0\leftarrow\mathbf{F}_0+(1-\alpha)\mathbf{F}_s,\quad
\mathbf{F}_{1,2}\leftarrow\mathbf{F}_{1,2}+\tfrac{\alpha}{2}\mathbf{F}_s,\quad
\mathbf{F}_s\leftarrow\mathbf{0}.
\]

Charge on M and H, zero on O; intramolecular pairs excluded via special_bonds
0 0 0 and O–H / O–M topology. Frame layout: 8 atoms (O H H M × 2); types
0=O, 1=H, 2=M; M mass 0.

#### OutOfPlane3 (TIP5P lone pair)

\[
\mathbf{r}_s=\mathbf{r}_0+a\,\mathbf{d}_1+b\,\mathbf{d}_2
  +c\,(\mathbf{d}_1\times\mathbf{d}_2)
\]

Equilibrium coefficients \((a,b,c)\) for Mahoney–Jorgensen TIP5P
(\(r_{\mathrm{OL}}=0.70\) Å, \(\theta_{\mathrm{LOL}}=109.47^\circ\), same
\(r_{\mathrm{OH}}\), \(\theta_{\mathrm{HOH}}\)) come from
`VirtualSites::tip5p_out_of_plane_coeffs` (`sign = ±1` selects L1 vs L2).
Force redistribution includes the cross-term Jacobian (GROMACS vsite3OUT).
TIP5P is **geometry-gated** only: the host LAMMPS has no `tip5p` pair style for
an energy pin.

#### Pipeline

Place M → neighbor + `PairLJCut` + `PairCoulCut` → redistribute M forces onto
O/H.

### Implementation

```cpp
const double alpha = cpu::md::VirtualSites::tip4p_alpha(qdist, r_oh, theta);

// Frame: 8 atoms O H H M × 2; tags 1..8; bonds O–H, O–M + angle H–O–H;
// special list — see regressions/md/constraints/virtual-site-multi-topology.cpp
cpu::md::VirtualSites vs;
vs.add_weighted3(/*M=*/4, /*O=*/1, /*H1=*/2, /*H2=*/3, alpha);
vs.add_weighted3(/*M=*/8, /*O=*/5, /*H1=*/6, /*H2=*/7, alpha);
vs.place(f);  // dual-gate: |OM − qdist| ≤ 1e-10 Å

cpu::md::VirtualSites::Place place{vs};
cpu::md::VirtualSites::Redistribute redist{vs};

cpu::PairLJCut lj{{{0, 0, eps, sigma}}, cut};
lj.set_special_bonds(0.0, 0.0, 0.0);
cpu::PairCoulCut coul{cut};
coul.set_special_bonds(0.0, 0.0, 0.0);

cpu::Evaluate eval;
eval.on(evaluate::Phase::prepare).add_ref(place, +100);
eval.set_neighbor(cpu::md::NeighborList<>{cut, 0.3, 1, 0, true});
eval.set_evaluator(std::move(lj));
eval.set_evaluator(std::move(coul));
eval.on(evaluate::Phase::finalize).add_ref(redist, -100);
eval.build();
eval.run(f);
const double E = f.meta.get(kEnergy);  // [eV]; site F = 0 after redistribute
```

TIP5P geometry dual-gate (no energy):

```cpp
double a1, b1, c1, a2, b2, c2;
cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(
    /*r_ol=*/0.70, /*theta_lol=*/109.47 * units::kDegreeInRad,
    r_oh, theta, +1, a1, b1, c1);
cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(
    0.70, 109.47 * units::kDegreeInRad, r_oh, theta, -1, a2, b2, c2);

cpu::md::VirtualSites tip5p;
tip5p.add_out_of_plane3(/*L1=*/4, /*O=*/1, /*H1=*/2, /*H2=*/3, a1, b1, c1);
tip5p.add_out_of_plane3(/*L2=*/5, 1, 2, 3, a2, b2, c2);
tip5p.place(w);  // gate r_OL and θ_LOL
```

MD binding (defaults on the nested Components): `Place` at
`md::Phase::post_integrate` (+100) and `pre_force` (+50); `Redistribute` at
`post_force` (+50). Pair evaluators never special-case tip4p geometry.
Sources: `virtual_sites.h` (public `VirtualSites`, `Place`, `Redistribute`),
`src/cpu/dynamics/virtual_sites.cpp` (place / Feenstra redistribute / TIP4P α /
TIP5P coeffs); unit coverage in `tests/cpu/dynamics/test_virtual_sites.cpp`.

### Validation (vs LAMMPS)

| Quantity | Atomiverse / pin | LAMMPS (offline) | Tol | Notes |
| -------- | ---------------: | ---------------: | --- | ----- |
| \(E\) [eV] | `0.52924228894955327451` | same | \(10^{-8}\) eV | total PE after place + LJ + Coulomb |
| OM [Å] | `0.15` (= qdist) | geometry | \(10^{-10}\) Å | after `place` |
| \(F_x,F_z\) parents O1 H1 H2 O2 H3 H4 [eV/Å] | common force table | same | \(10^{-6}\) eV/Å | after redistribute; \(F_y=0\) |
| Site force on M | \(\mathbf{0}\) | n/a (internal) | \(10^{-12}\) eV/Å | Feenstra residual |
| TIP5P \(r_{\mathrm{OL}}\) [Å] | `0.70` | geometry | \(10^{-8}\) Å | both lone pairs |
| TIP5P \(\theta_{\mathrm{LOL}}\) [rad] | \(109.47^\circ\) | geometry | \(10^{-8}\) rad | no energy pin |

### Performance

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | -----------------------------------------: | ---- | ------ |
| TIP4P dimer single-point (`run 0`) | 0.1260 | 0.01055 | **0.0837** (≈ 0.084) | ✓ | process-launch mean; small system |

### Limitations

- **TIP5P energy unpinned.** Host LAMMPS lacks a `tip5p` pair; only
  \(r_{\mathrm{OL}}\) / \(\theta_{\mathrm{LOL}}\) are gated. A full five-site
  energy twin needs a different oracle binary or a literature pin.
- **Construction catalog.** Only Weighted3 and OutOfPlane3; further
  GROMACS-style vsite kinds (2-body, 4-body, COG, …) are not implemented.
- **Topology is the caller's job.** Callers register sites, topology for
  special exclusions, and ordinary LJ/Coulomb; a wrong topology silently
  changes intramolecular exclusions.
- **Scope.** This is the multi-topology engine, not electrode coupling (the E9
  molecular-water map slice documents electrode context).

## four-site-tip4p-foothold — Four-site TIP4P pair: water dimer {#four-site-tip4p-foothold}

### Summary

`cpu::PairLJCutTip4pCut` (LAMMPS name `lj/cut/tip4p/cut`) is a pair Component
that reconstructs M from each water's O/H atoms, evaluates LJ on O–O and
Coulomb with the oxygen charge at M, and applies the Feenstra redistribution
itself. The section is a two-water single point (6 atoms, no explicit M atoms)
asserting energy, M-site geometry (\(\lvert\mathrm{OM}-q_{\mathrm{dist}}\rvert\le 10^{-10}\) Å)
and six-atom forces against the shared LAMMPS pin. Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} = \mathbf{0.338}\) (within 10×).

### Method

The pair reconstructs M with the common construction,
\(\mathbf{r}_M=\mathbf{r}_O+\tfrac{\alpha}{2}(\mathbf{d}_1+\mathbf{d}_2)\),
\(\alpha=q_{\mathrm{dist}}/(\cos(\theta/2)\,r_{\mathrm{OH}})\), and partitions
the M force as \((1-\alpha)\) onto O and \(\alpha/2\) onto each H
(`src/cpu/potentials/pair_lj_cut_tip4p_cut.cpp`). Topology: O–H bonds + H–O–H
angles; tags follow the LAMMPS tip4p order (the H atoms of the O with tag \(t\)
are \(t{+}1\), \(t{+}2\)).

### Implementation

```cpp
// Classic TIP4P; types 0=O, 1=H. 6-atom dimer Frame with type/mass/charge/tag,
// O–H bonds + H–O–H angles and special list —
// see regressions/md/constraints/four-site-tip4p-foothold.cpp
cpu::PairLJCutTip4pCut tip4p(/*type_O=*/0, /*type_H=*/1, qdist, r_oh, theta,
                             eps, sigma, cut, cut);
tip4p.set_special_bonds(0.0, 0.0, 0.0);

cpu::Evaluate eval;
eval.set_neighbor(
    cpu::md::NeighborList<>{tip4p.neighbor_cutoff(), 0.3, 1, 0, true});
eval.set_evaluator(std::move(tip4p));
eval.build();
eval.run(frame);

const double E = frame.meta.get(kEnergy);   // [eV]
// forces: frame.span(kAtom_fx/y/z)
```

Headers: `pair_lj_cut_tip4p_cut.h`, `evaluate.h`, `neighbor_list.h`.

### Validation (vs LAMMPS)

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E\) [eV] | `0.52924228894955327451` | same (hard-coded) | \(10^{-8}\) eV | total PE, metal units |
| \(E_{\mathrm{vdwl}}\) [eV] | — | `0.057185016777344513383` | (report only) | LJ portion of pin log |
| \(E_{\mathrm{coul}}\) [eV] | — | `0.4720572721722087195` | (report only) | Coulomb portion |
| OM geometry [Å] | reconstructed \(= q_{\mathrm{dist}}\) | \(q_{\mathrm{dist}}=0.15\) | \(10^{-10}\) Å | dual-gate |
| \(f_x, f_z\) (6 atoms) [eV/Å] | common force table | same | \(10^{-6}\) eV/Å per component | \(f_y = 0\) within \(10^{-10}\) |

### Performance

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | -------------------------: | ---------------------: | -----------------------------------------: | ---- | ------ |
| TIP4P dimer SP | 0.1548 | 0.0523 | **0.338** | ✓ | process-launch fair wall, not in-process kernel microbench |

### Limitations

- **Rigid classic TIP4P only.** Arbitrary 2/3/4-site constructions and
  out-of-plane sites are the job of
  [`virtual-site-multi-topology`](#virtual-site-multi-topology); TIP5P is not
  available through this pair.
- **Tag convention:** H atoms of oxygen with tag \(t\) must be \(t{+}1\) and
  \(t{+}2\) (LAMMPS tip4p convention).
- **No long-range TIP4P;** charged-slab / PME water needs a different pair
  style.
