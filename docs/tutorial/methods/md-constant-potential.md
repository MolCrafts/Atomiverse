# Constant-potential electrodes

**Constant-potential** (Born–Oppenheimer) electrode electrostatics: metal atom
groups are held at fixed applied potentials \(\Psi\), and every step the
electrode site charges \(Q\) are solved from a Gaussian elastance matrix so each
electrode atom sits at its target potential in the field of the electrolyte and
of the other electrode charges. This is the classical electrochemical MD model
of MetalWalls and of the LAMMPS ELECTRODE package (`fix electrode/conp`).

The public surface is a set of **Components** on `cpu::md::MD`, never a Driver
category and never a charge solve inside `KSpace*`:

- `cpu::md::ConstPotBO` — dense (or, with `use_cg`, CG) BO solve, mounted with
  `sim.set_observer_ref(conp)` / `sim.set_observer(std::move(conp))` and bound
  by default at `md::Phase::pre_force`, order −50, so pair and k-space forces
  see the updated charges. Parameters are the constructor argument \(\eta\)
  [1/Å] and public members (`add_electrode(type, Psi[, eta])`,
  `electroneutrality`, `xy_image_range`, `rebuild_matrix_every_step`, `use_cg`,
  `cg_tol`, `max_cg_iters`, `predictor_order`, …); results are
  `total_charge(e)` and `neutrality_residual()`, and solved charges land on
  `kAtom_charge` (electrolyte rows are never rewritten).
- `cpu::md::ConstPotCG` — Jacobi-PCG solve on the same elastance.
- `cpu::ConstPotentialBO` — the min-image Gaussian BO solve used in the short
  NVT product (default `pre_force` order −500).
- `cpu::md::DipoleElectrodeJoint` — PIM dipoles and electrode charges solved
  jointly.
- Companions: `cpu::ElectrodeChargeObserver` (measures \(Q_e\)),
  `cpu::md::FixFreeze` (immobile walls), the piston and predictor members, and
  the `updates_owned_charges()` halo protocol.

All sections run on the CPU backend (CPU login environment
`buildenv-gcc/2026.03-mpich`; `REQUIRES MD`, plus `KSPACE` for the PME
sections). Constant-charge control (`add_electrode_charge`)
is on [constant-charge electrodes](md-constant-charge.md#const-charge-electrode);
Thomas–Fermi metallicity (`l_tf`, `site_density`, `site_volume`) on
[Thomas–Fermi electrodes](md-thomas-fermi.md#thomas-fermi-electrode). The case
also runs the sections `const-pot-tip4p-edl`, `nanoporous-carbon-const-pot`,
`electrode-atom-forces` and `global-electroneutrality-bo`, whose oracle records
live only in the case `REFERENCE.md`.

Case: `ctest -R '^md/constant-potential$'` — sources and oracle record in [`regressions/md/constant-potential/`](../../../regressions/md/constant-potential/). One section of the case per `##` heading below: `atv_regressions --case md/constant-potential --section <name>`.

## Physics

Siepmann–Sprik Gaussian electrode charges of width \(\eta\) [1/Å]. For
electrode atoms \(i,j\) the elastance (potential response) matrix elements are
(metal units, \(k_e =\) `units::coulomb` \(= 14.399645\) eV·Å/e²):

\[
A_{ii} = k_e\,\eta\sqrt{\frac{2}{\pi}}
  + \sum_{(n_x,n_y)\neq(0,0)} k_e\,\frac{\operatorname{erf}\!\bigl((\eta/\sqrt{2})\,r_{n}\bigr)}{r_{n}},
\quad
A_{ij} = \sum_{n_x,n_y} k_e\,\frac{\operatorname{erf}\!\bigl(\eta_{ij}\,r_{ij,n}\bigr)}{r_{ij,n}}
\]

with \(\eta_{ij} = \eta_i\eta_j / \sqrt{\eta_i^2+\eta_j^2}\). The electrolyte
contribution to the potential on electrode site \(i\) is

\[
b_i = \sum_{j\notin\mathrm{elec}} \sum_{n_x,n_y}
  k_e\,q_j\,\frac{\operatorname{erf}(\eta\,r_{ij,n})}{r_{ij,n}}.
\]

The unconstrained BO solve is \(A Q = \Psi - b\). With **electroneutrality**
(`electroneutrality = true`, LAMMPS ELECTRODE `symm on`), charges are the
projection

\[
Q = \Bigl(A^{-1} - \frac{A^{-1}ee^{\mathsf T}A^{-1}}{e^{\mathsf T}A^{-1}e}\Bigr)(\Psi - b)
\]

so \(\sum_i Q_i = 0\). The dense path uses a Cholesky / SPD inverse
(`src/cpu/dynamics/const_pot_bo.cpp`); residual and group totals are
diagnostics on the component.

**Lattice treatment.** \(A\) and \(b\) use **real-space 2D \(xy\) Gaussian
lattice images** with half-range `xy_image_range` (0 = primary cell /
min-image). There is no reciprocal-space electrode matrix
(`ewald/electrode`) and no 2D Ewald electrode kernel: the truncated image sum
converges to the 2D-periodic elastance only as \(O(1/R)\), so each section
states the image range it runs and how far that sits from the oracle.

## Oracles

- **LAMMPS** wherever the feature overlaps the ELECTRODE package or core
  LAMMPS (BO plate, charge sums, `fix setforce 0`). The local `lmp` lacks
  `PKG_ELECTRODE`, so ELECTRODE pins come from the package's shipped logs.
- **MetalWalls** release 20.05, git `c8ad175d`, run offline for MetalWalls-only
  physics (CG electrodes, ASPC, per-electrode \(\eta\), piston, PIM ⊕ electrode,
  EDL boxes). Pins are numbers the binary printed (`total_charges.out`,
  `charges.out`, `electrode_forces.out`); MetalWalls' atomic unit of charge is
  e, so the unit work is on the inputs only. Values that re-implement
  `ConstPotBO`'s own truncated sum are a self-golden, never a pin.
- **Literature / analytic** where no binary applies (halo invariant, Gaussian
  kernels).

No oracle binary is ever launched from ctest.

## Common limitations

- **Truncated real-space elastance.** Real-space Gaussian + 2D \(xy\) images;
  no full electrode Ewald / MetalWalls reciprocal matrix.
- **Fully periodic ATV box.** LAMMPS slabs use `boundary p p f`; ATV shifts
  \(z\) into \((0,L_z)\) so VelocityVerlet PBC does not wrap the slab.
- **Dense \(O(N_{\mathrm{elec}}^3)\) default solve** and
  \(O(N_{\mathrm{elec}}^2 R^2)\) matrix build; CG and ASPC serve larger walls.
- **Frozen-wall default.** `rebuild_matrix_every_step = false` unless the
  caller sets it.
- **CPU MD only**; no CUDA electrode path.

## References

- Ahrens-Iwers, Janssen, Tee, Meißner, *J. Chem. Phys.* **157**, 084801 (2022),
  DOI [10.1063/5.0099239](https://doi.org/10.1063/5.0099239) — LAMMPS ELECTRODE.
- Marin-Laflèche et al., MetalWalls, *J. Open Source Softw.* **5**, 2373 (2020),
  DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373).
- Scalfi, Dufils, Reeves, Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704
  (2020), DOI [10.1063/5.0028232](https://doi.org/10.1063/5.0028232).
- J. I. Siepmann, M. Sprik, *J. Chem. Phys.* **102**, 511 (1995) — Gaussian
  fluctuating electrode charges at fixed potential.
- S. K. Reed, O. J. Lanning, P. A. Madden, *J. Chem. Phys.* **126**, 084704
  (2007) — constant-potential MD electrodes.
- L. Scalfi *et al.*, *Chem. Rev.* **122**, 10899 (2022) — constant-potential
  and electrochemical MD review.

## const-pot-bo-parallel-plate — BO solve, parallel-plate madelung twin {#const-pot-bo-parallel-plate}

The LAMMPS ELECTRODE **madelung** four-atom parallel-plate twin on
`ConstPotBO`. Asserts electroneutrality, charge sign, and
\(\lvert Q_{\mathrm{top}} - Q_{\mathrm{pin}}\rvert / \lvert Q_{\mathrm{pin}}\rvert
\le 1\%\) (2D-image Gaussian \(A\) vs full `ewald/electrode`), both standalone
and with the component bound on MD.

### System

| Item | Setting |
| ---- | ------- |
| Geometry | 4 atoms: bot/top electrode (types 1,2), electrolyte \(\pm 0.5\,e\) (type 3) |
| Relative \(z\) [Å] | electrode \(\{-2,+2\}\), electrolyte \(\{-1,+1\}\); shifted into \((0,L_z)\) as \(z=\{8,12,9,11\}\) |
| Box | \(L_x=L_y=1\) Å, \(L_z=20\) Å, orthorhombic, fully periodic ATV box |
| Applied \(\Psi\) | \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V (\(\Delta\Psi=1\) V) |
| Gaussian width | \(\eta = 2.0\) Å⁻¹ |
| Neutrality | on (`symm`) |
| Image range | `xy_image_range = 250` |

### Code

```cpp
cpu::Frame frame(4);
const double z[4]  = {8.0, 12.0, 9.0, 11.0};
const double q0[4] = {0.0, 0.0, 0.5, -0.5};
const int typ[4]   = {1, 2, 3, 3};
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_z)[i] = z[i];           // x = y = 0
    frame.span(kAtom_charge)[i] = q0[i];
    frame.span(kAtom_type)[i] = typ[i];
    frame.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.966553 : 1.0;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);

cpu::md::ConstPotBO conp(/*eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
conp.electroneutrality = true;
conp.xy_image_range = 250;
conp.rebuild_matrix_every_step = true;  // optional if geometry moves

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.set_observer_ref(conp);   // default binding: pre_force @ -50
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_bot = conp.total_charge(0);  // [e]
const double q_top = conp.total_charge(1);  // [e]
```

### Oracle

LAMMPS **3 Nov 2022**, ELECTRODE package example
`examples/PACKAGES/electrode/madelung` (`in.ewald-ew3dc`, `plate_cap.py`), log
`log.1Dec2022.madelung-ewald-ew3dc.g++.1` after `run 0` with
`fix electrode/conp 0 2 couple top 1 symm on`,
`kspace_style ewald/electrode 1e-8`, `kspace_modify slab 8.0`.

### Accuracy

| Quantity | Atomiverse gate / pin | LAMMPS pin | Tol | Notes |
| -------- | --------------------: | ---------: | --- | ----- |
| \(Q_{\mathrm{top}}\) [e] | within **1%** of pin | **+0.279214485147238** | rel ≤ **0.01** | 2D-image \(A\) vs full Ewald electrode |
| \(Q_{\mathrm{bot}}\) [e] | opposite of top under neutrality | **−0.279214485147238** | — | pin log; regression checks sign |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | ~0 (`symm on`) | **1e-10** | electroneutrality residual |
| \(\mathrm{sign}(Q_{\mathrm{top}})\) | \(+1\) at \(\Delta\Psi=+1\) V | \(+\) | exact | capacitive polarity |
| Electrolyte \(q\) | unchanged \(\{+0.5,-0.5\}\) | fixed | exact write-back | only electrode rows rewritten |
| PE (kcal/mol, real units) | **not compared** | 139.943964815502 | — | unit system differs |

**Literature cross-check (not the pin):** the pure real-space 2D Gaussian
lattice sum `plate_cap.py` in the same LAMMPS example directory (metal units,
image range ≳ 200) gives \(Q_{\mathrm{top}}^{\mathrm{Madelung}} \approx
+0.27941\,e\) — relative difference vs the Ewald-electrode log pin ≈ **0.07%**.
The 1% gate is therefore dominated by the missing reciprocal electrode matrix,
not by a wrong BO equation.

### Offline pin repro (never ctest)

```bash
# Requires LAMMPS built with -D PKG_ELECTRODE=ON
cd $LAMMPS_DIR/examples/PACKAGES/electrode/madelung
lmp -in in.ewald-ew3dc
# thermo: pe c_qbot c_qtop at step 0
```

### Performance

`perf_exempt`: the local LAMMPS binary lacks the ELECTRODE package, so there is
no fair twin binary for `fix electrode/conp`.

### Limitations

- ~1% level mismatch vs `ewald/electrode` pins on small cells from the
  real-space elastance — the gate is set accordingly.
- No fair speed ratio until ELECTRODE-enabled LAMMPS is on the pin host.

## const-pot-cg-large-electrode — CG solve, multi-site plate {#const-pot-cg-large-electrode}

`cpu::md::ConstPotCG` solves the same \(A Q = \Psi - b\) (plus neutrality
projector) with **Jacobi-preconditioned conjugate gradient** (`pcg_solve_`,
relative residual tolerance `cg_tol`, cap `max_cg_iter`) instead of a dense
inverse; it keeps the dense row-major \(A\). Same physics matrix → CG and BO
site charges must agree to machine-relevant tolerances. The section pins a
primary-cell \(Q_{\mathrm{top}}\) against MetalWalls and checks CG ≡ dense BO
on a \(4\times 4\) planar capacitor per wall (\(N_{\mathrm{elec}}=32\)) sharing
\(A\), \(\eta\), \(\Psi\) and the projector.

### System

| Item | Setting |
| ---- | ------- |
| Geometry (large) | Bottom/top walls: \(4\times 4\) grids at \(z=\{8,12\}\) Å; electrolyte \(\pm 0.5\,e\) at box-center \(xy\), \(z=\{9,11\}\) Å |
| \(N_{\mathrm{elec}}\) | **32** (16 + 16); gate \(\ge 18\) |
| Box | \(L_x=L_y=4\) Å, \(L_z=20\) Å, orthorhombic, fully periodic ATV |
| Applied \(\Psi\) | \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V |
| Gaussian width | \(\eta = 2.0\) Å⁻¹ |
| Neutrality | on (`symm`) |
| Image range | `xy_image_range = 0` |
| CG controls | `cg_tol = 1e-14` (standalone twin); `max_cg_iter = 2000` |

The pinned number comes from the primary-cell system: the 2-electrode madelung
geometry (4 particles on one \((x,y)\) line, cell \(1\times1\times20\) Å) at
`xy_image_range = 0`, where min-image is exactly the isolated cluster.

### Code

```cpp
cpu::md::ConstPotCG cg(/*eta [1/Å]=*/2.0);
cg.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
cg.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
cg.electroneutrality = true;
cg.xy_image_range = 0;
cg.cg_tol = 1e-14;
cg.max_cg_iter = 2000;

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.set_observer_ref(cg);
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_top = cg.total_charge(1);  // [e]
const int iters = cg.cg_iterations();
// Dual twin: cpu::md::ConstPotBO bo(2.0) with the same electrodes on a
// second Frame; assert max_i |Q_i^CG - Q_i^BO| <= 1e-8 e.
```

### Oracle

**MetalWalls** `c8ad175d` (`num_pbc 2`, `num_steps 0`,
`electrode_charges cg 1.0e-14 2000`, `charge_neutrality true electrodes`,
`coulomb_rtol/ktol 1.0e-12`). MetalWalls has no non-periodic mode, so the twin
takes \(L_x=L_y=L\to\infty\) and \(L_z\to\infty\) (its 2D scheme discretizes
the \(k_z\) integral at spacing \(2\pi/L_z\)). At \(L = L_z = 1600\) Å
MetalWalls prints \(Q_{\mathrm{top}} = 0.2717535546806\) e; the lateral
\(1/L^3\) convergence extrapolates to
\(Q_{\mathrm{top}}^{\mathrm{MW}}(\text{isolated}) = 0.2717535549687\) e.
ATV prints 0.2717535558024 e (\(3.07\times10^{-9}\) rel); the residual is the
truncated `units::coulomb` literal — with the exact
\(k_e = 14.399645478456\) eV·Å the same formula gives 0.271753554945 e.

### Accuracy

| Quantity | Atomiverse gate / pin | Twin / pin | Tol | Notes |
| -------- | --------------------: | ---------: | --- | ----- |
| \(N_{\mathrm{elec}}\) | \(\ge 18\) (product: **32**) | geometry | exact | \(4\times 4\) per wall |
| \(\max_i \lvert Q_i^{\mathrm{CG}} - Q_i^{\mathrm{BO}}\rvert\) [e] | \(\le 10^{-8}\) | dense BO dual | **1e-8** | site-wise dual twin |
| \(\lvert Q_{\mathrm{group}}^{\mathrm{CG}} - Q_{\mathrm{group}}^{\mathrm{BO}}\rvert\) [e] | \(\le 10^{-8}\) | dense BO dual | **1e-8** | bot and top groups |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | ~0 (`symm`) | **1e-10** | electroneutrality residual |
| Primary \(Q_{\mathrm{top}}\) [e] | within **1e-6** of pin | **0.27175356** (MetalWalls) | abs \(\le\) **1e-6** | 2-elec madelung, `xy_image_range=0` |
| Electrolyte \(q\) | unchanged \(\{+0.5,-0.5\}\) | fixed | exact write-back | only electrode rows rewritten |
| MD-bound \(Q_{\mathrm{top}}\) vs BO | \(\lvert\Delta\rvert \le 10^{-6}\) e | dense BO | **1e-6** | one VV step, `PairCoulCut` |

The large-plate numbers are a CG-vs-dense-BO equivalence on a shared truncated
\(A\), not a MetalWalls comparison: at `xy_image_range = 0` the plate sits
20.9% above MetalWalls' full 2D-Ewald answer (0.365146 vs 0.301964 e), and the
gap shrinks only as \(0.0260/R\).

### Performance

MetalWalls `c8ad175d` vs ATV on the same 32-site box, single rank,
`OMP_NUM_THREADS=1`, best of 5: ATV 0.01 s, MetalWalls 0.58 s, ratio 0.026.
MetalWalls' wall includes MPI start-up and Ewald k-tables and computes the
converged 2D-Ewald elastance, while ATV computes only the min-image truncation
(32-site CG solve 0.032 ms, dense BO 0.203 ms); it is not an algorithmic win.
The binary also prints internal `wall_ms CG` / `wall_ms BO`.

### Limitations

- **Primary-cell elastance for the large plate** — a multi-site correctness
  check, not 2D-image / electrode-Ewald parity with production cells.
- **Dense \(A\) storage.** Jacobi-PCG still builds a dense
  \(N_{\mathrm{elec}}\times N_{\mathrm{elec}}\) matrix; matrix-free / sparse /
  Ewald-electrode matvecs for very large walls are not available.
- **No ASPC warm-start on `ConstPotCG`** — the predictor is the BO `use_cg`
  path ([`electrode-charge-predictor`](#electrode-charge-predictor)).

## const-pot-charge-halo-mpi — Post-solve ghost-charge refresh {#const-pot-charge-halo-mpi}

Constant-potential MD rewrites electrode charges **mid-step**: the Q solve runs
at `md::Phase::pre_force`, *after* the step's forward halo has populated ghost
atoms. Without a second, post-solve charge communication, the force stages read
ghost charges that are **one solve stale** — and identically **zero on the
first step**, since electrode charges start at 0. Forces are then silently
wrong across every periodic face and rank cut.

Any Component that declares `updates_owned_charges()` (`ConstPotBO`,
`ConstPotential`, `ConstPotCG`, `ConstPotMaZe`, `DipoleElectrodeJoint`) posts
`ChargePacker` (`charge_packer.h`) through `ctx.md.comm->forward_comm` after
writing owned charges, before any force stage. This mirrors the documented
LAMMPS ELECTRODE contract: `fix electrode` forward-communicates the per-atom
charge after every conp solve so ghosts match owners before force evaluation.
The section gates the **default** MPI pipeline — nothing registers a
Plan-level refresh component.

### Invariant and oracle

\[
q_{\mathrm{ghost}}(\mathrm{tag}) \equiv q_{\mathrm{owner}}(\mathrm{tag})
\quad \text{at every force read, every step.}
\]

Because the halo transport is a memcpy of the owner value, the pin is
**bitwise** (tolerance 1e−15 e). Oracle = **literature**: no binary is run; the
contract is documented behaviour of the reference codes —

- A. P. Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022),
  DOI [10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171)
  — LAMMPS forward/reverse ghost communication semantics (owner → ghost
  refresh of per-atom state).
- L. J. V. Ahrens-Iwers, M. Janssen, S. R. Tee, R. H. Meißner, *ELECTRODE:
  An electrochemistry package for atomistic simulations* (2022) — per-step
  conp charge solve; solved charges are forward-communicated
  (`comm->forward_comm`) before force evaluation.
- A. Marin-Laflèche et al., *MetalWalls* (2020) — the constant-potential MD
  loop this capability maps.

### System (child process, np = 2)

38-atom parallel-plate capacitor, box 20 × 10 × 16 Å (orthorhombic). Both
ranks run the whole system as a replica (`sim.set_serial_on_replica()`): the
electrode solve requires a complete Frame. CommBrick creates explicit periodic
image rows at every rank count, so the ghosts under test are the self-images
at the ±x / ±y faces; a remote ghost rides the same `ChargePacker` pass.

| Species | Type | Count | z [Å] | q₀ [e] |
| ------- | ---- | ----- | ----- | ------ |
| Bottom plate (Ψ = 0 V) | 1 | 15 (5×3 grid) | 5 | 0 (solved) |
| Top plate (Ψ = 1 V) | 2 | 15 (5×3 grid) | 11 | 0 (solved) |
| Electrolyte ions | 3 | 8 | 7–9.5 | ±0.25 (neutral) |

Pipeline (all defaults): `VelocityVerlet{0.1 fs}` + `NeighborList{3.0, 0.3}` +
`PairCoulCut{3.0}` + `ConstPotBO{η = 2 Å⁻¹}` (electroneutrality on, matrix
rebuilt per step, `xy_image_range = 0`). 5 MD steps. An out-of-library
`GhostChargeChecker` Component — the user-extension path, defined in the
regression file — runs at the force phase and compares every ghost row's
charge against its owner's current value (tag-matched via allgather). The same
pipeline under a forced 2×1×1 spatial decomposition (`ATV_FORCE_DIMS=2,1,1`)
is attempted first and must be rejected at frame binding.

The regression is one binary in two modes: the parent re-launches itself
through the configured `mpiexec` (compile-time pin `ATV_MPIEXEC`) at np = 2;
the child runs the capacitor MD with the checker and rank 0 writes one result
line that the parent gates.

### Code

```cpp
cpu::md::ConstPotBO conp(/*eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
conp.electroneutrality = true;
conp.xy_image_range = 0;
conp.rebuild_matrix_every_step = true;

cpu::md::MD sim;
sim.set_serial_on_replica();
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff [Å]=*/3.0, /*skin=*/0.3});
sim.set_evaluator(cpu::PairCoulCut{3.0});
sim.set_observer_ref(conp);                          // pre_force @ -50
sim.on(ATV::md::Phase::force).add_ref(checker, +500);
sim.build();
sim.run(frame, /*nsteps=*/5);
```

A new charge-writing Component keeps the invariant by overriding one virtual
(`component.h`, default `false`) and posting `ChargePacker` after the write:

```cpp
class MyChargeSolver : public cpu::Component {
    bool updates_owned_charges() const override { return true; }
    // ... solve at pre_force; then ChargePacker::forward(ctx).
};
```

### Accuracy

| Quantity | ATV (gate) | Oracle pin (literature contract) | Tol | Notes |
| -------- | ---------: | -------------------------------: | --- | ----- |
| max \|q_ghost − q_owner\| over all ghosts, ranks, steps [e] | gated | **0** | **1e-15** | halo transport is a memcpy of the owner value — bitwise |
| electrode ghosts with \|q\| > 1e-12 e | gated | **> 0** | exact | non-vacuous: electrode charges start at 0; only the refreshed solve makes ghosts nonzero |
| unresolved ghost tags | gated | **0** | exact | every ghost must map to an owner |
| checker launches | gated | **≥ 5** (one per step) | exact | invariant enforced on every step |
| decomposed `ConstPotBO` run | gated | **rejected** (`requires a complete Frame`) | exact | no silent rank-local partial solve |

Without the refresh, the first gate fails at the first step with max \|dq\| ≈
the solved electrode charge magnitude (~1e-2 e in this geometry).

### Performance

`perf_exempt`: the gate is a self-consistency transport invariant with no
external binary, so there is no wall-clock twin; the 38-atom 5-step child runs
in seconds.

### Limitations

- **Transport only.** A DD-aware global electrode solve does not exist; the
  decomposed combination is refused explicitly. Only the invariant that ghosts
  carry exactly what owners solved is pinned.
- **No migration exercised.** dt = 0.1 fs over 5 steps keeps every atom in
  place.
- **2 ranks.** Higher rank counts reuse the same halo endpoints but are not
  separately pinned.
- **Primary-cell solve** (`xy_image_range = 0`) — the electrode physics is
  deliberately minimal; accuracy of the solve itself is pinned by the other
  sections.

## const-pot-charge-observer — Electrode charge observer {#const-pot-charge-observer}

`cpu::ElectrodeChargeObserver` (`electrode_charge.h`,
`src/cpu/observe/electrode_charge.cpp`) is a Frame-immutable `Observer` that
sums `kAtom_charge` over owned atoms whose `kAtom_type` matches each configured
electrode type id. It does **not** solve charges — it measures whatever is on
the Frame (fixed charges, or charges written by a prior const-pot Component
such as `ConstPotBO`). Default bind is MD `end_of_step`, order +40.

### Observable

For electrode \(e\) identified by atom type \(t_e\),

\[
Q_e = \sum_{i\,\in\,\mathrm{owned}:\;\mathrm{type}_i = t_e} q_i
\quad[\mathrm{e}].
\]

Only **owned** local atoms contribute; ghosts are excluded so domain
decomposition cannot double-count charge. With MPI, electrode totals are
`MPI_Allreduce`’d over the domain cart communicator. Non-listed types never
enter the sums. Totals are available as `obs.total_charge(e)` and as Frame meta
`kElectrodeCharge` (first ≤3 electrodes packed into `MetaF64x3`).

### System (constant-charge parallel-plate toy)

The section uses **constant-charge** walls (\(q_i\) fixed at construction),
isolating the observer from the solvers and matching the LAMMPS twin
(`variable atom q` + `compute reduce sum` on type groups).

| Quantity | Value |
| -------- | ----- |
| Box | orthorhombic \(20\times 10\times 10\) Å |
| Left wall | type 1, 4 atoms, \(q_i = +0.125\) e |
| Right wall | type 2, 4 atoms, \(q_i = -0.125\) e |
| Electrolyte placeholders | type 3, 2 atoms, \(q_i = +1\) e (excluded) |
| Ensemble | none (`run 0` twin) — pure charge accounting |

Analytic totals (identical to the reduce-sum pin):
\(Q_L = 4\times 0.125 = +0.5\,\mathrm{e}\),
\(Q_R = 4\times(-0.125) = -0.5\,\mathrm{e}\).

### Code

```cpp
cpu::ElectrodeChargeObserver obs({1, 2}); // left type, right type

cpu::md::MD sim;
// ... potentials, neighbor, integrator, const-pot / freeze as needed ...
sim.set_observer_ref(obs);   // default: end_of_step @ +40
sim.build();
sim.run(frame, /*nsteps=*/1);

const double Q_L = obs.total_charge(0);                     // [e]
const double meta_L = frame.meta.get(kElectrodeCharge)[0];  // same value
```

### Oracle

Core **LAMMPS** (30 Mar 2026 - Development - `patch_30Mar2026-1074-g7f680de296`),
metal units, `run 0`, no pair/kspace; the ELECTRODE package is not needed for a
group charge sum. Recorded pin lines: `PIN_QL 0.5`, `PIN_QR -0.5`.

### Accuracy

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(Q_L\) [e] | \(+0.5\) | \(+0.5\) | \(10^{-12}\) | type-1 wall sum |
| \(Q_R\) [e] | \(-0.5\) | \(-0.5\) | \(10^{-12}\) | type-2 wall sum |
| \(Q_L + Q_R\) [e] | \(0\) | \(0\) | \(10^{-12}\) | electrodes alone (electrolyte excluded) |
| `meta` \(Q_L,Q_R\) | same as members | — | \(10^{-12}\) | `kElectrodeCharge[0,1]` |

The pin is an **exact finite sum** of fixed charges; it catches wrong type
selection, ghost double-counting, or unit mistakes.

### Performance

`perf_exempt`: a single \(O(N)\) sum over 10 atoms; a wall ratio vs LAMMPS
would measure binary startup (~0.3 s for `run 0` with no pair/kspace), not
the reduce.

### Limitations

- **Measure-only.** Pair with `ConstPotBO` / freeze / long-range Components
  for a full const-pot stack.
- **Type-keyed electrodes.** Membership is by `kAtom_type` list; group masks /
  tag lists are not supported.
- **Meta pack depth.** `kElectrodeCharge` publishes the first ≤3 electrodes;
  more electrodes remain available via `total_charges()`.
- **MPI.** The allreduce requires a domain cart communicator on
  `ctx.runtime.md.dd` when `mpi_size > 1`.
- Capacitance / charge-fluctuation statistics and potential profiles live on
  [electrode observables](md-electrode-observables.md#capacitance-charge-fluctuation).

## const-pot-il-edl-box — Ionic-liquid EDL production box {#const-pot-il-edl-box}

The **production electric-double-layer (EDL) box** is a composition gate, not
a new solver: two planar honeycomb (graphene-like) carbon walls held at
\(\Psi = \mp 1\) V by `ConstPotBO` (CG solve + ASPC predictor, MetalWalls
graphite Gaussian width) sandwich a non-polarizable single-site coarse-grained
ionic liquid (12 cations + 12 anions, BMIM⁺/PF₆⁻-like), with LJ + `CoulLong` +
`KSpacePME` forces, frozen walls, and a short Nosé–Hoover NVT run. Between the
walls a charged electrolyte polarizes the electrodes far beyond the bare
vacuum-capacitor response — the EDL screening enhancement is the headline
observable.

Production path details (all public members):

- **CG solve** (`use_cg`, `cg_tol = 1e-10`, `max_cg_iters = 5000`) instead of
  the dense \(O(N^3)\) factorization.
- **ASPC predictor** (`predictor_order = 2`) warm-starts each step's CG from
  the extrapolated charge history — the MetalWalls production path.
- η = **1.805132 Å⁻¹**: the MetalWalls graphite Gaussian width
  (0.955234 bohr⁻¹, MW example inputs).
- `xy_image_range = 80` (truncated xy-image elastance; the empty capacitor
  approaches MetalWalls' 2D Ewald, the with-IL absolute \(Q\) does not).

### System

| Quantity | Value |
| -------- | ----- |
| Box | 17.04 × 17.216585 × 40 Å (orthorhombic, 3D PBC) |
| Walls | honeycomb C, \(a_{\mathrm{CC}}\) = 1.42 Å, 4-atom rect cell, 4×7 → **112 sites/wall** |
| Wall planes | z = 12 Å (type 2, Ψ = −1 V), z = 28 Å (type 3, Ψ = +1 V), frozen (`FixFreeze`) |
| IL gap | 16 Å interior < 24 Å periodic return path (EDL response dominates) |
| Ions | 12 + 12 single-site CG (±1 e, non-polarizable), deterministic 3×3(−centre)×3 grid, checkerboard + one swap per outer layer (nascent-EDL polarization) |
| Cation LJ | σ = 4.38 Å, ε = 0.026532 eV, m = 139.22 amu (BMIM⁺-like) |
| Anion LJ | σ = 5.06 Å, ε = 0.048815 eV, m = 144.96 amu (PF₆⁻-like) |
| Wall C LJ | σ = 3.37 Å, ε = 0.0023837 eV; **no wall-type diagonal** → zero intra-wall LJ (LAMMPS `neigh_modify exclude` analogue) |
| Electrostatics | `CoulLong{α=0.3 Å⁻¹, r_c=8 Å}` + `KSpacePME{α=0.3, spacing 0.5 Å, order 5}` |
| Neighbor / LJ cutoff | 10 Å + 0.5 skin / 10 Å |
| Const-pot | η = 1.805132 Å⁻¹, CG tol 1e-10, ASPC order 2, electrode-only neutrality |
| NVT | VelocityVerlet dt = 1 fs, Nosé–Hoover 400 K τ = 100 fs, 600 steps, `InitVelocity{400 K, seed 11}` |

Ion sizes/masses follow the coarse-grained imidazolium family of Roy &
Maroncelli, *J. Phys. Chem. B* **114**, 12629 (2010) and Merlet et al.,
*J. Phys. Chem. C* **116**, 7687 (2012) — each ion reduced to its dominant
interaction site. The section pins solver algebra and physics invariants,
**not** FF-sensitive observables.

### Code

```cpp
cpu::md::ConstPotBO cpot(/*eta [1/Å]=*/1.805132);
cpot.add_electrode(/*type=*/2, /*Psi [V]=*/-1.0);
cpot.add_electrode(/*type=*/3, /*Psi [V]=*/+1.0);
cpot.electroneutrality = true;
cpot.xy_image_range = 80;
cpot.use_cg = true;                   // large-electrode CG route
cpot.cg_tol = 1e-10;
cpot.max_cg_iters = 5000;
cpot.predictor_order = 2;             // ASPC warm start

cpu::ElectrodeChargeObserver qobs({2, 3});
cpu::PotentialProfile psi(/*n_bins=*/80, /*z_min=*/0.0, /*z_max=*/40.0);
cpu::SpeciesDensityProfile rho({0, 1}, /*n_bins=*/32, /*z_min=*/12.0,
                               /*z_max=*/28.0);

cpu::md::MD sim;
sim.set_observer(cpu::md::InitVelocity{/*T=*/400.0, /*seed=*/11u});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.set_thermostat(cpu::md::NoseHoover{400.0, /*tau=*/100.0, 1.0});
sim.set_evaluator(cpu::PairLJCut{{{0, 0, 0.026532, 4.38},   // cation
                                  {1, 1, 0.048815, 5.06},   // anion
                                  {0, 2, eps_cw, sig_cw},   // explicit LB
                                  {0, 3, eps_cw, sig_cw},   // ion–wall rows;
                                  {1, 2, eps_aw, sig_aw},   // no wall diagonal
                                  {1, 3, eps_aw, sig_aw}},
                                 /*cutoff=*/10.0});
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/10.0, /*skin=*/0.5});
sim.on(md::Phase::force).add(cpu::CoulLong{/*alpha=*/0.3, /*cutoff=*/8.0});
sim.set_evaluator(cpu::KSpacePME{/*alpha=*/0.3, /*spacing=*/0.5, /*order=*/5});
sim.set_observer(std::move(cpot));
sim.set_observer(cpu::md::FixFreeze{2});
sim.set_observer(cpu::md::FixFreeze{3});
sim.set_observer_ref(qobs);
sim.set_observer_ref(psi);
sim.set_observer_ref(rho);
sim.build();
sim.run(frame, 600);                  // 0.6 ps NVT at 400 K

double dpsi = psi.psi()[56] - psi.psi()[24];  // Ψ(z_R) − Ψ(z_L) [V]
double n0 = rho.mean_density(0, /*bin=*/0);   // cation run-mean ρ_s(z) [1/Å³]
```

### Oracle

MetalWalls `c8ad175d` on the empty and filled twin boxes:

| case | MW \(Q_{\mathrm{left}}\) [e] | MW \(Q_{\mathrm{right}}\) [e] |
|------|------------------------------|-------------------------------|
| empty | **−2.076457628129008×10⁻¹** | **+2.076457628129015×10⁻¹** |
| with-IL | −4.818386149307364 | +4.818386149307384 |

ATV's empty capacitor at `xy_image_range = 80` (\(|Q|\approx 0.2076\,e\))
matches MetalWalls to ~0.5 %.
The with-IL absolute \(Q\) under the image sum (~1.36 e) does not approach the
MetalWalls Ewald value (~4.82 e), so the with-IL solve is gated by screening
order, polarity and neutrality only. At `xy_image_range = 0` the min-image BO
algebra gives \(Q_{\mathrm{left}} =\) −4.965149459545e−01 e (empty) and
−1.552770528510e+00 e (with-IL, ratio ≈ 3.13; right plate opposite) — a self-golden of ATV's
own elastance, not a pin.

### Accuracy

Step-0 BO solves on the exact initial geometry (CG at tol 1e-12):

| Quantity | ATV gate | Pin | Tol | Notes |
| -------- | -------: | --: | --- | ----- |
| empty capacitor \(Q_{\mathrm{left}}\), \(Q_{\mathrm{right}}\) [e] | within tol of pin | **∓2.0765e−01** (MetalWalls) | rel ≤ **2 %** | IL charges zeroed |
| EDL screening | \(\lvert Q\rvert_{\mathrm{IL}} > \lvert Q\rvert_{\mathrm{empty}}\) | MetalWalls ordering | exact | electrolyte enhances induced charge |
| polarity | \(Q_{\mathrm{left}} < 0 < Q_{\mathrm{right}}\) | same | exact | matches ΔΨ = ∓1 V |
| electrode neutrality [e] | residual | ~0 | ≤ **1e−8** | both solves |

Run-time invariants over 600 fs NVT (trajectory-dependent, therefore asserted
as invariants):

| Invariant | Gate |
| --------- | ---- |
| finite \(E\), \(K > 0\), finite charges | every 100-step chunk |
| \(\lvert Q_L + Q_R\rvert\) [e] | ≤ **1e−6** each chunk |
| polarity after 600 fs | \(Q_L < 0 < Q_R\) |
| walls immobile | max \(\lvert\Delta z\rvert\) ≤ **1e−10** Å |
| \(\Psi(z)\) (80 bins over [0, 40 Å)) | \(\Psi(z_R) - \Psi(z_L) > 0\) (applied ΔΨ = +2 V) |
| ion confinement (32 bins over [12, 28 Å)) | all 12+12 ions between the walls at the final step |
| run-mean profile normalization | \(\sum_z \rho_s A\,dz = 12 \pm\) **1e−6** per species |
| EDL layering (4 Å first layer) | printed diagnostics (0.6 ps — not asserted) |

### Performance

`perf_exempt`: a product box, not a kernel twin; the wall ratio vs MetalWalls
is not measured.

### Limitations

- **Single-site CG IL, non-polarizable, unscaled ±1 e charges** — not
  FF-sensitive observables (density profiles, differential capacitance vs
  experiment). Polarizable ILs are on [polarizable](md-polarizable.md#pim-npt-polarizable-il).
- **0.6 ps trajectory**: EDL layering numbers are printed diagnostics, not
  converged structure.
- **Planar walls**; the slit-pore geometry is the `nanoporous-carbon-const-pot`
  section.
- Composition gate only: CG, ASPC, TF metallicity, per-electrode η and the MPI
  halo are pinned by their own sections.

## const-pot-md-nvt-short — Short NVT, parallel-plate NaCl toy {#const-pot-md-nvt-short}

Fluctuating electrode charges at **fixed potential**, **frozen walls**, and a
short **NVT** trajectory of a neutral electrolyte between parallel plates, on
`cpu::ConstPotentialBO` + `cpu::md::FixFreeze`. The dense Gaussian matrix
\(A\) is rebuilt and inverted each `pre_force` step; force evaluation stays
point-charge `CoulLong` + `KSpacePME` (Gaussian force kernels are
[`gaussian-electrode-charge`](#gaussian-electrode-charge)). Walls are held by
`FixFreeze` (force zero + velocity zero + position restore; LAMMPS
`fix setforce 0` family); mobile ions run Velocity-Verlet with Nosé–Hoover.

### Method

At fixed applied potentials \(\Psi\), charges \(Q\) minimise the quadratic
energy

\[
U = \tfrac12 Q^{\mathsf T} A Q - Q^{\mathsf T}(\Psi - \Phi_{\mathrm{electrolyte}}).
\]

Stationarity plus global electroneutrality yields

\[
A Q = \Psi - \Phi_{\mathrm{electrolyte}},\qquad
\sum_i Q_i = -Q_{\mathrm{electrolyte}},
\]

with Gaussian width \(w\) (\(\eta = 1/w\) in the form above):

\[
A_{ii} = k_e\sqrt{2/\pi}/w,\quad
A_{ij} = k_e\,\mathrm{erf}\!\big(r_{ij}/(w\sqrt2)\big)/r_{ij}\ (i\neq j).
\]

\(\Phi_i\) is the min-image potential at electrode site \(i\) from
**non-electrode** (point) charges. For a neutral electrolyte,
\(\sum_i Q_i = 0\).

### System

| Item | Value |
| ---- | ----- |
| Species | Na⁺ type 0, Cl⁻ type 1, left wall type 2, right wall type 3 |
| Box | orthorhombic \(L_x=L_y=14\) Å, \(L_z=20\) Å |
| Walls | \(2\times 2\) sites at \(z=2\) Å (type 2) and \(z=18\) Å (type 3) |
| Electrolyte | 4 Na⁺ + 4 Cl⁻ on staggered mid-slab lattice (\(z\in\{6,14\}\) Å) |
| \(\Psi_L, \Psi_R\) | −0.5 V, +0.5 V |
| \(w\) | 0.5 Å (η = 2 Å⁻¹) |
| Soft LJ | \(\varepsilon=0.05\) eV, \(\sigma=2.0\) Å, cut 5 Å |
| Coulomb / PME | \(\alpha=0.3\) Å⁻¹, real-space cutoff 6 Å, PME spacing 0.5 Å, order 5 |
| NVT | `VelocityVerlet` + `NoseHoover`, \(T=300\) K, \(\tau_T=100\) fs, \(dt=0.5\) fs |
| Trajectory | 400 steps (8 chunks of 50 after 1 seed step) |
| Freeze | wall types 2 and 3 |

### Code

```cpp
cpu::ConstPotentialBO cpot(
    /*electrode_types*/ std::vector<int>{2, 3},
    /*potentials_V*/    std::vector<double>{-0.5, +0.5},
    /*gaussian_width_A*/ 0.5);

cpu::md::MD sim;
sim.set_observer(cpu::md::InitVelocity{/*T=*/300.0, /*seed=*/7u});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.set_thermostat(cpu::md::NoseHoover{/*T=*/300.0, /*tau_T=*/100.0, /*dt=*/0.5});
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.05, /*sigma=*/2.0, /*cut=*/5.0});
sim.set_neighbor(cpu::md::NeighborList<>{/*cut=*/6.0, /*skin=*/0.5});
sim.on(md::Phase::force).add(cpu::CoulLong{/*alpha=*/0.3, /*cutoff=*/6.0});
sim.set_evaluator(cpu::KSpacePME{/*alpha=*/0.3, /*spacing=*/0.5, /*order=*/5});
sim.set_observer(std::move(cpot));           // default: pre_force @ -500
sim.set_observer(cpu::md::FixFreeze{2});     // left wall type
sim.set_observer(cpu::md::FixFreeze{3});     // right wall type
sim.build();
sim.run(frame, 400);
// Electrode totals: sum kAtom_charge over types 2 / 3; Q_L < 0 < Q_R.
```

### Oracle

MetalWalls `c8ad175d`, `num_pbc 3` twin (matches ATV's 3D PME cell),
`total_charges.out`:

| step | Q_C1 [e] | Q_C2 [e] |
|------|----------|----------|
| 0 | **−9.711155787094686×10⁻²** | **+9.711155787094686×10⁻²** |
| 1 | −9.742254974821957×10⁻² | +9.742254974821957×10⁻² |

ATV uses PME where MetalWalls uses Ewald; the ~3 % residual is the model gap,
hence the soft 5 % gate. Continuum diagnostic: the vacuum parallel-plate
capacitance \(C \approx A/(4\pi k_e d)\) with \(A=L_x L_y\), \(d\approx 16\) Å,
\(\Delta\Psi = 1\) V gives \(\lvert Q\rvert \sim 0.05\,e\) — same sign and
scale.

### Accuracy

| Quantity | Atomiverse / pin | Oracle | Bound | Notes |
| -------- | ---------------: | ------ | ----- | ----- |
| Polarity (after first BO solve) | \(Q_L < 0 < Q_R\) | MetalWalls sign | hard | Higher-\(\Psi\) plate positive |
| \(\lvert Q_L\rvert\) vs step 0 | 9.711155787094686e−2 e | MetalWalls | rel ≤ **5 %** | PME vs Ewald residual ~3 % |
| Electroneutrality (step 1) | \(\lvert Q_L+Q_R\rvert\) | MW neutrality | \(\le 10^{-8}\) e | Neutral electrolyte |
| Neutrality during NVT chunks | \(\lvert Q_L+Q_R\rvert\) | MW neutrality | \(\le 10^{-6}\) e | Looser for PME / min-image \(\Phi\) noise |
| Thermo | \(E\), \(K\) | finite every chunk | finite | no NaN |
| Charges | all \(q_i\) | finite | finite | Charge-solve stability |
| Freeze position | wall \(\Delta x\) | FixFreeze / MW `mobile=false` | \(\le 10^{-10}\) Å | Per wall atom vs build snapshot |
| Freeze velocity | wall \(\lvert v\rvert\) | same | \(\le 10^{-12}\) Å/fs | After NVT |
| End polarity | \(Q_L < 0 < Q_R\) | MW | hard | Polarity survives 400 steps |
| Fluid not frozen | ion \(\lvert v\rvert\) or \(K\) | selective freeze | any ion \(v^2>10^{-12}\) or \(K>0.01\) eV | Walls only |

### Performance

`perf_exempt`: a short product trajectory, not a kernel microbench; no fair
wall ratio vs MetalWalls is measured.

### Limitations

- **Short NVT smoke product**, not production EDL sampling: 16 atoms, 400
  steps, soft LJ, toy \(2\times 2\) electrode grids.
- **Point-charge force path** after the BO \(Q\) write.
- **Dense \(A^{-1}\)** only — large electrodes need CG
  ([`const-pot-cg-large-electrode`](#const-pot-cg-large-electrode)).

## per-electrode-eta — Per-electrode Gaussian width η {#per-electrode-eta}

Each electrode group carries its own Gaussian width \(\eta_e\) [1/Å] instead
of one shared value — MetalWalls' per-`species` `charge gaussian` input —
through the 3-argument `ConstPotBO::add_electrode(type, Psi, eta)` overload;
the 2-argument overload inherits the shared constructor `eta` (a group `eta`
of 0 means "inherit"). Dissimilar widths mix in the off-diagonals via
\(\eta_{ij}=\eta_i\eta_j/\sqrt{\eta_i^2+\eta_j^2}\); only the elastance
elements (\(A_{ii}\), \(A_{ij}\), \(b_i\)) change, so electroneutrality, TF
hardness (`l_tf`), CG and ASPC compose unchanged. Widths are per electrode
*group*, not per atom.

### Method

Closed-form elements (Reed, Lanning, Madden, *J. Chem. Phys.* **126**, 084704
(2007), DOI [10.1063/1.2464084](https://doi.org/10.1063/1.2464084); Scalfi et
al. (2020); MetalWalls *JOSS* **5**(53), 2373 (2020)):

\[
A_{ii} = k_e\sqrt{2/\pi}\;\eta_i,
\qquad
A_{ij} = k_e\,\frac{\mathrm{erf}(\eta_{ij} r_{ij})}{r_{ij}},
\qquad
\eta_{ij} = \frac{\eta_i\eta_j}{\sqrt{\eta_i^2+\eta_j^2}},
\]

with RHS \(b_i = k_e\,q_j\,\mathrm{erf}(\eta_i r)/r\). Smaller \(\eta\) (wider
Gaussian) lowers \(A_{ii}\), so at fixed \(\Delta\Psi\) the induced charge —
and the capacitance — grows monotonically as either wall's width shrinks.
MetalWalls uses exactly this convention (`etaij` in its 2D short-range
kernel, the \(\sqrt{2}\,\eta_i\) self term, and the same RHS).

### Case 1 — two-site vacuum capacitor, dissimilar η (exact pin)

| Quantity | Value |
| -------- | ----- |
| \(\eta_1\) (bottom) | **1.979** Å⁻¹ (LAMMPS ELECTRODE default, Ahrens-Iwers et al. 2022; MetalWalls `charge gaussian 1.04724169991593` bohr⁻¹) |
| \(\eta_2\) (top) | **0.955** Å⁻¹ (a deliberately dissimilar width; MetalWalls' graphite width is 0.955234657 bohr⁻¹ = 1.805131886 Å⁻¹) |
| Separation \(d\) | 4 Å (sites at \(z = 18, 22\) Å; cubic box 40 Å) |
| \(\Psi_{\mathrm{bot}},\Psi_{\mathrm{top}}\) | 0 V, 1 V (\(\Delta\Psi=1\) V) |
| Electroneutrality | on (\(Q_2=-Q_1\)); default `xy_image_range = 0` (image-free) |

\[
Q = \frac{\Delta\Psi}{A_{11}+A_{22}-2A_{12}} = \mathbf{0.037722100829787}\ e,
\]

with \(A_{11}=22.737234510126967\), \(A_{22}=10.972237977347778\),
\(A_{12}=3.599907138217925\), \(\eta_{12}=0.860091344088193\) Å⁻¹. Shared-η
twins of the same geometry bracket it:
\(Q(\eta{=}1.979\ \text{shared}) = 0.026126955854989\ e <
Q_{\mathrm{mixed}} < Q(\eta{=}0.955\ \text{shared}) \approx 0.0678\ e\).

**MetalWalls `c8ad175d` confirmation.** MetalWalls is always periodic, so its
twin carries the 2D image lattice (\(L_z = 20\,L_{xy}\)) and the pin is the
\(L\to\infty\) limit after removing the absolutely convergent dipole-lattice
term. At \(L_{xy} = 320\) Å MetalWalls prints
\(Q_{\mathrm{top}} = 0.037722009201931\) e, image-corrected
\(Q_{\mathrm{iso}} = 0.037722099576710\) e, which equals the closed form with
the exact \(k_e = 14.399645478456184\) eV·Å/e² to +8.3e-12; the −3.322e-08
offset from the pin is entirely the 8-digit `units::coulomb` literal. Both
shared-η twins close the same way (MetalWalls 0.026126954987021 and
0.067816772335944 e), so the per-electrode \(\eta_{ij}\) mixing rule itself is
confirmed.

### Case 2 — multi-atom two-wall, mixed η, 2D xy Gaussian images

2×2-atom square walls (\(a = 2\) Å, \(L_x=L_y=4\) Å) at \(z = 8\) and 13 Å
(separation 5 Å) in a 4×4×24 Å box; \(\eta_{\mathrm{bot}}=1.979\),
\(\eta_{\mathrm{top}}=0.955\) Å⁻¹, \(\Delta\Psi=1\) V,
`xy_image_range = 100`. Asserted: electroneutrality, the sign of
\(Q_{\mathrm{top}}\) follows \(\Delta\Psi\), the mixed-η solution lands
**strictly between** its two shared-η twins, and the 3-arg overload at equal
η reproduces the 2-arg shared path (≤ 1e-14 e/site). The charge itself is not
pinned: at \(R = 100\) it sits +0.631 % above the converged MetalWalls value
\(1.980450776415\times10^{-2}\) e (3×3 supercell), and the image sum
Richardson-extrapolates to it within 1.7e-8.

### Code

```cpp
cpu::Frame frame(2);
const double z[2] = {18.0, 22.0};          // x = y = 0
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_type)[i] = i + 1;
    frame.span(kAtom_mass)[i] = 196.97;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(40.0);

cpu::md::ConstPotBO conp(/*shared eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0, /*eta [1/Å]=*/1.979);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0, /*eta [1/Å]=*/0.955);
conp.electroneutrality = true;
// ... bind with sim.set_observer_ref(conp) and run one step ...

const double q_top = conp.total_charge(1);        // [e] → 0.037722100829787
const double resid = conp.neutrality_residual();  // ≤ 1e-12
```

### Accuracy

| Quantity | Atomiverse gate | Pin | Tol | Notes |
| -------- | --------------: | --: | --- | ----- |
| Case 1 \(Q_{\mathrm{top}}\) [e] | within rel 1e-9 of pin | **+0.037722100829787** | rel ≤ **1e-9** | \(\eta_1=1.979\), \(\eta_2=0.955\) Å⁻¹, \(d=4\) Å, \(\Delta\Psi=1\) V |
| Case 1 \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-12}\) | 0 (electroneutral) | **1e-12** | projector on |
| Case 2 \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | 0 (electroneutral) | **1e-10** | 2×2 walls, `xy_image_range = 100` |
| Case 2 \(\mathrm{sign}(Q_{\mathrm{top}})\) | \(+1\) at \(\Delta\Psi=+1\) V | \(+\) | exact | capacitive polarity |
| Case 2 bracket \(Q^{\eta=1.979} < Q^{\mathrm{mixed}} < Q^{\eta=0.955}\) | required | strict ordering | exact | smaller η ⇒ smaller \(A_{ii}\) ⇒ more charge |
| 3-arg `add_electrode` at equal η vs 2-arg shared path [e/site] | identical | 0 | **1e-14** | overload equivalence |

The 1e-9 tolerance is a code-vs-its-own-closed-form check at ATV's \(k_e\);
the MetalWalls agreement above is the oracle statement.

### Performance

Matched twin (4×4×24 Å box, 2×2 walls, one BO solve, single rank,
`OMP_NUM_THREADS=1`): MetalWalls 0.32–0.76 s (median 0.44) for one solve plus
2D Ewald setup and I/O; the ATV section 0.04 s for five BO solves. Ratio
≈ 0.09, start-up dominated on both sides — "same order, ATV not slower", not a
kernel benchmark.

### Limitations

- **Vacuum capacitor pins only** — no electrolyte RHS exercised at mixed η
  (the mixed-η \(b_i\) uses the same published erf form).
- **Dense path** in the pins (CG is optional on the same component; not
  exercised here).

## gaussian-electrode-charge — Gaussian electrode charge kernels {#gaussian-electrode-charge}

Electrode atoms are modelled as **normalized 3D Gaussian charge densities** of
width \(\eta\) [Å⁻¹], not point charges. This section covers the closed-form
Coulomb Green function (Reed–Lanning–Madden; MetalWalls electrode Gaussians)
as a real-space pair potential `cpu::PairGaussianCoul`
(`pair_gaussian_coul.h`, `src/cpu/potentials/pair_gaussian_coul.cpp`) and the
header-only math `cpu::GaussianChargeKernel` (`gaussian_charge.h`) behind the
BO matrix \(A\). `PairGaussianCoul` takes a hard cutoff, a per-`kAtom_type`
\(\eta\) table (\(\eta\le 0\) ⇒ point ion) and a self-energy flag (default
**on**); it is a `NeighborConsumer`, so the driver requires a neighbor list.

### Method

\[
\rho_i(\mathbf{r}) = q_i \frac{\eta_i^3}{\pi^{3/2}}
  e^{-\eta_i^2|\mathbf{r}-\mathbf{R}_i|^2},
\qquad
\eta_{ij}=\frac{\eta_i\eta_j}{\sqrt{\eta_i^2+\eta_j^2}},
\quad
G_{ij}(r)=\frac{\mathrm{erf}(\eta_{ij} r)}{r},
\quad
G_{ii}=\eta_i\sqrt{\frac{2}{\pi}}.
\]

\[
U=\texttt{coulomb}\Big[
  \sum_{i<j} q_i q_j G_{ij}(r_{ij})
  +\tfrac12\sum_i q_i^2 G_{ii}\Big].
\]

| Pair kind | Kernel |
| --------- | ------ |
| Gaussian–Gaussian | \(G=\mathrm{erf}(\eta_{ij} r)/r\); self \(G_{ii}\) when enabled |
| Gaussian–point | \(\eta_j\le 0\) flag ⇒ \(\eta_{ij}=\eta_i\) |
| Point–point | \(G=1/r\); **no** self term |

Forces use the analytic radial derivative

\[
\frac{dG}{dr}
  = \frac{2\eta_{\mathrm{eff}}}{\sqrt{\pi}}
    \frac{e^{-(\eta_{\mathrm{eff}} r)^2}}{r}
    - \frac{\mathrm{erf}(\eta_{\mathrm{eff}} r)}{r^2}
\]

(point–point: \(-1/r^2\)).

**Why not point-charge \(A\).** Replacing the Gaussian kernel by
\(A_{ij}=\texttt{coulomb}/r_{ij}\) (no self diagonal) drops \(G_{ii}\) and
replaces \(\mathrm{erf}(\eta r)/r\) by \(1/r\); on the two-site toy below
\(Q\) changes by \(O(1)\) and **flips sign**. Electrode \(A\) must use the
Gaussian kernel; point-charge \(A\) is a documented approximation only.

### System

Fixed-\(Q\) energy: cubic box \(L=100\) Å, cutoff **40** Å (every pair inside
the cutoff, so the neighbor list changes which loop runs, never which pairs
are summed).

| Atom | type | \(\eta\) [Å⁻¹] | \(q\) [e] | \((x,y,z)\) [Å] |
| ---- | ---- | -------------- | --------- | --------------- |
| L0 | 1 | 1.0 | +0.25 | (0, 0, 0) |
| L1 | 1 | 1.0 | +0.25 | (2, 0, 0) |
| R0 | 1 | 1.0 | −0.25 | (0, 0, 10) |
| R1 | 1 | 1.0 | −0.25 | (2, 0, 10) |
| ion | 0 | point (\(\le 0\)) | +1 | (1, 0, 5) |

Toy BO: two Gaussian sites at \(z=0\) and \(z=L=10\) Å, \(\eta=1.0\) Å⁻¹,
\(\Psi=(+0.5,-0.5)\) eV/e, neutrality \(Q_1+Q_2=0\):

\[
Q_1=\frac{\Psi_1-\Psi_2}{2(A_{11}-A_{12})},\quad
A_{11}=\texttt{coulomb}\,G_{ii},\quad
A_{12}=\texttt{coulomb}\,G(L).
\]

| Model | \(A_{11}\) | \(A_{12}\) | \(Q_1\) [e] |
| ----- | ---------- | ---------- | ----------- |
| Gaussian | \(\texttt{coulomb}\,\eta\sqrt{2/\pi}\) | \(\texttt{coulomb}\,\mathrm{erf}(\eta_{\mathrm{eff}} L)/L\) | **+0.0497547591576** |
| Point-charge \(A\) (approx) | 0 (no self) | \(\texttt{coulomb}/L\) | −0.347230782426 |

### Code

```cpp
cpu::Evaluate eval;
eval.set_evaluator(cpu::PairGaussianCoul{
    /*cutoff [Å]=*/40.0,
    /*eta_by_type=*/{0.0, 1.0},  // type 0 point, type 1 Gaussian
    /*include_self=*/true});
eval.set_neighbor(cpu::md::NeighborList<>{40.0, /*skin=*/1.0});
eval.build();
eval.run(f);
const double E = f.meta.get(kEnergy);  // [eV]

// Math-only (BO matrix elements; no Frame):
const double Gii = cpu::GaussianChargeKernel::self_greens(/*eta=*/1.0);
const double Gij = cpu::GaussianChargeKernel::greens(/*r=*/10.0, 1.0, 1.0);
// A_ij = units::coulomb * G_ij  [eV/e²]
```

### Oracle and accuracy

Oracle = **literature**: Reed, Lanning, Madden, *J. Chem. Phys.* **126**,
084704 (2007) Gaussian-charge Coulomb and the MetalWalls / Scalfi electrode
form. The pins are closed-form evaluations of the same published Green
function MetalWalls uses, re-evaluable with `units::coulomb`.

| Quantity | Atomiverse / pin | Oracle (literature) | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E\) [eV] (fixed \(Q\)) | **1.938689655050537** | same closed form | \(\lvert\Delta E\rvert < 10^{-9}\) eV | Direct eval of Reed/MW \(G\) |
| \(Q_1\) [e] (Gaussian \(A\)) | **+0.0497547591576** | same closed form | \(\lvert\Delta Q\rvert < 10^{-9}\) e | Two-site BO + neutrality |
| \(\lvert Q_1^{\mathrm{gauss}}-Q_1^{\mathrm{point}}\rvert\) | \(> 10^{-6}\) e (required) | documents point-\(A\) error | — | Point \(A\): \(Q_1=-0.347230782426\) e |
| \(\lim_{r\to 0} G(r)\) vs \(G_{ii}\) | matched | analytic limit | \(10^{-6}\) | Self-consistency sanity |

### Performance

`perf_exempt`: the pin is an analytic closed-form kernel, not a wall-time
twin.

### Limitations

- **Kernel + pair only.** The BO loop, observers and electrode MD are the
  other sections.
- **Formula identity, not bit parity** with a MetalWalls binary.
- No reciprocal-space / slab 2D Ewald for Gaussian electrodes
  ([slab 2D Ewald](longrange-ewald-pme.md#slab-2d-ewald-foothold)).

## electrode-charge-predictor — Kolafa ASPC warm-start for const-pot CG {#electrode-charge-predictor}

The electrode charge vector \(Q(t)\) is re-solved every step as the
electrolyte moves; over MD \(A\) is often quasi-static (frozen walls) while
\(b\) changes smoothly. With iterative CG, cold-starting from zero (or a stale
\(Q\)) burns unnecessary Krylov iterations. `cpu::electrodes::ChargeASPC`
(`charge_aspc.h`, order as constructor argument) applies Kolafa's
Always-Stable Predictor-Corrector to the electrode charges, and
`ConstPotBO` with `use_cg = true` and `predictor_order = K` warm-starts CG from
the extrapolated guess, then records the converged unconstrained solution
(`last_iters()`, `last_used_predictor()` report the result). LAMMPS ELECTRODE
has no ASPC path, so the oracle is MetalWalls.

### Method

With order \(K\), the last \(K+1\) converged charge vectors
\(\{Q_{n-K},\ldots,Q_n\}\) (oldest → newest) predict

\[
Q^{\mathrm{pred}}_{n+1} = \sum_{k=0}^{K} \omega_k\, Q_{n-K+k},
\]

with Kolafa's coefficients (J. Comput. Chem. **25**, 335 (2004), Eq. 10)

\[
B_j = (-1)^{j+1}\, j\, \frac{\binom{2K+2}{K+1-j}}{\binom{2K}{K}},
\qquad \omega_{K+1-j} = B_j .
\]

For \(K=2\): \(\omega = \{1/2,\ -2,\ 5/2\}\), i.e.
\(Q^{\mathrm{pred}}_{n+1} = \tfrac12 Q_{n-2} - 2 Q_{n-1} + \tfrac52 Q_n\); for
MetalWalls' \(K=5\),
\(B = \{22/7,\,-55/14,\,55/21,\,-22/21,\,5/21,\,-1/42\}\) newest → oldest.

These are **not** Lagrange (polynomial) extrapolation coefficients. ASPC is
exact only through degree 1 — \(\sum_j B_j = 1\) and \(\sum_j j B_j = 0\), so a
linear history \(Q_m = m\) predicts \(Q_{n+1} = n+1\) exactly, while a quadratic
history \(Q_m = m^2\) is deliberately damped (\(K=2\) gives 8, not the Lagrange
9). That damping makes the predictor-corrector pair unconditionally stable.

`ChargeASPC::correct` implements Kolafa's time-reversibility blend

\[
Q_{\mathrm{corr}} =
  \frac{K+2}{2K+3}\,Q_{\mathrm{scf}}
  + \frac{K+1}{2K+3}\,Q_{\mathrm{pred}}
\]

for callers that want the full corrector; the `ConstPotBO` CG path uses
`predict` and records the unconstrained SCF solution, and the section gates
prediction + CG iteration reduction, not the blend.

### System

| Item | Setting |
| ---- | ------- |
| ASPC order | \(K=5\) coeffs == MetalWalls `B1..B6` (rel ≤ 1e-15); \(K=2\) coeffs \(\{1/2,-2,5/2\}\) (exact) |
| CG vs dense | 4-atom madelung family, `xy_image_range = 0`, \(\eta=2\) Å⁻¹, \(\Psi=(0,1)\) V |
| CG iter comparison | 3×3 atoms/wall (**18** electrode sites) + 4 ions, \(L_{xy}=6\) Å, \(L_z=20\) Å |
| Drift protocol | electrolyte \(z\) shifts each launch so \(b\) changes; primary cell only; 16 steps |
| Neutrality | on; \(\lvert\sum Q_i\rvert \le 10^{-8}\,e\) |

### Code

```cpp
cpu::electrodes::ChargeASPC aspc(/*order K=*/2);
// coefficient(k): k=0 oldest in the K+1 window → {1/2, -2, 5/2}
cpu::electrodes::ChargeASPC mw(/*order K=*/5);
assert(mw.coefficient(5) == 22.0 / 7.0);   // B1
for (int n = 0; n <= 2; ++n)
    aspc.record(std::vector<double>{static_cast<double>(n)});
std::vector<double> pred;
const bool ok = aspc.predict(pred);  // needs K+1 history points; pred[0] == 3

cpu::md::ConstPotBO conp(/*eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
conp.electroneutrality = true;
conp.xy_image_range = 0;
conp.use_cg = true;            // iterative path required for ASPC gain
conp.cg_tol = 1e-10;
conp.max_cg_iters = 500;
conp.predictor_order = 2;      // K=2; 0 = cold start every step

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
sim.set_evaluator(cpu::PairCoulCut{12.0});
sim.set_integrator(cpu::md::VelocityVerlet{0.0});
sim.set_observer_ref(conp);
sim.build();
for (int s = 0; s < 16; ++s) {
    // ... shift electrolyte z on the Frame ...
    sim.run(frame, 1);         // continuable: BO/ASPC state persists
}
// after K+1 records: conp.last_used_predictor() == true;
// conp.last_iters() = CG iterations of the last solve
```

### Oracle

MetalWalls `c8ad175d` hard-codes the \(K=5\) row in
`src/electrode_charge.F90:572-577` (read with `sed -n '556,596p'` at that
commit) and applies it to the six stored charge vectors. The binary exposes no
switch to disable its predictor (it turns on at step 7, `src/system.F90:1430`)
and its non-predictor branch initialises from the previous step rather than
from zero, so the "warm < cold" claim is not A/B-testable there; the
coefficient row is a source-level oracle read. Other algorithm sources:
A. Coretti et al., *J. Chem. Phys.* **157**, 184801 (2022) (const-pot /
electrode dynamics context).

### Accuracy

| Quantity | Atomiverse gate / pin | Oracle / theory | Tol | Notes |
| -------- | --------------------: | --------------: | --- | ----- |
| ASPC \(K=5\) coeffs | \(\{22/7,-55/14,55/21,-22/21,5/21,-1/42\}\) | MetalWalls c8ad175d `electrode_charge.F90:572-577` | rel ≤ **1e-15** | `ChargeASPC::coefficient` |
| ASPC \(K=2\) coeffs \(\omega\) | \(\{1/2,-2,5/2\}\) | Kolafa 2004 Eq. 10 | **exact** | `ChargeASPC::coefficient` |
| Linear history predict | \(Q_k=k\) → \(Q^{\mathrm{pred}}=3\) after \(\{0,1,2\}\) | degree-1 exact | abs ≤ **1e-12** | |
| Quadratic history damped | \(Q_k=k^2\) → 8 (Lagrange would give 9) | ASPC-vs-Lagrange discriminator | abs ≤ **1e-12** | |
| CG \(Q_{\mathrm{top}}\) vs dense Cholesky | same madelung RHS | dense BO twin | abs ≤ **1e-8** e | `use_cg` on/off |
| Mean CG iters, post history fill | warm **<** cold | MW ASPC intent | **strict inequality** | observed cold≈6, warm≈5 on 18-site plate |
| ASPC engaged after fill | `last_used_predictor() == true` for \(s\ge 6\) at \(K=2\) and \(K=5\) | needs \(K+1\) records | exact | 16-step drift protocol |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-8}\) | neutrality on | **1e-8** | every warm launch |
| Polarity at \(\Delta\Psi=+1\) V | \(Q_{\mathrm{top}}>0\), \(Q_{\mathrm{bot}}<0\) | capacitive | sign only | multi-site end state |

### Performance

`perf_exempt`: the two codes' charge solves are not comparable work (see
above), so there is no fair wall ratio. The iteration gate (≈5 vs ≈6) is an
algorithm gate, not \(T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\).

### Limitations

- **No black-box parity.** MetalWalls prints no predictor output, so there is
  no charge-by-charge comparison against a MetalWalls log; CG vs dense and the
  iteration reduction are internal-consistency gates.
- **ASPC gain requires `use_cg = true`.** Dense Cholesky ignores
  `predictor_order` for iteration counts (`last_iters()` is 0 on dense).
- **History fill latency.** Prediction needs \(K+1\) successful `record`
  calls; the first \(K\) steps after `build` / `reset` are cold.
- **Unconstrained record.** History stores the unconstrained CG solution used
  for warm-start; the electroneutrality projection still applies to the
  written-back charges.

## freeze-electrode-atoms — Freeze wall atoms (`FixFreeze`), LJ NVE {#freeze-electrode-atoms}

Immobile electrode (wall) atoms under classical LJ MD: selected atom **type**
rows keep \(v=\mathbf{0}\) and fixed positions while the fluid evolves under
NVE velocity-Verlet. `cpu::md::FixFreeze{type_id}` (`fix_freeze.h`,
`src/cpu/dynamics/fix_freeze.cpp`) is a `Constraint` / `ConstraintProvider`
bound into the MD preset's stages — the LAMMPS twin of
`fix setforce 0.0 0.0 0.0` on a wall group plus `velocity wall set 0 0 0`, and
the same end-state as MetalWalls' `mobile = false` electrodes.

### Method

LAMMPS (`fix_setforce.cpp`, type groups):

\[
\mathbf{F}_i \leftarrow \mathbf{0}
\quad\text{for } i\in\text{wall group, after force accumulate (post\_force).}
\]

With \(\mathbf{v}_i(0)=\mathbf{0}\) and \(\mathbf{F}_i=\mathbf{0}\),
velocity-Verlet leaves wall positions unchanged. `FixFreeze` is a superset:

| Stage | Binding | Action |
| ----- | ------- | ------ |
| `post_force` | order \(+1000\) | zero \(\mathbf{F}\) on frozen type (after physical + Langevin at \(+500\)) |
| `post_integrate` | order \(0\) | restore build-time \(\mathbf{x}\) by tag; zero \(\mathbf{v}\) |
| `final_integrate` | order \(+200\) | re-zero \(\mathbf{v}\) after the final half-kick |

At `build()` wall rows are resolved, positions snapshotted keyed by stable
global **tag**, and wall \(v\)/\(F\) zeroed (so `InitVelocity` cannot leave
wall momentum). `ConstraintProvider::n_constraints` reports **3 DOF per frozen
atom** so Ndof-consuming thermostats stay consistent. Selection is one integer
type id; multi-type freeze composes several `FixFreeze`.

### System

| Quantity | Value |
| -------- | ----- |
| Box | cubic \(L = 10\) Å, PBC |
| Wall (type 2) | 4 atoms at \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å; mass 196.9665 amu |
| Fluid (type 1) | 4 atoms at \(z=3.5\) above each wall site; mass 39.948 amu |
| Pair | LJ 12-6, \(\varepsilon = 0.010341\) eV, \(\sigma = 3.4\) Å, \(r_c = 8.0\) Å (all type pairs) |
| Neighbor | skin 0.3 Å |
| Ensemble | NVE velocity-Verlet, \(\Delta t = 1\) fs |
| Steps (correctness) | 1000 |
| Wall initial \(v\) | \(\mathbf{0}\) |
| Fluid initial \(T\) | Maxwell–Boltzmann 200 K (seed 99; RNG **not** bit-matched to LAMMPS) |

LAMMPS `units metal` time is ps; the pin inputs use `timestep 0.001` (ps)
\(\Leftrightarrow\) `dt = 1.0` fs.

### Code

```cpp
cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{/*eps [eV]=*/0.010341, /*sigma [Å]=*/3.4,
                                 /*cutoff [Å]=*/8.0});
sim.set_neighbor(cpu::md::NeighborList<>{8.0, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.set_observer(cpu::md::InitVelocity{/*T [K]=*/200.0, /*seed=*/99u});
sim.set_observer(cpu::md::FixFreeze{/*wall type=*/2});
sim.build();
sim.run(frame, 1000);
```

### Oracle and accuracy

LAMMPS develop / `patch_30Mar2026-1074-g7f680de296` (30 Mar 2026
Development), `fix setforce 0.0 0.0 0.0` on group `wall`; the dump after 1000
steps (`wall_final.lammpstrj`) shows the wall atoms at create coordinates with
zero velocities.

| Observable | Atomiverse / pin | LAMMPS | Tol / gate | Notes |
| ---------- | ---------------: | -----: | ---------- | ----- |
| Wall positions after 1000 steps | create \((2,2,0)\), \((2,6,0)\), \((6,2,0)\), \((6,6,0)\) Å | identical (dump) | \(\max\|x_i(t)-x_i(0)\| \le 10^{-10}\) Å | by stable tag (sort-safe) |
| Wall velocities | all components 0 | all components 0 | \(\max\|v\| \le 10^{-12}\) Å/fs | metal dump v is Å/ps; pin is zero either way |
| Wall \(z\) span | \(z\in[0,0]\) | thermo `c_min_dz`/`c_max_dz` | exact | throughout |
| Fluid displacement | \(\ge 1\) atom moves | fluid free under LJ | \(\max\) fluid disp \(> 10^{-3}\) Å | freeze is **selective**, not global |

Fluid coordinates are **not** golden-matched (RNG and velocity-init
conventions differ between engines).

### Performance

Same geometry, **20000** NVE steps, `OMP_NUM_THREADS=1`, no dump/thermo
(`in.bench`); ATV time instrumented in the regression binary.

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ------------------------------------------: | ---- | ------ |
| `FixFreeze` + VV + LJ vs `fix setforce 0` + NVE | **0.0186** (1 rank, OMP=1) | **0.0740** | **3.98** | ✓ | Tiny \(N=8\); a fair-twin ceiling, not a production throughput claim |

### Limitations

- **Eight atoms** pin algorithm correctness and a fair ceiling, not large-slab
  throughput.
- **Type-only selection**; no region/id group predicates.
- **Superset vs pure setforce.** Position restore + multi-stage \(v\) zeroing
  is stricter than LAMMPS post_force alone; intermediate force arrays are not
  claimed bit-identical.
- **Single-rank regression.** MPI migration re-resolves rows each launch in
  product code; this section is single-rank.

## piston-electrode-pz — Piston electrode at constant normal pressure \(P_z\) {#piston-electrode-pz}

**Constant normal pressure** on a rigid electrode wall: one atom type is a
**rigid piston** free only along \(z\), with external force
\(F_{\mathrm{ext}}=\pm P_{\mathrm{target}}A\) so that
\(\langle P_z\rangle\to P_{\mathrm{target}}\). `cpu::md::PistonElectrode`
(`piston_electrode.h`, `src/cpu/dynamics/piston_electrode.cpp`; constructor
`(type, P [eV/Å³], Side)`, optional member `tau_damp`) is a Component on
`cpu::md::MD`, orthogonal to the charge solve — a rigid electrode plate, not a
box barostat (`fix npt`), and not LAMMPS `fix wall/piston` (SHOCK package
shock kinematics).

### Method

With wall area \(A=L_x L_y\):

\[
F_z^{\mathrm{atoms}}=\sum_{i\in t} F_{z,i},\quad
F_{\mathrm{ext}}=\begin{cases}
-P_{\mathrm{target}} A & \text{Upper}\\
+P_{\mathrm{target}} A & \text{Lower}
\end{cases}
\]

\[
F_{\mathrm{net}}=F_z^{\mathrm{atoms}}+F_{\mathrm{ext}}(+F_{\mathrm{damp}}),\quad
F_{z,i}\leftarrow F_{\mathrm{net}}\,m_i/M,\quad F_{x,i}=F_{y,i}=0.
\]

Optional COM friction (equilibration aid, default off):

\[
F_{\mathrm{damp}}=-\frac{M\,v_{\mathrm{com},z}}{\tau_{\mathrm{damp}}\,\texttt{force\_to\_accel}}
\quad(\tau_{\mathrm{damp}}>0).
\]

Diagnostic pressure (atom forces only): `Side::Upper`
\(P_z=F_z^{\mathrm{atoms}}/A\), `Side::Lower` \(P_z=-F_z^{\mathrm{atoms}}/A\).
Rigid-body enforcement equalizes \(v_z\) (mass-weighted COM), zeros
\(v_x,v_y\), and restores in-plane / relative-\(z\) offsets snapshotted at
`build`. Default bindings:

| Phase | Order | Role |
| ----- | ----: | ---- |
| `md::Phase::post_force` | +900 | redistribute \(F_{\mathrm{net}}\) after physical forces; before `FixFreeze` (+1000) |
| `md::Phase::post_integrate` | 0 | rigid-body position/velocity enforcement |
| `md::Phase::final_integrate` | +200 | re-equalize \(v_z\), zero \(v_{xy}\) after second VV half-kick |

This matches MetalWalls `c8ad175d` (source-verified): one scalar velocity per
electrode, velocity Verlet on a rigid body (`piston_positions_update` /
`piston_velocities_update`), \(M\) the total electrode mass, \(A = L_x L_y\),
\(P_{\mathrm{sig}} = \text{dir}\times P\) (upper plate `dir = -1`), and the
`pressure` column `force_ions(3,1)/area`. Differences: ATV flips the lower
plate's displayed sign so both plates read positive under compression
(identical mechanics), `tau_damp` is an ATV extension, and the piston work
term of MetalWalls' Hamiltonian is not reported.

### System

| Species | type | role |
| --- | ---: | --- |
| Ar-like fluid | 1 | mobile LJ fluid (16 atoms, two layers) |
| lower wall | 2 | frozen (`FixFreeze`) at \(z=2\) Å — never freeze the piston type |
| upper wall | 3 | **piston** (`PistonElectrode::Side::Upper`) at \(z=8\) Å |

- Box: orthorhombic 10×10×20 Å (\(L_x=L_y=10\) Å, \(L_z=20\) Å) → \(A=100\) Å²
- Walls: \(2\times2\) atoms each, at \(x,y\in\{2.5,7.5\}\)
- Fluid: \(x=(i\%4+0.5)\cdot 2.5\), \(y=((i/4)\%2+0.5)\cdot 5\),
  \(z = 4.2\) for \(i<8\) else \(5.8\) (\(z_{\mathrm{lo}}+2.2\) /
  \(z_{\mathrm{hi}}-2.2\), repulsive contact)
- Pair: LJ \(\varepsilon=0.10\) eV, \(\sigma=3.0\) Å, \(r_c=4.5\) Å, skin 0.3 Å
  (\(r_c+\text{skin} < L/2\); at \(r_c = 6\) Å the single-rank min-image fold
  would drop second images, \(F_z = 82.660592\) vs the true lattice sum
  82.630417 eV/Å, and MetalWalls rejects `lj rcut > min(a,b,c)/2`)
- NVT fluid: VelocityVerlet + NoseHoover \(T=300\) K, \(\tau_T=50\) fs,
  \(dt=0.5\) fs
- Piston: \(P_{\mathrm{target}}=\) the probe \(P_z\) at the compressed start;
  \(\tau_{\mathrm{damp}}=20\) fs
- Warm 400 + prod 600 steps; \(P_z\) chunk-sampled every 20 steps

### Code

```cpp
// Probe: P_target = 0 → last_P_z() = F_z^atoms / A on the initial geometry.
cpu::md::PistonElectrode probe(/*type=*/3, /*P=*/0.0,
                               cpu::md::PistonElectrode::Side::Upper);
{
    cpu::md::MD probe_sim;
    probe_sim.set_evaluator(cpu::PairLJCut{0.10, 3.0, 4.5});
    probe_sim.set_neighbor(cpu::md::NeighborList<>{4.5, 0.3});
    probe_sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    probe_sim.on(md::Phase::post_force).add_ref(probe, +900);
    probe_sim.on(md::Phase::post_integrate).add_ref(probe, 0);
    probe_sim.on(md::Phase::final_integrate).add_ref(probe, +200);
    probe_sim.build();
    probe_sim.run(frame, 0);   // setup force evaluation = step-0 single point
}
const double P_pin = probe.last_P_z();  // [eV/Å³]

cpu::md::PistonElectrode piston(/*type=*/3, /*P [eV/Å³]=*/P_pin,
                                cpu::md::PistonElectrode::Side::Upper);
piston.tau_damp = 20.0;  // [fs], optional

cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{0.10, 3.0, 4.5});
sim.set_neighbor(cpu::md::NeighborList<>{4.5, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.set_thermostat(cpu::md::NoseHoover{/*T [K]=*/300.0, /*tau_T=*/50.0, /*dt=*/0.5});
sim.set_observer(cpu::md::InitVelocity{300.0, /*seed=*/42u});
sim.set_observer(cpu::md::FixFreeze{/*lower type=*/2});
sim.set_observer_ref(piston);
sim.build();
sim.run(frame, /*nsteps=*/1000);
// diagnostics: piston.last_P_z(), last_F_z_atoms(), last_F_net(), last_area()
```

### Oracle

MetalWalls `c8ad175d` twin of this 24-atom system, step 0,
`electrode_forces.out`, electrode 2 (`Wup` = the piston), column 15
`fz_lennard_jones = +1.609547904622151` Ha/Bohr (`com_z = 15.11780900366263`
Bohr \(= 8.000\) Å):

\[
F_z = 1.609547904622151 \times 51.42206749896734 = 82.7662809943017\ \mathrm{eV/Å},
\qquad
P_z = F_z / A = 8.27662809943017\times10^{-1}\ \mathrm{eV/Å^3}.
\]

The factor is 27.211386245988 eV/Ha ÷ 0.52917721067 Å/Bohr; the
energy/volume route (\(4.507193180712053\times10^{-3}\) Ha/Bohr³) gives the
same number, and a from-scratch min-image LJ sum gives 82.76628232744 eV/Å
(1.6e-8 relative, the CODATA drift). MetalWalls' piston exists only on a
constant-potential electrode, so its twin applies \(\Delta\psi = 0.5\) a.u.
to keep the CG solve well posed; the LJ column depends on geometry alone, and
electrode–electrode LJ (absent in MetalWalls) contributes exactly 0 here.

### Accuracy

| Quantity | Atomiverse gate / pin | Oracle / identity | Tol | Notes |
| -------- | --------------------: | ----------------: | --- | ----- |
| Probe \(P_z\) [eV/Å³] | **8.27662809943017e−01** | MetalWalls `c8ad175d` | rel ≤ **1e-6** | observed ≈ 1e-8 (unit round-trip) |
| Probe \(P_z\) [eV/Å³] | \(>10^{-8}\) and \(>0\) | compressive LJ on upper wall | gate | compressed gap at \(z_{\mathrm{hi}}=8\) Å |
| \(\lvert\langle P_z\rangle-P_{\mathrm{target}}\rvert/P_{\mathrm{target}}\) | **≤ 0.10** | constant-\(P_z\) equilibration | **0.10** | \(P_{\mathrm{target}}=\) probe pin |
| Upper wall coplanar \(\Delta z\) [Å] | \(\le 10^{-9}\) | rigid piston | **1e-9** | all type-3 \(z\) equal |
| Upper \(\lvert v_x\rvert,\lvert v_y\rvert\) [Å/fs] | \(\le 10^{-12}\) | \(z\)-only motion | **1e-12** | rigid in plane |
| Lower wall \(\Delta z\) [Å] | \(\le 10^{-10}\) | `FixFreeze` | **1e-10** | type 2 fixed |
| Energy finite | true | smoke stability | finite | `kEnergy` |

Unit tests (`tests/cpu/dynamics/test_piston_electrode.cpp`) pin the static
identities: \(P_z=F_z^{\mathrm{atoms}}/A\) (e.g. \(F=4\) eV/Å, \(A=100\) →
\(P_z=0.04\) eV/Å³); \(F_{\mathrm{net}}=0\) when
\(P_{\mathrm{target}}=P_z^{\mathrm{atoms}}\) (\(\sim 10^{-12}\));
mass-weighted \(F_z\) share with \(F_x=F_y=0\); the lower-side sign; rigid
\(v_z\) equalization.

### Performance

`perf_exempt`: no configuration does the same work on both sides. Every
MetalWalls piston step also runs a CG electrode solve and a 2D Ewald sum, and
MetalWalls cannot propagate this compressed start (closest fluid–fluid
contact 1.6 Å, \(r/\sigma = 0.53\)): NVE reports a NaN temperature around step
12 and the Nosé–Hoover chain fails to converge.

### Limitations

- **Small LJ parallel plate** (16 fluid + 8 wall atoms); not a production EDL
  cell.
- **Pair forces only.** No charge solve on the piston.
- **Compressed start.** The two fluid layers overlap at \(r/\sigma = 0.53\);
  the gates hold and the pinned quantity is a step-0 static force, but the MD
  half rides on an overlap no other engine integrates.
- **Opposite wall frozen**, not a second piston.
- `cpu::md::NeighborList` accepts \(r_c > L/2\) silently (library-wide
  behaviour).

## dipole-electrode-joint — Coupled PIM dipoles ⊕ electrode \(Q\) {#dipole-electrode-joint}

Joint self-consistent solve of constant-potential BO electrode charges \(Q\)
and **Polarizable Ion Model (PIM)** induced dipoles \(\boldsymbol{\mu}\) on
electrolyte sites: the electrode RHS includes the permanent electrolyte
potential **and** the potential of the induced dipoles; the PIM field includes
all charges (electrode \(Q\) + permanent \(q\)). `cpu::md::DipoleElectrodeJoint`
(`dipole_electrode_joint.h`, `src/cpu/dynamics/dipole_electrode_joint.cpp`;
constructor `(eta, alpha_by_type)`) is a Component at `md::Phase::pre_force`
that writes electrode charges to `kAtom_charge` and exposes \(\boldsymbol{\mu}\)
via `last_mu()` (diagnostics `total_charge`, `last_joint_*`). It reuses the
`ConstPotBO` electrode kernels and the bulk `cpu::PimInducedDipole` Thole
kernels ([pim-induced-dipole-bulk](md-polarizable.md#pim-induced-dipole-bulk)).
LAMMPS has no joint twin (ELECTRODE is non-polarizable; Drude is a different
model).

### Method

\[
A Q = \Psi - \Phi^{\mathrm{perm}} - \Phi^{\mu},\quad
\Phi_i^{\mu} = k_e\sum_j\frac{\boldsymbol{\mu}_j\cdot\mathbf{r}_{ij}}{r_{ij}^3},
\qquad
\boldsymbol{\mu}_i = \alpha_i\mathbf{E}_i^{\mathrm{tot}}(q_{\mathrm{all}},\boldsymbol{\mu}).
\]

Outer alternating solve: rebuild the electrode RHS with the current
\(\boldsymbol{\mu}\), solve \(Q\) (optional electroneutrality projector), write
electrode charges, run the inner Thole-damped PIM SCC for \(\boldsymbol{\mu}\),
until \(\max|\Delta Q|\) and \(\max|\Delta\mu|\) \(\le\) `joint_tol`.

### System

| Item | Setting |
| ---- | ------- |
| Geometry | 4 sites: bot/top electrode (types 1,2), IL cation/anion (types 3,4) |
| Positions \((x,y,z)\) [Å] | bot \((0,0,2)\), top \((0,0,18)\), cation \((0,0,8)\), anion \((0,0,12)\) |
| Permanent \(q\) [e] | electrodes solved; cation \(+1\), anion \(-1\) |
| \(\alpha\) [Å³] | electrodes 0; type 3 → 1.0; type 4 → 1.5 (`alpha_by_type={0,0,0,1.0,1.5}`) |
| Box | orthorhombic \(L_x=L_y=10\) Å, \(L_z=20\) Å |
| Applied \(\Psi\) | \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V |
| Gaussian width | \(\eta = 2.0\) Å⁻¹ |
| Neutrality | on |
| Image range | `xy_image_range = 0` (min-image) |
| Thole \(a\) | 0.39 |
| Outer / inner | `joint_tol = 10^{-12}`, `pol_tol = 10^{-12}` e·Å; budgets 100 / 200 |

### Code

```cpp
cpu::Frame frame(4);
const double z[4]  = {2.0, 18.0, 8.0, 12.0};
const double q0[4] = {0.0, 0.0, 1.0, -1.0};
const int typ[4]   = {1, 2, 3, 4};
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_z)[i] = z[i];           // x = y = 0
    frame.span(kAtom_charge)[i] = q0[i];
    frame.span(kAtom_type)[i] = typ[i];
    frame.span(kAtom_mass)[i] = (typ[i] < 3) ? 12.0 : 1.0;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);

cpu::md::DipoleElectrodeJoint joint(
    /*eta [1/Å]=*/2.0, /*alpha_by_type=*/{0.0, 0.0, 0.0, 1.0, 1.5});
joint.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
joint.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
joint.thole_a = 0.39;
joint.electroneutrality = true;
joint.xy_image_range = 0;
joint.joint_tol = 1e-12;
joint.pol_tol = 1e-12;
joint.joint_max_iter = 100;
joint.pol_max_iter = 200;

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/8.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/8.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.set_observer_ref(joint);   // pre_force @ -50
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_top = joint.total_charge(1);  // [e]
const auto& mu = joint.last_mu();            // packed xyz [e·Å]
// mu[6..8] = cation; mu[9..11] = anion
```

### Pins and their provenance

The hard-coded pins are **ATV's own minimum-image algebra**, not MetalWalls
output. A MetalWalls `c8ad175d` twin (`dipoles_and_electrodes`, `cg 1.0e-13
5000`, Jacobi preconditioner, `num_pbc 3`) cannot express this system:
MetalWalls has no minimum-image electrostatics (only `num_pbc` 2 or 3) and no
Thole damping (Tang–Toennies on charge–dipole only, run undamped here).
Against it the pins differ:

| Observable | ATV pin | MW `num_pbc 3` | (pin − MW)/\|MW\| |
| --- | --- | --- | --- |
| \(Q_{\mathrm{top}}\) joint | +7.3285008958883610e−02 | +6.7072614574458830e−02 | **+9.26 %** |
| \(Q_{\mathrm{top}}\), \(\alpha=0\) | +7.5339627935767609e−02 | +6.9135349400319443e−02 | **+8.97 %** |
| \(\mu_{\mathrm{cat},z}\) [e·Å] | +5.8951916148990660e−02 | +7.0207488022e−02 | **−16.03 %** |
| \(\mu_{\mathrm{an},z}\) [e·Å] | +8.7432701391015569e−02 | +1.0360432203e−01 | **−15.61 %** |

The ATV kernels under a full 3D Ewald sum give
\(Q_{\mathrm{top}}(\alpha=0) = 6.9135350235340731\times10^{-2}\) e, matching
MetalWalls to 1.21e-8 relative (the `units::coulomb` truncation), so the
kernels are MetalWalls' kernels and the gap is the min-image truncation of a
conditionally convergent lattice sum (plus the damping-model difference for
\(\mu\)). Under `num_pbc 2` MetalWalls gives 1.302179493406125e−01 (joint) and
1.362412229597930e−01 (\(\alpha=0\)). This section is therefore an algebra
pin, not an electrode-physics pin; references for the form: Coretti et al.,
*J. Chem. Phys.* (2022) (MetalWalls `dipoles_and_electrodes`); Thole, *Chem.
Phys.* **59**, 341 (1981).

### Accuracy

| Quantity | Atomiverse / pin | Pin | Tol | Notes |
| -------- | ---------------: | --: | --- | ----- |
| \(Q_{\mathrm{bot}}\) [e] | **−7.3285008958883610e−02** | ATV min-image | rel \(\le 10^{-6}\) | joint \(\alpha>0\) |
| \(Q_{\mathrm{top}}\) [e] | **+7.3285008958883610e−02** | same | rel \(\le 10^{-6}\) | opposite under neutrality |
| \(\lvert\sum Q\rvert\) [e] | 0 | 0 | \(\le 10^{-10}\) | electroneutrality |
| \(\mu_{\mathrm{cation},z}\) [e·Å] | **+5.8951916148990660e−02** | same | \(\lvert\Delta\mu_z\rvert\le 10^{-8}\) | collinear \(z\) geometry |
| \(\mu_{\mathrm{anion},z}\) [e·Å] | **+8.7432701391015569e−02** | same | \(\lvert\Delta\mu_z\rvert\le 10^{-8}\) | packed in `last_mu()` |
| \(\mu_{*,x}\), \(\mu_{*,y}\) [e·Å] | 0 | 0 | \(\lvert\mu\rvert < 10^{-10}\) | transverse |
| \(\alpha=0\) \(Q_{\mathrm{top}}\) [e] | **+7.5339627935767609e−02** | same | rel \(\le 10^{-6}\) | reduction |
| \(\alpha=0\) vs `ConstPotBO` | \(\lvert\Delta Q\rvert\le 10^{-10}\) | parity | absolute | joint degenerates to BO |
| Coupling \(\lvert Q_{\mathrm{top}}-Q_{\alpha=0}\rvert/\lvert Q_{\alpha=0}\rvert\) | \(\ge 10^{-6}\) | expected shift | lower bound | induced \(\mu\) moves \(Q\) |
| Electrolyte \(q\) | unchanged \(\{+1,-1\}\) | fixed | exact write-back | only electrode rows rewritten |
| MD-bound \(Q_{\mathrm{top}}\) | same joint pin | same | rel \(\le 10^{-6}\) | one-step MD path |

### Performance

`perf_exempt`: measured walls (MetalWalls 0.52 / 0.73 / 0.67 s; whole ATV
section 0.04 s) are fixed start-up overhead for 4 atoms, and the two codes
solve different linear systems, so the ~0.07 ratio is not a speed result.

### Limitations

- **Min-image elastance and PIM field.** ~9 % on \(Q\) and ~16 % on \(\mu\)
  versus Ewald for this cell; `xy_image_range > 0` adds only a bare real-space
  \(xy\) image sum to \(A\) and \(b\) and does not touch the PIM field.
- **Four-site collinear parallel plate**; the cation–top and anion–bottom
  separations sit exactly on the \(L_z/2\) minimum-image tie.
- **All-pairs joint SCC**; no NeighborConsumer path.
- **Isotropic \(\alpha\) by type.** Electrode types keep \(\alpha=0\); no
  anisotropic polarizability, no Drude oscillators.
