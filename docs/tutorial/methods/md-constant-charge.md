# Constant-charge electrodes

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

**Constant-charge (conq) electrode control** is the dual of constant potential:
each electrode's **total charge** \(Q_e\) is prescribed, the per-site
distribution is solved from the same Gaussian elastance matrix, and the uniform
electrode potential \(\Psi_e\) is read back as the **response** (LAMMPS
ELECTRODE `fix electrode/conq`, whose outputs `f_conq[e]` / `v_dv` map to
`electrode_potential(e)` / `delta_psi()` here). This is *not* "fixed uniform
charge + freeze": the prescribed total redistributes over the electrode sites
self-consistently in the field of the electrolyte every step.

The public surface is the constant-potential component
`cpu::md::ConstPotBO` (see [constant-potential electrodes](md-constant-potential.md#const-pot-bo-parallel-plate))
extended by one member method `add_electrode_charge(type, Q)`. It is mounted on
`cpu::md::MD` with `set_observer_ref` and runs at `md::Phase::pre_force`; there
is no separate Driver or solver class and no charge solve inside `KSpace*`. The
case runs on the CPU backend only (MD-only configure, CPU login environment
`buildenv-gcc/2026.03-mpich`) and covers the conp↔conq duality, the potential
response and the MD-bound path.

Case: `ctest -R '^md/constant-charge$'` — sources and oracle record in [`regressions/md/constant-charge/`](../../../regressions/md/constant-charge/). One section of the case per `##` heading below: `atv_regressions --case md/constant-charge --section <name>`.

## Oracle

**LAMMPS** ELECTRODE package, 3 Nov 2022: the madelung example log
`log.1Dec2022.madelung-ewald-ew3dc.g++.1` plus `graph-il/in.conq`
`fix electrode/conq` semantics, as an offline log pin (never live LAMMPS in
ctest). The local `lmp` lacks `PKG_ELECTRODE` (installed packages: EXTRA-*,
KSPACE, MANYBODY, MISC, MOLECULE, OPENMP, QEQ, REPLICA, RIGID), so the pin uses
the shipped official logs plus the package's documented conq↔conp inverse
contract — the same sourcing as `const-pot-bo-parallel-plate`. LAMMPS is the
oracle because constant-charge control overlaps the ELECTRODE package;
MetalWalls is reserved for MetalWalls-only physics (e.g. the
[Thomas–Fermi](md-thomas-fermi.md#thomas-fermi-electrode) electrodes).

## const-charge-electrode — Parallel-plate conq twin {#const-charge-electrode}

### Method

Ahrens-Iwers, Janssen, Tee, Meißner, *J. Chem. Phys.* **157**, 084801 (2022),
DOI [10.1063/5.0099239](https://doi.org/10.1063/5.0099239) (the LAMMPS
ELECTRODE package; `fix electrode/conq` semantics). With indicator vector
\(d_e\) of const-Q group \(e\) and unknown uniform group potentials
\(\psi_e\), the BO solve becomes the constrained system

\[
Q = A^{-1}\bigl(\Psi_{\text{base}} + \textstyle\sum_e \psi_e d_e - b\bigr),
\qquad d_e^{\mathsf T} Q = Q_e,
\]

with \(A\) the Siepmann–Sprik Gaussian elastance matrix (width \(\eta\),
2D \(xy\) lattice images for planar PBC) and \(b\) the electrolyte potential
vector. Eliminating \(Q\) reduces it to the small SPD system

\[
M\psi = r, \qquad
M_{ef} = d_e^{\mathsf T}A^{-1}d_f, \qquad
r_e = Q_e - d_e^{\mathsf T}A^{-1}\bigl(\Psi_{\text{base}} - b\bigr),
\]

solved on both the dense-Cholesky and CG paths of `ConstPotBO`
(`src/cpu/dynamics/const_pot_bo.cpp`). The solved \(\psi_e\) **is** the
electrode-potential response (LAMMPS `f_conq[e]`); `delta_psi()` is the
two-electrode convenience `f_conq[2] − f_conq[1]` (LAMMPS `v_dv`). Registering
any const-Q group supersedes the `electroneutrality` projection — choose
targets summing to \(-q_{\text{elyt}}\) for global neutrality. Mixed
const-\(\Psi\) / const-\(Q\) groups on one component are supported.

### System (LAMMPS ELECTRODE madelung twin)

Same geometry as `const-pot-bo-parallel-plate`:

| Species | Type | z [Å] | q [e] |
| --- | --- | --- | --- |
| Bottom electrode | 1 | 8 | solved |
| Top electrode | 2 | 12 | solved |
| Electrolyte + | 3 | 9 | +0.5 |
| Electrolyte − | 3 | 11 | −0.5 |

Box \(1\times1\times20\) Å, \(\eta = 2.0\) Å⁻¹, 2D \(xy\) Gaussian image
half-range 250 (real-space elastance — not full `ewald/electrode`).

### Pins

1. **LAMMPS conp anchor** (madelung example log, `run 0`,
   `fix electrode/conp 0 2 couple top 1 symm on`):
   \(\Delta\Psi = 1\) V \(\Leftrightarrow Q_{\mathrm{top}} =
   +0.279214485147238\) e.
2. **conq inverse contract** (`graph-il/in.conq` semantics): feeding the conp
   totals \(\pm Q^{*}\) back as conq targets must return the conp potential
   difference exactly. The shipped input pins the invariant: *"symm on and
   off give different electrode potentials, but identical potential
   difference"* — conq is the exact inverse of conp up to a uniform shift.
3. **Literature Madelung-sum plate capacitor** (empty cell, no electrolyte,
   \(Q = \pm1\) e, \(A_{\mathrm{cell}} = 1\) Å², \(d = 4\) Å,
   \(\eta = 2\) Å⁻¹): the potential difference between the two **discrete**
   Gaussian-charge lattices is

   \[
   \Delta\Psi = 2 k_e\Bigl[\,2\pi\sigma d
     + \eta\sqrt{2/\pi} + M_\square + g(\eta)\Bigr]
   = \mathbf{651.8103\ V},
   \]

   with \(k_e = 14.399645\) eV·Å/e², sheet term \(2\pi\sigma d\) (Griffiths,
   *Introduction to Electrodynamics*, 4th ed., §2.5.4), the square-lattice 2D
   Madelung constant \(M_\square = -3.90026492\) (Glasser & Zucker,
   *Theoretical Chemistry: Advances and Perspectives* **5**, 67 (1980)) and the
   Gaussian correction \(g(\eta{=}2) = -0.195386\). The bare-sheet textbook
   value \(4\pi k_e\sigma d = 723.88\) V is **not** the pin — it omits the
   on-site discrete-lattice term (−72.07 V here). \(R\to\infty\)
   extrapolation of the lattice sum reproduces 651.81 V; at \(R = 250\) the
   real-space tail leaves −0.8%.

### Code

```cpp
// Madelung geometry: two plates + two electrolyte ions.
cpu::Frame frame(4);
const double z[4] = {8.0, 12.0, 9.0, 11.0};
const double q0[4] = {0.0, 0.0, 0.5, -0.5};
const int typ[4] = {1, 2, 3, 3};
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_z)[i] = z[i];          // x = y = 0
    frame.span(kAtom_charge)[i] = q0[i];
    frame.span(kAtom_type)[i] = typ[i];
    frame.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.966553 : 1.0;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);

cpu::md::ConstPotBO conq(/*eta [1/Å]=*/2.0);
conq.add_electrode_charge(/*type=*/1, /*Q [e]=*/-0.279214485147238);
conq.add_electrode_charge(/*type=*/2, /*Q [e]=*/+0.279214485147238);
conq.xy_image_range = 250;
conq.rebuild_matrix_every_step = true;   // electrodes move / small twin

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.set_observer_ref(conq);   // default binding: pre_force @ -50
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_top = conq.total_charge(1);          // [e] — exactly the target
const double dv = conq.delta_psi();                 // [V] — LAMMPS v_dv
const double psi_bot = conq.electrode_potential(0); // [V] — f_conq[1]
// frame.span(kAtom_charge) holds the solved per-site distribution on the
// electrode rows; electrolyte rows are untouched.
```

Parameters are constructor arguments and public members (no Config bag);
`add_electrode_charge` is ref-qualified for rvalue chaining. Solved per-site
charges land on `kAtom_charge`; electrolyte rows are never mutated.

### Accuracy

| Quantity | Atomiverse gate | Oracle pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| conp \(Q^{*}_{\mathrm{top}}\) [e] | within **1%** of pin | **+0.279214485147238** | rel ≤ **0.01** | LAMMPS conp log anchor (real-space A-matrix gate) |
| conq group totals [e] | equal the targets \(\mp Q^{*}\) | targets ("Q fixed") | **1e-12** | constraint exact by construction |
| conq \(\Delta\Psi(\pm Q^{*})\) [V] | \(= 1\) V | **1.0** (inverse contract) | **1e-8** | the tight, well-conditioned gate |
| conq per-site charges [e] | equal conp per-site charges | duality | **1e-10** | same distribution both directions |
| conq \(\Delta\Psi(Q_{\mathrm{pin}})\) [V] | \(1 \pm 0.25\) | **1.0** | **±0.25** | coarse gate; see error budget |
| Empty-capacitor \(\Delta\Psi\) at \(Q=\pm1\) e [V] | within **1.5%** | **651.8103** (Madelung sum) | rel ≤ **0.015** | R=250 real-space tail ≈ 0.8% |
| MD-bound path (1 Verlet step) | totals exact / \(\Delta\Psi\approx1\) V | — | **1e-10 e** / **±0.05 V** | Component on `cpu::md::MD` at `pre_force` |
| Electrolyte charges [e] | ±0.5 untouched | — | **1e-14** | solver never mutates non-electrode rows |

**Error budget (why the raw-\(Q_{\mathrm{pin}}\) gate is ±0.25 V):** the
plate stiffness \(d\Delta\Psi/dQ \approx 4\pi k_e d / A_{\mathrm{cell}}
\approx 724\) V/e amplifies the real-space-A vs `ewald/electrode` charge
difference (~2×10⁻⁴ e, 0.07%) to ≈ 0.14 V at fixed \(Q\). The
well-conditioned oracle observable is therefore the inverse contract
\(\Delta\Psi(\pm Q^{*}) = 1\) V (1e−8), with \(Q^{*}\) itself gated at 1%
against the LAMMPS log; the raw-\(Q_{\mathrm{pin}}\) response is coarse-gated
to catch sign/scale regressions only.

### Offline pin repro (never ctest)

```bash
# Requires LAMMPS built with -D PKG_ELECTRODE=ON
cd $LAMMPS_DIR/examples/PACKAGES/electrode/madelung
lmp -in in.ewald-ew3dc            # conp: Q_top at dV = 1 V (run 0)
cd ../graph-il && lmp -in in.conq # conq: f_conq[e] potentials, v_dv
```

### Performance

No fair wall-clock ratio: the madelung conq BO solve is `perf_exempt` because
the local `lmp` lacks `PKG_ELECTRODE`, so there is no `fix electrode/conq` twin
binary. When a `PKG_ELECTRODE` LAMMPS binary is available, the fair twin is
re-run offline and its version and walls recorded in `REFERENCE.md`.

### Limitations

- **Real-space elastance only.** Real-space Gaussian + 2D \(xy\) images; no
  full `ewald/electrode` reciprocal matrix — the ~0.07% charge-scale difference
  is amplified by the plate stiffness (error budget above), so the tight gate
  is the inverse contract, not the raw pin response.
- **Uniform-shift gauge.** Individual `electrode_potential(e)` values carry
  the same gauge freedom LAMMPS documents for `symm on/off` — only potential
  *differences* are physical observables.
- **Toy twin geometry.** Four atoms, one madelung cell; a production EDL box
  is [`const-pot-il-edl-box`](md-constant-potential.md#const-pot-il-edl-box).
- **Dense \(O(N_{\mathrm{elec}}^3)\) default** for the BO solve (CG + ASPC
  warm-start serve large electrodes; the reduced \(M\psi = r\) system itself is
  tiny — one row per const-Q group).
- Perfect-metal conp, Thomas–Fermi metallicity, CG large electrodes, MPI halo
  and MaZe dynamics are covered by the
  [constant-potential](md-constant-potential.md#const-pot-bo-parallel-plate),
  [Thomas–Fermi](md-thomas-fermi.md#thomas-fermi-electrode) and
  [polarizable](md-polarizable.md#maze-symplectic-charge-dynamics) pages.
