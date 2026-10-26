# Thomas–Fermi electrode metallicity

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

**Thomas–Fermi (TF) electrode metallicity** gives constant-potential electrodes
a finite screening length \(l_{\mathrm{TF}}\): it hardens the diagonal of the
Gaussian electrode elastance so the metal is imperfect, lowering the induced
charge and the integral capacitance at fixed potential difference. Perfect
metal is recovered at \(l_{\mathrm{TF}}=0\).

The public surface is the constant-potential component `cpu::md::ConstPotBO`
(see [constant-potential electrodes](md-constant-potential.md#const-pot-bo-parallel-plate)),
mounted on `cpu::md::MD` with `set_observer_ref` at `md::Phase::pre_force`,
with public members `l_tf` [Å] and either `site_density` [1/Å³] (one global
site density) or `site_volume` [Å³] (per-row Voronoi volumes), and the result
accessors `total_charge(e)` and `capacitance()`. Per-site volumes come from the
setup-time domain type `cpu::VoronoiVolume::volumes(frame)`
(`cpu/observe/voronoi_volume.h`). There is no separate Driver or solver class
and no charge solve inside `KSpace*`; `l_tf > 0` without a density or volume
throws.

Both sections run on the CPU backend only, on the same empty gold capacitor:
[`thomas-fermi-electrode`](#thomas-fermi-electrode) pins the induced charge and
capacitance with a global site density, and
[`voronoi-volume-tf`](#voronoi-volume-tf) pins the Voronoi tessellation and the
per-site-volume form.

Case: `ctest -R '^md/thomas-fermi$'` — sources and oracle record in [`regressions/md/thomas-fermi/`](../../../regressions/md/thomas-fermi/). One section of the case per `##` heading below: `atv_regressions --case md/thomas-fermi --section <name>`.

## Physics

Scalfi, Dufils, Reeves, Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704
(2020), DOI [10.1063/5.0028232](https://doi.org/10.1063/5.0028232)
(arXiv:1910.13341). The semiclassical TF self-energy is per-site,

\[
U_{\mathrm{TF}} = \sum_i \frac{l_{\mathrm{TF}}^2}{2\,\epsilon_0\,v_i}\,q_i^2
\quad\Rightarrow\quad
A_{ii}\;\mathrel{+}=\; 4\pi\,k_e\,\frac{l_{\mathrm{TF}}^{2}}{v_i}
\;=\; 4\pi\,k_e\,l_{\mathrm{TF}}^{2}\,d \quad (v_i = 1/d),
\]

in metal units (\(e\), Å, eV; \(k_e =\) `units::coulomb` \(= 14.399645\)
eV·Å/e²), with \(v_i\) the Voronoi volume of electrode atom \(i\) or \(d\) the
site number density. Larger \(l_{\mathrm{TF}}\) or smaller \(v_i\) increases
\(A_{ii}\), reduces the induced electrode charge at fixed \(\Delta\Psi\), and
therefore **lowers** the integral capacitance \(C=|Q|/|\Delta\Psi|\).

The underlying BO solve is the perfect-metal one
([`const-pot-bo-parallel-plate`](md-constant-potential.md#const-pot-bo-parallel-plate)):
Siepmann–Sprik Gaussian electrode charges of width \(\eta\), 2D \(xy\) lattice
images for planar PBC, optional electroneutrality projector

\[
Q = \Bigl(A^{-1} - \frac{A^{-1}ee^{\mathsf T}A^{-1}}{e^{\mathsf T}A^{-1}e}\Bigr)(\Psi - b)
\]

so \(\sum_i Q_i = 0\). Both sections have **no electrolyte** (\(b=0\)): an
empty vacuum capacitor.

## Oracle

**MetalWalls** release 20.05, git `c8ad175d`, `src/thomas_fermi.f90`
(`gradQ_V(i,i) += 4*pi*ltf**2 / voronoi_volume`) — the atomic-unit form of the
Scalfi model, identical to Atomiverse's diagonal shift in metal units. The
binary is run offline (never in ctest) on twin inputs (`num_pbc 2`,
`electrode_charges matrix_inversion`, `charge_neutrality true`) as
`mpirun -n 1 mw`; pins are `total_charges.out` verbatim (MetalWalls' atomic
unit of charge *is* e, so no conversion — the unit work is on the inputs:
Å→bohr, V→hartree/e, Å⁻¹→bohr⁻¹). The same binary reproduces its own shipped
`tests/thomas_fermi/LiCl-Al-inv` reference to 1e−12 relative, which is the
check that the oracle run is sane. MetalWalls rather than LAMMPS: LAMMPS
ELECTRODE implements perfect-conductor electrodes only, so finite TF
metallicity and its Voronoi-volume input are MetalWalls-only physics.

## Common system — gold capacitor toy

Empty parallel-plate capacitor (no electrolyte):

| Quantity | Value |
| -------- | ----- |
| Au lattice \(a_0\) | 4.07 Å |
| Site density \(d = 4/a_0^3\) | **0.059330329963** Å⁻³ |
| Plane spacing \(a = a_0/2\) | 2.035 Å |
| Planes per electrode | 3 (1 site per plane at cell centre) |
| \(L_x = L_y\) | \(a_0 = 4.07\) Å |
| First-plane gap \(L\) | 20.0 Å |
| \(\Psi_{\mathrm{bot}},\Psi_{\mathrm{top}}\) | 0 V, 1 V (\(\Delta\Psi=1\) V) |
| Gaussian \(\eta = 1/0.56\) | **1.785714285714** Å⁻¹ (MetalWalls \(w=0.56\) Å) |
| 2D \(xy\) image half-range | 1280 (convergence knob — see below) |
| Electroneutrality | on |
| Box \(L_z\) | 80 Å (electrode path uses 2D \(xy\) images only) |

Atom \(z\) [Å]: bot `{5.0, 7.035, 9.07}`, top `{29.07, 31.105, 33.14}`
(types 1 / 2). Six Au sites total.

```cpp
constexpr double kEta = 1.0 / 0.56;              // [1/Å]
constexpr double kA0 = 4.07;                     // [Å]
constexpr double kD = 4.0 / (kA0 * kA0 * kA0);   // site density [1/Å³]
constexpr double kPlane = kA0 / 2.0;
constexpr double kLGap = 20.0;

cpu::Frame gold_capacitor() {
    cpu::Frame f(6);
    const double z_surf_bot = 5.0 + 2.0 * kPlane;
    const double z_surf_top = z_surf_bot + kLGap;
    const double z[6] = {
        z_surf_bot - 2.0 * kPlane, z_surf_bot - kPlane, z_surf_bot,
        z_surf_top, z_surf_top + kPlane, z_surf_top + 2.0 * kPlane,
    };
    const int typ[6] = {1, 1, 1, 2, 2, 2};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_z)[i] = z[i];           // x = y = 0, q = 0
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = 196.966553;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(kA0, kA0, 80.0);
    return f;
}

// One BO solve: ConstPotBO bound on MD (default pre_force order −50).
cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
sim.set_evaluator(cpu::PairCoulCut{12.0});
sim.set_integrator(cpu::md::VelocityVerlet{0.0});
sim.set_observer_ref(conp);
sim.build();
sim.run(frame, 1);
```

### `xy_image_range` is a convergence knob, not a system property

`ConstPotBO` builds \(A\) as a direct square-image lattice sum truncated at
half-range \(R\). The exact 2D-periodic elastance (MetalWalls `num_pbc 2`,
Ewald) is the \(R\to\infty\) limit, and each pair misses a lattice tail
\(\simeq \pi z_{ij}^{2}/(R L A_{\mathrm{cell}})\) — an \(O(1/R)\) error that
grows as \(z^2\). This toy's 20 Å gap over a 4.07 Å lateral cell is the worst
case:

| \(R\) | \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=0)\) [e] | vs MetalWalls |
| ----: | ------------------------------------------: | ------------: |
| 40 | 4.620945546104e−03 | **+6.128 %** |
| 160 | 4.417905982930e−03 | +1.465 % |
| 320 | 4.385778337944e−03 | +0.727 % |
| 640 | 4.369898601631e−03 | +0.362 % |
| **1280** | **4.362004736725e−03** | **+0.181 %** |
| 2560 | 4.358069301166e−03 | +0.091 % |
| exact 2D Ewald | 4.354141528379e−03 | +0.0004 % |

At \(R = 40\) the truncated sum is **+6.13 % high** at every
\(l_{\mathrm{TF}}\); the model is correct, the numerics are not converged.
Values computed by re-implementing `ConstPotBO`'s own truncated sum are a
self-golden, never a pin. Both sections run at \(R = 1280\).

## Common limitations

- **Discrete 1-site/plane lattice.** Under-samples lateral Au density vs
  continuum capacitor formulas; continuum numbers are trend cross-checks only —
  the product gate is the discrete MetalWalls pin.
- **Real-space elastance only.** Real-space Gaussian + 2D \(xy\) images; no
  full electrode Ewald / MetalWalls reciprocal matrix. The \(O(1/R)\)
  truncation means `xy_image_range` must be raised until the answer stops
  moving (1280 here for 0.18 %); a 2D Ewald elastance would remove the knob.
- **Empty vacuum only** — no electrolyte \(b\), no MD trajectory product for TF
  walls.
- **Dense \(O(N_{\mathrm{elec}}^3)\) path** for the BO solve (CG is optional on
  the same component for larger electrodes; not exercised here).
- Perfect-metal BO, CG large electrodes, the ASPC predictor and the piston
  \(P_z\) live on the [constant-potential](md-constant-potential.md#const-pot-bo-parallel-plate)
  page; constant-charge control on
  [constant-charge](md-constant-charge.md#const-charge-electrode).

## References

- Scalfi, Dufils, Reeves, Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704
  (2020), DOI [10.1063/5.0028232](https://doi.org/10.1063/5.0028232)
  (arXiv:1910.13341) — MetalWalls TF hardness model.
- Marin-Laflèche et al., *J. Open Source Softw.* **5**, 2373 (2020), DOI
  [10.21105/joss.02373](https://doi.org/10.21105/joss.02373) — MetalWalls,
  per-species `thomas_fermi_length` + `voronoi_volume` inputs.

## thomas-fermi-electrode — Gold capacitor, global site density {#thomas-fermi-electrode}

Dual gate on the gold capacitor toy: perfect metal vs \(l_{\mathrm{TF}}=2\) Å
with one global site density \(d = 4/a_0^3\). Asserts electroneutrality, charge
polarity, \(C(l_{\mathrm{TF}}=2)<C(0)\), and relative agreement ≤ **1%** on
\(Q_{\mathrm{top}}\) and integral capacitance \(C\) at both
\(l_{\mathrm{TF}}\).

### Code

```cpp
for (double l_tf : {0.0, 2.0}) {
    auto frame = gold_capacitor();
    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
    conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
    conp.electroneutrality = true;
    conp.xy_image_range = 1280;  // O(1/R) convergence knob
    conp.l_tf = l_tf;            // [Å]; 0 = perfect metal
    conp.site_density = kD;      // [1/Å³]; required when l_tf > 0
    // ... bind on MD and run one step (common setup) ...
    const double q_top = conp.total_charge(1);  // [e]
    const double c = conp.capacitance();        // [e/V]
    // frame.span(kAtom_charge)[0..5] updated for electrode rows
}
```

### MetalWalls pin

| \(l_{\mathrm{TF}}\) [Å] | \(Q_{\mathrm{top}}\) [e] | \(C=\|Q\|/\Delta\Psi\) [e/V] |
| ----------------------: | -----------------------: | ---------------------------: |
| **0.0** (perfect metal) | **+4.354124119406114e−03** | **4.354124119406114e−03** |
| **2.0** (finite metallicity) | **+3.707555065364343e−03** | **3.707555065364343e−03** |

The \(l_{\mathrm{TF}}=2\) run uses `thomas_fermi_length` 3.779452249251540
bohr. At \(R = 40\) the truncated sum gives 3.934601758679e−03 e at
\(l_{\mathrm{TF}}=2\) (+6.124 %).

Continuum cross-check (paper Eq. 8 with \(L_{\mathrm{vac}}=L-a\)):
\(C_{\mathrm{cont}}(0)\approx 5.10\times 10^{-3}\),
\(C_{\mathrm{cont}}(2)\approx 4.17\times 10^{-3}\) [e/V] on this area — same
trend (the discrete 1-site/plane lattice under-samples lateral density vs
continuum; the gate is the MetalWalls pin).

### Accuracy

| Quantity | Atomiverse gate / pin | MetalWalls c8ad175d | Tol | Notes |
| -------- | --------------------: | --------------------------: | --- | ----- |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=0)\) [e] | within **1%** of pin | **+4.354124119406114e−03** | rel ≤ **0.01** | perfect metal (ATV +0.181 %) |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=2)\) [e] | within **1%** of pin | **+3.707555065364343e−03** | rel ≤ **0.01** | finite metallicity (ATV +0.183 %) |
| \(C(l_{\mathrm{TF}}=0)\) [e/V] | dual-gate vs pin | **4.354124119406114e−03** | rel ≤ **0.01** | \(\Delta\Psi=1\) V |
| \(C(l_{\mathrm{TF}}=2)\) [e/V] | dual-gate vs pin | **3.707555065364343e−03** | rel ≤ **0.01** | \(\Delta\Psi=1\) V |
| \(C(l_{\mathrm{TF}}=2) < C(l_{\mathrm{TF}}=0)\) | required | exact ordering | exact | metallicity lowers \(C\) |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | ~0 (neutrality) | **1e-10** | both \(l_{\mathrm{TF}}\) |
| \(\mathrm{sign}(Q_{\mathrm{top}})\) | \(+1\) at \(\Delta\Psi=+1\) V | \(+\) | exact | capacitive polarity |
| Frame write-back \(\sum q_{\mathrm{top}}\) | equals `total_charge(1)` | — | **1e-12** | rows 3..5 |

### Offline pin repro (never ctest)

```bash
# oracle: copy mw_twin/, substitute @LTF@ (0.0 or 3.779452249251540 bohr), run
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd /tmp/tf && mpirun -n 1 /path/to/metalwalls/mw      # -> total_charges.out

# truncation diagnosis (needs numpy + scipy)
python3 tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py
```

### Performance

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| gold-cap TF BO solve (per elastance build + solve, 1 rank, `OMP_NUM_THREADS=1`) | 0.20 | 0.885 | **4.4** | yes | MetalWalls' 0.20 s is MPI-startup dominated, so \(r\) flatters Atomiverse |

The gap is algorithmic: MetalWalls sums the 2D-periodic elastance by Ewald,
Atomiverse evaluates \(21\times2561^2\approx1.4\times10^{8}\) `erf` images per
solve just to reach 0.18 %. Giving `ConstPotBO` a 2D Ewald elastance closes
both the accuracy tail and this ratio at once.

## voronoi-volume-tf — Voronoi site volumes for TF metallicity {#voronoi-volume-tf}

MetalWalls takes the **Voronoi volume of the electrode site** as a per-species
input (`voronoi_volume` next to `thomas_fermi_length`) instead of one global
site density. This section covers both pieces of that form:
`cpu::VoronoiVolume` computes periodic Voronoi cell volumes from geometry
(typically once, on a bulk electrode crystal frame), and `ConstPotBO` accepts
per-site volumes through `site_volume`, applying
\(A_{ii}\mathrel{+}= 4\pi k_e\, l_{\mathrm{TF}}^2/v_i\). The fixed-density form
of [`thomas-fermi-electrode`](#thomas-fermi-electrode) is the special case
\(v_i = 1/d\); for FCC gold \(v = a_0^3/4\) **exactly** (4 sites tile the
conventional cube; the FCC Voronoi cell is the rhombic dodecahedron). Perfect
metal (\(l_{\mathrm{TF}} = 0\)) ignores the volumes entirely.

### Tessellation

`cpu::VoronoiVolume` computes cells by convex bisector clipping: the initial
cell is the box cuboid centred on the site (the self-image bisectors of an
orthorhombic lattice), clipped by the perpendicular-bisector half-spaces of
every neighbor image that can reach the cell, nearest first with early
termination. The tessellation is generated by **all** atoms in the frame
(min-imaged), so \(\sum_i v_i = V_{\mathrm{box}}\) exactly by construction.
Orthorhombic boxes only; throws on triclinic and degenerate boxes, empty
frames, and coincident sites (separation < 1e−6 Å). It is a **setup-time
companion** — a plain domain type, not a hot-path Component (MetalWalls
likewise treats `voronoi_volume` as a static input).

### System

1. **Bulk FCC Au**, 2×2×2 conventional cells (32 sites), \(a_0 = 4.07\) Å,
   cubic box \(2a_0\), fully periodic — input to `cpu::VoronoiVolume`.
2. **Gold capacitor toy** (common system), \(l_{\mathrm{TF}} = 2\) Å, with
   MetalWalls twin inputs substituting `@LTF@` / `@VORONOI@`:

| case | `thomas_fermi_length` [bohr] | `voronoi_volume` [bohr³] |
| ---- | --------------------------- | ------------------------ |
| perfect metal | 0 | \(a_0^3/4\) (unused at \(l_{\mathrm{TF}}=0\)) |
| TF bulk \(v\) | \(2\,\mathrm{Å}/a_B\) = 3.779452249251540 | 113.741732075823890 |
| TF half \(v\) | same | 56.870866037911945 |

### Code

```cpp
// 1. Site volumes from the bulk electrode crystal (MetalWalls workflow):
//    bulk FCC Au 2x2x2, a0 = 4.07 Å, box = cubic(2*a0).
const auto v = cpu::VoronoiVolume{}.volumes(bulk_gold);  // [Å³]; each = a0³/4

// 2. Feed the TF hardness per site (indexed by the run frame's rows).
cpu::md::ConstPotBO conp(kEta);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
conp.electroneutrality = true;
conp.xy_image_range = 1280;
conp.l_tf = 2.0;                                  // [Å]; 0 = perfect metal
conp.site_volume = std::vector<double>(6, v[0]);  // per-row v_i [Å³]
// ... bind on MD and run one step (common setup) ...
const double q_top = conp.total_charge(1);        // [e]
const double c = conp.capacitance();              // [e/V]
```

Exactly one of `site_density` / `site_volume` may be set when `l_tf > 0`
(throws otherwise); non-electrode rows of `site_volume` are ignored. Solved
charges land on `kAtom_charge`.

### Pins

| Quantity | Pin | Units |
| -------- | --: | ----- |
| FCC Voronoi volume \(v = a_0^3/4\) (exact geometry) | **16.854785750000** | Å³ |
| \(Q_{\mathrm{top}}\), \(l_{\mathrm{TF}}=2\) Å, \(v_i = a_0^3/4\) (MetalWalls) | **+3.707555065364343e−03** | e |
| \(Q_{\mathrm{top}}\), \(l_{\mathrm{TF}}=2\) Å, \(v_i = a_0^3/8\) (MetalWalls) | **+3.321360199637453e−03** | e |
| \(Q_{\mathrm{top}}\), \(l_{\mathrm{TF}}=0\) (perfect metal, MetalWalls) | **+4.354124119406114e−03** | e |

The \(v_i = a_0^3/4\) and perfect-metal pins equal the
[`thomas-fermi-electrode`](#thomas-fermi-electrode) fixed-density pins — the
volume and density forms are the same algebra; the regression asserts that
equivalence in-binary at 1e−10 rel. A numpy 2.5.1 transcription of
`ConstPotBO`'s truncated elastance at \(R = 40\) (`pin_voronoi_tf.py`) gives
+3.934601758679e−03 / +3.509853715789e−03 / +4.620945546104e−03 e for the
same three charges, ~6.13 % high versus MetalWalls — a self-golden, never a
pin.

### Accuracy

| Quantity | Atomiverse gate | Oracle pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| every bulk-FCC \(v_i\) [Å³] | min & max of 32 sites | **16.854785750000** (\(a_0^3/4\)) | rel ≤ **1e−9** | rhombic dodecahedron, exact geometry |
| \(\sum_i v_i\) [Å³] | complete tessellation | \(8a_0^3\) (box volume) | rel ≤ **1e−10** | \(\sum v_i = V_{\mathrm{box}}\) by construction |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}{=}2,\,v_i{=}a_0^3/4)\) [e] | computed \(v_i\) fed to `site_volume` | **+3.707555065364343e−03** | rel ≤ **0.01** | MetalWalls pin |
| \(Q_{\mathrm{top}}(v_i)\) vs fixed-density twin [e] | in-binary equivalence | \(d = 4/a_0^3\) twin | rel ≤ **1e−10** | volume ↔ density same algebra |
| \(C(v_i)\) [e/V] | `capacitance()` | **3.707555065364343e−03** | rel ≤ **0.01** | \(\Delta\Psi = 1\) V |
| \(Q_{\mathrm{top}}(v_i/2)\) [e] | halved volume | **+3.321360199637453e−03** | rel ≤ **0.01** | harder diagonal, less charge |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}{=}0)\) [e] | volumes ignored at \(l_{\mathrm{TF}}=0\) | **+4.354124119406114e−03** | rel ≤ **0.01** | perfect-metal pin |
| \(C(v/2) < C(v) < C(l_{\mathrm{TF}}{=}0)\) | required | exact ordering | exact | metallicity lowers \(C\) |
| \(\lvert\sum_i Q_i\rvert\) [e] | each solve | ~0 (neutrality) | **1e−10** | electroneutrality projector |

### Performance

`perf_exempt`: a 32-site tessellation plus sub-second algebraic gold-capacitor
TF solves on a tiny elastance — a wall ratio would measure Ewald setup, not the
TF/Voronoi companion; LAMMPS has no TF electrode feature.

### Limitations

- **Orthorhombic boxes only** for `VoronoiVolume::volumes`.
- **Static volumes.** Computed once from the bulk crystal, as MetalWalls'
  static `voronoi_volume` input; no per-step re-tessellation on the MD hot path.
- **Plain Voronoi only** — no radical (Laguerre) weighting for multi-species
  electrodes with unequal radii.
