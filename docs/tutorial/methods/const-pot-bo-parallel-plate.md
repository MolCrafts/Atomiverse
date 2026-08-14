# Constant-potential BO electrode solve — parallel-plate madelung twin

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `const-pot-bo-parallel-plate` |
| Regression | `regressions/md/const-pot-bo-parallel-plate/` |
| Public API | `cpu::md::ConstPotBO` (Component on `cpu::md::MD`, default `pre_force`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **E1 done** (wave 1, 2026-07-23); map slice E1 |
| Oracle | **LAMMPS** ELECTRODE package (`fix electrode/conp`, madelung example) |

## Summary

**Constant-potential** (Born–Oppenheimer) electrode electrostatics: given fixed
applied potentials \(\Psi\) on metal atom groups, solve electrode site charges
\(Q\) from a Gaussian elastance matrix so each electrode atom sits at its
target potential in the field of the electrolyte and the other electrode
charges. Public surface is `cpu::md::ConstPotBO` — a **Component** bound at
`md::Phase::pre_force` on `cpu::md::MD` (not a Driver category; not inside
`KSpace*`). It writes solved charges to `kAtom_charge` before pair / k-space
force. This regression is the **E1 foothold**: the LAMMPS ELECTRODE
**madelung** four-atom parallel-plate twin (Ahrens-Iwers et al., *J. Chem.
Phys.* **157**, 084801 (2022)). Offline pin is the shipped LAMMPS log
`log.1Dec2022.madelung-ewald-ew3dc.g++.1` (LAMMPS 3 Nov 2022). Product gate
`reg_const-pot-bo-parallel-plate` asserts electroneutrality, charge sign, and
\(\lvert Q_{\mathrm{top}} - Q_{\mathrm{pin}}\rvert / \lvert Q_{\mathrm{pin}}\rvert
\le 1\%\) (v1 2D-image Gaussian \(A\) vs full `ewald/electrode`). Fair wall vs
LAMMPS is **`perf_exempt`** (local `lmp` lacked `PKG_ELECTRODE`).

## Method

**Physics.** Siepmann–Sprik Gaussian electrode charges of width \(\eta\)
[1/Å]. For electrode atoms \(i,j\) the elastance (potential response) matrix
elements are (metal units, \(k_e =\) `units::coulomb` \(= 14.399645\) eV·Å/e²):

\[
A_{ii} = k_e\,\eta\sqrt{\frac{2}{\pi}}
  + \sum_{(n_x,n_y)\neq(0,0)} k_e\,\frac{\operatorname{erf}\!\bigl((\eta/\sqrt{2})\,r_{n}\bigr)}{r_{n}},
\quad
A_{ij} = \sum_{n_x,n_y} k_e\,\frac{\operatorname{erf}\!\bigl(\eta_{ij}\,r_{ij,n}\bigr)}{r_{ij,n}}
\]

with \(\eta_{ij} = \eta_i\eta_j / \sqrt{\eta_i^2+\eta_j^2}\) (v1 shares one
\(\eta\) for all electrode sites). The electrolyte contribution to the
potential on electrode site \(i\) is

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

so \(\sum_i Q_i = 0\). Dense Cholesky / SPD inverse; residual and group totals
are diagnostics on the component.

**v1 lattice treatment (this pin).** \(A\) and \(b\) use **real-space 2D
\(xy\) Gaussian lattice images** with half-range `xy_image_range` (regression:
250). There is **no** reciprocal-space electrode matrix (`ewald/electrode`) and
**no** MetalWalls-style 2D Ewald electrode kernel here — that is queue **E2**
(`slab-2d-ewald-foothold` / related). The 1% gate vs the full Ewald-electrode
LAMMPS pin is deliberate for this foothold.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry | 4 atoms: bot/top electrode (types 1,2), electrolyte \(\pm 0.5\,e\) (type 3) |
| Relative \(z\) [Å] | electrode \(\{-2,+2\}\), electrolyte \(\{-1,+1\}\); shifted into \((0,L_z)\) as \(z=\{8,12,9,11\}\) |
| Box | \(L_x=L_y=1\) Å, \(L_z=20\) Å, orthorhombic, fully periodic ATV box |
| Applied \(\Psi\) | \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V (\(\Delta\Psi=1\) V) |
| Gaussian width | \(\eta = 2.0\) Å⁻¹ |
| Neutrality | on (`symm`) |
| Image range | `xy_image_range = 250` |
| Binding | standalone `build`/`launch`, and MD `add_ref` → default `pre_force` @ −50 |

**Key exclusions (non-goals of this slug)**

- Full `kspace_style ewald/electrode` / slab correction parity (E2).
- MetalWalls-only BO or finite-field / single-electrode product (E3+).
- Movable electrode geometry every step as the default (v1 walls frozen;
  `rebuild_matrix_every_step` is optional).
- Charge solve inside PME/`KSpace*` — forbidden by architecture.
- Live LAMMPS or MetalWalls inside ctest; PE pin in kcal/mol (unit system
  differs — not compared).
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `ConstPotBO` ctor arg / members; no Config
bag). Minimal standalone solve matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame frame(4);
const double z[4]  = {8.0, 12.0, 9.0, 11.0};
const double q0[4] = {0.0, 0.0, 0.5, -0.5};
const int typ[4]   = {1, 2, 3, 3};
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_x)[i] = frame.span(kAtom_y)[i] = 0.0;
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_charge)[i] = q0[i];
    frame.span(kAtom_type)[i] = typ[i];
    frame.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.966553 : 1.0;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);

// Component defaults to pre_force order −50 so pair/kspace see updated charges.
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
sim.add_ref(conp);   // or add(std::move(conp)) to own
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_bot = conp.total_charge(0);  // [e]
const double q_top = conp.total_charge(1);  // [e]
// frame.span(kAtom_charge)[0..1] updated; electrolyte charges unchanged
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Electrodes are Components; never a new
  Driver category.
- Parameters are **constructor arguments and public members** — no
  `ConstPotBOConfig`.
- Frame is the sole I/O currency; outputs land on `kAtom_charge`.

Build and run the regression (MD-only configure; CPU login pattern from
`AGENTS.md` / metalwalls loop):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_const-pot-bo-parallel-plate -j"$(nproc)"
  ./build/regressions/reg_const-pot-bo-parallel-plate
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Public `ConstPotBO`, `ElectrodeGroup` |
| `src/cpu/dynamics/const_pot_bo.cpp` | Dense \(A\), neutrality projector, write-back |
| `include/atomiverse/cpu/dynamics/md.h` | Host preset `cpu::md::MD` |
| `regressions/md/const-pot-bo-parallel-plate/reg_const-pot-bo-parallel-plate.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/const-pot-bo-parallel-plate.notes.md` | Offline pin provenance |

## Validation (vs LAMMPS)

Source:
[`regressions/md/const-pot-bo-parallel-plate/REFERENCE.md`](../../../regressions/md/const-pot-bo-parallel-plate/REFERENCE.md)
(offline pin; **never** live LAMMPS in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md),
LAMMPS notes under
`tools/metalwalls_oracle/inputs/lammps/const-pot-bo-parallel-plate.notes.md`.

**Oracle:** LAMMPS **3 Nov 2022**, ELECTRODE package example
`examples/PACKAGES/electrode/madelung`, log
`log.1Dec2022.madelung-ewald-ew3dc.g++.1` after `run 0` with
`fix electrode/conp 0 2 couple top 1 symm on`,
`kspace_style ewald/electrode 1e-8`, `kspace_modify slab 8.0`.

**Why LAMMPS (not MetalWalls):** the algorithm/feature exists in LAMMPS
ELECTRODE; binding oracle priority for this loop is **LAMMPS first** when the
feature overlaps. MetalWalls is reserved for MW-only physics.

### Accuracy

| Quantity | Atomiverse gate / pin | LAMMPS pin | Tol | Notes |
| -------- | --------------------: | ---------: | --- | ----- |
| \(Q_{\mathrm{top}}\) [e] | within **1%** of pin | **+0.279214485147238** | rel ≤ **0.01** | v1 2D-image \(A\) vs full Ewald electrode |
| \(Q_{\mathrm{bot}}\) [e] | opposite of top under neutrality | **−0.279214485147238** | — | pin log; regression checks sign |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | ~0 (`symm on`) | **1e-10** | electroneutrality residual |
| \(\mathrm{sign}(Q_{\mathrm{top}})\) | \(+1\) at \(\Delta\Psi=+1\) V | \(+\) | exact | capacitive polarity |
| Electrolyte \(q\) | unchanged \(\{+0.5,-0.5\}\) | fixed | exact write-back | only electrode rows rewritten |
| PE (kcal/mol, real units) | **not compared** | 139.943964815502 | — | unit system differs |

**Literature cross-check (not the ctest pin):** pure real-space 2D Gaussian
lattice sum `plate_cap.py` in the same LAMMPS example directory (metal units,
image range ≳ 200) gives \(Q_{\mathrm{top}}^{\mathrm{Madelung}} \approx
+0.27941\,e\) — relative difference vs the Ewald-electrode log pin ≈ **0.07%**.
Supports that the 1% ATV gate is dominated by missing reciprocal electrode
matrix, not a wrong BO equation.

Geometry, \(\eta\), \(\Psi\), image range, and tolerances: **Method** table
and REFERENCE (match `reg_const-pot-bo-parallel-plate`).

Related unit surface (not the e2e pin):
`tests/cpu/dynamics/test_const_pot_bo.cpp`,
`tests/cpu/dynamics/test_const_potential_bo.cpp`.

### Offline pin repro (never ctest)

```bash
# Requires LAMMPS built with -D PKG_ELECTRODE=ON
cd $LAMMPS_DIR/examples/PACKAGES/electrode/madelung
lmp -in in.ewald-ew3dc
# thermo: pe c_qbot c_qtop at step 0
```

## Performance

Source: REFERENCE.md dual-gate summary; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| madelung `run 0` / BO solve | — | — | **perf_exempt** | n/a | Local LAMMPS binary lacked `PKG_ELECTRODE`; no fair twin binary for `fix electrode/conp` |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | local LAMMPS binary lacks ELECTRODE package; no fair twin binary for `fix electrode/conp` |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When an ELECTRODE-enabled `lmp` is
available, re-run offline fair bench, record version + walls, and replace
`perf_exempt` in REFERENCE.

## Limitations

- **v1 elastance only.** Real-space Gaussian + 2D \(xy\) images; no full
  electrode Ewald / MetalWalls reciprocal matrix (E2). Expect ~1% level
  mismatch vs `ewald/electrode` pins on small cells — gate is set accordingly.
- **Fully periodic ATV box.** LAMMPS madelung uses `boundary p p f`; ATV
  shifts \(z\) into \((0,L_z)\) so VelocityVerlet PBC does not wrap the slab.
  Slab BC product is a later queue item.
- **Dense \(O(N_{\mathrm{elec}}^3)\) solve** and \(O(N_{\mathrm{elec}}^2 R^2)\)
  matrix build — fine for few electrode sites; large electrodes need sparse /
  Ewald electrode follow-ons.
- **Frozen-wall default.** `rebuild_matrix_every_step = false` unless the
  caller sets it; movable walls and electrode forces (piston \(P_z\), etc.)
  are later E-series work.
- **No fair speed ratio** until ELECTRODE-enabled LAMMPS is on the pin host.
- **Scope of E1 wave only** for this page: charge observer, freeze, charging
  work, short NVT, Gaussian charge form, etc. are sibling slugs with their own
  pages when written.

## References

- Offline pin: [`regressions/md/const-pot-bo-parallel-plate/REFERENCE.md`](../../../regressions/md/const-pot-bo-parallel-plate/REFERENCE.md)
- Product gate: [`regressions/md/const-pot-bo-parallel-plate/reg_const-pot-bo-parallel-plate.cpp`](../../../regressions/md/const-pot-bo-parallel-plate/reg_const-pot-bo-parallel-plate.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/inputs/lammps/const-pot-bo-parallel-plate.notes.md`](../../../tools/metalwalls_oracle/inputs/lammps/const-pot-bo-parallel-plate.notes.md)
- Literature: Ahrens-Iwers et al., *J. Chem. Phys.* **157**, 084801 (2022), DOI [10.1063/5.0099239](https://doi.org/10.1063/5.0099239) (LAMMPS ELECTRODE); Siepmann–Sprik Gaussian electrode charges (method basis of `plate_cap` / BO matrix)
- LAMMPS example: `examples/PACKAGES/electrode/madelung` (`in.ewald-ew3dc`, `plate_cap.py`)
