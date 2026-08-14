# Constant-charge electrode control — parallel-plate conq twin

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `const-charge-electrode` |
| Regression | `regressions/md/const-charge-electrode/` |
| Public API | `cpu::md::ConstPotBO::add_electrode_charge` (Component on `cpu::md::MD`, default `pre_force`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → wave 5 (2026-07-27); map §1 "Constant-charge electrodes (control)" |
| Oracle | **LAMMPS** ELECTRODE package (`fix electrode/conq`; madelung conp log anchor) |

## Summary

**Constant-charge (conq) electrode control** is the dual of constant
potential: prescribe each electrode's **total charge** \(Q_e\), solve the
per-site distribution from the same Gaussian elastance matrix, and read the
uniform electrode potential \(\Psi_e\) as the **response** (LAMMPS ELECTRODE
`fix electrode/conq`, whose outputs `f_conq[e]` / `v_dv` map to
`electrode_potential(e)` / `delta_psi()` here). This is *not* "fixed uniform
charge + freeze": the prescribed total redistributes over the electrode sites
self-consistently in the field of the electrolyte every step. Public surface
stays `cpu::md::ConstPotBO` — a **Component** on `cpu::md::MD` at
`md::Phase::pre_force` — extended by one member method
`add_electrode_charge(type, Q)`; no new Driver category and no charge solve
inside `KSpace*`. The regression is the LAMMPS ELECTRODE **madelung
parallel-plate twin**: the conp log pins the self-consistent charge scale, the
package's documented conq↔conp inverse contract carries the tight potential
gate, and an empty Gaussian-lattice capacitor cross-checks the V response
against a literature Madelung sum. Fair wall vs LAMMPS is **`perf_exempt`**
(local `lmp` lacks `PKG_ELECTRODE`).

## Method

**Physics.** Ahrens-Iwers, Janssen, Tee, Meißner, *J. Chem. Phys.* **157**,
084801 (2022), DOI
[10.1063/5.0099239](https://doi.org/10.1063/5.0099239) (the LAMMPS ELECTRODE
package; `fix electrode/conq` semantics). With indicator vector \(d_e\) of
const-Q group \(e\) and unknown uniform group potentials \(\psi_e\), the BO
solve becomes the constrained system

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

solved on both the dense-Cholesky and CG paths of `ConstPotBO`. The solved
\(\psi_e\) **is** the electrode-potential response (LAMMPS `f_conq[e]`);
`delta_psi()` is the two-electrode convenience `f_conq[2] − f_conq[1]`
(LAMMPS `v_dv`). Registering any const-Q group supersedes the
`electroneutrality` projection — choose targets summing to
\(-q_{\text{elyt}}\) for global neutrality. Mixed const-\(\Psi\) /
const-\(Q\) groups are supported.

**Why LAMMPS (not MetalWalls):** constant-charge control overlaps the LAMMPS
ELECTRODE package (`fix electrode/conq`), and this loop's oracle priority
binds to LAMMPS whenever the feature overlaps; MetalWalls is reserved for
MW-only physics (e.g. the Thomas–Fermi sibling).

### System (LAMMPS ELECTRODE madelung twin)

Same geometry as `regressions/md/const-pot-bo-parallel-plate/`:

| Species | Type | z [Å] | q [e] |
| --- | --- | --- | --- |
| Bottom electrode | 1 | 8 | solved |
| Top electrode | 2 | 12 | solved |
| Electrolyte + | 3 | 9 | +0.5 |
| Electrolyte − | 3 | 11 | −0.5 |

Box \(1\times1\times20\) Å, \(\eta = 2.0\) Å⁻¹, 2D \(xy\) Gaussian image
half-range 250 (v1 real-space elastance — not full `ewald/electrode`).

### Offline pins

1. **LAMMPS conp anchor** (madelung example log
   `log.1Dec2022.madelung-ewald-ew3dc.g++.1`, `run 0`,
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
   Madelung constant \(M_\square = -3.90026492\) (Glasser & Zucker 1980) and
   the Gaussian correction \(g(\eta{=}2) = -0.195386\). The bare-sheet
   textbook value \(4\pi k_e\sigma d = 723.88\) V is **not** the pin — it
   omits the on-site discrete-lattice term (−72.07 V here). \(R\to\infty\)
   extrapolation of the v1 lattice sum reproduces 651.81 V; at \(R = 250\)
   the real-space tail leaves −0.8%.

### Key exclusions (non-goals of this slug)

- Live LAMMPS inside ctest (offline log pin only).
- Full electrode Ewald / `ewald/electrode` reciprocal matrix (v1 2D-image
  \(A\); the ~0.07% charge-scale difference is the documented error budget).
- conq as a separate Driver or solver class — only a member method on
  `ConstPotBO`.
- Charge solve inside PME / `KSpace*`.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Fair wall-clock twin vs LAMMPS (`PKG_ELECTRODE` absent locally).

## Implementation

Public surface only (`cpu::Frame` + `ConstPotBO` ctor / members; no Config
bag). Minimal standalone conq solve matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

// LAMMPS ELECTRODE madelung geometry: two plates + two electrolyte ions.
cpu::Frame madelung_frame() {
    cpu::Frame f(4);
    const double z[4] = {8.0, 12.0, 9.0, 11.0};
    const double q0[4] = {0.0, 0.0, 0.5, -0.5};
    const int typ[4] = {1, 2, 3, 3};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.966553 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);
    return f;
}

auto frame = madelung_frame();
// Defaults to pre_force order −50.
cpu::md::ConstPotBO conq(/*eta [1/Å]=*/2.0);
conq.add_electrode_charge(/*type=*/1, /*Q [e]=*/-0.279214485147238);
conq.add_electrode_charge(/*type=*/2, /*Q [e]=*/+0.279214485147238);
conq.xy_image_range = 250;
conq.rebuild_matrix_every_step = true;   // electrodes move / small twin

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.add_ref(conq);   // or add(std::move(conq)) to own; pre_force @ -50
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_top = conq.total_charge(1);   // [e] — exactly the target
const double dv = conq.delta_psi();          // [V] — LAMMPS v_dv response
const double psi_bot = conq.electrode_potential(0);  // [V] — f_conq[1]
// frame.span(kAtom_charge)[0..1] hold the solved per-site distribution;
// electrolyte rows are untouched.
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** conq is the same `ConstPotBO` Component;
  never a new Driver category.
- Parameters are **constructor arguments and public members** — no
  `ConstPotBOConfig` / conq bag; `add_electrode_charge` is ref-qualified for
  rvalue chaining.
- Frame is the sole I/O currency; solved per-site charges land on
  `kAtom_charge`, electrolyte rows are never mutated.
- Any const-Q group supersedes the `electroneutrality` projector (pick
  targets summing to \(-q_{\text{elyt}}\) for a neutral cell).
- Mixed const-\(\Psi\) / const-\(Q\) groups on one component are supported;
  both the dense-Cholesky and CG paths solve the same reduced SPD system.

Build and run the regression (MD-only configure; CPU login pattern):

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
  cmake --build build --target reg_const-charge-electrode -j"$(nproc)"
  ./build/regressions/reg_const-charge-electrode
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Public `ConstPotBO` (`add_electrode_charge`, `electrode_potential`, `delta_psi`) |
| `src/cpu/dynamics/const_pot_bo.cpp` | Reduced SPD system \(M\psi = r\) on the dense and CG paths |
| `regressions/md/const-charge-electrode/reg_const-charge-electrode.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/in.conq_madelung` | Exact conq twin input (re-pin on any `PKG_ELECTRODE` host) |

## Validation (vs LAMMPS ELECTRODE)

Source:
[`regressions/md/const-charge-electrode/REFERENCE.md`](../../../regressions/md/const-charge-electrode/REFERENCE.md)
(offline log pin; **never** live LAMMPS in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
and
[`tools/metalwalls_oracle/pin_notes_const-charge-electrode.md`](../../../tools/metalwalls_oracle/pin_notes_const-charge-electrode.md).

**Oracle:** LAMMPS ELECTRODE package, 3 Nov 2022 (madelung example log
`log.1Dec2022.madelung-ewald-ew3dc.g++.1` + `graph-il/in.conq`
`fix electrode/conq` semantics). Local `lmp` lacks `PKG_ELECTRODE`, so the
pin uses the **shipped official logs** plus the package's documented
conq↔conp inverse contract — same sourcing as `const-pot-bo-parallel-plate`.

### Accuracy

| Quantity | Atomiverse gate | Oracle pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| conp \(Q^{*}_{\mathrm{top}}\) [e] | within **1%** of pin | **+0.279214485147238** | rel ≤ **0.01** | LAMMPS conp log anchor (v1 A-matrix gate) |
| conq group totals [e] | equal the targets \(\mp Q^{*}\) | targets ("Q fixed") | **1e-12** | constraint exact by construction |
| conq \(\Delta\Psi(\pm Q^{*})\) [V] | \(= 1\) V | **1.0** (inverse contract) | **1e-8** | the tight, well-conditioned gate |
| conq per-site charges [e] | equal conp per-site charges | duality | **1e-10** | same distribution both directions |
| conq \(\Delta\Psi(Q_{\mathrm{pin}})\) [V] | \(1 \pm 0.25\) | **1.0** | **±0.25** | coarse gate; see error budget |
| Empty-capacitor \(\Delta\Psi\) at \(Q=\pm1\) e [V] | within **1.5%** | **651.8103** (Madelung sum) | rel ≤ **0.015** | R=250 real-space tail ≈ 0.8% |
| MD-bound path (1 Verlet step) | totals exact / \(\Delta\Psi\approx1\) V | — | **1e-10 e** / **±0.05 V** | Component on `cpu::md::MD` at `pre_force` |
| Electrolyte charges [e] | ±0.5 untouched | — | **1e-14** | solver never mutates non-electrode rows |

**Error budget (why the raw-\(Q_{\mathrm{pin}}\) gate is ±0.25 V):** the
plate stiffness \(d\Delta\Psi/dQ \approx 4\pi k_e d / A_{\mathrm{cell}}
\approx 724\) V/e amplifies the v1 real-space-A vs `ewald/electrode` charge
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
# Exact twin of this regression:
#   tools/metalwalls_oracle/inputs/lammps/in.conq_madelung
```

## Performance

Source: REFERENCE.md perf block; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| madelung conq BO solve | — | — | **perf_exempt** | n/a | local `lmp` lacks `PKG_ELECTRODE`; no fair `fix electrode/conq` twin binary |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | local `lmp` lacks `PKG_ELECTRODE` (verified 2026-07-27: installed = EXTRA-*, KSPACE, MANYBODY, MISC, MOLECULE, OPENMP, QEQ, REPLICA, RIGID); no fair `fix electrode/conq` twin binary |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a `PKG_ELECTRODE` LAMMPS binary is
available, re-run the fair twin offline
(`tools/metalwalls_oracle/inputs/lammps/in.conq_madelung`), record version +
walls, and replace `perf_exempt` in REFERENCE.

## Limitations

- **v1 elastance only.** Real-space Gaussian + 2D \(xy\) images; no full
  `ewald/electrode` reciprocal matrix — the ~0.07% charge-scale difference is
  amplified by the plate stiffness (documented error budget), so the tight
  gate is the inverse contract, not the raw pin response.
- **Uniform-shift gauge.** Individual `electrode_potential(e)` values carry
  the same gauge freedom LAMMPS documents for `symm on/off` — only potential
  *differences* are physical observables.
- **Toy twin geometry.** Four atoms, one madelung cell; a production EDL box
  is the `const-pot-il-edl-box` sibling.
- **Dense \(O(N_{\mathrm{elec}}^3)\) default** for the BO solve (CG + ASPC
  warm-start are the large-electrode siblings; the reduced \(M\psi = r\)
  system itself is tiny — one row per const-Q group).
- **No fair speed ratio** until a `PKG_ELECTRODE` binary is on the pin host.
- **Scope of this slug only:** perfect-metal conp (E1), Thomas–Fermi
  metallicity, CG large electrode, MPI halo, MaZe dynamics are sibling slugs.

## References

- Offline pin: [`regressions/md/const-charge-electrode/REFERENCE.md`](../../../regressions/md/const-charge-electrode/REFERENCE.md)
- Product gate: [`regressions/md/const-charge-electrode/reg_const-charge-electrode.cpp`](../../../regressions/md/const-charge-electrode/reg_const-charge-electrode.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h)
- Pin notes: [`tools/metalwalls_oracle/pin_notes_const-charge-electrode.md`](../../../tools/metalwalls_oracle/pin_notes_const-charge-electrode.md); twin input `tools/metalwalls_oracle/inputs/lammps/in.conq_madelung`
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (§1 constant-charge control)
- Literature: Ahrens-Iwers, Janssen, Tee, Meißner, *J. Chem. Phys.* **157**, 084801 (2022), DOI [10.1063/5.0099239](https://doi.org/10.1063/5.0099239) (LAMMPS ELECTRODE, `fix electrode/conq`); Griffiths, *Introduction to Electrodynamics*, 4th ed., §2.5.4 (sheet term); Glasser & Zucker, *Theoretical Chemistry: Advances and Perspectives* **5**, 67 (1980) (square-lattice Madelung constant)
- Related siblings: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md) (conp E1 foothold), [`thomas-fermi-electrode.md`](thomas-fermi-electrode.md) (finite metallicity)
