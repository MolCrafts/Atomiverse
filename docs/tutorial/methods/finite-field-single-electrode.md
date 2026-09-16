# Finite-field + single electrode — Dufils-style 3D foothold

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `finite-field-single-electrode` |
| Regression | `regressions/md/finite-field-single-electrode/` |
| Public API | `cpu::md::ConstPotBO` (`Ex`/`Ey`/`Ez` on BO RHS) + `cpu::md::FixEfield` + `cpu::md::ClassicalPolarization` on `cpu::md::MD` |
| Roadmap | map slice E3 |
| Oracle | **MetalWalls / literature analytic** (Dufils et al. *PRL* **123**, 195501 (2019); Reed/Siepmann–Sprik Gaussian kernels) |

## Summary

**Finite-field constant-potential electrode** in **3D PBC with a single wall**:
one Gaussian electrode group at applied \(\Psi\), free total charge
(`electroneutrality = false`), and a uniform external field \(\mathbf{E}\)
folded into the BO right-hand side so induced electrode charge responds to
\(\Phi_{\mathrm{ext}} = -\mathbf{E}\cdot\mathbf{r}\). Product surface composes
three Components on `cpu::md::MD` (never a new Driver category; never a charge
solve inside `KSpace*`):

1. `cpu::md::ConstPotBO` — solves \(A Q = \Psi - b + \mathbf{E}\cdot\mathbf{r}\)
   at default `pre_force` (−50), writes \(Q\) to `kAtom_charge`.
2. `cpu::md::FixEfield` — electrolyte / free-ion forces \(\mathbf{F}=q\mathbf{E}\)
   at default `post_force` (+100).
3. `cpu::md::ClassicalPolarization` — observer
   \(\mathbf{P}=V^{-1}\sum q\mathbf{r}^{u}\) at default `end_of_step` (+50).

This is the **E3 single-electrode** foothold (Dufils-class). Offline pin is a
**vacuum single-site analytic** Reed self-term (no live MetalWalls binary on
the pin host). Fair wall vs MetalWalls is **`perf_exempt`**.

## Method

**Physics.** Siepmann–Sprik / Reed Gaussian electrode charges of width \(\eta\)
[1/Å]. With metal units \(k_e =\) `units::coulomb` \(= 14.399645\) eV·Å/e², the
self elastance is

\[
A_{ii} = k_e\,\eta\sqrt{\frac{2}{\pi}}.
\]

For a uniform external field the BO equation on electrode sites is

\[
(A Q)_i = \Psi_i - b_i + \mathbf{E}\cdot\mathbf{r}_i,
\]

where \(b_i\) is the electrolyte contribution to the potential at site \(i\)
(zero in the vacuum pin). **Electroneutrality is off** for a free total \(Q\)
on a single wall (MetalWalls / Dufils single-electrode geometry).
\(\mathbf{E}\) on `ConstPotBO` only updates charges; forces on free charges
come from a separate `FixEfield` Component:

\[
\mathbf{F}_i \mathrel{+}= q_i\mathbf{E},\qquad
U \mathrel{+}= -q_i\,\mathbf{E}\cdot\mathbf{r}_i^{u}
\quad\text{(eV, eV/Å; metal units, }qe2f=1\text{)}.
\]

Classical polarization density (MetalWalls / Dufils observable):

\[
\mathbf{P} = \frac{1}{V}\sum_i q_i\,\mathbf{r}_i^{u}
\quad [e/\mathrm{Å}^2].
\]

**Why MetalWalls / literature, not LAMMPS.** LAMMPS ELECTRODE `ffield on`
implements the **two-electrode** finite-field path
(\(E_z=\Delta\Psi/L_z\), requires exactly two groups). This slug is the
**single-wall** Dufils product — MetalWalls-only geometry under the loop
oracle priority rule.

### Vacuum single-site pin (this regression)

| Item | Setting |
| ---- | ------- |
| Box | \(L=20\) Å cubic, 3D PBC |
| \(\eta\) | \(2.0\) Å⁻¹ |
| \(\Psi\) | \(0\) V |
| \(\mathbf{E}\) | \((0,\,0,\,0.1)\) V/Å |
| Electrode | type 1 at \(\mathbf{r}=(0,0,10)\) Å |
| Neutrality | **off** |
| \(xy\) images | `0` (primary cell only) |
| Electrolyte | none (vacuum pin); dual-gate ion test is a separate one-atom frame |

Analytic numbers (offline; `REFERENCE.md`):

\[
A_{ii} = 22.978508853084353\ \mathrm{V}/e,
\quad
Q_{\mathrm{pin}} = \frac{E_z z}{A_{ii}} = 0.04351892485250505\,e,
\]

\[
P_{z,\mathrm{pin}} = \frac{Q z}{V} = 5.439865606563132\times 10^{-5}\ e/\mathrm{Å}^2
\quad (V=L^3=8000\,\mathrm{Å}^3).
\]

Field force on a test ion \(q=+1\): \(F_z = q E_z = 0.1\) eV/Å.

**Key exclusions (non-goals of this slug)**

- Two-electrode LAMMPS ELECTRODE `ffield on` product (different geometry /
  constraint).
- Full electrolyte continuum or MetalWalls liquid input as the ctest pin
  (vacuum analytic only).
- Oscillating / AC field (`oscillating-efield-md`, queue residual).
- Charge solve inside PME/`KSpace*` — forbidden by architecture.
- Live MetalWalls or LAMMPS inside ctest.
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + Components with ctor args / members; no
Config bag). Standalone BO + polarization matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/classical_polarization.h"

using namespace ATV;

cpu::Frame frame(1);
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 10.0;   // [Å]
frame.span(kAtom_charge)[0] = 0.0;
frame.span(kAtom_type)[0] = 1;
frame.span(kAtom_mass)[0] = 196.966553;
frame.span(kAtom_tag)[0] = 1;
frame.box = cpu::Box::cubic(20.0);

// BO + polarization + optional field force on one MD step (Frame only).
cpu::md::ConstPotBO conp(/*eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.Ez = 0.1;                   // [V/Å] — BO RHS only
conp.electroneutrality = false;  // free total Q (single wall)
conp.rebuild_matrix_every_step = true;

cpu::md::ClassicalPolarization pol;
cpu::md::FixEfield field(0.0, 0.0, 0.1);

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/2.0, /*skin=*/0.3});
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0, /*sigma=*/1.0, /*cutoff=*/2.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.add_ref(conp);    // pre_force −50
sim.add_ref(field);   // post_force +100
sim.add_ref(pol);     // end_of_step +50
sim.build();
sim.run(frame, /*nsteps=*/1);

const double Q = conp.total_charge(0);  // [e] → pin 0.04351892485250505
// pol.Pz() → pin 5.439865606563132e-5 [e/Å²]
```

Dual-gate field force alone (separate from charge solve):

```cpp
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::md::FixEfield{/*Ex=*/0.0, /*Ey=*/0.0, /*Ez=*/0.1});
eval.build();
eval.run(frame_ion);   // q = +1 → Fz = 0.1 eV/Å
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Electrodes and field are Components.
- `ConstPotBO::{Ex,Ey,Ez}` and charge write-back are **orthogonal** to
  `FixEfield` forces — both needed for a full finite-field MD step.
- Parameters are **constructor arguments and public members** — no Config bags.
- Frame is the sole I/O currency; \(Q\) lands on `kAtom_charge`.

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
  cmake --build build --target reg_finite-field-single-electrode -j"$(nproc)"
  ./build/regressions/reg_finite-field-single-electrode
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Public `ConstPotBO`, `Ex`/`Ey`/`Ez` members |
| `include/atomiverse/cpu/dynamics/fix_efield.h` | `FixEfield` (LAMMPS-like constant field) |
| `include/atomiverse/cpu/observe/classical_polarization.h` | \(\mathbf{P}\) observer |
| `include/atomiverse/cpu/dynamics/md.h` | Host preset `cpu::md::MD` |
| `regressions/md/finite-field-single-electrode/reg_finite-field-single-electrode.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/metalwalls/finite-field-single-electrode/` | Offline pin notes |

## Validation (vs MetalWalls / literature analytic)

Source:
[`regressions/md/finite-field-single-electrode/REFERENCE.md`](../../../regressions/md/finite-field-single-electrode/REFERENCE.md)
(offline pin; **never** live MetalWalls or LAMMPS in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md),
[`tools/metalwalls_oracle/inputs/metalwalls/finite-field-single-electrode/README.md`](../../../tools/metalwalls_oracle/inputs/metalwalls/finite-field-single-electrode/README.md).

**Oracle:** literature analytic — Dufils et al., *Phys. Rev. Lett.* **123**,
195501 (2019) finite-field + single constant-potential electrode; Reed /
Siepmann–Sprik Gaussian electrode kernels (same family as MetalWalls JOSS **5**,
2373 (2020) / Atomiverse `ConstPotBO`). `oracle: metalwalls`,
`oracle_version: literature analytic`.

**Why not LAMMPS:** ELECTRODE `ffield on` is two-electrode only; single-wall
finite-field is MW-only under the binding oracle rule.

### Accuracy

| Quantity | Atomiverse gate / pin | Analytic pin | Tol | Notes |
| -------- | --------------------: | -----------: | --- | ----- |
| \(A_{ii}\) [V/e] | \(k_e\eta\sqrt{2/\pi}\) | **22.978508853084353** | identity | Reed self-term |
| \(Q\) [e] | within **1e-10** of pin | **0.04351892485250505** | abs ≤ **1e-10** | \(Q=(E_z z)/A_{ii}\), \(\Psi=0\), \(b=0\) |
| \(P_z\) [e/Å²] | within **1e-12** of pin | **5.439865606563132×10⁻⁵** | abs ≤ **1e-12** | standalone `ClassicalPolarization` |
| \(P_x,P_y\) | ~0 | 0 | abs ≤ **1e-14** | vacuum pin orientation |
| \(F_z\) ion [eV/Å] | within **1e-12** of \(E_z\) | **0.1** | abs ≤ **1e-12** | `FixEfield` dual-gate, \(q=+1\) |
| MD \(Q\) (1 VV step) | same as pin | **0.04351892485250505** | abs ≤ **1e-10** | `ConstPotBO` + `FixEfield` + pol on `MD` |
| MD \(P_z\) | same as pin | **5.439865606563132×10⁻⁵** | abs ≤ **1e-10** | large mass / \(v=0\); looser than standalone |

Geometry, \(\eta\), \(\mathbf{E}\), \(\Psi\), and tolerances: **Method** table
and REFERENCE (match `reg_finite-field-single-electrode`).

Related unit surface (not the e2e pin):
`tests/cpu/dynamics/test_const_pot_ffield.cpp`,
`tests/cpu/dynamics/test_fix_efield.cpp`,
`tests/cpu/observe/test_classical_polarization.cpp`.

### Offline pin repro (never ctest)

```python
import math
coulomb = 14.399645
eta, Ez, z, L = 2.0, 0.1, 10.0, 20.0
Aii = coulomb * eta * math.sqrt(2 / math.pi)
Q = (Ez * z) / Aii
Pz = Q * z / (L ** 3)
print(Aii, Q, Pz)
# 22.978508853084353  0.04351892485250505  5.439865606563132e-05
```

## Performance

Source: REFERENCE.md dual-gate summary; handoff
the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{MW}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ----------------------: | -----------------------: | ----: | ---- | ------ |
| single-electrode finite-field | — | — | **perf_exempt** | n/a | No MetalWalls binary on agent host; LAMMPS `ffield` is two-electrode only — no fair twin wall |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | no MetalWalls binary on agent host; single-electrode finite-field is MW-only (LAMMPS ffield requires two electrodes) — no fair twin wall |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls build is available for
an equivalent single-wall finite-field input, re-run offline fair bench, record
version + walls, and replace `perf_exempt` in REFERENCE.

Sibling constant-field MD (two-path product, LAMMPS oracle) is documented under
[`external-efield-md`](external-efield-md.md) when that page exists — fair ratio
there is independent of this MW-only slug.

## Limitations

- **Vacuum single-site pin.** No electrolyte \(b\) term, no multi-site wall,
  no liquid MetalWalls input as the ctest oracle — only the Reed self-term
  identity for \(A Q = \Psi + \mathbf{E}\cdot\mathbf{r}\).
- **Primary-cell elastance.** `xy_image_range = 0` on this pin; planar image
  sums and full electrode Ewald remain E1/E2 product surfaces, not this
  foothold’s gate.
- **Free total \(Q\).** Electroneutrality projection is intentionally off;
  parallel-plate `symm on` is the E1 path (`const-pot-bo-parallel-plate`).
- **Constant \(\mathbf{E}\) only.** Oscillating / variable field is
  `oscillating-efield-md` (queue residual).
- **No fair speed ratio** until a MetalWalls twin binary is on the pin host.
- **Scope of E3 wave-2 only** for this page: constant external field MD
  (`external-efield-md`), BO plate, CG/ASPC, piston, TF, etc. are sibling
  slugs with their own pages when written.

## References

- Offline pin: [`regressions/md/finite-field-single-electrode/REFERENCE.md`](../../../regressions/md/finite-field-single-electrode/REFERENCE.md)
- Product gate: [`regressions/md/finite-field-single-electrode/reg_finite-field-single-electrode.cpp`](../../../regressions/md/finite-field-single-electrode/reg_finite-field-single-electrode.cpp)
- Public headers: [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h), [`include/atomiverse/cpu/dynamics/fix_efield.h`](../../../include/atomiverse/cpu/dynamics/fix_efield.h), [`include/atomiverse/cpu/observe/classical_polarization.h`](../../../include/atomiverse/cpu/observe/classical_polarization.h)
- Capability map: [`.claude/notes/metalwalls-capability-map.md`](../../../.claude/notes/metalwalls-capability-map.md) (E3)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/inputs/metalwalls/finite-field-single-electrode/README.md`](../../../tools/metalwalls_oracle/inputs/metalwalls/finite-field-single-electrode/README.md)
- Related E1 tutorial: [`const-pot-bo-parallel-plate`](const-pot-bo-parallel-plate.md)
- Literature: Dufils et al., *Phys. Rev. Lett.* **123**, 195501 (2019); MetalWalls JOSS **5**, 2373 (2020); Reed / Siepmann–Sprik Gaussian electrode charges
