# Coupled PIM dipoles ⊕ electrode \(Q\) — parallel-plate IL ion pair (E5)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dipole-electrode-joint` |
| Regression | `regressions/md/dipole-electrode-joint/` |
| Public API | `cpu::md::DipoleElectrodeJoint` (Component on `cpu::md::MD`, default `pre_force`) |
| Roadmap | map slice E5 |
| Oracle | **MetalWalls** multipole + Gaussian electrode BO form (offline algebraic pin; not LAMMPS) |

## Summary

Classical **joint** of constant-potential Born–Oppenheimer electrode charges
\(Q\) and **Polarizable Ion Model (PIM)** induced dipoles \(\boldsymbol{\mu}\)
on electrolyte sites: the electrode RHS includes the permanent electrolyte
potential **and** the potential of induced dipoles; the PIM field includes all
charges (electrode \(Q\) + permanent \(q\)). The outer alternating loop runs
until \(\max|\Delta Q|\) and \(\max|\Delta\mu|\) fall below `joint_tol`. Public
surface is `cpu::md::DipoleElectrodeJoint` — a **Component** bound at
`md::Phase::pre_force` on `cpu::md::MD` (not a Driver category; not inside
`KSpace*`). It writes electrode charges to `kAtom_charge` and exposes
\(\boldsymbol{\mu}\) via `last_mu()`. This regression is the **E5 foothold**:
two graphene-like electrode sites, one IL cation, one IL anion, offline
MetalWalls multipole ⊕ Gaussian-electrode algebra. **Oracle is MetalWalls** —
LAMMPS has no SCF PIM ⊕ const-pot joint twin (ELECTRODE is non-polarizable;
Drude is a different model). Fair wall vs MetalWalls is **`perf_exempt`** (no
MW binary on the pin host).

## Method

**Physics.** Self-consistent joint of:

1. **Constant-potential BO** electrode charges \(Q\) with Gaussian elastance
   \(A\) (same kernels as `cpu::md::ConstPotBO`).
2. **PIM induced dipoles** \(\boldsymbol{\mu}\) on polarizable electrolyte
   sites with Thole damping (same kernels as bulk `cpu::PimInducedDipole`).

Coupling (MetalWalls multipole + Gaussian electrode form; \(k_e =\)
`units::coulomb` \(= 14.399645\) eV·Å/e²):

\[
A Q = \Psi - \Phi^{\mathrm{perm}} - \Phi^{\mu},\quad
\Phi_i^{\mu} = k_e\sum_j\frac{\boldsymbol{\mu}_j\cdot\mathbf{r}_{ij}}{r_{ij}^3}
\]

\[
\boldsymbol{\mu}_i = \alpha_i\mathbf{E}_i^{\mathrm{tot}}(q_{\mathrm{all}},\boldsymbol{\mu})
\]

Outer alternating solve: rebuild electrode RHS with current \(\boldsymbol{\mu}\),
solve \(Q\) (optional electroneutrality projector), write electrode charges,
inner Thole-damped PIM SCC for \(\boldsymbol{\mu}\), until
\(\max|\Delta Q|\) and \(\max|\Delta\mu|\) \(\le\) `joint_tol`.

**What this pin includes**

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
| Binding | standalone `build`/`launch`, and MD `add_ref` → default `pre_force` @ −50 |

**Key exclusions (non-goals of this slug)**

- LAMMPS ELECTRODE or Drude as oracle (different physics; not a joint twin).
- Full electrode Ewald / MetalWalls reciprocal elastance (v1 shares ConstPotBO
  real-space Gaussian \(A\); pin uses min-image).
- Bulk-only PIM without electrodes (E4 `pim-induced-dipole-bulk`).
- Neighbor-list / production \(O(N)\) pair path for the joint SCC.
- Live MetalWalls binary inside ctest (offline pin only).
- Public `*Config` bags or `make_*` / `build_*` factories.
- Charge solve inside PME / `KSpace*` — forbidden by architecture.

## Implementation

Public surface only (`cpu::Frame` + `DipoleElectrodeJoint` ctor args / members;
no Config bag). Minimal standalone solve matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/dipole_electrode_joint.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame frame(4);
const double z[4]  = {2.0, 18.0, 8.0, 12.0};
const double q0[4] = {0.0, 0.0, 1.0, -1.0};
const int typ[4]   = {1, 2, 3, 4};
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_x)[i] = frame.span(kAtom_y)[i] = 0.0;
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_charge)[i] = q0[i];
    frame.span(kAtom_type)[i] = typ[i];
    frame.span(kAtom_mass)[i] = (typ[i] < 3) ? 12.0 : 1.0;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);

// Component defaults to pre_force order −50 so pair/kspace see updated charges.
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
joint.rebuild_matrix_every_step = true;  // optional if geometry moves

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/8.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/8.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.add_ref(joint);   // or add(std::move(joint)) to own
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_bot = joint.total_charge(0);  // [e]
const double q_top = joint.total_charge(1);  // [e]
const auto& mu = joint.last_mu();            // packed xyz [e·Å]
// mu[6..8] = cation; mu[9..11] = anion
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Electrodes + PIM joint is a Component; never a
  new Driver category.
- Parameters are **constructor arguments and public members** — no
  `DipoleElectrodeJointConfig`.
- Frame is the sole I/O currency; electrode \(Q\) lands on `kAtom_charge`;
  diagnostics via `total_charge`, `last_mu()`, `last_joint_*`.
- Charge solve is **not** inside `KSpace*` / PME.

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
  cmake --build build --target reg_dipole-electrode-joint -j"$(nproc)"
  ./build/regressions/reg_dipole-electrode-joint
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/dipole_electrode_joint.h` | Public `DipoleElectrodeJoint` |
| `src/cpu/dynamics/dipole_electrode_joint.cpp` | Outer joint, BO \(A Q\), inner PIM SCC, write-back |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Shared electrode BO kernels / \(\alpha=0\) parity |
| `include/atomiverse/cpu/dynamics/md.h` | Host preset `cpu::md::MD` |
| `regressions/md/dipole-electrode-joint/reg_dipole-electrode-joint.cpp` | Product gate |
| `tests/cpu/dynamics/test_dipole_electrode_joint.cpp` | Unit surface |
| `tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/pin_joint.py` | Offline pin script |

## Validation (vs MetalWalls-form offline pin)

Source:
[`regressions/md/dipole-electrode-joint/REFERENCE.md`](../../../regressions/md/dipole-electrode-joint/REFERENCE.md)
(offline pin; **never** live MetalWalls in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md);
pin notes in `tools/metalwalls_oracle/pin_notes.md` (§ `dipole-electrode-joint`).

**Oracle:** MetalWalls multipole + Gaussian electrode BO form — Coretti et al.,
*J. Chem. Phys.* (2022) (polarizable interfaces / MetalWalls
`dipoles_and_electrodes`); Marin-Laflèche et al., *J. Open Source Softw.* **5**,
2373 (2020); Thole, *Chem. Phys.* **59**, 341 (1981). **Offline algebraic pin**
via `pin_joint.py` (no MetalWalls binary on the agent host).

**Why MetalWalls (not LAMMPS):** joint SCF PIM ⊕ const-pot is **MW-only** in
this loop’s oracle priority. LAMMPS ELECTRODE is non-polarizable; Drude is a
different polarizable model — neither is a joint twin for this pin.

### Accuracy

| Quantity | Atomiverse / pin | Oracle pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(Q_{\mathrm{bot}}\) [e] | **−7.3285008958883610e−02** | same hard pin | rel \(\le 10^{-6}\) | joint \(\alpha>0\) |
| \(Q_{\mathrm{top}}\) [e] | **+7.3285008958883610e−02** | same | rel \(\le 10^{-6}\) | opposite under neutrality |
| \(\lvert\sum Q\rvert\) [e] | 0 | 0 | \(\le 10^{-10}\) | electroneutrality |
| \(\mu_{\mathrm{cation},z}\) [e·Å] | **+5.8951916148990660e−02** | same | \(\lvert\Delta\mu_z\rvert\le 10^{-8}\) | collinear \(z\) geometry |
| \(\mu_{\mathrm{anion},z}\) [e·Å] | **+8.7432701391015569e−02** | same | \(\lvert\Delta\mu_z\rvert\le 10^{-8}\) | packed in `last_mu()` |
| \(\mu_{*,x}\), \(\mu_{*,y}\) [e·Å] | 0 | 0 | \(\lvert\mu\rvert < 10^{-10}\) | transverse |
| \(\alpha=0\) \(Q_{\mathrm{top}}\) [e] | **+7.5339627935767609e−02** | same | rel \(\le 10^{-6}\) | dual-gate reduction |
| \(\alpha=0\) vs `ConstPotBO` | \(\lvert\Delta Q\rvert\le 10^{-10}\) | parity | absolute | joint degenerates to BO |
| Coupling \(\lvert Q_{\mathrm{top}}-Q_{\alpha=0}\rvert/\lvert Q_{\alpha=0}\rvert\) | \(\ge 10^{-6}\) | expected shift | lower bound | induced \(\mu\) moves \(Q\) |
| Electrolyte \(q\) | unchanged \(\{+1,-1\}\) | fixed | exact write-back | only electrode rows rewritten |
| MD-bound \(Q_{\mathrm{top}}\) | same joint pin | same | rel \(\le 10^{-6}\) | one-step `add_ref` path |

**Dual gate:** electrode \(Q\) **and** induced \(\mu_z\), plus \(\alpha=0\)
reduction vs `ConstPotBO` and a measurable coupling shift. Geometry,
\(\eta\), \(\Psi\), \(\alpha\), Thole \(a\), and tolerances: **Method** table
and REFERENCE (match `reg_dipole-electrode-joint`).

Related unit surface (not the e2e pin):
`tests/cpu/dynamics/test_dipole_electrode_joint.cpp`.

### Offline pin repro (never ctest)

```bash
python3 tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/pin_joint.py
# prints joint Q and mu pins; no MetalWalls binary required
```

## Performance

Source: REFERENCE.md; handoff
the case `REFERENCE.md`
(wave 3, 2026-07-24).

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| joint \(Q\)–\(\mu\) parallel-plate / multipole wall | — | — | **perf_exempt** | n/a | No MetalWalls binary on PATH for a fair joint \(Q\)–\(\mu\) wall twin; pin is published MW multipole ⊕ Gaussian-electrode algebra |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | No MetalWalls binary on PATH for a fair joint \(Q\)–\(\mu\) wall twin; pin is the published MW multipole ⊕ Gaussian-electrode algebra (same path as thomas-fermi / bulk-PIM offline pins) |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls twin binary is
available, record \(T_{\mathrm{atv}}/T_{\mathrm{MW}}\) in REFERENCE and require
\(\le 10\).

## Limitations

- **Foothold geometry.** Four-site collinear parallel plate; not a production
  multi-layer electrode + dense electrolyte.
- **v1 elastance.** Real-space Gaussian \(A\) shared with `ConstPotBO`; pin
  uses `xy_image_range = 0` (min-image). Full electrode Ewald / MW reciprocal
  matrix is out of scope for this slug.
- **All-pairs joint SCC.** No NeighborConsumer production path for the coupled
  solve.
- **Isotropic \(\alpha\) by type.** Electrode types must keep \(\alpha=0\);
  no anisotropic polarizability, no Drude oscillators.
- **No fair speed ratio** until a MetalWalls joint twin is on the pin host.
- **Depends on E4 bulk PIM kernels and E1 BO electrode form** — those pages
  document the decoupled pieces; this page is the **coupled** product only.

## References

- Offline pin: [`regressions/md/dipole-electrode-joint/REFERENCE.md`](../../../regressions/md/dipole-electrode-joint/REFERENCE.md)
- Product gate: [`regressions/md/dipole-electrode-joint/reg_dipole-electrode-joint.cpp`](../../../regressions/md/dipole-electrode-joint/reg_dipole-electrode-joint.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/dipole_electrode_joint.h`](../../../include/atomiverse/cpu/dynamics/dipole_electrode_joint.h)
- Pin script: [`tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/pin_joint.py`](../../../tools/metalwalls_oracle/inputs/metalwalls/dipole-electrode-joint/pin_joint.py)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/pin_notes.md`](../../../tools/metalwalls_oracle/pin_notes.md)
- Sibling methods: [pim-induced-dipole-bulk](pim-induced-dipole-bulk.md) (E4), [const-pot-bo-parallel-plate](const-pot-bo-parallel-plate.md) (E1 BO form)
- Literature: Coretti et al., *J. Chem. Phys.* (2022) — polarizable electrolyte
  + metallic electrodes (MetalWalls `dipoles_and_electrodes`); Marin-Laflèche
  et al., *J. Open Source Softw.* **5**, 2373 (2020); Thole, *Chem. Phys.*
  **59**, 341 (1981)
