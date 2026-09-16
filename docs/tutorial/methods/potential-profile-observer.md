# Slab electrostatic potential profile \(\Psi(z)\) — parallel-plate sheets

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `potential-profile-observer` |
| Regression | `regressions/md/potential-profile-observer/` |
| Public API | `cpu::PotentialProfile` (`Observer` on `cpu::md::MD`, default `end_of_step` @ +45) |
| Roadmap | map slice E10 ecosystem |
| Oracle | **literature** (planar Poisson / parallel-plate; Jackson continuum; MetalWalls potential-profile lineage; no LAMMPS or MetalWalls binary twin) |

## Summary

**Planar slab electrostatic potential profile** \(\Psi(z)\) for classical
electrode / electric-double-layer (EDL) slabs. Public surface is
`cpu::PotentialProfile` — a frame-immutable **Observer** that bins
`kAtom_charge` along \(z\) into a planar density \(\rho(z)\) and integrates the
planar Poisson equation in MD metal units to \(E_z(z)\) [V/Å] and \(\Psi(z)\)
[V]. Gauge: \(\Psi(z_{\min})=0\), \(E_z(z_{\min})=0\) at the left edge of the
bin window. It does **not** solve constant-potential charges — it measures
whatever is already on the Frame (typically after `ConstPotBO` or a fixed-charge
setup). Architecture: Component on `cpu::md::MD` — never a new Driver category;
never charge solve inside `KSpace*`. Product gate
`reg_potential-profile-observer` pins the discrete piecewise-constant Poisson
scheme on a synthetic two-sheet capacitor against the closed-form continuum
identity \(E_{\mathrm{gap}}=4\pi k_{\mathrm{e}}\sigma\). Fair wall vs an oracle
binary is **`perf_exempt`**.

## Method

**Physics.** For planar symmetry the electrostatic field and potential satisfy

\[
\frac{\mathrm{d} E_z}{\mathrm{d} z}
  = 4\pi\,k_{\mathrm{e}}\,\rho(z)
  = \frac{\rho(z)}{\varepsilon_0},
\qquad
E_z = -\frac{\mathrm{d}\Psi}{\mathrm{d} z},
\]

with

\[
k_{\mathrm{e}} = 14.399645\,\mathrm{eV\cdot Å\cdot e^{-2}}
\]

(`units::coulomb`, LAMMPS metal `qqr2e`), \(\rho\) [e/Å³], \(E_z\) [V/Å],
\(\Psi\) [V]. Ideal infinite parallel-plate capacitor with surface charge
\(\pm\sigma\):

\[
E_{\mathrm{gap}} = 4\pi k_{\mathrm{e}}\,\sigma
  \qquad\text{(between plates; \(0\) outside)}.
\]

**Discrete scheme (implementation algorithm).** Charges on owned atoms are
histogrammed into \(n_{\mathrm{bins}}\) bins over \([z_{\min}, z_{\max})\).
Per bin,

\[
\rho_i = \frac{Q_i}{A\,\Delta z},\qquad
A = L_x L_y,\qquad
\Delta z = \frac{z_{\max}-z_{\min}}{n_{\mathrm{bins}}}.
\]

Assuming piecewise-constant \(\rho\) in each bin, the left-edge field and
potential are advanced analytically; samples at bin centers are

\[
E_i = E_{\mathrm{left}} + 4\pi k_{\mathrm{e}}\,\rho_i\,\tfrac{\Delta z}{2},
\]

\[
\Psi_i = \Psi_{\mathrm{left}}
  - \Bigl(
      E_{\mathrm{left}}\,\tfrac{\Delta z}{2}
      + \tfrac12\,4\pi k_{\mathrm{e}}\,\rho_i\,
        \bigl(\tfrac{\Delta z}{2}\bigr)^2
    \Bigr),
\]

then the right edge is advanced by a full bin step. The reported potential
drop is \(\Delta\Psi = \Psi(z_{\max})-\Psi(z_{\min})\) with left-edge gauge
zero. MPI: charge histograms are all-reduced when a domain decomposition is
active.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry | Two point sheets (proxy for planar walls): \(Q=+1\,e\) at \(z=5.5\) Å, \(Q=-1\,e\) at \(z=14.5\) Å |
| Box | orthorhombic \(L_x=L_y=10\) Å, \(L_z=20\) Å |
| Area \(A\) | \(100\) Å² |
| Bin window | \(z\in[0,20)\) Å |
| \(n_{\mathrm{bins}}\) | \(20\) |
| \(\Delta z\) | \(1\) Å |
| \(\sigma = Q/A\) | \(0.01\) e/Å² |
| Binding | standalone `launch` (regression); MD default `end_of_step` @ +45 |

Closed form / discrete pin (REFERENCE):

\[
E_{\mathrm{gap}}^{\mathrm{pin}}
  = 4\pi\times 14.399645\times 0.01
  = 1.80951275785204\,\mathrm{V/Å},
\]

\[
\Psi(z_{\max})^{\mathrm{pin}}
  = -E_{\mathrm{gap}}\times 9
  = -16.28561482066836\,\mathrm{V}
\]

(center-to-center plate separation \(9\) Å under this binning). Mid-gap field
is bin 9 (\(z=9.5\)); vacuum bins 0 and 19 have \(E_z=0\); \(\rho\) at the
charged bins is \(\pm 0.01\) e/Å³.

**Key exclusions (non-goals of this slug)**

- Solving electrode charges (that is `ConstPotBO` / E1–E2); this Observer only
  **reads** `kAtom_charge` and `kAtom_z`.
- Live MetalWalls post-process scripts or LAMMPS planar Poisson \(\Psi(z)\)
  computes inside ctest.
- Non-\(z\) axes, cylindrical / spherical profiles, or full 3D Poisson.
- Time-averaged running \(\langle\rho\rangle\), \(\langle\Psi\rangle\) over MD
  trajectories (single-shot / per-launch profiles only in this foothold).
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + ctor args; no Config bag). Minimal sequence
matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/potential_profile.h"

using namespace ATV;

cpu::Frame frame(2);
// atom 0: z=5.5 Å, q=+1 e; atom 1: z=14.5 Å, q=-1 e
// frame.box = cpu::Box::orthorhombic(10.0, 10.0, 20.0);

// After charge update so profiles see electrode q (default end_of_step @ +45).
cpu::md::ConstPotBO conp(/*eta [1/Å]=*//*…*/);
// … electrodes, electroneutrality, …

cpu::PotentialProfile psi_z(/*n_bins=*/20, /*z_min=*/0.0, /*z_max=*/20.0);

cpu::md::MD sim;
// … neighbor, potentials, integrator …
sim.add_ref(conp);
sim.add_ref(psi_z);   // or add(std::move(psi_z))
sim.build();
sim.run(frame, /*nsteps=*/1);

// Members [e/Å³], [V/Å], [V]
const auto& rho = psi_z.rho();              // bin 5 → +0.01; bin 14 → −0.01
const auto& Ez  = psi_z.field();            // mid-gap bin 9 → E_gap
const auto& Psi = psi_z.psi();              // bin 0 → 0; bin 19 → ΔΨ
const double drop = psi_z.potential_drop(); // Ψ(z_max) − Ψ(z_min)
// Frame meta: kPotentialDrop
const double drop_meta = frame.meta.get(kPotentialDrop);
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Potential profile is an Observer Component;
  never a new Driver category.
- Parameters are **constructor arguments** (`n_bins`, `z_min`, `z_max`) — no
  `PotentialProfileConfig`.
- Frame is the sole I/O currency; scalar drop lands on `kPotentialDrop`; full
  profiles live on component members (`rho()`, `field()`, `psi()`).
- MPI: per-bin charge sums are all-reduced when a domain decomposition is
  active (`owned_count` + `MPI_Allreduce`).

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
  cmake --build build --target reg_potential-profile-observer -j"$(nproc)"
  ./build/regressions/reg_potential-profile-observer
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/observe/potential_profile.h` | Public `PotentialProfile` |
| `src/cpu/observe/potential_profile.cpp` | Bin charges, planar Poisson, meta write-back |
| `include/atomiverse/core/frame_schema.h` | `kPotentialDrop` |
| `regressions/md/potential-profile-observer/reg_potential-profile-observer.cpp` | Product gate |
| `tests/cpu/observe/test_potential_profile.cpp` | Unit surface (not the e2e pin) |
| `tools/metalwalls_oracle/pin_notes_potential-profile-observer.md` | Offline pin provenance |

## Validation (vs literature)

Source:
[`regressions/md/potential-profile-observer/REFERENCE.md`](../../../regressions/md/potential-profile-observer/REFERENCE.md)
(offline pin; **never** live MetalWalls or LAMMPS in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md),
[`tools/metalwalls_oracle/pin_notes_potential-profile-observer.md`](../../../tools/metalwalls_oracle/pin_notes_potential-profile-observer.md).

**Oracle:** **literature** — planar Poisson / parallel-plate continuum (Jackson,
*Classical Electrodynamics*, ch. 1); MetalWalls JOSS **5**(53), 2373 (2020)
potential-profile analysis tools; Scalfi et al., *Phys. Chem. Chem. Phys.*
**22**, 10480 (2020) EDL profiles. Offline pin = closed-form \(E_{\mathrm{gap}}
= 4\pi k_{\mathrm{e}}\sigma\) plus the discrete piecewise-constant \(\rho\)
integration (same algorithm as `cpu::PotentialProfile`) on the synthetic
two-sheet geometry above.

**Why literature (not LAMMPS / live MetalWalls):**

- LAMMPS core has no built-in planar Poisson \(\Psi(z)\) compute matching this
  named post-process on the agent host.
- MetalWalls analysis tools implement related potential-profile workflows; no MW
  binary was installed for a live twin.
- Binding oracle priority for this loop is **LAMMPS when the feature overlaps**;
  here it does not — continuum planar Poisson is textbook electrostatics.

### Accuracy

| Quantity | Atomiverse gate / pin | Literature closed form | Tol | Notes |
| -------- | --------------------: | ---------------------: | --- | ----- |
| Continuum \(E_{\mathrm{gap}}\) [V/Å] | **1.80951275785204** | \(4\pi k_{\mathrm{e}}\sigma\) | **1e-14** | `units::coulomb` identity |
| \(E_z\) mid-gap bin 9 [V/Å] | **1.80951275785204** | same \(E_{\mathrm{gap}}\) | **1e-12** | \(z=9.5\) Å |
| \(E_z\) vacuum left bin 0 | **0** | 0 outside plates | **1e-12** | left of \(+Q\) sheet |
| \(E_z\) vacuum right bin 19 | **0** | 0 outside plates | **1e-12** | right of \(-Q\) sheet |
| \(\Psi\) bin 0 [V] | **0** | gauge \(\Psi(z_{\min})=0\) | **1e-12** | left-edge zero |
| \(\Psi\) bin 19 [V] | **−16.28561482066836** | \(-E_{\mathrm{gap}}\times 9\) | **1e-12** | potential drop |
| `potential_drop()` [V] | same as \(\Psi\) bin 19 | same | **1e-12** | member |
| meta `kPotentialDrop` [V] | same | same | **1e-12** | Frame write-back |
| \(\rho\) bin 5 [e/Å³] | **0.01** | \(Q/(A\Delta z)\) | **1e-12** | left sheet |
| \(\rho\) bin 14 [e/Å³] | **−0.01** | \(-Q/(A\Delta z)\) | **1e-12** | right sheet |
| \(\Psi\) drop vs \(-E\times 9\) | identity | algebraic | **1e-12** | center-to-center 9 Å |

Tolerance is machine-eps scale: exact double arithmetic of the discrete scheme
plus the shared `units::coulomb` constant. Catches wrong area, inverted Poisson
sign, gauge mistakes, or bin-edge / bin-center off-by-one errors.

Recorded offline constants (REFERENCE):

```text
PIN_E_GAP           1.80951275785204
PIN_PSI_DROP        -16.28561482066836
PIN_RHO_LEFT        0.01
PIN_RHO_RIGHT       -0.01
```

Related unit surface (not the e2e pin):
`tests/cpu/observe/test_potential_profile.cpp`.

### Offline pin repro (never ctest)

```bash
# Closed form (Python / same ke as ATV units::coulomb) — no MW/LAMMPS binary
python3 - <<'PY'
import math
ke = 14.399645
sigma = 0.01
E_gap = 4.0 * math.pi * ke * sigma
print(E_gap, -E_gap * 9.0)
PY
# → 1.80951275785204  -16.28561482066836
```

## Performance

Source: REFERENCE.md dual-gate summary; handoff
the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| two-sheet planar Poisson | — | — | **perf_exempt** | n/a | No MetalWalls or LAMMPS twin binary for this named potential-profile post-process |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | Analysis Observer is \(O(N_{\mathrm{atoms}})\) charge binning + \(O(n_{\mathrm{bins}})\) planar Poisson integration on a 2-atom synthetic capacitor. No MetalWalls or LAMMPS twin binary for this named potential-profile post-process on the agent host; fair wall vs an oracle binary is not applicable. |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. A fair twin would require either a
MetalWalls analysis-script harness or a custom LAMMPS post-process with the
same binning and metal units; neither shipped as a binary pin for this slug.

## Limitations

- **Observer only.** Does not solve or enforce constant potential; pair with
  `ConstPotBO` (or fixed charges) for physical electrode \(q\) on the Frame.
- **\(z\)-axis foothold.** No \(x\)/\(y\) profiles, no cylindrical / spherical
  Poisson.
- **Synthetic pin, not a production EDL trajectory.** The e2e gate uses two
  static point sheets — it validates planar Poisson algebra and units, not
  sampling quality, electrolyte layering, or long-run \(\langle\Psi(z)\rangle\).
- **Per-launch snapshot.** No built-in running average / block average of
  \(\rho\) or \(\Psi\); time averages need external accumulation or a later
  extension.
- **Left-edge gauge.** \(\Psi(z_{\min})=0\) and \(E_z(z_{\min})=0\) assume a
  vacuum / neutralized reference at the left window edge; asymmetric or charged
  systems need a gauge choice consistent with that assumption.
- **No fair speed ratio** until a documented oracle binary twin exists.
- **E10 ecosystem slice.** Capacitance from charge fluctuations
  (`capacitance-charge-fluctuation`) is a sibling; thermodynamic integration of
  \(C\) and production EDL paper workflows remain follow-ons.

## References

- Offline pin: [`regressions/md/potential-profile-observer/REFERENCE.md`](../../../regressions/md/potential-profile-observer/REFERENCE.md)
- Product gate: [`regressions/md/potential-profile-observer/reg_potential-profile-observer.cpp`](../../../regressions/md/potential-profile-observer/reg_potential-profile-observer.cpp)
- Public header: [`include/atomiverse/cpu/observe/potential_profile.h`](../../../include/atomiverse/cpu/observe/potential_profile.h)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/pin_notes_potential-profile-observer.md`](../../../tools/metalwalls_oracle/pin_notes_potential-profile-observer.md)
- Capability map: [`.claude/notes/metalwalls-capability-map.md`](../../../.claude/notes/metalwalls-capability-map.md) (E10)
- Literature: Jackson, *Classical Electrodynamics* (planar Poisson); Marin-Laflèche et al., *J. Open Source Softw.* **5**(53), 2373 (2020) (MetalWalls); Scalfi et al., *Phys. Chem. Chem. Phys.* **22**, 10480 (2020)
