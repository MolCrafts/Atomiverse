# Constant-D polarization feedback — Zhang–Sprik finite-field ensemble

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `constant-d-polarization-feedback` |
| Regression | `regressions/md/constant-d-polarization-feedback/` |
| Public API | `cpu::md::FixDfield(Dx, Dy, Dz)` with member `polarization_feedback` (default `false`); accessors `last_energy()`, `last_Pz()`, `last_Ez()` (Component, `Bias` kind, on `cpu::md::MD`, default `post_force` +100) |
| Roadmap | map "Displacement D-field" → ✓★ (P2) |
| Oracle | **literature** intentional — Zhang & Sprik 2016 / Cox & Sprik 2019 closed-form constant-D; not a 1:1 MW twin |

## Summary

**Constant-D polarization feedback** upgrades the wave-4 `FixDfield` vacuum
foothold (\(\mathbf{E}\equiv\mathbf{D}\), no screening) into the
self-consistent **constant-\(D_z\) ensemble** of Zhang & Sprik: with
`polarization_feedback = true` the Maxwell field along z responds to the
instantaneous itinerant polarization of the cell,
\(E_z = D_z - 4\pi k_e P_z\), and the electric-enthalpy term
\(U_D = V E_z^2/(8\pi k_e)\) enters the energy so the force
\(F_{i,z} = q_i E_z\) is exactly \(-\partial U_D/\partial z_i\). Free charges
now *screen* the applied displacement field — the physical content of
MetalWalls' finite-field route for bulk electrolytes — instead of riding an
unscreened external \(E\). The public surface stays one Component on
`cpu::md::MD`: no new Driver, no solver class, no `*Config` bag. The
regression pins a ±1 e pair in a cubic cell against the closed-form
constant-D expressions (static screening at \(D_z=0.5\) V/Å, the Cox–Sprik
plasma-like polarization oscillation at \(D_z=0\), and the vacuum-foothold
twin that must not move).

## Method

**Physics.** Zhang & Sprik, *Phys. Rev. B* **93**, 144201 (2016), DOI
[10.1103/PhysRevB.93.144201](https://doi.org/10.1103/PhysRevB.93.144201);
Cox & Sprik, *J. Chem. Phys.* **151**, 064506 (2019), DOI
[10.1063/1.5099207](https://doi.org/10.1063/1.5099207); constant-\(D\)
functional origin: Stengel, Spaldin, Vanderbilt, *Nat. Phys.* **5**, 304
(2009). In metal units (eV, Å, e, fs; \(k_e =\) `units::coulomb`
\(= 14.399645\) eV·Å/e²), along the cell z axis:

\[
P_z = \frac{1}{V}\sum_i q_i\, z_i^{\mathrm u} \quad [\mathrm{e/\AA^2}]
\qquad\text{(itinerant polarization, unwrapped coordinates)}
\]

\[
E_z = D_z - 4\pi k_e P_z \quad [\mathrm{V/\AA}],
\qquad
U_D = \frac{V\,E_z^2}{8\pi k_e} \quad [\mathrm{eV}],
\qquad
F_{i,z} = -\frac{\partial U_D}{\partial z_i} = q_i E_z .
\]

This is the electric-enthalpy functional
\(F = E_{\mathrm{PBC}} + V(D-4\pi P)^2/8\pi\) that MetalWalls implements for
its finite-field route. Because the force is the exact gradient of \(U_D\),
constant-D MD conserves \(KE + U_D\) when the field does the only work.

Two consequences the regression turns into gates:

- **Static screening.** At fixed geometry, a polarized cell reduces the
  Maxwell field below the applied \(D_z\) (the foothold twin keeps
  \(E_z = D_z\) exactly).
- **Plasma-like oscillation.** At \(D_z = 0\) the itinerant polarization of
  a ±q pair sits in a harmonic well \(U = 2\pi k_e q^2 d^2/V\)
  (Cox & Sprik), so the pair separation oscillates as
  \(d(t) = d_0\cos\omega t\) with
  \(\omega = \sqrt{4\pi k_e q^2/(\mu V)\cdot\texttt{force\_to\_accel}}\).

x and y keep the foothold coupling — MetalWalls applies \(D\) along the cell
normal only. Berry-phase / electronic polarization is out of scope.

**Why MetalWalls (literature pin, not LAMMPS):** mainline LAMMPS has no
constant-D ensemble, and no MetalWalls binary is installed on the agent
host. The pins are therefore the closed-form constant-D expressions from
the finite-field papers — the same \(E = D - 4\pi P\) functional MetalWalls
implements — evaluated with the project's Coulomb constant
(`tools/metalwalls_oracle/inputs/literature/constant_d_pins.py`).

### System (±1 e pair, cubic cell)

| Quantity | Value |
| -------- | ----- |
| Charges | +1 e at \(z_0 + d/2\), −1 e at \(z_0 - d/2\) (on the z axis) |
| Separation \(d\) | 2 Å |
| Box | cubic \(L = 10\) Å (\(V = 1000\) Å³) |
| Masses | 1 amu each (reduced mass \(\mu = 0.5\) amu) |
| Gate A field | \(D_z = 0.5\) V/Å, static (no MD) |
| Gate B field | \(D_z = 0\), VV dt = 0.1 fs, 532 steps (half period) |
| Gate C | feedback **off**, \(D = 0\) — vacuum-foothold twin |

### Offline closed-form pins

Script: `tools/metalwalls_oracle/inputs/literature/constant_d_pins.py`.

Gate A (static, \(D_z = 0.5\) V/Å):

| Quantity | Value | Units |
| -------- | ----: | ----- |
| \(P_z\) | 2.0e−3 | e/Å² |
| \(E_z\) | 1.3809744842959204e−1 | V/Å |
| \(U_D\) | 5.2696244279044818e−2 | eV |
| \(F_z(+q)\) | 1.3809744842959204e−1 | eV/Å |
| foothold twin \(E_z\) | 0.5 (= \(D_z\), unscreened) | V/Å |

Gate B (dynamics, \(D_z = 0\)): \(\omega =\) **5.9091697154893236e−2** fs⁻¹
(period 106.329 fs), \(d(t) = d_0\cos\omega t\) with \(d_0 = 2\) Å, total
energy \(U_0 =\) **3.6190255157040796e−1** eV.

### Key exclusions (non-goals of this slug)

- Live MetalWalls or LAMMPS inside ctest (closed-form literature pin only).
- Constant-D along x/y (MetalWalls applies \(D\) along the cell normal; x,y
  stay foothold).
- Berry-phase / electronic polarization (point charges only).
- Coupling to induced dipoles or electrode charge solvers in this pin
  (compose `FixDfield` with `ConstPotBO` separately; see
  `regressions/md/displacement-dfield-md`).
- A new Driver / solver class — only a member flag on `FixDfield`.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Fair wall-clock twin vs a MetalWalls binary (none on agent host).

## Implementation

Public surface only (`cpu::Frame` + `FixDfield` ctor / members; no Config
bag). Static single evaluation matching Gate A:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_dfield.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame f(2);                       // ±1 e pair on z, cubic L = 10 Å
f.span(kAtom_z)[0] = 6.0;  f.span(kAtom_charge)[0] = +1.0;
f.span(kAtom_z)[1] = 4.0;  f.span(kAtom_charge)[1] = -1.0;
// ... x/y/mass/type/tag/image as in the regression ...
f.box = cpu::Box::cubic(10.0);

cpu::md::FixDfield d(/*Dx=*/0.0, /*Dy=*/0.0, /*Dz [V/Å]=*/0.5);
d.polarization_feedback = true;        // constant-D ensemble (default: foothold)

// Static single evaluation (Gate A) — Bias on Evaluate; Frame is the only token.
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(d);
eval.build();
eval.run(f);

double Pz = d.last_Pz();       // [e/Å²]  itinerant polarization
double Ez = d.last_Ez();       // [V/Å]   D_z − 4π k_e P_z
double U  = d.last_energy();   // [eV]    V E_z² / (8π k_e)
// forces q_i E_z added to frame.span(kAtom_fz)
```

Constant-D MD (Gate B shape) — the Component binds itself at
`md::Phase::post_force` order +100:

```cpp
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
cpu::md::FixDfield dfield(0.0, 0.0, 0.0);
dfield.polarization_feedback = true;
sim.add_ref(dfield);           // or add(std::move(dfield)) to own
sim.build();
sim.run(frame, 532);           // half a polarization-oscillation period
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** The constant-D ensemble is a member flag
  on the existing `FixDfield` Bias Component; never a new Driver category.
- Parameters are **constructor arguments and public members** — no
  `FixDfieldConfig` / `ConstantDConfig`.
- Frame is the sole I/O currency; forces land on `kAtom_fz`, the enthalpy
  term accumulates into `frame.meta.get(kEnergy)`.
- \(P_z\) uses **unwrapped** coordinates (`kAtom_image`) — itinerant
  polarization must track charge transport across the boundary.

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
  cmake --build build --target reg_constant-d-polarization-feedback -j"$(nproc)"
  ./build/regressions/reg_constant-d-polarization-feedback
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/fix_dfield.h` | Public `FixDfield` (`Dx/Dy/Dz`, `polarization_feedback`, `last_Pz/last_Ez/last_energy`) |
| `src/cpu/dynamics/fix_dfield.cpp` | \(P_z\) from unwrapped coords; \(E_z = D_z - 4\pi k_e P_z\); \(U_D\) + exact-gradient forces |
| `regressions/md/constant-d-polarization-feedback/reg_constant-d-polarization-feedback.cpp` | Product gate (Gates A/B/C) |
| `tools/metalwalls_oracle/inputs/literature/constant_d_pins.py` | Offline closed-form pin derivation |

## Validation (vs MetalWalls / Zhang–Sprik constant-D literature pins)

Source:
[`regressions/md/constant-d-polarization-feedback/REFERENCE.md`](../../../regressions/md/constant-d-polarization-feedback/REFERENCE.md)
(offline closed-form pins; **never** live MetalWalls in ctest). Oracle
tooling: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

**Oracle:** the constant-D electric-enthalpy functional
\(E = D - 4\pi P\) from Zhang & Sprik (2016) / Cox & Sprik (2019) — the same
functional MetalWalls implements for its finite-field route — evaluated in
closed form with the project's Coulomb constant. No MetalWalls binary is
installed on this machine, and mainline LAMMPS has no constant-D ensemble.

### Accuracy

| Quantity | ATV | Oracle pin (closed-form constant-D) | Tol | Notes |
| -------- | --- | ----------------------------------: | --- | ----- |
| A: \(P_z\) [e/Å²] | `last_Pz()` | **2.0e−3** | abs ≤ **1e-13** | \(q\,d/V\), static |
| A: \(E_z\) [V/Å] | `last_Ez()` | **1.3809744842959204e−1** | abs ≤ **1e-13** | \(D_z - 4\pi k_e P_z\), screened below \(D_z=0.5\) |
| A: \(U_D\) [eV] | `last_energy()` | **5.2696244279044818e−2** | abs ≤ **1e-13** | \(V E_z^2/(8\pi k_e)\) |
| A: \(F_z(+q)\) [eV/Å] | `span(kAtom_fz)[0]` | **1.3809744842959204e−1** | abs ≤ **1e-13** | \(q_i E_z\) = exact gradient |
| A: twin \(E_z\) [V/Å] | feedback off | **0.5** (= \(D_z\), unscreened) | exact | vacuum foothold |
| B: \(d(t{=}26.6\,\mathrm{fs})\) [Å] | VV, dt = 0.1 fs | \(d_0\cos(\omega\,26.6)\), \(\omega=\) **5.9091697154893236e−2** fs⁻¹ | abs ≤ **5e-5** | VV error ≤ 4.6e−6 Å |
| B: \(d(t{=}53.2\,\mathrm{fs})\) [Å] | VV, dt = 0.1 fs | \(d_0\cos(\omega\,53.2)\approx-d_0\) | abs ≤ **5e-5** | half period (106.329 fs / 2) |
| B: \(KE + U_D\) [eV] | at step 532 | **3.6190255157040796e−1** | abs ≤ **1e-6** | energy conservation, \(U_0\) |
| B: \(E_z + 4\pi k_e P_z\) [V/Å] | every probe | **0** (self-consistency at \(D_z=0\)) | abs ≤ **1e-13** | field tracks motion |
| C: \(z_+, z_-\) [Å] | after 532 steps | **6.0, 4.0** (unmoved) | exact | feedback off, \(D=0\) → zero force |
| C: \(E_z\) [V/Å] | after 532 steps | **0** (= \(D_z\)) | exact | foothold twin |

Geometry, masses, dt, and tolerances: **Method** table and REFERENCE
(match `reg_constant-d-polarization-feedback`; runtime ~1064 MD steps of a
2-atom system, well under 60 s).

### Offline pin repro (never ctest)

```bash
python3 tools/metalwalls_oracle/inputs/literature/constant_d_pins.py
```

## Performance

Source: REFERENCE.md; handoff
the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| constant-D pair MD | — | — | **perf_exempt** | n/a | No MetalWalls binary on agent host and no LAMMPS constant-D ensemble for a fair twin; the O(N) field evaluation on 2 atoms is sub-second |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | no fair twin binary: no MetalWalls on agent host; mainline LAMMPS has no constant-D ensemble |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls binary is available,
re-run an offline fair bench, record version + walls, and replace
`perf_exempt` in REFERENCE.

## Limitations

- **z axis only** for the self-consistent ensemble; x,y keep the vacuum
  foothold coupling (MetalWalls applies \(D\) along the cell normal).
- **Point charges / itinerant polarization only** — no Berry-phase or
  electronic polarization, no induced-dipole contribution to \(P_z\) in
  this pin (PIM dipoles are sibling slugs).
- **Uniform-field coupling** — the constant-D term is on top of whatever
  Coulomb pipeline the simulation composes; this pin runs it stand-alone
  (no PME) so the closed-form well is exact.
- **No constant-E ↔ constant-D Legendre-transform toggle** as a separate
  API — constant-E at \(\varepsilon=1\) is the foothold path
  (`polarization_feedback = false`).
- **No fair speed ratio** until MetalWalls is on the pin host.
- **Scope of this slug only:** the vacuum D-field foothold
  (`displacement-dfield-md`), oscillating E-fields
  (`oscillating-efield-md`), and finite-field + electrode compositions
  (`finite-field-single-electrode`) are sibling pages.

## References

- Offline pin: [`regressions/md/constant-d-polarization-feedback/REFERENCE.md`](../../../regressions/md/constant-d-polarization-feedback/REFERENCE.md)
- Product gate: [`regressions/md/constant-d-polarization-feedback/reg_constant-d-polarization-feedback.cpp`](../../../regressions/md/constant-d-polarization-feedback/reg_constant-d-polarization-feedback.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/fix_dfield.h`](../../../include/atomiverse/cpu/dynamics/fix_dfield.h)
- Pin script: [`tools/metalwalls_oracle/inputs/literature/constant_d_pins.py`](../../../tools/metalwalls_oracle/inputs/literature/constant_d_pins.py)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (Displacement D-field, P2)
- Literature: C. Zhang, M. Sprik, *Phys. Rev. B* **93**, 144201 (2016), DOI [10.1103/PhysRevB.93.144201](https://doi.org/10.1103/PhysRevB.93.144201) (constant-D electric enthalpy \(F = E_{\mathrm{PBC}} + V(D-4\pi P)^2/8\pi\)); S. J. Cox, M. Sprik, *J. Chem. Phys.* **151**, 064506 (2019), DOI [10.1063/1.5099207](https://doi.org/10.1063/1.5099207) (finite-field bulk electrolyte formalism); M. Stengel, N. A. Spaldin, D. Vanderbilt, *Nat. Phys.* **5**, 304 (2009), DOI [10.1038/nphys1185](https://doi.org/10.1038/nphys1185) (constant-D functional origin)
- Related foothold: `regressions/md/displacement-dfield-md`
