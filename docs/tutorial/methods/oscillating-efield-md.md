# Oscillating (AC) external E-field MD — LAMMPS variable `fix efield`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `oscillating-efield-md` |
| Regression | `regressions/md/oscillating-efield-md/` |
| Public API | `cpu::md::FixOscillatingEfield` (`Bias`) on `cpu::md::MD` (default `post_force` +100) |
| Roadmap | map slice E3 oscillating residual |
| Oracle | **LAMMPS** (`oracle: lammps`; `oracle_version: 30 Mar 2026 - Development`, patch `30Mar2026-1074-g7f680de296`) |

## Summary

**Time-dependent uniform electric field** on classical point charges: an AC
cosine envelope \(\mathbf{E}(t)=\mathbf{E}_0\cos(2\pi t/T+\varphi)\) with
LAMMPS-compatible forces \(\mathbf{F}_i=q_i\mathbf{E}(t)\) and constant-path
potential energy at frozen \(E(t)\). Product surface is a single **Component**
`cpu::md::FixOscillatingEfield` (kind `Bias`), never a new Driver category and
never a charge solve inside `KSpace*`. Bound by default at
`md::Phase::post_force` (order +100), matching LAMMPS `POST_FORCE`.

This is the **E3 oscillating / AC** foothold after the constant-field product
(`external-efield-md`, `FixEfield`) and the MW-only single-electrode finite-field
path (`finite-field-single-electrode`). Offline pin and fair wall use **LAMMPS**
variable `fix efield` (equal-style `v_Ez`); Atomiverse public time is **fs**
(metal-unit twin: \(T=1\) ps, \(\Delta t=0.001\) ps \(\Leftrightarrow\)
\(T=1000\) fs, \(\Delta t=1\) fs). Fair ratio
\(r=T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\approx\mathbf{2.12}\) (≤ 10×).

## Method

**Physics.** Uniform field on charges (metal units, \(qe2f=1\)):

\[
\mathbf{E}(t) = \mathbf{E}_0\cos\!\bigl(2\pi\, t / T + \varphi\bigr),
\qquad t = n_{\mathrm{step}}\,\Delta t,
\]

\[
\mathbf{F}_i \mathrel{+}= q_i\,\mathbf{E}(t),\qquad
U \mathrel{+}= -q_i\,\mathbf{E}(t)\cdot\mathbf{r}_i^{\mathrm{u}}
\quad\text{[eV, eV/Å]}.
\]

Unwrapped \(\mathbf{r}^{\mathrm{u}}\) uses orthorhombic image flags
(\(\mathbf{r}+\mathrm{image}\cdot\mathbf{L}\)); triclinic throws. Instantaneous
energy freezes \(\mathbf{E}\) over the timestep (LAMMPS variable-path convention;
oracle pins energy with atom-style `potential` \(\phi=-\mathbf{E}\cdot\mathbf{r}\)).

**Why LAMMPS, not MetalWalls.** The algorithm is LAMMPS `fix efield` with
equal-style variables (`lammps/src/fix_efield.cpp`). Under the metalwalls-loop
oracle priority rule, overlap features pin and fair-bench against **LAMMPS**.
MetalWalls is not the oracle for this slug.

### Three-charge multi-step pin (this regression)

| Item | Setting |
| ---- | ------- |
| Box | \(L=20\) Å cubic, 3D PBC |
| \(\mathbf{E}_0\) | \((0,\,0,\,0.2)\) V/Å |
| Period \(T\) | \(1000\) fs (LAMMPS metal: \(1.0\) ps) |
| Timestep \(\Delta t\) | \(1\) fs (LAMMPS metal: \(0.001\) ps) |
| Phase \(\varphi\) | \(0\) |
| Pair | none in ATV pin launch; LAMMPS `pair_style zero` |
| Atom 1 | \(q=+1\), \(\mathbf{r}=(5,10,10)\) Å |
| Atom 2 | \(q=-1\), \(\mathbf{r}=(7,10,10)\) Å |
| Atom 3 | \(q=+0.5\), \(\mathbf{r}=(10,12,8)\) Å |
| Image flags | all zero |

For this geometry \(\sum_i q_i z_i = 4\) Å·e, so

\[
U(t) = -4\,E_z(t)\ \mathrm{eV},\qquad
F_{z,i}(t) = q_i E_z(t).
\]

Offline pin table (LAMMPS `run 0` at selected steps; full table in
`REFERENCE.md`):

| step \(n\) | \(t\) (metal ps / ATV fs) | \(E_z\) [V/Å] | \(U\) [eV] | \(F_{z,1}\) | \(F_{z,2}\) | \(F_{z,3}\) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | **0.2** | **−0.8** | 0.2 | −0.2 | 0.1 |
| 125 | 0.125 / 125 | **0.14142135623731000** | **−0.5656854249492400** | \(q_i E_z\) | … | … |
| 250 | 0.25 | ~0 (≤ 2e−17) | ~0 | ~0 | ~0 | ~0 |
| 375 | 0.375 | **−0.14142135623731000** | **+0.5656854249492400** | … | … | … |
| 500 | 0.5 | **−0.2** | **+0.8** | −0.2 | 0.2 | −0.1 |
| 750 | 0.75 | ~0 | ~0 | ~0 | ~0 | ~0 |
| 1000 | 1.0 | **0.2** | **−0.8** | 0.2 | −0.2 | 0.1 |

Regression dual-gates steps **0, 125, 500** (max / mid-quarter / min):
per-atom \(\mathbf{F}\) and `frame.meta.get(kEnergy)` vs pin.

**Key exclusions (non-goals of this slug)**

- Atom-style spatial fields, LAMMPS `region` keyword, dipole torque.
- Arbitrary equal-style expression engine beyond the AC cosine envelope.
- Constant DC field only (`FixEfield` / `external-efield-md`).
- Finite-field + single constant-potential electrode (MW / Dufils;
  `finite-field-single-electrode`).
- Charge solve inside PME / `KSpace*` — forbidden by architecture.
- Live LAMMPS or MetalWalls inside ctest.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Triclinic boxes (orthorhombic image unwrap only).

## Implementation

Public surface only (`cpu::Frame` + Component with ctor args / public members;
no Config bag). Multi-step pin matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_oscillating_efield.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame frame(3);
// Atom 1: q=+1 at (5,10,10); atom 2: q=-1 at (7,10,10);
// atom 3: q=+0.5 at (10,12,8)  — [e], [Å]
frame.span(kAtom_charge)[0] = 1.0;
frame.span(kAtom_x)[0] = 5.0;
frame.span(kAtom_y)[0] = 10.0;
frame.span(kAtom_z)[0] = 10.0;
// … fill atoms 1–2; tags/types/mass as needed
frame.box = cpu::Box::cubic(20.0);

// Amplitudes [V/Å], period and dt [fs], phase default 0.
// Component default binding is post_force +100; time is t = n·dt from the step.
cpu::md::FixOscillatingEfield osc(
    /*Ex0=*/0.0, /*Ey0=*/0.0, /*Ez0=*/0.2,
    /*period [fs]=*/1000.0, /*dt [fs]=*/1.0);

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/2.0, /*skin=*/0.3});
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0, /*sigma=*/1.0, /*cutoff=*/2.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.add_ref(osc);  // or add(cpu::md::FixOscillatingEfield{…})
sim.build();
sim.run(frame, /*nsteps=*/1);   // pin checks also at n ∈ {125, 500, …}
// osc.last_Ez() → 0.2 [V/Å] at step 0
// frame.meta.get(kEnergy) → −0.8 [eV]
// Fz[i] → q[i] * Ez
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** The AC field is a Component (`Bias`).
- Parameters are **constructor arguments and public members**
  (`Ex0`/`Ey0`/`Ez0`, `period`, `dt`, `phase`) — no Config bags.
- Frame is the sole I/O currency; forces accumulate on `kAtom_f*`, energy on
  `kEnergy`.
- Time is formed as \(t=n\cdot\Delta t\) from the launch/step index; callers
  must pass a consistent `dt` with the integrator timestep.

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
  cmake --build build --target reg_oscillating-efield-md -j"$(nproc)"
  ./build/regressions/reg_oscillating-efield-md
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/fix_oscillating_efield.h` | Public `FixOscillatingEfield` |
| `src/cpu/dynamics/fix_oscillating_efield.cpp` | Force / energy launch |
| `include/atomiverse/cpu/dynamics/fix_efield.h` | Sibling constant-field `FixEfield` |
| `include/atomiverse/cpu/dynamics/md.h` | Host preset `cpu::md::MD` |
| `regressions/md/oscillating-efield-md/reg_oscillating-efield-md.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/oscillating-efield-md/in.oscillating_efield` | Offline LAMMPS twin |

## Validation (vs LAMMPS)

Source:
[`regressions/md/oscillating-efield-md/REFERENCE.md`](../../../regressions/md/oscillating-efield-md/REFERENCE.md)
(offline pin; **never** live LAMMPS or MetalWalls in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md),
input
[`tools/metalwalls_oracle/inputs/lammps/oscillating-efield-md/in.oscillating_efield`](../../../tools/metalwalls_oracle/inputs/lammps/oscillating-efield-md/in.oscillating_efield).

**Oracle:** LAMMPS variable `fix efield` equal-style cosine
(`docs.lammps.org/fix_efield.html`; `src/fix_efield.cpp`).
`oracle: lammps`,
`oracle_version: 30 Mar 2026 - Development (patch_30Mar2026-1074-g7f680de296)`;
binary recorded in REFERENCE as `/home/jicli594/.local/x86_64/bin/lmp`.

**Why not MetalWalls:** feature is classical uniform AC field on charges — same
algorithm as LAMMPS; MW is reserved for MW-only electrode physics.

### Accuracy

| Quantity | Atomiverse gate | LAMMPS pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| \(E_z(0)\) [V/Å] | `last_Ez()` | **0.2** | abs ≤ **1e-12** | \(\cos 0 = 1\) |
| \(U(0)\) [eV] | `kEnergy` | **−0.8** | abs ≤ **1e-12** | \(U=-4 E_z\) |
| \(F_{z,1..3}(0)\) [eV/Å] | per-atom \(q_i E_z\) | **0.2, −0.2, 0.1** | abs ≤ **1e-12** | dual force gate |
| \(E_z(125)\) [V/Å] | `last_Ez()` | **0.14142135623731000** | abs ≤ **1e-12** | \(0.2\cos(\pi/4)\) |
| \(U(125)\) [eV] | `kEnergy` | **−0.5656854249492400** | abs ≤ **1e-12** | mid-quarter |
| \(F_{z,i}(125)\) | \(q_i E_z\) | \(q_i\times\) pin \(E_z\) | abs ≤ **1e-12** | force dual-gate |
| \(E_z(500)\) [V/Å] | `last_Ez()` | **−0.2** | abs ≤ **1e-12** | half-period min |
| \(U(500)\) [eV] | `kEnergy` | **+0.8** | abs ≤ **1e-12** | sign flip |
| \(F_{z,1..3}(500)\) | \(q_i E_z\) | **−0.2, 0.2, −0.1** | abs ≤ **1e-12** | dual force gate |

Geometry, period/dt conversion, and tolerances: **Method** table and REFERENCE
(match `reg_oscillating-efield-md`). Near-zero crossings in the full offline
table use looser notes (\(10^{-10}\)) in REFERENCE; the ctest regression asserts
the three high-signal steps above at \(10^{-12}\).

Related unit surface (not the e2e pin):
`tests/cpu/dynamics/test_fix_oscillating_efield.cpp`. Constant DC sibling:
`tests/cpu/dynamics/test_fix_efield.cpp`,
`regressions/md/external-efield-md/`.

### Offline pin repro (never ctest)

```bash
# With a matching LAMMPS binary on PATH:
lmp -in tools/metalwalls_oracle/inputs/lammps/oscillating-efield-md/in.oscillating_efield
# PIN lines: step, time, Ez, Ufield from fix scalar + potential v_phi
```

Metal ↔ ATV time map: \(1\) metal time unit = \(1\) ps; ATV uses fs, so
`period=1.0` ps / `dt=0.001` ps maps to `period=1000` fs / `dt=1` fs with the
same integer step index \(n\).

## Performance

Source: [`regressions/md/oscillating-efield-md/REFERENCE.md`](../../../regressions/md/oscillating-efield-md/REFERENCE.md)
fair-wall section; handoff
the case `REFERENCE.md`
(`ratio` **2.12**).

Comparable pure oscillating-efield workload (pair zero / inert, NVE, AC field
on charges; same login node, OMP=1, 2026-07-24):

| Twin | Workload | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| oscillating efield | 1000 atoms, 5000 steps; LAMMPS `pair zero` + variable `fix efield 0 0 v_Ez` + `fix nve`, dt=0.001 ps; ATV inert `PairLJCut` (rc=2 Å ≪ spacing) + `FixOscillatingEfield{0,0,0.05,1000,1}` + VV, dt=1 fs | **0.0585** (Loop time) | **0.124** | **2.12** | ✓ | Single-thread; field amp 0.05 V/Å on fair twin (not the 0.2 pin) |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **false** |
| `ratio_atv_over_oracle` | **2.12** (\(=0.124/0.0585\)) |
| `within_10x` | **true** |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\). Numbers are from
REFERENCE only — do not invent alternate walls. Re-run offline fair bench when
re-pinning against a new LAMMPS `oracle_version`.

Sibling constant-field MD fair wall (when documented under
`external-efield-md`) is independent of this AC ratio.

## Limitations

- **Uniform AC cosine only.** No atom-style \(E(\mathbf{r})\), no region mask,
  no dipole / molecular torque, no arbitrary equal-style expression beyond
  \(\mathbf{E}_0\cos(2\pi t/T+\varphi)\).
- **Orthorhombic unwrap.** Triclinic boxes throw; image-based unwrapping matches
  the constant `FixEfield` foothold.
- **Caller supplies `dt`.** The Component does not read the integrator’s
  timestep automatically — `period`/`dt` must stay consistent with the MD
  schedule for \(t=n\Delta t\) to match the pin.
- **No pair / electrolyte in the accuracy pin.** Multi-step force+energy gate is
  pure field on frozen charges; production MD composes with real potentials
  separately.
- **Not finite-field electrodes.** Coupling oscillating \(E\) into
  `ConstPotBO` RHS (Dufils AC + constant potential) is out of scope for this
  slug.
- Scope of E3 wave-3 AC residual only; constant field and single-electrode
  finite-field have their own pages.

## References

- Offline pin: [`regressions/md/oscillating-efield-md/REFERENCE.md`](../../../regressions/md/oscillating-efield-md/REFERENCE.md)
- Product gate: [`regressions/md/oscillating-efield-md/reg_oscillating-efield-md.cpp`](../../../regressions/md/oscillating-efield-md/reg_oscillating-efield-md.cpp)
- Public header / impl: [`include/atomiverse/cpu/dynamics/fix_oscillating_efield.h`](../../../include/atomiverse/cpu/dynamics/fix_oscillating_efield.h), [`src/cpu/dynamics/fix_oscillating_efield.cpp`](../../../src/cpu/dynamics/fix_oscillating_efield.cpp)
- Capability map: [`.claude/notes/metalwalls-capability-map.md`](../../../.claude/notes/metalwalls-capability-map.md) (E3)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/inputs/lammps/oscillating-efield-md/in.oscillating_efield`](../../../tools/metalwalls_oracle/inputs/lammps/oscillating-efield-md/in.oscillating_efield)
- Related: constant field `external-efield-md` / `FixEfield`; finite-field single electrode [`finite-field-single-electrode`](finite-field-single-electrode.md)
- LAMMPS docs: <https://docs.lammps.org/fix_efield.html> (equal-style variable path)
