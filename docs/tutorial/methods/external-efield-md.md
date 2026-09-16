# Uniform external electric field — classical MD charges (LAMMPS `fix efield`)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `external-efield-md` |
| Regression | `regressions/md/external-efield-md/` |
| Public API | `cpu::md::FixEfield{Ex, Ey, Ez}` (`Bias` kind); `cpu::Evaluate` or default `md::Phase::post_force` on `cpu::md::MD` |
| Queue / map | **E3**; map slice E3 in `.claude/notes/metalwalls-capability-map.md` |
| Dual gate | Force + energy vs offline LAMMPS pin; fair wall \(r \le 10\) |
| Oracle | **LAMMPS** (not MetalWalls) — feature also exists as `fix efield` |

## Summary

**Uniform external electric field** applied to classical point charges in metal
units: \(\mathbf{F}_i = q_i\mathbf{E}\) and
\(U = -\sum_i q_i\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}\). The public
component is `cpu::md::FixEfield` (constructor args for \(\mathbf{E}\)
components in V/Å; no `*Config` bag). It is a `Bias` component bound by
default at `md::Phase::post_force` (after pair/bonded/kspace forces), and can
be registered on `cpu::Evaluate` for a single-point force/energy check.

This is the **E3 constant-field foothold** for electrochemical MD (finite-field
electrode work builds on it). The regression is **not** a self-golden of
Atomiverse output: offline pins come from LAMMPS `fix efield` on
`pair_style zero`. Product gate: `regressions/md/external-efield-md`
(`reg_external-efield-md`). Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} = 0.60\) (≤10×).

## Method

**Physical model.** Charges \(q_i\) [e] feel a spatially uniform, time-constant
field \(\mathbf{E}\) [V/Å]. In LAMMPS **metal** units the force conversion
factor is \(q_{\mathrm{e2f}} = 1\), so forces land directly in eV/Å:

\[
\mathbf{F}_i = q_i\,\mathbf{E},
\qquad
U = -\sum_i q_i\,\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}.
\]

Unwrapped positions follow the orthorhombic image convention used by LAMMPS
`domain->unmap` (and by Frame image flags):

\[
\mathbf{r}_i^{\mathrm{u}}
  = \mathbf{r}_i + \mathrm{image}_i \odot \mathbf{L},
\]

with \(\mathbf{L}\) the orthorhombic box lengths. The energy identity is the
same as LAMMPS `fix_efield.cpp` constant path
(`fsum[0] -= F·unwrap`).

**Regression system (offline pin geometry).**

| Quantity | Value |
| -------- | ----- |
| Box | cubic \(L = 20\) Å, PBC |
| \(\mathbf{E}\) | \((0.1,\,-0.05,\,0.2)\) V/Å |
| Atom 1 | \(q=+1\), \(\mathbf{r}=(5,10,10)\) Å |
| Atom 2 | \(q=-1\), \(\mathbf{r}=(7,10,10)\) Å |
| Atom 3 | \(q=+0.5\), \(\mathbf{r}=(10,12,8)\) Å |
| Pair style | none (`pair_style zero` twin); image flags all zero |
| Pin workflow | LAMMPS `run 0`; dump forces + fix scalar `f_1` |

Expected forces: \(\mathbf{F}_i = q_i\mathbf{E}\) exactly; field energy
\(U = -0.8\) eV.

**Architecture placement.** Electrodes / const-pot / external field are
**Components** on `cpu::md::MD` (or Evaluate for static checks). There is no
new Driver category. Charge solving never lives inside `KSpace*`; efield is
pure bias on existing charges.

**Excluded (this foothold).** Variable (`v_`) fields; region keyword;
dipole torque; minimization-only variable energy keyword; triclinic image
unwrap (throws if the box is non-orthorhombic); time-varying or gradient
fields; finite-field single-electrode product (queue slug
`finite-field-single-electrode`, oracle MetalWalls).

## Implementation

Public surface only: `cpu::Frame` charges + positions + (optional) image flags;
`cpu::md::FixEfield(Ex, Ey, Ez)` with public members `Ex`/`Ey`/`Ez` [V/Å].
Headers: `include/atomiverse/cpu/dynamics/fix_efield.h`,
`include/atomiverse/cpu/evaluate/evaluate.h`. Unit gate (not e2e):
`tests/cpu/dynamics/test_fix_efield.cpp`.

**Single-point Evaluate (matches the regression):**

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame frame(3);
// set kAtom_x/y/z, kAtom_charge, kAtom_mass, kAtom_type, kAtom_tag, zero forces
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::md::FixEfield{/*Ex=*/0.1, /*Ey=*/-0.05, /*Ez=*/0.2});
eval.build();
eval.run(frame);

const double U = frame.meta.get(kEnergy);  // [eV]
// forces on frame.span(kAtom_fx/fy/fz) [eV/Å]
```

**On classical MD** the component self-binds at `post_force` (order +100). For
an explicit MD force pipeline, construct `FixEfield` and either rely on
default bindings when registered through the plan, or add it at
`md::Phase::post_force` so pair/kspace forces accumulate first.

```bash
# MD-only configure (binding for this loop)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_external-efield-md -j"$(nproc)"
./build/regressions/reg_external-efield-md
```

Relevant sources: `src/cpu/dynamics/fix_efield.cpp` (OMP force + unwrapped
energy; accumulates into `kEnergy`). Offline twin input:
`tools/metalwalls_oracle/inputs/lammps/in.external_efield`.

## Validation (oracle: LAMMPS)

Source: `regressions/md/external-efield-md/REFERENCE.md` (offline pin; **never**
live LAMMPS in ctest).

| Engine | Role |
| ------ | ---- |
| **LAMMPS** | Primary oracle (`fix efield` constant path; metal units) |
| MetalWalls | Not used for this slug (MW is oracle only for MW-only physics) |

**Oracle version:** LAMMPS 30 Mar 2026 - Development
(`patch_30Mar2026-1074-g7f680de296`); binary recorded in REFERENCE as
`/home/jicli594/.local/x86_64/bin/lmp`. Input:
`tools/metalwalls_oracle/inputs/lammps/in.external_efield` (`lmp -in …`,
`run 0`).

### Accuracy (hard-coded pins)

| Atom | \(q\) [e] | \(F_x\) | \(F_y\) | \(F_z\) [eV/Å] | Tol |
| ---: | --------: | ------: | ------: | -------------: | --- |
| 1 | +1 | 0.1 | −0.05 | 0.2 | \(10^{-12}\) |
| 2 | −1 | −0.1 | 0.05 | −0.2 | \(10^{-12}\) |
| 3 | +0.5 | 0.05 | −0.025 | 0.1 | \(10^{-12}\) |

| Quantity | Atomiverse / pin | LAMMPS | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| Field energy \(U\) [eV] | −0.8 | −0.8 (`f_1`) | \(10^{-12}\) | \(U=-\sum q_i\mathbf{E}\cdot\mathbf{r}_i^{\mathrm{u}}\) |
| Pair PE [eV] | 0 | 0 | — | twin uses `pair_style zero` |

Dual gate: **force and energy** must both match (regression fails either side).
With zero image flags, wrapped ≡ unwrapped for the energy pin; unit tests cover
nonzero image unwrap separately.

## Performance

Source: same REFERENCE fair-wall table (host login node, OMP=1, 2026-07-23).
Comparable pure-efield workload: pair contribution inert / zero, NVE-like
integration, efield only on charges.

| Twin | Workload | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\) | ≤10× |
| ---- | -------- | --------------------------: | -----------------------: | -----------------------------------------: | ---- |
| Pure efield MD | \(N=1000\), 5000 steps, dt = 1 fs; LAMMPS `pair zero` + `fix efield 0.05 0 0` + `fix nve`; ATV inert `PairLJCut` (rc = 2 Å ≪ spacing) + `FixEfield{0.05,0,0}` + VelocityVerlet | 0.21 | 0.126 | **0.60** | ✓ |

Gate: \(r \le 10\) unless exempted. This slug is **within_10x: true** at
\(r \approx 0.60\) (ATV faster on this pure-bias wall). Caveat: fair wall is
not a full neighbor/kspace-heavy EDL bench; it isolates efield bookkeeping + MD
step overhead against the LAMMPS twin.

## Limitations

- **Constant field only.** No `v_` variable field, no spatial region mask, no
  dipole/torque path from LAMMPS `fix efield`.
- **Orthorhombic unwrap only.** Triclinic boxes throw at launch in this
  foothold.
- **Charges are inputs.** FixEfield does not induce or solve electrode charges;
  const-pot / finite-field electrode coupling is other queue slugs (E1 / E3
  `finite-field-single-electrode`).
- **Single-rank classical path** documented here; MPI DD ownership uses
  `owned_count` but multi-rank efield energy reduction is not the regression
  pin.
- **Not a MetalWalls twin.** Oracle priority: if the feature exists in LAMMPS,
  offline pin + fair bench use LAMMPS.

## References

- Offline pin + fair wall: `regressions/md/external-efield-md/REFERENCE.md`
- Regression consumer: `regressions/md/external-efield-md/reg_external-efield-md.cpp`
- Public API: `include/atomiverse/cpu/dynamics/fix_efield.h`
- Unit tests: `tests/cpu/dynamics/test_fix_efield.cpp`
- LAMMPS input twin: `tools/metalwalls_oracle/inputs/lammps/in.external_efield`
- LAMMPS docs: <https://docs.lammps.org/fix_efield.html>
- LAMMPS source (constant charge path): `src/fix_efield.cpp`
- Capability map: `.claude/notes/metalwalls-capability-map.md` (E3)
- Oracle tooling notes: `tools/metalwalls_oracle/README.md`
