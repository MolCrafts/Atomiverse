# Constant surface-tension NPT — couple \(xy\) (NPγT)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `surface-tension-coupling` |
| Regression | `regressions/md/surface-tension-coupling/` |
| Public API | `cpu::md::SurfaceTensionNPT` (Integrator on `cpu::md::MD`; thin NPγT front over `NoseHooverNPT` couple-\(xy\)) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **done** (wave 3, 2026-07-24); `docs/roadmap/molecular-dynamics.md` constant-surface-tension ✓ |
| Oracle | **LAMMPS** `fix npt … couple xy` (offline pin; no live `lmp` in ctest) |

## Summary

**Constant surface tension** (NPγT / semi-isotropic NPT with couple \(xy\))
controls the lateral pressure so that the planar surface tension \(\gamma\) is
held at a target while the normal pressure \(P_n = P_{zz}\) is controlled
independently. Atomiverse ships this as `cpu::md::SurfaceTensionNPT` — an
**Integrator** Component on `cpu::md::MD` (Martyna–Tobias–Klein Trotter; not a
new Driver category; not inside `KSpace*`). It is a thin constructor front
over `NoseHooverNPT` that enables LAMMPS-style `couple xy` and the GROMACS- /
literature lateral offset \(P_\perp(t)=P_n-n_{\mathrm{surf}}\,\gamma/L_z(t)\).
The product gate `reg_surface-tension-coupling` is a compressed LJ FCC slab
(108 atoms, 200 steps) with structural duals (lockstep \(L_x=L_y\), anisotropic
response, finite \(\gamma_{\mathrm{inst}}\)). Offline pin and fair wall are
vs **LAMMPS** develop (2026-07-24); ratio \(T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\approx\mathbf{2.88}\) (≤10).

## Method

**Physics.** For a planar interface normal to \(z\) with \(n_{\mathrm{surf}}\)
surfaces in the cell (typically 2 under PBC), the mechanical surface tension
and NPγT barostat targets are

\[
\gamma_{\mathrm{inst}}=\frac{L_z}{2}\Bigl(P_{zz}-\tfrac12(P_{xx}+P_{yy})\Bigr),
\]

\[
P_\perp(t)=P_n-\frac{n_{\mathrm{surf}}\,\gamma}{L_z(t)},\qquad P_z=P_n,
\]

with \(P\) in eV/Å³ (LAMMPS sign: positive compressive), \(\gamma\) in eV/Å²,
\(L_z\) in Å. **Couple \(xy\)** averages the lateral stresses
\((P_{xx}+P_{yy})/2\), forces \(\dot\varepsilon_x=\dot\varepsilon_y\) so
\(L_x=L_y\) remains exact, and leaves \(z\) independent (no three-axis
isotropic pressure average). The barostat itself is MTK NPT
(Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994)) as already
implemented in `NoseHooverNPT`.

**LAMMPS twin vs true NPγT.** LAMMPS `fix npt` cannot take equal-style
variables for `Pstart`/`Pstop`, so the offline twin uses a **fixed**
\(P_\perp=P_n-n_{\mathrm{surf}}\,\gamma/L_z(0)\) for the short pin.
Atomiverse updates \(P_\perp(L_z)\) **each step** (true NPγT). Short-run e2e
gates are therefore **structural** (lockstep, expansion, anisotropy, finite
\(\gamma_{\mathrm{inst}}\)), not bit-identical trajectories.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Atoms | 108 (FCC \(3^3\) cells × 4 basis; mass 39.948 amu) |
| Lattice | \(a=5.1\) Å; \(L_0=15.3\) Å cubic start (compressed) |
| Pair | LJ \(\varepsilon=0.0104\) eV, \(\sigma=3.4\) Å, \(r_c=8.5\) Å, skin \(0.5\) Å |
| \(T\) | 100 K; \(\tau_T=50\) fs; InitVelocity seed 777 |
| \(P_n\) | 0 eV/Å³ (normal target) |
| \(\gamma\) | 0.001 eV/Å²; \(n_{\mathrm{surf}}=2\) |
| \(P_\perp(0)\) | \(-1.30719\times 10^{-4}\) eV/Å³ (\(= -209.434\) bar in metal units) |
| \(\tau_P\) | 500 fs |
| \(dt\) | 1 fs |
| steps | 200 (e2e); fair wall uses 5000 atom-steps |
| Stress | `Virial` (auto / explicit) for barostat feedback |

**Key exclusions (non-goals of this slug)**

- Full equilibrium \(\langle\gamma\rangle=\gamma_{\mathrm{target}}\) production
  sampling (short structural foothold only).
- Orthogonal barostats without couple \(xy\) / without \(\gamma\) offset
  (plain isotropic `NoseHooverNPT`, Berendsen).
- MetalWalls as oracle (feature exists in LAMMPS).
- Live LAMMPS inside ctest; bit-identical trajectory match to LAMMPS RNG path.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Electrode / const-pot charge solve (orthogonal E1/E2 Components).

## Implementation

Public surface only (`cpu::Frame` + `SurfaceTensionNPT` ctor args; no Config
bag). Minimal MD binding matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/surface_tension_npt.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

// frame: 108-atom LJ FCC at a=5.1 Å, cubic L0=15.3 Å — see
// reg_surface-tension-coupling.cpp

cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0104, /*sigma=*/3.4, /*rc=*/8.5});
sim.set_neighbor(cpu::md::NeighborList<>{8.5, 0.5});
sim.set_integrator(cpu::md::SurfaceTensionNPT{
    /*T [K]=*/100.0,
    /*tau_T [fs]=*/50.0,
    /*P_n [eV/Å³]=*/0.0,
    /*gamma [eV/Å²]=*/0.001,
    /*tau_P [fs]=*/500.0,
    /*dt [fs]=*/1.0,
    /*n_surfaces=*/2});
sim.add(cpu::md::InitVelocity{100.0, /*seed=*/777u});
sim.add(cpu::md::Virial{});
sim.build();
sim.run(frame, /*nsteps=*/200);

// diagnostics from box + stress meta:
//   Lx, Ly, Lz = frame.box.length(0|1|2)
//   gamma_inst = 0.5 * Lz * (Pzz - 0.5*(Pxx+Pyy))
//   with P_aa = (kStress[a] + kStressKinetic[a]) / V
```

Equivalent via the base integrator (chaining members, still no Config bag):

```cpp
cpu::md::NoseHooverNPT npt{100.0, 50.0, /*P=*/0.0, 500.0, /*dt=*/1.0};
npt.set_surface_tension(/*gamma=*/0.001, /*n_surfaces=*/2);  // implies couple xy
sim.set_integrator(std::move(npt));
```

Default bindings (inherited MTK split on `NoseHooverNPT`):

| Phase | Order | Role |
| ----- | ----: | ---- |
| `md::Phase::initial_integrate` | 0 | first half-step Trotter |
| `md::Phase::final_integrate` | 0 | second half-step |
| `md::Phase::end_of_step` | −40 | rewrite kinetic after second NHC |

**Replaces** (do not add alongside) `VelocityVerlet`, `NoseHoover`, isotropic
`Barostat`, and plain isotropic `NoseHooverNPT`.

Architecture rules:

- **Driver stays `cpu::md::MD`.** Surface-tension coupling is an Integrator
  Component; never a new Driver category.
- Parameters are **constructor arguments** (and chainable setters on
  `NoseHooverNPT`) — no `SurfaceTensionNPTConfig`.
- Frame is the sole I/O currency; box edges and velocities are updated in
  place; stress is read from Frame meta (`requires_system_stress()`).
- Units on the public surface: eV, Å, fs, eV/Å³, eV/Å², K.

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
  cmake --build build --target reg_surface-tension-coupling -j"$(nproc)"
  ./build/regressions/reg_surface-tension-coupling
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/surface_tension_npt.h` | Public `SurfaceTensionNPT` |
| `include/atomiverse/cpu/dynamics/nose_hoover_npt.h` | MTK NPT + `set_couple_xy` / `set_surface_tension` |
| `include/atomiverse/cpu/dynamics/md.h` | Host preset `cpu::md::MD` |
| `regressions/md/surface-tension-coupling/reg_surface-tension-coupling.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling` | Offline LAMMPS twin |

## Validation (vs LAMMPS)

Source:
[`regressions/md/surface-tension-coupling/REFERENCE.md`](../../../regressions/md/surface-tension-coupling/REFERENCE.md)
(offline pin; **never** live LAMMPS in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).
Twin input:
[`tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling`](../../../tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling).

**Oracle:** LAMMPS **30 Mar 2026 - Development**
(`patch_30Mar2026-1074-g7f680de296`; binary recorded in REFERENCE), metal
units, `fix npt temp 100 100 0.05 x Plat Plat 0.5 y Plat Plat 0.5 z 0 0 0.5
couple xy tchain 3 pchain 3 mtk yes` with fixed
\(\mathrm{Plat}=-209.434\) bar.

**Why LAMMPS (not MetalWalls):** the algorithm exists as LAMMPS
`fix npt … couple xy` (surface-tension community path). Binding oracle
priority for this loop is **LAMMPS first** when the feature overlaps.
MetalWalls is reserved for MW-only physics.

### Accuracy

| Quantity | Atomiverse gate / pin | Oracle / identity | Tol | Notes |
| -------- | --------------------: | ----------------: | --- | ----- |
| \(\lvert L_x-L_y\rvert\) | \(\le 10^{-9}\max(1,\lvert L_x\rvert)\) | LAMMPS couple \(xy\) exact | **1e-9** (rel) | Gate A2/A4 |
| Volume | \(0 < V < 10\,V_0\), finite | no runaway | window | Gate A4 |
| Box motion | \(\lvert L_x-L_0\rvert+\lvert L_z-L_0\rvert > 10^{-4}\) Å | compressed start expands | structural | |
| Anisotropy / expansion | \(\lvert L_z-L_x\rvert>10^{-4}\) **or** \(\lvert L_x-L_0\rvert>10^{-3}\) | LAMMPS \(L_z\neq L_x\) path | soft dual | true NPγT vs fixed Plat |
| \(\gamma_{\mathrm{inst}}\) | finite | diagnostic | finite | Gate A5 |
| LAMMPS snapshot \(L_x=L_y\) [Å] | — | **16.527384** @ step 200 | pin | \(L_0=15.3\) |
| LAMMPS snapshot \(L_z\) [Å] | — | **16.249165** @ step 200 | pin | \(L_z\neq L_x\) |

ATV need not match LAMMPS edge lengths at step 200 (different RNG path and
dynamic \(P_\perp(L_z)\)). Unit surface (not the e2e statistical pin) —
`tests/cpu/dynamics/test_surface_tension_npt.cpp`: Integrator API (no Config),
couple-\(xy\) keeps \(L_x=L_y\), `NoseHooverNPT::set_surface_tension` enables
couple \(xy\).

Geometry, LJ parameters, trajectory lengths, and tolerances: **Method** table
and REFERENCE (match `reg_surface-tension-coupling`).

### Offline pin repro (never ctest)

```bash
# Requires a local LAMMPS binary with fix npt (standard package)
lmp -in tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling
# Pin Lx=Ly / Lz recorded in REFERENCE.md (2026-07-24)
```

## Performance

Source: REFERENCE.md; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md);
queue
[`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md).

Comparable workload: **108 atoms, 5000 MD steps**, OMP=1, same LJ + couple-\(xy\)
NPT settings as the pin (fixed Plat on LAMMPS; `SurfaceTensionNPT` on ATV).
Host: same login node, 2026-07-24.

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| 108-atom couple-\(xy\) NPT (5000 steps) | **0.140** | **0.403** | **2.88** | ✓ | ATV wall = \(25\times\) full `reg_surface-tension-coupling` (200 steps each = 5000 atom-steps; includes 25× `build()` — conservative vs pure loop) |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **false** |
| `ratio_atv_over_oracle` | **2.88** |
| `within_10x` | **true** |
| `oracle_version` | 30 Mar 2026 - Development (`patch_30Mar2026-1074-g7f680de296`) |

Do **not** invent wall-clock ratios. Numbers above are exactly those in
REFERENCE / handoff.

## Limitations

- **Structural e2e only.** The regression does not pin
  \(\langle\gamma\rangle=\gamma_{\mathrm{target}}\) over long sampling; it
  gates lockstep, expansion, anisotropy, and finite \(\gamma_{\mathrm{inst}}\).
- **Dynamic vs fixed \(P_\perp\).** Atomiverse updates \(P_\perp(L_z)\) each
  step; the LAMMPS twin freezes Plat at \(L_z(0)\). Trajectory numbers are not
  interchangeable for short runs.
- **Foothold geometry.** Compressed argon-like LJ FCC (108 atoms); not a
  production liquid–vapour slab force field or interfacial free-energy protocol.
- **Single public front.** `SurfaceTensionNPT` is couple-\(xy\) + \(\gamma\)
  only; fully anisotropic NPH / other couple modes are out of scope for this
  slug.
- **CPU classical MD path.** CUDA parity is not claimed.
- **Scope of wave-3 MD coupling only** for this page: electrode const-pot,
  pistons, continuum walls are sibling slugs with their own pages.

## References

- Offline pin: [`regressions/md/surface-tension-coupling/REFERENCE.md`](../../../regressions/md/surface-tension-coupling/REFERENCE.md)
- Product gate: [`regressions/md/surface-tension-coupling/reg_surface-tension-coupling.cpp`](../../../regressions/md/surface-tension-coupling/reg_surface-tension-coupling.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/surface_tension_npt.h`](../../../include/atomiverse/cpu/dynamics/surface_tension_npt.h)
- Base integrator: [`include/atomiverse/cpu/dynamics/nose_hoover_npt.h`](../../../include/atomiverse/cpu/dynamics/nose_hoover_npt.h)
- Unit tests: [`tests/cpu/dynamics/test_surface_tension_npt.cpp`](../../../tests/cpu/dynamics/test_surface_tension_npt.cpp)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Roadmap: [`docs/roadmap/molecular-dynamics.md`](../../roadmap/molecular-dynamics.md) (constant surface-tension coupling)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), twin [`tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling`](../../../tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling)
- Literature: Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994); LAMMPS `fix nh` / `fix npt` documentation (`couple xy`) — <https://docs.lammps.org/fix_nh.html>
