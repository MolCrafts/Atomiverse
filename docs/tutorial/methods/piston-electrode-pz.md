# Piston electrode — constant normal pressure \(P_z\) (E6)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `piston-electrode-pz` |
| Regression | `regressions/md/piston-electrode-pz/` |
| Public API | `cpu::md::PistonElectrode` (Component on `cpu::md::MD`; default `post_force` / rigid integrate hooks) |
| Roadmap | map slice E6 |
| Oracle | **MetalWalls** rigid-piston constant-\(P_z\) algorithm (JOSS; no live MW binary on pin host) |

## Summary

**Constant normal pressure** on a rigid electrode wall: one atom type is a
**rigid piston** free only along \(z\), with external force
\(F_{\mathrm{ext}}=\pm P_{\mathrm{target}}A\) so that the diagnostic
\(\langle P_z\rangle\to P_{\mathrm{target}}\). Public surface is
`cpu::md::PistonElectrode` — a **Component** on `cpu::md::MD` (not a Driver
category; not inside `KSpace*`; orthogonal to E1 charge solve). This
regression is the **E6 foothold**: MetalWalls-class electrode mechanics
(Marin-Laflèche et al., *JOSS* **5**, 2373 (2020)). Offline pin is the
MW force identity plus a self-consistent probe → target equilibration gate
(\(\lvert\langle P_z\rangle-P_{\mathrm{target}}\rvert/P_{\mathrm{target}}\le 0.10\)).
Fair wall vs MetalWalls is **`perf_exempt`** (no MW binary on the offline
worker; LAMMPS `fix wall/piston` is shock kinematics, not electrode \(P_z\)).

## Method

**Physics.** MetalWalls treats electrodes as **rigid pistons** for
constant-\(P_z\) runs (JOSS summary; capability map E6). Atomiverse
`cpu::md::PistonElectrode` implements the same textbook force balance on
atoms of a chosen type. With wall area \(A=L_x L_y\):

\[
F_z^{\mathrm{atoms}}=\sum_{i\in t} F_{z,i},\quad
F_{\mathrm{ext}}=\begin{cases}
-P_{\mathrm{target}} A & \text{Upper}\\
+P_{\mathrm{target}} A & \text{Lower}
\end{cases}
\]

\[
F_{\mathrm{net}}=F_z^{\mathrm{atoms}}+F_{\mathrm{ext}}(+F_{\mathrm{damp}}),\quad
F_{z,i}\leftarrow F_{\mathrm{net}}\,m_i/M,\quad F_{x,i}=F_{y,i}=0.
\]

Optional COM friction (equilibration aid, default off):

\[
F_{\mathrm{damp}}=-\frac{M\,v_{\mathrm{com},z}}{\tau_{\mathrm{damp}}\,\texttt{force\_to\_accel}}
\quad(\tau_{\mathrm{damp}}>0).
\]

Instantaneous diagnostic pressure (atom forces only; external load is the
target):

- `Side::Upper`: \(P_z=F_z^{\mathrm{atoms}}/A\)
- `Side::Lower`: \(P_z=-F_z^{\mathrm{atoms}}/A\)

Equilibrium: \(\langle P_z\rangle\to P_{\mathrm{target}}\). Rigid-body
enforcement equalizes \(v_z\) (mass-weighted COM), zeros \(v_x,v_y\), and
restores in-plane / relative-\(z\) offsets snapshotted at `build`.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry | Ar-like LJ fluid (16 atoms, two layers) + lower wall type 2 + upper piston type 3 |
| Walls | \(2\times2\) atoms each; lower frozen at \(z=2\) Å; upper piston at \(z=8\) Å |
| Box | orthorhombic \(L_x=L_y=10\) Å, \(L_z=20\) Å → \(A=100\) Å² |
| Fluid layers | \(z_{\mathrm{lo}}+2.2\) and \(z_{\mathrm{hi}}-2.2\) (repulsive contact) |
| Pair | LJ \(\varepsilon=0.10\) eV, \(\sigma=3.0\) Å, \(r_c=6\) Å, skin \(0.5\) Å |
| Integrator / thermostat | VelocityVerlet \(dt=0.5\) fs; NoseHoover \(T=300\) K, \(\tau_T=50\) fs (fluid NVT) |
| Piston | `Side::Upper`, type 3; \(P_{\mathrm{target}}=\) probe \(P_z>0\) at compressed geometry; \(\tau_{\mathrm{damp}}=20\) fs |
| Lower wall | `FixFreeze` type 2 (must **not** freeze the piston type) |
| Trajectory | warm 400 + prod 600 steps; chunk-sampled \(P_z\) every 20 steps |

**Key exclusions (non-goals of this slug)**

- Charge solve / const-pot BO or CG on the piston (E1/E2 — orthogonal).
- Generic isotropic / anisotropic **barostat** on the simulation box — this is
  a **rigid electrode plate**, not `fix npt`.
- LAMMPS `fix wall/piston` parity (SHOCK package: driven shock wall, not
  electrochemical \(F_{\mathrm{ext}}=\pm PA\)).
- Live MetalWalls or LAMMPS inside ctest.
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `PistonElectrode` ctor args / members; no
Config bag). Minimal MD binding matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/piston_electrode.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

// frame: type 1 fluid, type 2 lower wall, type 3 upper piston;
// box orthorhombic 10×10×20 Å — see reg_piston-electrode-pz.cpp

// Probe: P_target=0 → last_P_z = F_z^atoms / A (must be >0 at compressed gap)
cpu::md::PistonElectrode probe(/*type=*/3, /*P=*/0.0,
                               cpu::md::PistonElectrode::Side::Upper);
{
    cpu::md::MD probe_sim;
    probe_sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.10, /*sigma=*/3.0, /*rc=*/6.0});
    probe_sim.set_neighbor(cpu::md::NeighborList<>{6.0, 0.5});
    probe_sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
    probe_sim.add_ref(probe);
    probe_sim.build();
    probe_sim.run(frame, 1);
}
const double P_pin = probe.last_P_z();  // [eV/Å³]

// Production piston at probe pin + COM damp; freeze the other wall
cpu::md::PistonElectrode piston(/*type=*/3, /*P [eV/Å³]=*/P_pin,
                                cpu::md::PistonElectrode::Side::Upper);
piston.tau_damp = 20.0;  // [fs], optional

cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{0.10, 3.0, 6.0});
sim.set_neighbor(cpu::md::NeighborList<>{6.0, 0.5});
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.set_thermostat(cpu::md::NoseHoover{/*T [K]=*/300.0, /*tau_T=*/50.0, /*dt=*/0.5});
sim.add(cpu::md::InitVelocity{300.0, /*seed=*/42u});
sim.add(cpu::md::FixFreeze{/*lower type=*/2});
sim.add_ref(piston);  // or add(std::move(piston)) to own
sim.build();
sim.run(frame, /*nsteps=*/1000);
// diagnostics: piston.last_P_z(), last_F_z_atoms(), last_F_net(), last_area()
```

Default bindings (from the public header):

| Phase | Order | Role |
| ----- | ----: | ---- |
| `md::Phase::post_force` | +900 | redistribute \(F_{\mathrm{net}}\) after physical forces; before `FixFreeze` (+1000) |
| `md::Phase::post_integrate` | 0 | rigid-body position/velocity enforcement |
| `md::Phase::final_integrate` | +200 | re-equalize \(v_z\), zero \(v_{xy}\) after second VV half-kick |

Architecture rules:

- **Driver stays `cpu::md::MD`.** Electrode mechanics are Components; never a
  new Driver category.
- Parameters are **constructor arguments and public members** — no
  `PistonElectrodeConfig`.
- Frame is the sole I/O currency; forces / velocities rewritten on piston rows
  only.
- **Do not** `FixFreeze` the same type as the piston; freeze the opposite wall
  only.

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
  cmake --build build --target reg_piston-electrode-pz -j"$(nproc)"
  ./build/regressions/reg_piston-electrode-pz
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/piston_electrode.h` | Public `PistonElectrode`, `Side` |
| `src/cpu/dynamics/piston_electrode.cpp` | Force balance, damp, rigid enforce |
| `include/atomiverse/cpu/dynamics/fix_freeze.h` | Opposite-wall freeze |
| `include/atomiverse/cpu/dynamics/md.h` | Host preset `cpu::md::MD` |
| `regressions/md/piston-electrode-pz/reg_piston-electrode-pz.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/README.md` | Offline pin provenance |

## Validation (vs MetalWalls)

Source:
[`regressions/md/piston-electrode-pz/REFERENCE.md`](../../../regressions/md/piston-electrode-pz/REFERENCE.md)
(offline pin; **never** live MetalWalls in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md),
MW notes under
`tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/README.md`.

**Oracle:** MetalWalls algorithm (Marin-Laflèche et al., *JOSS* **5**, 2373
(2020), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373)) —
constant \(P_z\) by rigid electrode pistons. No MetalWalls binary on the
offline worker; pin is the MW force identity plus self-consistent probe →
target equilibration.

**Why MetalWalls (not LAMMPS):** LAMMPS `fix wall/piston` (SHOCK) is
shock-wave kinematics, not electrode rigid-body \(F_{\mathrm{ext}}=\pm P A\).
Loop oracle priority is LAMMPS when the feature overlaps; this slug is
**MW-only** mechanics (map E6).

### Accuracy

| Quantity | Atomiverse gate / pin | Oracle / identity | Tol | Notes |
| -------- | --------------------: | ----------------: | --- | ----- |
| Probe \(P_z\) [eV/Å³] | \(>10^{-8}\) and \(>0\) | compressive LJ on upper wall | gate | compressed gap at \(z_{\mathrm{hi}}=8\) Å |
| \(\lvert\langle P_z\rangle-P_{\mathrm{target}}\rvert/P_{\mathrm{target}}\) | **≤ 0.10** | MW constant-\(P_z\) equilibration | **0.10** | textbook 10% e2e gate; \(P_{\mathrm{target}}=\) probe pin |
| Upper wall coplanar \(\Delta z\) [Å] | \(\le 10^{-9}\) | rigid piston | **1e-9** | all type-3 \(z\) equal |
| Upper \(\lvert v_x\rvert,\lvert v_y\rvert\) [Å/fs] | \(\le 10^{-12}\) | \(z\)-only motion | **1e-12** | rigid in plane |
| Lower wall \(\Delta z\) [Å] | \(\le 10^{-10}\) | `FixFreeze` | **1e-10** | type 2 fixed |
| Energy finite | true | smoke stability | finite | `kEnergy` |

**Unit surface (not the e2e statistical pin)** —
`tests/cpu/dynamics/test_piston_electrode.cpp`:

| Identity | Bound | Notes |
| -------- | ----- | ----- |
| \(P_z=F_z^{\mathrm{atoms}}/A\) (Upper) | machine precision | e.g. \(F=4\) eV/Å, \(A=100\) → \(P_z=0.04\) eV/Å³ |
| \(F_{\mathrm{net}}=0\) when \(P_{\mathrm{target}}=P_z^{\mathrm{atoms}}\) | \(\sim 10^{-12}\) | \(F_{\mathrm{ext}}=-PA\) cancels atom sum |
| Mass-weighted \(F_z\) share; \(F_x=F_y=0\) | \(\sim 10^{-12}\) | type-selective; fluid forces untouched |
| Lower side sign: \(P_z=-F_z^{\mathrm{atoms}}/A\), balance | \(\sim 10^{-12}\) | \(F_{\mathrm{ext}}=+PA\) |
| Rigid equalize \(v_z\); zero \(v_{xy}\) | \(\sim 10^{-12}\) | mass-weighted COM \(v_z\) |

Geometry, LJ parameters, trajectory lengths, and tolerances: **Method** table
and REFERENCE (match `reg_piston-electrode-pz`).

### Offline pin repro (never ctest)

```bash
# Self-consistent probe pin (no MW binary required)
./build/regressions/reg_piston-electrode-pz
# When MetalWalls is available offline: run a comparable rigid-piston
# example from ampere2/metalwalls example/, tabulate <P_z>, then fair-bench.
```

## Performance

Source: REFERENCE.md; handoff
the case `REFERENCE.md`;
| Twin | \(T_{\mathrm{MW}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ----------------------: | -----------------------: | ----: | ---- | ------ |
| rigid-piston constant-\(P_z\) | — | — | **perf_exempt** | n/a | No MetalWalls binary on offline worker PATH; cannot fair-bench vs MW twin |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | no MetalWalls binary on the offline worker PATH; cannot fair-bench vs MW twin for this MW-only rigid-piston path. (LAMMPS `fix wall/piston` is shock-wave kinematics, not electrode \(P_z\).) |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls binary is available,
re-run offline fair bench, record version + walls, and replace `perf_exempt`
in REFERENCE.

## Limitations

- **Foothold geometry only.** Small LJ parallel plate (16 fluid + 8 wall
  atoms); not a production EDL cell or electrolyte force field.
- **Pair forces only.** No const-pot charge solve on the piston; movable
  electrode + BO/CG re-solve is later composition (E1/E2 + E6).
- **Self-consistent probe pin.** Without a live MW trajectory, \(P_{\mathrm{target}}\)
  is the instantaneous probe \(P_z\) at the compressed start geometry — not a
  published MW table value. Unit tests pin the static identities independently.
- **No fair speed ratio** until MetalWalls is on the pin host.
- **Opposite wall is frozen**, not a second piston; dual-piston NPT-style
  electrodes are out of scope for this slug.
- **Scope of E6 wave only** for this page: Thomas–Fermi metallicity,
  capacitance observers, etc. are sibling slugs with their own pages when
  written.

## References

- Offline pin: [`regressions/md/piston-electrode-pz/REFERENCE.md`](../../../regressions/md/piston-electrode-pz/REFERENCE.md)
- Product gate: [`regressions/md/piston-electrode-pz/reg_piston-electrode-pz.cpp`](../../../regressions/md/piston-electrode-pz/reg_piston-electrode-pz.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/piston_electrode.h`](../../../include/atomiverse/cpu/dynamics/piston_electrode.h)
- Unit tests: [`tests/cpu/dynamics/test_piston_electrode.cpp`](../../../tests/cpu/dynamics/test_piston_electrode.cpp)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/README.md`](../../../tools/metalwalls_oracle/inputs/metalwalls/piston-electrode-pz/README.md)
- Literature: Marin-Laflèche et al., *J. Open Source Softw.* **5**, 2373 (2020), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373) (MetalWalls; constant \(P_z\) via rigid electrode pistons)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (E6 electrode mechanics)
