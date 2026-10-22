# Constant-pressure MD (NPT)

Isothermal–isobaric molecular dynamics: a Martyna–Tobias–Klein (MTK) Trotter
barostat coupled to a Nosé–Hoover chain thermostat, run as an **Integrator**
Component on the `cpu::md::MD` preset (never a separate Driver category). The
case also carries the isotropic NPT, NPH, Parrinello–Rahman and PME NaCl
sections; this page documents the constant surface-tension variant,
`cpu::md::SurfaceTensionNPT`, a constructor front over `cpu::md::NoseHooverNPT`
with LAMMPS-style `couple xy` and a lateral pressure offset set by the target
surface tension.

Case: `ctest -R '^md/npt$'` — sources and oracle record in [`regressions/md/npt/`](../../../regressions/md/npt/). One section of the case per `##` heading below: `atv_regressions --case md/npt --section <name>`.

## Common setup

- **Barostat:** MTK NPT (Martyna, Tobias, Klein, *J. Chem. Phys.* **101**,
  4177 (1994)) as implemented in `NoseHooverNPT`. Parameters are constructor
  arguments (plus chainable setters on `NoseHooverNPT`); there is no Config bag.
- **Frame contract:** Frame is the sole I/O currency; box edges and velocities
  are updated in place; stress is read from Frame meta
  (`requires_system_stress()`), so a `Virial` observer must be attached.
- **Units:** eV, Å, fs, eV/Å³, eV/Å², K. Pressure sign follows LAMMPS
  (positive compressive).
- **Backend:** CPU classical MD; MD-only build (`buildenv-gcc/2026.03-mpich`,
  `ATV_BUILD_MD=ON`).

## surface-tension-coupling — Constant surface-tension NPT, couple \(xy\) (NPγT) {#surface-tension-coupling}

### Summary

**Constant surface tension** (NPγT / semi-isotropic NPT with couple \(xy\))
controls the lateral pressure so that the planar surface tension \(\gamma\) is
held at a target while the normal pressure \(P_n = P_{zz}\) is controlled
independently. `cpu::md::SurfaceTensionNPT` enables LAMMPS-style `couple xy`
on `NoseHooverNPT` and applies the GROMACS- / literature lateral offset
\(P_\perp(t)=P_n-n_{\mathrm{surf}}\,\gamma/L_z(t)\). The section runs a
compressed LJ FCC slab (108 atoms, 200 steps) with structural gates (lockstep
\(L_x=L_y\), anisotropic response, finite \(\gamma_{\mathrm{inst}}\)). Offline
pin and fair wall are against **LAMMPS**; ratio
\(T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\approx\mathbf{2.88}\) (≤10).

### Method

**Physics.** For a planar interface normal to \(z\) with \(n_{\mathrm{surf}}\)
surfaces in the cell (typically 2 under PBC), the mechanical surface tension
and NPγT barostat targets are

\[
\gamma_{\mathrm{inst}}=\frac{L_z}{2}\Bigl(P_{zz}-\tfrac12(P_{xx}+P_{yy})\Bigr),
\]

\[
P_\perp(t)=P_n-\frac{n_{\mathrm{surf}}\,\gamma}{L_z(t)},\qquad P_z=P_n,
\]

with \(P\) in eV/Å³, \(\gamma\) in eV/Å², \(L_z\) in Å. **Couple \(xy\)**
averages the lateral stresses \((P_{xx}+P_{yy})/2\), forces
\(\dot\varepsilon_x=\dot\varepsilon_y\) so \(L_x=L_y\) remains exact, and
leaves \(z\) independent (no three-axis isotropic pressure average).

**LAMMPS twin vs true NPγT.** LAMMPS `fix npt` cannot take equal-style
variables for `Pstart`/`Pstop`, so the offline twin uses a **fixed**
\(P_\perp=P_n-n_{\mathrm{surf}}\,\gamma/L_z(0)\) for the short pin.
Atomiverse updates \(P_\perp(L_z)\) **each step** (true NPγT). Short-run
gates are therefore **structural** (lockstep, expansion, anisotropy, finite
\(\gamma_{\mathrm{inst}}\)), not bit-identical trajectories.

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
| steps | 200 (regression); fair wall uses 5000 atom-steps |
| Stress | `Virial` for barostat feedback |

**Out of scope:** full equilibrium
\(\langle\gamma\rangle=\gamma_{\mathrm{target}}\) production sampling;
orthogonal barostats without couple \(xy\) / without the \(\gamma\) offset
(plain isotropic `NoseHooverNPT`, Berendsen); MetalWalls as oracle (the feature
exists in LAMMPS); live LAMMPS inside ctest or bit-identical trajectory match to
the LAMMPS RNG path; electrode / const-pot charge solve (orthogonal
Components).

### Implementation

```cpp
// frame: 108-atom LJ FCC at a=5.1 Å, cubic L0=15.3 Å — see
// regressions/md/npt/surface-tension-coupling.cpp
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
sim.set_observer(cpu::md::InitVelocity{100.0, /*seed=*/777u});
sim.set_observer(cpu::md::Virial{});
sim.build();
sim.run(frame, /*nsteps=*/200);

// diagnostics from box + stress meta:
//   Lx, Ly, Lz = frame.box.length(0|1|2)
//   gamma_inst = 0.5 * Lz * (Pzz - 0.5*(Pxx+Pyy))
//   with P_aa = (kStress[a] + kStressKinetic[a]) / V
```

Equivalent via the base integrator:

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

`SurfaceTensionNPT` **replaces** (do not add alongside) `VelocityVerlet`,
`NoseHoover`, isotropic `Barostat`, and plain isotropic `NoseHooverNPT`.
Headers: `surface_tension_npt.h` (public `SurfaceTensionNPT`),
`nose_hoover_npt.h` (MTK NPT + `set_couple_xy` / `set_surface_tension`),
`md.h` (preset `cpu::md::MD`).

### Validation (vs LAMMPS)

**Oracle:** LAMMPS **30 Mar 2026 - Development**
(`patch_30Mar2026-1074-g7f680de296`; binary recorded in REFERENCE), metal
units, `fix npt temp 100 100 0.05 x Plat Plat 0.5 y Plat Plat 0.5 z 0 0 0.5
couple xy tchain 3 pchain 3 mtk yes` with fixed
\(\mathrm{Plat}=-209.434\) bar. Offline pin only; LAMMPS never runs inside
ctest. Twin input: `tools/metalwalls_oracle/inputs/lammps/in.surface_tension_coupling`.

**Why LAMMPS (not MetalWalls):** the algorithm exists as LAMMPS
`fix npt … couple xy` (surface-tension community path); LAMMPS is the first
oracle when the feature overlaps, MetalWalls is reserved for MW-only physics.

#### Accuracy

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
dynamic \(P_\perp(L_z)\)).

Offline pin repro (never ctest), with a local LAMMPS binary carrying `fix npt`
(standard package): `lmp -in <LAMMPS oracle input for this case>`; the
\(L_x=L_y\) / \(L_z\) pin is recorded in REFERENCE.md.

### Performance

Comparable workload: **108 atoms, 5000 MD steps**, OMP=1, same LJ + couple-\(xy\)
NPT settings as the pin (fixed Plat on LAMMPS; `SurfaceTensionNPT` on ATV),
same login node.

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| 108-atom couple-\(xy\) NPT (5000 steps) | **0.140** | **0.403** | **2.88** | ✓ | ATV wall = \(25\times\) the full section (200 steps each = 5000 atom-steps; includes 25× `build()` — conservative vs pure loop) |

Not perf-exempt; oracle version as above.

### Limitations

- **Structural gates only.** The section does not pin
  \(\langle\gamma\rangle=\gamma_{\mathrm{target}}\) over long sampling; it
  gates lockstep, expansion, anisotropy, and finite \(\gamma_{\mathrm{inst}}\).
- **Dynamic vs fixed \(P_\perp\).** Atomiverse updates \(P_\perp(L_z)\) each
  step; the LAMMPS twin freezes Plat at \(L_z(0)\). Trajectory numbers are not
  interchangeable for short runs.
- **Foothold geometry.** Compressed argon-like LJ FCC (108 atoms); not a
  production liquid–vapour slab force field or interfacial free-energy protocol.
- **Single public front.** `SurfaceTensionNPT` is couple-\(xy\) + \(\gamma\)
  only; fully anisotropic NPH / other couple modes are out of scope.
- **CPU only.** CUDA parity is not claimed.
- **No unit test.** `SurfaceTensionNPT` is covered only by this regression
  section.

### References

- Roadmap: [`docs/roadmap/molecular-dynamics.md`](../../roadmap/molecular-dynamics.md) (constant surface-tension coupling)
- Martyna, Tobias, Klein, *J. Chem. Phys.* **101**, 4177 (1994).
- LAMMPS `fix nh` / `fix npt` documentation (`couple xy`) — <https://docs.lammps.org/fix_nh.html>
