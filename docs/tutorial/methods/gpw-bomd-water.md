# GPW BOMD — H₂O NVE energy-drift foothold

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-bomd-water` |
| Regression | `regressions/gpw/gpw-bomd-water/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::gpw::GPWEvaluator{…, compute_forces=true}` at `md::Phase::force` |
| Roadmap | MD product path in [ch09-gpw.md](../ch09-gpw.md) / [ch10-classical-md.md](../ch10-classical-md.md) |
| Dual gate | Correctness A4–A6; speed **`perf_exempt`** (no CP2K AIMD twin wall) |
| Wave | CP2K capability loop — wave 2 **done** (merged 2026-07-24) |

## Summary

Born–Oppenheimer molecular dynamics (BOMD) with **Gaussian-and-plane-waves
(GPW / Quickstep)** forces on every ionic step: the outer loop is the MD
preset (`cpu::md::MD` + NVE `VelocityVerlet`); the force/energy slot binds
`cpu::gpw::GPWEvaluator` with analytic forces (`compute_forces = true`).
Public units are eV, Å, fs, eV/Å; Hartree energy is available on
`last_energy_hartree()` only.

This page documents the **product multi-atom GPW BOMD foothold**: isolated
**H₂O** at the CP2K getting-started geometry (same twin family as
`gpw-forces-water` / `gpw-water-pbe-mt`), cubic 10 Å cell, Martyna–Tuckerman
nonperiodic Poisson, PBE / DZVP-MOLOPT-GTH / GTH-PBE. The pin is **not** a
golden Atomiverse total energy and **not** a live CP2K AIMD twin — it is the
literature symplectic / BOMD **total-energy conservation envelope** plus
finite-trajectory diagnostics. Absolute force quality vs **CP2K 2026.1** is
already offline-pinned in `gpw-forces-water` (max |ΔF| ≈ 3.4×10⁻⁴ Ha/Bohr);
this slug gates **MD binding + NVE conservation** on the product evaluator
path. Enables later `xl-bomd-water`.

## Method

**Level.** Classical Born–Oppenheimer MD: nuclei integrated with a fixed-step
symplectic Velocity Verlet; electrons re-solved self-consistently each step
(no Car–Parrinello fictitious electronic dynamics, no XL-BOMD / ASPC extended
Lagrangian).

**Electronic structure.** GPW Kohn–Sham (VandeVondele et al., *Comput. Phys.
Commun.* **167**, 103 (2005)): density matrix in contracted Gaussians
(DZVP-MOLOPT-GTH); Hartree / XC / local short-range GTH PP on a uniform
real-space grid; MT isolated Poisson (`nonperiodic = true`). Forces are the
product analytic path (Hellmann–Feynman + Pulay + grid/core) written to
`frame.span(kAtom_f*)` [eV/Å] when `compute_forces = true`. The evaluator
may warm-start SCF from the previous step’s MO coefficients when available
(product BOMD path).

**Nuclear dynamics (NVE).**

\[
\mathbf{v}\!\left(t+\tfrac{\Delta t}{2}\right)
  = \mathbf{v}(t) + \tfrac{\Delta t}{2}\,\mathbf{a}(t),\quad
\mathbf{r}(t+\Delta t)
  = \mathbf{r}(t) + \Delta t\,\mathbf{v}\!\left(t+\tfrac{\Delta t}{2}\right),
\]

\[
\mathbf{a}(t+\Delta t)
  = \mathbf{F}\!\big(\mathbf{r}(t+\Delta t)\big)/m,\quad
\mathbf{v}(t+\Delta t)
  = \mathbf{v}\!\left(t+\tfrac{\Delta t}{2}\right)
  + \tfrac{\Delta t}{2}\,\mathbf{a}(t+\Delta t).
\]

Total energy \(E(t)=E_{\mathrm{pot}}(t)+E_{\mathrm{kin}}(t)\) is conserved up
to bounded, non-secular error when \(\mathbf{F}\approx -\nabla E\) and the
integrator is symplectic (Frenkel & Smit §4.3.4; Allen & Tildesley §3.2;
Marx & Hutter, *Ab Initio Molecular Dynamics*, BOMD chapter).

**Primary observable (envelope, not absolute energy):**

\[
\delta = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)}
  \le 0.05,
\]

with \(E_0\) taken after one warm-up step (energies populated). Additional
gates: PE finite and bound (\(< -100\) eV after warm-up for PBE water); KE
finite, \(\ge 0\), and non-trivial after warm-up; coordinates finite; soft
O–H bond ceiling \(r_{\mathrm{OH}} < 1.8\) Å after production.

**System (regression settings).**

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started (same as `gpw-forces-water`) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | **200 Ry** (BOMD envelope; forces pin uses 400 Ry) |
| SCF | `tol = 1e-5`, `max_iter = 60`, `mix = 0.3` |
| Ensemble | NVE (no thermostat / barostat) |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

**Why cheaper cutoff than the forces pin?** One ENERGY_FORCE at 400 Ry is
~40 s wall on login-class ATV (see `gpw-forces-water` REFERENCE). Multi-step
BOMD at that cost exceeds the ≤60 s regression gate. 200 Ry keeps the same
method family (PBE / MOLOPT / GTH / MT / analytic forces) while staying inside
the time budget; absolute E/F parity remains the forces-pin’s job.

**Excluded.** XL-BOMD / CPMD / ASPC (`xl-bomd-water`, `cpmd-water`); GPW cell
stress (`gpw-stress-water`); thermostatted production AIMD / long traj; PW
BOMD foothold (`pw-bomd-h` — toy UPF H atom); open-shell UKS GPW, k-point
GPW, hybrid/GAPW (other queue rows); live CP2K from ctest; MPI multi-rank;
public `*Config` bags / `make_*` factories; changes to analytic force math
(already gated by `gpw-forces-water`).

## Implementation

Public surface only: `cpu::Frame`, `cpu::gpw::Kind` / lookup constructors,
`GPWEvaluator` constructor args / members, MD preset, explicit force-phase
binding. Headers: `include/atomiverse/cpu/dynamics/md.h`,
`velocity_verlet.h`, `include/atomiverse/cpu/gpw/gpw_evaluator.h`,
`gaussian_basis.h`, `gth_potential.h`. Unit gate (not e2e):
`tests/cpu/gpw/test_gpw_bomd.cpp`.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

cpu::Frame frame(3);
const int z[3] = {8, 1, 1};
const double x0[3] = {5.0000, 5.7586, 4.2414};
const double y0[3] = {5.0000, 5.0000, 5.0000};
const double z0[3] = {5.0000, 5.5043, 5.5043};
const double mass[3] = {15.999, 1.008, 1.008};
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_type)[i] = z[i];
    frame.span(kAtom_mass)[i] = mass[i];
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_x)[i] = x0[i];
    frame.span(kAtom_y)[i] = y0[i];
    frame.span(kAtom_z)[i] = z0[i];
    frame.span(kAtom_vx)[i] = 0.0;
    frame.span(kAtom_vy)[i] = 0.0;
    frame.span(kAtom_vz)[i] = 0.0;
}
// COM-preserving H stretch [Å/fs]
frame.span(kAtom_vx)[1] = 0.0015;
frame.span(kAtom_vx)[2] = -0.0015;
frame.box = cpu::Box::cubic(10.0);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

// Product twin family (forces pin is 400 Ry / 1e-7). BOMD uses a cheaper
// same-method envelope so the ≤60 s gate holds.
cpu::gpw::GPWEvaluator eval(
    /*kinds*/ {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.on(md::Phase::force).add(std::move(eval));
sim.build();
sim.run(frame, 1);   // warm-up: populate PE/KE/forces
// E0 = frame.meta.get(kEnergy) + frame.meta.get(kKinetic)  [eV]
sim.run(frame, 4);   // production NVE
```

Rules (architecture):

- Parameters are **constructor arguments / members** of `GPWEvaluator` — no
  public `*Config` bag.
- Force binding is explicit `on(md::Phase::force).add(...)` — no categorical
  electronic-structure setter.
- `compute_forces = true` fills per-atom forces [eV/Å] via the product
  analytic path each MD force phase.
- Frame is the sole I/O currency (`blocks` + `meta` + `box`).
- Fixed cell (NVE): GPW rejects box changes after `build()`.

```bash
# configure with GPW (+ HF + PW deps) + regressions (CPU login node pattern)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
    -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=OFF \
    -DATV_BUILD_REGRESSIONS=ON -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_gpw-bomd-water -j"$(nproc)"
  ./build/regressions/reg_gpw-bomd-water
'
```

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-bomd-water/REFERENCE.md`](../../../regressions/gpw/gpw-bomd-water/REFERENCE.md)
(offline pin; **never** live CP2K in ctest). Oracle tooling:
[`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md).
Suggested twin input (when a binary is available):
`tools/cp2k_oracle/inputs/h2o_bomd_mt.inp`.

### Offline CP2K status

| Field | Value |
| ----- | ----- |
| CP2K binary on pin host | **unavailable** (no `cp2k` / `cp2k.psmp` / `cp2k.sopt`, no CP2K Lmod module on Arrhenius login-class inventory 2026-07-23) |
| Twin design (when available) | Short NVE AIMD / MD Quickstep GPW on the same H₂O cell (PBE / DZVP-MOLOPT-GTH / GTH / MT); re-run offline via `tools/cp2k_oracle/`; record version + measured drift envelope + fair wall |
| Current pin basis | Literature NVE/BOMD energy conservation envelope + finite/bounded trajectory diagnostics |
| Force twin (absolute) | `gpw-forces-water` vs **CP2K 2026.1** (max \|ΔF\| ≈ 3.4×10⁻⁴ Ha/Bohr at 400 Ry) |

Do **not** invent absolute total-energy digits or a CP2K trajectory energy
series. When a CP2K AIMD twin is recorded, replace the status block with
version string + input digest + measured \(\delta\).

### Correctness gates (hard-coded in `reg_gpw-bomd-water`)

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| Relative total-energy drift \(\delta\) | \(\le 0.05\) (envelope); measured \(\approx 1.2\times 10^{-5}\) | n/a (no binary) | **0.05** | \(\max_t |E-E_0|/\max(1,|E_0|)\); A5 of closed `gpw-bomd-water` spec |
| PE after warm-up | finite; \(< -100\) eV | n/a | bound | PBE water O(−10²) eV; A4 |
| KE | finite; \(\ge 0\); non-trivial after warm-up | n/a | — | A4 |
| Soft O–H ceiling | both \(r_{\mathrm{OH}} < 1.8\) Å | n/a | **1.8 Å** | No dissociation on short traj (A6) |
| Coordinates | finite over traj | n/a | — | Trajectory non-explosion (A6) |
| Absolute \(E_{\mathrm{tot}}\) | **not pinned** | — | — | Envelope only; force table is `gpw-forces-water` |

Geometry, cutoff, SCF tolerances, \(\Delta t\), step count: table in
**Method** / REFERENCE (match the regression binary).

Related unit surface (not the e2e pin): `tests/cpu/gpw/test_gpw_bomd.cpp` —
≥2 MD steps, finite PE/KE, cheap 200 Ry settings.

## Performance

Source: REFERENCE.md dual-gate summary; queue/handoff
``,
``.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| NVE drift foothold (AIMD) | — | ≈ **35** (login-class, OMP=4, 2026-07-23; 1+4 steps @ 200 Ry) | **perf_exempt** | n/a | No offline CP2K binary for multi-step AIMD fair wall |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step AIMD |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |
| Regression wall budget | ≤ 60 s (spec A10; login-node class CPU) — measured ≈ 35 s |

When a CP2K twin becomes available, record wall times for the same step count
/ comparable settings and require
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) or document blocked-speed after
one optimize pass.

Note: single-point force wall vs CP2K is already measured in
`gpw-forces-water` (ratio **3.56**, within_10×) — that is ENERGY_FORCE, not
this multi-step BOMD twin.

## Limitations

- **No CP2K absolute energy / drift twin** until an offline binary pin is
  recorded (oracle path + `h2o_bomd_mt.inp` documented; not invented).
- **Cheaper cutoff than the force pin.** 200 Ry / `tol=1e-5` / 60 SCF iters —
  enough for the NVE envelope inside ≤60 s, not production AIMD accuracy or
  absolute E/F vs CP2K at the forces-pin settings (400 Ry / 1e-7).
- **Short traj only.** 1 warm-up + 4 production steps; not equilibration,
  thermostats, or statistics.
- **Fixed cubic cell NVE.** No barostat; GPW rejects box changes after
  `build()`. Cell stress is a separate slug (`gpw-stress-water`).
- **Closed-shell Γ RKS.** No UKS, k-points, hybrid, or GAPW on this page.
- **Follow-ons:** `xl-bomd-water` (dep: this slug ✓); production thermostatted
  AIMD; CP2K AIMD twin when binary available; tighter cutoff once wall budget
  allows.
- Analytic forces are product (`compute_forces = true`); stress write
  (`compute_stress`) and multi-rank GPW MD are out of this slug’s scope.

## References

- Offline pin: [`regressions/gpw/gpw-bomd-water/REFERENCE.md`](../../../regressions/gpw/gpw-bomd-water/REFERENCE.md)
- Regression source: [`regressions/gpw/gpw-bomd-water/reg_gpw-bomd-water.cpp`](../../../regressions/gpw/gpw-bomd-water/reg_gpw-bomd-water.cpp)
- Force absolute twin: [`gpw-forces-water`](gpw-forces-water.md) /
  `regressions/gpw/gpw-forces-water/`
- Unit bind test: `tests/cpu/gpw/test_gpw_bomd.cpp`
- Suggested CP2K input: `tools/cp2k_oracle/inputs/h2o_bomd_mt.inp`
- Oracle: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Narrative GPW / MD: [ch09-gpw.md](../ch09-gpw.md),
  [ch10-classical-md.md](../ch10-classical-md.md)
- Related PW foothold: [pw-bomd-h](pw-bomd-h.md)
- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD chapter
- VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
  Quickstep
- Closed acceptance (git history): `.claude/specs/gpw-bomd-water/spec.md`
  (A1–A10; deleted after merge on wave 2)
