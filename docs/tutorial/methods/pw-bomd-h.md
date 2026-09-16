# Plane-wave BOMD — H atom NVE foothold

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-bomd-h` |
| Regression | `regressions/pw/pw-bomd-h/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::pw::PWEvaluator{PW}.with_gradient(true)` at `md::Phase::force` |
| Roadmap | MD product path in [ch08-plane-wave.md](../ch08-plane-wave.md) / [ch10-classical-md.md](../ch10-classical-md.md) |
| Dual gate | Correctness A4–A6; speed **`perf_exempt`** (no CP2K twin wall) |

## Summary

Born–Oppenheimer molecular dynamics (BOMD) with **plane-wave** electronic
structure on every ionic step: the outer loop is the MD preset
(`cpu::md::MD` + NVE `VelocityVerlet`); the force/energy slot binds
`cpu::pw::PWEvaluator` with analytic ionic forces enabled
(`with_gradient(true)`). Public units are eV, Å, fs, eV/Å; the PW method keeps
Ry/Ha on `last_result()` only.

This page documents the **dep-safe foothold** regression: a **single H** at the
origin of a cubic cell with the **empty-`upf_path` toy local potential** (same
foothold as `h_force_*`). Lattice symmetry keeps \(\mathbf{F}\approx\mathbf{0}\);
a hand-set \(v_x\) supplies non-trivial KE so the NVE total-energy envelope is
meaningful. The pin is **not** a golden Atomiverse total energy and **not** a
live CP2K AIMD twin — it is the literature symplectic / BOMD conservation
envelope plus the symmetry force residual. Production Si/UPF BOMD and GPW BOMD
are separate queue rows.

## Method

**Level.** Classical Born–Oppenheimer MD: nuclei integrated with a fixed-step
symplectic Velocity Verlet; electrons re-solved self-consistently each step
(no Car–Parrinello fictitious electronic dynamics, no XL-BOMD / ASPC extended
Lagrangian).

**Electronic structure.** Concrete method `cpu::pw::PW` (plane-wave DFT
SCF: Poisson → XC → \(V_{\mathrm{eff}}\) → Davidson → density mix). Empty
`upf_path` selects the **toy local** \(V_{\mathrm{loc}}\) (no UPF file, no
Kleinman–Bylander nonlocal). Forces from `pw_forces_analytic`: local
Hellmann–Feynman + Ewald ionic contribution [eV/Å].

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
gates: PE finite; KE finite and \(\ge 0\); coordinates finite; warm-up
\(|\mathbf{F}| < 0.5\) eV/Å (symmetry residual, shared with
`h_force_analytic`).

**System (regression settings).**

| Quantity | Value |
| -------- | ----- |
| Species | 1 × H (\(Z=1\)) at the origin |
| Mass | 1.008 amu |
| Cell | cubic \(L = 6\) Å |
| Pseudopotential | empty `upf_path` (toy local \(V\)) |
| \(E_{\mathrm{cut}}\) | 5 Ry |
| k-mesh | Γ-only (`k_mesh = 1`) |
| SCF | `tol = 1e-4`, `max_iter = 12`, `mix_beta = 0.5` |
| Ensemble | NVE (no thermostat / barostat) |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | \(v_x = 0.002\) Å/fs (hand-set; `InitVelocity` COM-zeros \(N=1\)) |
| Steps | 1 warm-up + 10 production (chunked) |

**Why not H₂?** With the toy empty-UPF local potential the bare nuclear
Coulomb of two protons is not cancelled by a physical bonding curve; a short
NVE traj dissociates violently and is a poor drift pin. Product force math is
covered by `h_force_analytic` / FD unit tests; this slug pins **MD binding +
NVE conservation** on the symmetry-stable single-H foothold.

**Excluded.** GPW BOMD (`gpw-bomd-water`); XL-BOMD / CPMD; Si bulk UPF long
trajectories; production thermostats; true GGA / spin / metallic smearing PW
(other queue slugs); live CP2K from ctest; MPI multi-rank; public `*Config`
bags / `make_*` factories; changes to analytic force math.

## Implementation

Public surface only: `cpu::Frame`, concrete `cpu::pw::PW` members, MD preset,
explicit force-phase binding. Headers:
`include/atomiverse/cpu/dynamics/md.h`,
`velocity_verlet.h`,
`include/atomiverse/cpu/planewave/pw_solver.h`,
`pw_evaluator.h`. Unit gate (not e2e):
`tests/cpu/planewave/test_pw_bomd.cpp`.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;

cpu::Frame frame(1);
frame.span(kAtom_type)[0] = 1;
frame.span(kAtom_mass)[0] = 1.008;
frame.span(kAtom_tag)[0] = 1;
frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] = 0.0;
// Hand-set velocity [Å/fs] — InitVelocity COM-zeros N=1 and kills KE.
frame.span(kAtom_vx)[0] = 0.002;
frame.span(kAtom_vy)[0] = 0.0;
frame.span(kAtom_vz)[0] = 0.0;
frame.box = cpu::Box::cubic(6.0);

cpu::pw::PW pw;
pw.ecut_ry = 5.0;
pw.k_mesh = 1;
pw.tol = 1e-4;
pw.max_iter = 12;
pw.mix_beta = 0.5;
// empty upf_path → toy V_loc

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.on(md::Phase::force)
    .add(cpu::pw::PWEvaluator{std::move(pw)}.with_gradient(true));
sim.build();
sim.run(frame, 1);   // warm-up: populate PE/KE/forces
// E0 = frame.meta.get(kEnergy) + frame.meta.get(kKinetic)  [eV]
sim.run(frame, 10);  // production NVE
```

Rules (architecture):

- Parameters are **members** of `cpu::pw::PW` — no public `*Config` bag.
- Force binding is explicit `on(md::Phase::force).add(...)` — no categorical
  electronic-structure setter.
- `with_gradient(true)` (default **false**) fills per-atom forces [eV/Å] via
  analytic local HF + Ewald.
- Frame is the sole I/O currency (`blocks` + `meta` + `box`).

```bash
# configure with plane-wave + regressions (CPU login node pattern)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON -DATV_BUILD_CUDA=OFF \
  -DATV_BUILD_REGRESSIONS=ON -DBLA_VENDOR=OpenBLAS
cmake --build build --target reg_pw-bomd-h -j"$(nproc)"
./build/regressions/reg_pw-bomd-h
```

## Validation (CP2K)

Source: [`regressions/pw/pw-bomd-h/REFERENCE.md`](../../../regressions/pw/pw-bomd-h/REFERENCE.md)
(offline pin; **never** live CP2K in ctest). Oracle tooling:
[`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md).

### Offline CP2K status

| Field | Value |
| ----- | ----- |
| CP2K binary on pin host | **unavailable** (no `cp2k` / `cp2k.psmp` / `cp2k.sopt`, no CP2K Lmod module on Arrhenius login-class inventory 2026-07-23) |
| Twin design (when available) | Short NVE AIMD for single H (or H₂ with a real PP) in a comparable cell; re-run offline via `tools/cp2k_oracle/`; record version + measured drift envelope |
| Current pin basis | Literature NVE/BOMD energy conservation + symmetry force residual |

Do **not** invent absolute total-energy digits. When a CP2K twin is recorded,
replace the status block with version string + input digest + measured
\(\delta\).

### Correctness gates (hard-coded in `reg_pw-bomd-h`)

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| Relative total-energy drift \(\delta\) | \(\le 0.05\) (envelope) | n/a (no binary) | **0.05** | \(\max_t |E-E_0|/\max(1,|E_0|)\); A5 of closed `pw-bomd-h` spec |
| PE, KE | finite; KE \(\ge 0\) | n/a | — | A4; after warm-up and each production chunk |
| Warm-up \(\lvert\mathbf{F}\rvert\) | \(< 0.5\) eV/Å | n/a | **0.5 eV/Å** | Cubic high-symmetry residual; same bound as `h_force_analytic` (A6) |
| Coordinates | finite over traj | n/a | — | Trajectory non-explosion (A6) |
| Absolute \(E_{\mathrm{tot}}\) | **not pinned** | — | — | Toy PP; envelope only |

Geometry, cutoff, k-mesh, SCF tolerances, \(\Delta t\), step count: table in
**Method** / REFERENCE (match the regression binary).

Related unit surface (not the e2e pin): `tests/cpu/planewave/test_pw_bomd.cpp`
— ≥2 MD steps, finite PE/KE, residual force bound.

## Performance

Source: REFERENCE.md dual-gate summary; queue/handoff
``,
``.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| NVE drift foothold | — | — | **perf_exempt** | n/a | No offline CP2K (or QE) binary on pin host for a fair wall twin |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K/QE binary for a fair wall twin; correctness is the NVE drift envelope |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |
| Regression wall budget | ≤ 60 s (spec A10; login-node class CPU) |

When a CP2K twin becomes available, record wall times for the same step count
/ comparable settings and require
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) or document blocked-speed after
one optimize pass.

## Limitations

- **Toy PP only.** Empty `upf_path` is a non-physical local potential for
  foothold force/MD binding — not production hydrogen chemistry.
- **Single ion, \(\mathbf{F}\approx\mathbf{0}\).** Free-particle NVE with
  near-flat PE; does not stress multi-atom bonding, cell stress, or
  long-range force cancellation on a molecule.
- **No CP2K absolute energy / drift twin** until an offline binary pin is
  recorded (oracle path documented; not invented).
- **Short traj / loose SCF.** 5 Ry, 12 SCF iters, `tol=1e-4` — enough for the
  envelope, not production AIMD accuracy.
- **Follow-ons:** Si/UPF BOMD after bulk energy re-pins (`pw-lda-si` and
  friends); GPW BOMD water (`gpw-bomd-water`); XL-BOMD / CPMD queue rows.
- Analytic forces are product (`with_gradient(true)`); stress write
  (`with_stress`) and multi-rank PW MD are out of this slug’s scope.

## References

- Offline pin: [`regressions/pw/pw-bomd-h/REFERENCE.md`](../../../regressions/pw/pw-bomd-h/REFERENCE.md)
- Regression source: [`regressions/pw/pw-bomd-h/reg_pw-bomd-h.cpp`](../../../regressions/pw/pw-bomd-h/reg_pw-bomd-h.cpp)
- Force symmetry foothold: `regressions/pw/h_force_analytic/`
- Unit bind test: `tests/cpu/planewave/test_pw_bomd.cpp`
- Oracle: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Narrative PW / MD: [ch08-plane-wave.md](../ch08-plane-wave.md),
  [ch10-classical-md.md](../ch10-classical-md.md)
- Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4
- Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — BOMD chapter
- Closed acceptance (git history): `.claude/specs/pw-bomd-h/spec.md`
  (H-atom NVE foothold; A1–A10; deleted after merge on wave 1)
