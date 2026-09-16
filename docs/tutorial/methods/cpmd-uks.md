# Car–Parrinello MD on Γ-UKS — O₂ triplet NVE (GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-uks` |
| Regression | `regressions/dynamics/cpmd-uks/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{…, compute_forces=true, n_alpha, n_beta}, mu_au}` bound with `set_evaluator_ref`; dual propagators `cpu::electronic::Cpmd` (α/β). There is no `cpmd_dt`: `MD` pushes the integrator's dt into the wrapper (`TimestepProvider` → `TimestepConsumer`) |
| Roadmap | MD roadmap Car–Parrinello (Γ-UKS) |
| Dual gate | Correctness: \(E_{\mathrm{cons}}\) drift envelope + fixed-density + spin/structure; speed **`perf_exempt`** |
| Wave | CP2K capability loop — 2026-07-26 |
| Oracle | Offline CP2K twin input only — **never** live in ctest |

## Summary

**Car–Parrinello molecular dynamics (CPMD)** with **collinear Γ-point unrestricted
Kohn–Sham (UKS)** on the GPW path: nuclei integrate with NVE `VelocityVerlet`;
occupied MO coefficients for **α and β** are classical fictitious degrees of
freedom with mass \(\mu\), advanced by electronic velocity-Verlet + **RATTLE**
on \(C_\sigma^\top S C_\sigma = I\). After one cold UKS SCF seed, production
force steps evaluate dual-spin KS energy/forces at the **current** densities
without re-converging SCF (`iters==0`).

This is **not** Born–Oppenheimer MD (BOMD: SCF every step), **not** XL-BOMD
(`xl-bomd-uks`: auxiliary densities + SCF), and **not** second-generation CP
(`second-gen-cpmd-uks`: ASPC + partial SCF, no fictitious mass). The literature
conserved quantity is

\[
E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}},
\quad
T_{\mathrm{e}} = T_{\mathrm{e}}^\alpha + T_{\mathrm{e}}^\beta
= \tfrac12\mu\sum_{\sigma\in\{\alpha,\beta\}}\sum_i \lVert\dot c_{i\sigma}\rVert^2,
\]

with \(T_{\mathrm{e}}\) exposed as `last_electronic_ke_ev()` [eV] and
\(E_{\mathrm{KS}}\) on `frame.meta.get(kEnergy)` [eV]. Public energy remains
**eV** on Frame; internal KS work may cite Hartree.

Product gate: `reg_cpmd-uks` — isolated O₂ triplet NVE twin family (geometry
shared with `gpw-uks-o2-gamma` / `gpw-uks-forces` / `xl-bomd-uks`). Offline
CP2K twin input: `tools/cp2k_oracle/inputs/cpmd_uks_o2_nve.inp`. Closed-shell
CPMD twins: `cpmd-water`, `cpmd-rattle-water`.

## Method

**Extended Lagrangian (collinear UKS).** Separate fictitious electronic
dynamics per spin channel \(\sigma\):

\[
\mathcal{L}
= \sum_I \tfrac12 M_I \dot R_I^2
+ \sum_{\sigma}\sum_i \tfrac12 \mu \lVert \dot c_{i\sigma}\rVert^2
- E_{\mathrm{KS}}[C_\alpha,C_\beta,R]
- \sum_{\sigma}\text{constraints}\bigl(C_\sigma^\top S C_\sigma = I\bigr).
\]

For occupancy-1 UKS, the S-tangent electronic force on the occupied block is
of the product form

\[
\mu\,\ddot C_\sigma
= -(I - S C_\sigma C_\sigma^\top)\,(2 H_\sigma C_\sigma)
\]

(with electronic RATTLE restoring orthonormality after the unconstrained
Verlet step). Nuclei feel analytic GPW forces at the instantaneous electronic
state — **fixed density** after the cold seed, so production must not re-enter
the SCF loop.

**Γ-UKS GPW setup (this pin).** Spin densities \(\rho_\alpha,\rho_\beta\);
Hartree from total density; spin-resolved **PADE LSD** XC; dual KS matrices
\(H_\alpha,H_\beta\). Occupations are explicit members `n_alpha` / `n_beta`
(not a multiplicity bag). O GTH-q6 → 6 valence e⁻ per O → 12 valence e⁻;
multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\). Isolated electrostatics:
Martyna–Tuckerman (`nonperiodic=true` / CP2K `PSOLVER MT`). Molecule must be
**cell-centered**.

**Literature.**

- Car & Parrinello, *Phys. Rev. Lett.* **55**, 2471 (1985)
- H. C. Andersen, *J. Comput. Phys.* **52**, 24 (1983) — RATTLE
- Marx & Hutter, *Ab Initio Molecular Dynamics* (CUP) — CP conserved energy
- Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE LSD
- Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson

**Key exclusions (spec / product scope).**

- No k-point CPMD; Γ only
- No OT seed (`use_ot` mutually exclusive with `cpmd`)
- No simultaneous `xl_bomd` or `second_gen_cp`
- No electronic thermostat / NVT-CP
- No live CP2K binary energy twin in ctest (literature \(E_{\mathrm{cons}}\)
  envelope + structural / spin gates)
- Absolute single-point E/F parity is owned by `gpw-uks-o2-gamma` /
  `gpw-uks-forces`, not this trajectory envelope

## Implementation

Public surface only (`cpu::Frame`, evaluator members; **no** `*Config` bag,
**no** `make_*` / `build_*` factories):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// O₂ triplet, R = 1.2075 Å, centered in 12 Å cubic MT cell
cpu::Frame frame(2);
constexpr double L = 12.0;
constexpr double R = 1.2075;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 8;
    frame.span(kAtom_mass)[i] = 15.999;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_x)[i] = 0.5 * L;
    frame.span(kAtom_y)[i] = 0.5 * L;
    frame.span(kAtom_vx)[i] = 0.0;
    frame.span(kAtom_vy)[i] = 0.0;
    frame.span(kAtom_vz)[i] = 0.0;
}
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
// Antisymmetric stretch [Å/fs] — no COM translation
frame.span(kAtom_vz)[0] = -0.0008;
frame.span(kAtom_vz)[1] = 0.0008;
frame.box = cpu::Box::cubic(L);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

constexpr double dt = 0.1; // fs — nuclear and electronic share one step
cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
    /*mix*/ 0.4, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
gpw.n_alpha = 7; // multiplicity 3, 12 valence e⁻
gpw.n_beta = 5;
// AO residual on DZVP O₂ is stiffer than PW textbook ~400 a.u. mu is a
// constructor argument of the CP wrapper; the mode bits are private.
cpu::electronic::CarParrinello eval{std::move(gpw), /*mu_au=*/2000.0};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});  // the ONE owner of dt —
                                                  // MD pushes it to the wrapper
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, n_steps);

// CP conserved energy [eV]:
const double e_cons = frame.meta.get(kKinetic)
                    + frame.meta.get(kEnergy)
                    + eval.last_electronic_ke_ev();
// Spin diagnostics live on the wrapped evaluator:
const double s2 = eval.method().last_s2(); // Löwdin ⟨S²⟩ from last full SCF
```

Build (GPW needs FFTW3) and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_cpmd-uks
  ./build/regressions/reg_cpmd-uks
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator members | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| CPMD propagator | `include/atomiverse/cpu/electronic/cpmd.h` |
| CP wrapper | `include/atomiverse/cpu/electronic/car_parrinello.h` |
| Dual α/β path | `src/cpu/gpw/gpw_evaluator.cpp` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_gpw_cpmd*` (guards + RATTLE) |
| Regression | `regressions/dynamics/cpmd-uks/reg_cpmd-uks.cpp` |
| Pin | `regressions/dynamics/cpmd-uks/REFERENCE.md` |
| CP2K twin input | `tools/cp2k_oracle/inputs/cpmd_uks_o2_nve.inp` |

## Validation (CP2K / literature envelope)

Source: `regressions/dynamics/cpmd-uks/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Gate class: **literature conserved-energy envelope** +
structural / spin diagnostics — same inventory as `cpmd-rattle-water` and
`xl-bomd-uks`. Absolute total-energy twin vs CP2K for the static UKS surface
lives on `reg_gpw-uks-o2-gamma` (not re-asserted here).

| Observable | Pin / gate | Tol / window | Notes |
| ---------- | ---------: | ------------ | ----- |
| \(E_{\mathrm{cons}}\) relative drift | literature envelope | **≤ 0.10** | After Te-engaging baseline step; scale \(\max(1,\lvert E_0\rvert)\) |
| Production SCF iters | **0** | exact | Fixed-density dual-KS — must not be a BOMD rename |
| PE after warm-up | bound | **&lt; −100 eV** | Sanity on cold UKS seed |
| Nuclear KE after warm-up | nontrivial | **&gt; 1e−12 eV** | Stretch initial velocities |
| \(\langle S^2\rangle\) (warm-up SCF) | triplet window | **[1.5, 2.5]** | Ideal \(S(S+1)=2\); not refreshed on fixed-density steps |
| O–O distance | structural | **[0.9, 1.6] Å** | No explode / collapse on short NVE |
| Finite trajectory | PE, KE, Te, coords | `isfinite` | All production chunks |

**Setup (must match regression / twin input).**

| Item | Value |
| ---- | ----- |
| System | Isolated O₂ triplet, \(R=1.2075\) Å |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) — cell-centered |
| Cell | 12 × 12 × 12 Å, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`nonperiodic` / `PSOLVER MT`) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q6 (ATV); twin lists GTH-PADE-q6 |
| XC | PADE LSD (`XcKind::PADE` / CP2K `&XC_FUNCTIONAL PADE` + `LSD`) |
| Cutoff | **200 Ry** (envelope run; absolute E/F pin uses 400 Ry elsewhere) |
| Spin | UKS, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| dt | **0.1 fs** (nuclear = electronic) |
| \(\mu\) | **2000 a.u.** |
| Protocol | 1 SCF seed + 1 CP baseline (Te on) + 4 CP drift steps |

**Offline CP2K twin input** (operator only; not ctest):

```text
tools/cp2k_oracle/inputs/cpmd_uks_o2_nve.inp
```

Sketch: `RUN_TYPE MD`, `ENSEMBLE NVE`, `STEPS 10`, `TIMESTEP 0.1`,
`MULTIPLICITY 3`, `LSD`, PADE, DZVP-MOLOPT-GTH, MT 12 Å cell, same O–O
geometry as the static UKS family. CP2K version string for any future binary
energy twin must be recorded in REFERENCE when re-pinned; current gate does
**not** require a numerical \(E_{\mathrm{tot}}\) match from that MD run.

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| O₂ CPMD-UKS short NVE | — | envelope ≤60 s design | **perf_exempt** | n/a | No CP2K CPMD-UKS wall twin on the worker host; literature drift + structural/spin gates |

Gate: dual-gate speed is **`perf_exempt`** (documented in
the case `REFERENCE.md` and queue row `cpmd-uks`). Same class
as `cpmd-rattle-water` / `xl-bomd-uks` — trajectory conservation, not a fair
wall ratio.

## Limitations

| In | Out |
| -- | --- |
| Collinear Γ-UKS GPW + dual fictitious e⁻ DOF | k-point CPMD; non-collinear |
| Electronic RATTLE (pos + vel) | Nuclear RATTLE (`FixRattle`); electronic thermostat |
| Short O₂ NVE drift envelope | Multi-ps production; live CP2K in ctest |
| PADE LSD + MT isolated | Hybrid XC CPMD; OT-seeded CP |
| RKS CPMD remains on same `cpmd` flag | Simultaneous XL / second-gen CP flags |

Follow-ons (other queue rows): `second-gen-cpmd-uks` (ASPC + partial SCF UKS),
electronic thermostat / NVT-CP, tighter absolute twin if a CP2K CP binary
energy protocol is standardized offline.

## References

- Offline pin: `regressions/dynamics/cpmd-uks/REFERENCE.md`
- Regression: `regressions/dynamics/cpmd-uks/reg_cpmd-uks.cpp`
- CP2K oracle contract: `tools/cp2k_oracle/README.md`
- Twin input: `tools/cp2k_oracle/inputs/cpmd_uks_o2_nve.inp`
- Related tutorials: [cpmd-water](cpmd-water.md),
  [cpmd-rattle-water](cpmd-rattle-water.md),
  [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md),
  [xl-bomd-water](xl-bomd-water.md)
- MD roadmap: `docs/roadmap/molecular-dynamics.md` §AIMD / Car–Parrinello
- Car & Parrinello, *PRL* **55**, 2471 (1985); Andersen, *J. Comput. Phys.*
  **52**, 24 (1983); Marx & Hutter (CUP)
