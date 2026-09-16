# Second-generation Car–Parrinello MD — H₂O NVE (ASPC + partial SCF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `second-gen-cpmd-water` |
| Regression | `regressions/dynamics/second-gen-cpmd-water/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{…}, aspc_k, corrector_max_iter}` bound with `set_evaluator_ref`; ASPC primitive `cpu::electronic::ASPCPredictor` |
| Roadmap | MD roadmap second-generation CP |
| Dual gate | Correctness envelope + method-identity (partial SCF); speed **`perf_exempt`** |
| Wave | CP2K capability loop — harness wave 1 **done** (merged 2026-07-25) |
| Oracle | **CP2K** offline sketch (`tools/cp2k_oracle/inputs/second_gen_cpmd_water_nve.inp`); literature envelope on pin host |

## Summary

**Second-generation Car–Parrinello MD** (SGCP / 2nd-gen CP; Kühne, Krack,
Mohamed & Parrinello, *Phys. Rev. Lett.* **98**, 066401 (2007)) keeps nuclei on
ordinary Velocity Verlet while electrons use a **Kolafa ASPC density predictor**
plus a **partial SCF corrector** each MD step. Production steps do **not** run
full BOMD SCF: after the ASPC history is warm, SCF is capped at the wrapper's
`corrector_max_iter` (≪ the evaluator's `max_iter`). There is **no** fictitious
electronic mass and **no** electronic kinetic energy \(T_e\).

This is distinct from:

| Method | Electronics each MD step | Public type |
| ------ | ------------------------ | ----------- |
| BOMD (`gpw-bomd-water`) | Full SCF to `tol` / `max_iter` | bare `GPWEvaluator`, forces on |
| XL-BOMD (`xl-bomd-water`) | Full SCF; Niklasson auxiliary density as guess | `cpu::electronic::XlBomdForce{gpw, k}` |
| Classical CPMD (`cpmd-water`) | Fixed-density KS + electronic Verlet; \(T_e\) | `cpu::electronic::CarParrinello{gpw, mu_au}` |
| **SGCP (this page)** | ASPC predict + **budgeted** SCF corrector | `cpu::electronic::SecondGenCp{gpw, aspc_k, corrector_max_iter}` |

Scope: closed-shell RKS Γ-point GPW diagonalization path. UKS, OT, and a
simultaneous `CarParrinello` / `XlBomdForce` wrapper are refused loudly at
`build()`.

## Method

**Level.** Born–Oppenheimer-like nuclear MD with a second-generation CP
electronic corrector: the density is extrapolated in time (ASPC), then refined
with a short SCF, then analytic nuclear forces are taken at that density.

**ASPC predictor (Kolafa, *J. Comput. Chem.* **25**, 335 (2004)).** Order \(K\)
stores \(K+1\) previous densities and builds a Lagrange extrapolation
\(P^{\mathrm{pred}}\) for the next ionic step. Until history is full,
`predict()` returns false and the evaluator runs full SCF (seed phase). After
warm-up, each production force step:

1. ASPC → initial density guess
2. SCF with iteration cap `corrector_max_iter`
3. Optional corrector blend of SCF and predicted \(P\) (Kolafa)
4. Record density into ASPC history
5. Analytic forces at the post-corrector density → `frame.span(kAtom_f*)` [eV/Å]

**Nuclear dynamics.** NVE `VelocityVerlet` on nuclei only; conserved quantity
for the pin is the ordinary MD total

\[
E = E_{\mathrm{pot}} + E_{\mathrm{kin}}
\quad\text{(public units eV on Frame)},
\]

not the classical CP \(E_{\mathrm{cons}} = T_n + T_e + E_{\mathrm{KS}}\).

**Primary observables (hard-coded gates):**

\[
\delta = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)}
  \le 0.10,
\]

with \(E_0\) after the first ASPC-warm production step; production
`last_result().iters ≤ corrector_max_iter`; soft O–H ceiling
\(r_{\mathrm{OH}} < 1.8\) Å; PE after seed \(< -100\) eV.

Soft \(\delta\) vs BOMD’s 0.05 reflects partial-SCF heat on a short trajectory.

**System (regression settings — match `reg_second-gen-cpmd-water`).**

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started (same twin family as `gpw-bomd-water` / `xl-bomd-water` / `cpmd-water`) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; `nonperiodic = true` (Martyna–Tuckerman) |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 200 Ry |
| SCF envelope | `tol = 1e-5`, `max_iter = 60` (seed), `mix = 0.3` |
| ASPC \(K\) | **1** (needs \(K+1=2\) seed densities) |
| Corrector | `corrector_max_iter = 5` |
| Ensemble | NVE (no thermostat / barostat) |
| \(\Delta t\) | 0.5 fs (BOMD-class; classical CP uses ~0.1 fs) |
| Initial velocity | COM-preserving H stretch \(v_x(\mathrm{H1})=+0.0015\), \(v_x(\mathrm{H2})=-0.0015\) Å/fs |
| Steps | 2 seed (full SCF) + 1 baseline production + 2 drift production |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

**CP2K twin surface (offline only).** Quickstep GPW with
`EXTRAPOLATION ASPC`, limited `MAX_SCF`, PBE / MOLOPT / GTH / MT — sketch at
`tools/cp2k_oracle/inputs/second_gen_cpmd_water_nve.inp`. Never executed from
ctest.

**Excluded.** Classical fictitious-mass CPMD; Niklasson XL-BOMD; full BOMD
production SCF; UKS / OT / k-points; electronic thermostat; NVT/NPT;
multi-rank AIMD; public `*Config` bags / `make_*` factories; live CP2K in
ctest.

## Implementation

Public surface only: `cpu::Frame`, `cpu::gpw::Kind` / lookup constructors,
`GPWEvaluator` constructor args + members, MD preset, explicit force-phase
binding. Headers: `include/atomiverse/cpu/dynamics/md.h`,
`velocity_verlet.h`, `include/atomiverse/cpu/gpw/gpw_evaluator.h`,
`gaussian_basis.h`, `gth_potential.h`,
`include/atomiverse/cpu/electronic/second_gen_cp.h`,
`include/atomiverse/cpu/electronic/aspc_predictor.h`.
Unit gates: `tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp`.

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
// … fill Z, mass, tags, CP2K getting-started x/y/z, COM-preserving H stretch …
frame.box = cpu::Box::cubic(10.0);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

constexpr double dt = 0.5; // fs
constexpr int corrector_budget = 5;
cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
// Second-generation CP is a wrapper TYPE around the pure evaluator: the ASPC
// order K (K+1 seed densities) and the corrector budget are constructor
// arguments, and the mode bits they set are private on GPWEvaluator.
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/corrector_budget};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});
sim.set_evaluator_ref(eval);   // borrow: last_result() stays readable
sim.build();

sim.run(frame, 2);   // seed: full SCF fills ASPC history (K+1 = 2)
sim.run(frame, 1);   // first ASPC-warm production step → baseline E0
// assert eval.last_result().iters <= corrector_budget
sim.run(frame, 2);   // production drift window
// E = frame.meta.get(kEnergy) + frame.meta.get(kKinetic)  [eV]
```

Rules (architecture):

- Parameters are **constructor arguments / members** — no public `*Config` bag,
  no `make_*` / `build_*` factories. The AIMD mode itself is a wrapper type
  (`cpu::electronic::SecondGenCp`), not a flag on `GPWEvaluator`.
- Force binding is `set_evaluator_ref(eval)` to borrow (or
  `set_evaluator(std::move(eval))` to hand ownership over); the explicit
  `on(md::Phase::force).add_ref(...)` form is still available when you need a
  specific order.
- The wrapper sets `compute_forces = true` itself — analytic nuclear forces are
  the product.
- Mutual exclusion: `build()` throws if combined with a `CarParrinello` or
  `XlBomdForce` wrapper, `use_ot`, or UKS occupations.
- Frame is the sole I/O currency (`blocks` + `meta` + `box`); public energy
  is eV on `kEnergy` / `kKinetic` (Hartree on `last_energy_hartree()` only).
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
  cmake --build build -j"$(nproc)" --target atomiverse \
    test_cpu_gpw_second_gen_cpmd reg_second-gen-cpmd-water
  ctest --test-dir build -R "^test_cpu_gpw_second_gen_cpmd$" --output-on-failure
  cmake --build build --target run_reg_second-gen-cpmd-water
'
```

## Validation (CP2K)

Source:
[`regressions/dynamics/second-gen-cpmd-water/REFERENCE.md`](../../../regressions/dynamics/second-gen-cpmd-water/REFERENCE.md)
(offline pin; **never** live CP2K in ctest). Oracle tooling:
[`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md). Twin
input (when a binary is available):
`tools/cp2k_oracle/inputs/second_gen_cpmd_water_nve.inp`
(`EXTRAPOLATION ASPC`, limited `MAX_SCF`, PBE / MOLOPT / GTH / MT).

### Offline CP2K status

| Field | Value |
| ----- | ----- |
| CP2K binary on pin host | **unavailable** (no `cp2k` / Lmod module on Arrhenius login-class inventory 2026-07-25; same as `cpmd-water` / `xl-bomd-water`) |
| Twin design (when available) | Short NVE Quickstep GPW SGCP on the same H₂O cell; re-run offline via `tools/cp2k_oracle/`; record version + `.ener` drift + fair wall |
| Current pin basis | Literature SGCP method identity (ASPC + partial SCF) + NVE energy envelope + trajectory diagnostics |
| Related absolute force twin | `gpw-forces-water` vs **CP2K 2026.1** (ENERGY_FORCE, not multi-step SGCP) |

Do **not** invent absolute total-energy digits or a CP2K trajectory energy
series. When a CP2K SGCP twin is recorded, replace the status block with
version string + input digest + measured \(\delta\).

### Correctness gates (hard-coded in `reg_second-gen-cpmd-water`)

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| Relative total-energy drift \(\delta\) | \(\le 0.10\) (envelope) | n/a (no binary) | **0.10** | Soft vs BOMD 0.05 / CP 0.10; partial-SCF heat |
| Production SCF iters | \(\le 5\) once ASPC warm | n/a | **`corrector_max_iter`** | Method identity: not full BOMD |
| PE after seed | finite; \(< -100\) eV | n/a | bound | PBE water O(−10²) eV |
| KE | finite; \(\ge 0\); non-trivial after seed | n/a | — | COM-preserving H stretch |
| Soft O–H ceiling | both \(r_{\mathrm{OH}} < 1.8\) Å | n/a | **1.8 Å** | No dissociation on short traj |
| Coordinates | finite over traj | n/a | — | Trajectory non-explosion |
| Absolute \(E_{\mathrm{tot}}\) | **not pinned** | — | — | Envelope + identity only |

Geometry, cutoff, SCF, ASPC \(K\), corrector budget, \(\Delta t\), step counts:
**Method** table / REFERENCE (match the regression binary).

Related unit surface (not the e2e pin):
`tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp` — `build()` rejects UKS / OT /
`cpmd` / `xl_bomd` / forces-off / bad \(K\) or budget; accepts default members.

## Performance

Source: REFERENCE.md dual-gate summary; queue/handoff
``,
``.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| NVE SGCP (ASPC + partial SCF) | — | ≤ **60** budget (login-class) | **perf_exempt** | n/a | No offline CP2K binary for multi-step AIMD fair wall |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on pin host for a fair wall twin of multi-step SGCP AIMD |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |
| Regression wall budget | ≤ 60 s (short seed + production at 200 Ry) |

When a CP2K twin becomes available, record wall times for the same step count
/ comparable settings and require
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) or document blocked-speed after
one optimize pass.

Note: single-point force wall vs CP2K remains on `gpw-forces-water` (ENERGY_FORCE
ratio **3.56**, within_10×) — not this multi-step SGCP twin.

## Limitations

- **No CP2K absolute energy / drift twin** until an offline binary pin is
  recorded (oracle path + `second_gen_cpmd_water_nve.inp` documented; not invented).
- **Partial-SCF heat.** Drift tol is softer than full BOMD; short traj only
  (2 seed + 1 baseline + 2 production).
- **ASPC \(K=1\) in the regression** (fast seed \(K+1=2\)); product default
  member is \(K=3\) / corrector 5 — callers may raise \(K\) for production.
- **Diagonalization corrector**, not OT single-step.
- **Closed-shell Γ RKS only.** UKS / OT / k-points refused or deferred.
- **Fixed cubic cell NVE.** No barostat; no electronic / nuclear thermostat.
- **Mutual exclusion** with classical CPMD and XL-BOMD on the same evaluator.
- **Follow-ons:** CP2K SGCP twin when binary available; longer traj / NVT;
  OT-corrector path; UKS SGCP if ever productized.

## References

- Offline pin: [`regressions/dynamics/second-gen-cpmd-water/REFERENCE.md`](../../../regressions/dynamics/second-gen-cpmd-water/REFERENCE.md)
- Regression source: [`regressions/dynamics/second-gen-cpmd-water/reg_second-gen-cpmd-water.cpp`](../../../regressions/dynamics/second-gen-cpmd-water/reg_second-gen-cpmd-water.cpp)
- Unit gates: `tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp`
- Suggested CP2K input: `tools/cp2k_oracle/inputs/second_gen_cpmd_water_nve.inp`
- ASPC primitive: `include/atomiverse/cpu/electronic/aspc_predictor.h`
- SGCP wrapper: `include/atomiverse/cpu/electronic/second_gen_cp.h`
- Sibling AIMD pages: [gpw-bomd-water](gpw-bomd-water.md),
  [xl-bomd-water](xl-bomd-water.md), [cpmd-water](cpmd-water.md)
- Roadmap: [`docs/roadmap/molecular-dynamics.md`](../../roadmap/molecular-dynamics.md) §AIMD
- Oracle: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- CP2K exercise `exercises:common:sgcp` — `EXTRAPOLATION ASPC` + limited SCF
- Closed acceptance (git history): `.claude/specs/second-gen-cpmd-water/` (deleted
  after merge on harness wave 1, 2026-07-25)
