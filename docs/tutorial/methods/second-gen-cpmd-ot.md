# Second-generation CP with OT corrector — H₂O NVE (GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `second-gen-cpmd-ot` |
| Regression | `regressions/dynamics/second-gen-cpmd-ot/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{…, use_ot=true}, aspc_k, corrector_max_iter}` bound with `set_evaluator_ref` |
| Roadmap | MD roadmap SGCP OT corrector |
| Dual gate | Correctness: NVE drift + method-identity (partial OT budget); speed **`perf_exempt`** |
| Wave | CP2K capability loop — wave 9 (2026-07-26) |
| Oracle | Offline CP2K sketch only — never live in ctest |

## Summary

**Second-generation Car–Parrinello MD (SGCP)** (Kühne et al., *PRL* **98**,
066401 (2007)) replaces the classical CP fictitious electronic mass with a
**Kolafa ASPC density predictor** plus a **partial SCF corrector** each MD
step. This product page pins the **closed-shell Γ-point GPW** path where the
corrector is **Orbital Transformation (OT)** rather than diagonalization/DIIS:
ASPC predicts \(P\), the OT manifold is warm-started from that density
(natural-orbital seed), OT runs for at most `corrector_max_iter` steps, and
analytic nuclear forces are taken at the post-corrector electronic state.

**Not BOMD** — production steps cap OT iterations well below a full
self-consistency budget.
**Not classical CPMD** — no \(\mu\,\ddot C\), no \(T_e\), no electronic RATTLE.
**Not DIIS SGCP** — that is `second-gen-cpmd-water` (`use_ot=false`).
**Not XL-BOMD** — ASPC extrapolation, not Niklasson auxiliary Verlet
(`cpu::electronic::XlBomdForce`).

Public energy remains eV on `frame.meta.get(kEnergy)`; Hartree method channel
is on `eval.last_energy_hartree()` / `last_result()`.

## Method

### SGCP outer structure

Nuclei advance with ordinary NVE velocity-Verlet. At each force evaluation:

1. **Seed phase** (ASPC history not yet full): full-budget OT SCF, record
   density \(P_n\). Needs \(K+1\) records for ASPC order \(K\).
2. **Production (warm):** ASPC **predict** \(\tilde P_{n+1}\) from the history
   → OT corrector with `max_iter = corrector_max_iter` → analytic
   forces at the post-corrector density → ASPC **correct** + **record**.

Kolafa ASPC (order \(K\); Kolafa, *J. Comput. Chem.* **25**, 335 (2004)) is the
same predictor primitive used elsewhere (`cpu::electronic::ASPCPredictor`).
Typical product default: \(K=3\) (the wrapper's `aspc_k` default); this
regression uses \(K=1\) (two seed densities) for a short drift gate.

### OT corrector (closed-shell)

OT (VandeVondele & Hutter, *JCP* **118**, 4365 (2003)) optimizes occupied
coefficients on \(C^\top S C = I\) without a dense generalized eigensolve each
outer step. Parameterization for reference \(C_0\) and \(S\)-tangent \(x\):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\top S x)^{1/2}.
\]

Closed-shell band-energy gradient consistent with \(P = 2\,C C^\top\):

\[
g = (I - S C_0 C_0^\top)\,(4 H C_0).
\]

On the SGCP path the ASPC density becomes an OT **warm start** (natural-orbital
seed from predicted \(P\)); the corrector budget replaces full BOMD
convergence. Forces use the OT-exported orbitals / \(H_{\mathrm{KS}}\), not a
`max_iter=0` fixed-density re-eval (OT forbids that route — the solver must
export \(H_{\mathrm{KS}}\) / \(S\) when needed).

### Conserved quantity (gate)

Short NVE twin reports nuclear kinetic + electronic potential energy on Frame:

\[
E_{\mathrm{cons}} = T_{\mathrm{n}} + E_{\mathrm{KS}},
\quad
\frac{\lvert E_{\mathrm{cons}} - E_0\rvert}{\max(1,\,|E_0|)} \le 0.10.
\]

Soft relative drift (vs full-BO \(0.05\)) absorbs partial-SCF heat on a few
production steps. There is **no** classical CP \(T_e\) term.

### Method map (siblings)

| Slug | Electronics | Corrector | Fictitious mass |
| ---- | ----------- | --------- | --------------- |
| `gpw-bomd-water` | full SCF each step | full DIIS/OT budget | no |
| `xl-bomd-water` | Niklasson XL aux \(P\) + full SCF | full | no |
| `cpmd-water` / `cpmd-rattle-water` | VV + Löwdin/RATTLE on \(C\) | none after seed | yes \(\mu\) |
| `second-gen-cpmd-water` | ASPC + partial SCF | **diag/DIIS** | no |
| **`second-gen-cpmd-ot`** | ASPC + partial OT | **OT** | no |
| `second-gen-cpmd-uks` | dual ASPC + partial SCF | diag UKS | no |
| UKS OT + SGCP | — | residual (refused at `build()`) | — |

### Exclusions (binding)

- Γ-UKS + `SecondGenCp` + `use_ot` — throws at `build()` (no dual
  natural-orbital OT seed product path yet).
- Mutual exclusion: a `CarParrinello` or `XlBomdForce` wrapper with
  `SecondGenCp` (one AIMD wrapper per evaluator).
- k-points, smearing, Langevin / NVT-CP, electronic thermostat.
- Classical CP hybrid (`CarParrinello` + `use_ot`) — refused.
- CP2K binary energy twin on the pin host — literature envelope + method
  identity only (`perf_exempt`); offline input kept for re-pin.

## Implementation

Public surface only (`cpu::Frame` + `GPWEvaluator` members; **no** `*Config`
bags, **no** factories). Match the regression:

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

// Isolated water, CP2K getting-started geometry [Å]
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

constexpr double dt = 0.5; // fs — BOMD-class step (vs classical CP ~0.1)
constexpr int corrector_budget = 5;
cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
gpw.use_ot = true;                        // OT partial-SCF corrector
// SGCP is the wrapper type; ASPC order K (needs K+1 seeds) and the corrector
// budget are its constructor arguments.
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/corrector_budget};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});
sim.set_evaluator_ref(eval);
sim.build();

sim.run(frame, /*n_seed=*/2);  // warm ASPC (K+1)
sim.run(frame, 1);             // first warm production → baseline E0
// assert eval.last_result().iters <= corrector_budget
sim.run(frame, 2);             // drift window
// E_cons = kEnergy + kKinetic; relative |dE|/scale <= 0.10
```

Build and run the product gate:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_second-gen-cpmd-ot
  cmake --build build --target run_reg_second-gen-cpmd-ot
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/electronic/second_gen_cp.h` | `SecondGenCp` wrapper (ASPC order + corrector budget) |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Public `use_ot`; private SGCP mode bits |
| `src/cpu/gpw/gpw_evaluator.cpp` | ASPC predict/correct, OT force path, guards |
| `include/atomiverse/cpu/electronic/aspc_predictor.h` | Kolafa ASPC primitive |
| `include/atomiverse/cpu/gpw/internal/ot.h` | OT parameterization (internal) |
| `tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp` | Unit: RKS OT accepted; UKS+OT rejected; mutex with the CP / XL wrappers |
| `regressions/dynamics/second-gen-cpmd-ot/` | E2E NVE gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/second-gen-cpmd-ot/h2o_sgcp_ot_nve.inp` | Offline CP2K twin sketch |

## Validation (vs CP2K)

Source:
[`regressions/dynamics/second-gen-cpmd-ot/REFERENCE.md`](../../../regressions/dynamics/second-gen-cpmd-ot/REFERENCE.md).
Oracle: offline **CP2K** input sketch (`EXTRAPOLATION ASPC`,
`EXTRAPOLATION_ORDER 1`, `&OT`, tight `MAX_SCF`); **never** live in ctest.
No `cp2k` binary required on the pin host (same class as
`second-gen-cpmd-water` / `cpmd-water`) — re-pin when inventory allows.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O, charge 0, closed shell |
| Geometry [Å] | O (5, 5, 5); H (5.7586, 5, 5.5043); H (4.2414, 5, 5.5043) |
| Cell | 10 Å cubic, `PERIODIC NONE`, Martyna–Tuckerman |
| Basis / PP | DZVP-MOLOPT-GTH; GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| XC | PBE |
| Cutoff | **200 Ry** |
| dt | **0.5 fs** |
| ASPC \(K\) | **1** |
| Corrector | OT, `corrector_max_iter = 5` |
| Seed | 2 full-OT steps (\(K+1\)) |
| Production | 1 baseline + 2 drift steps |

**Hard-coded acceptance gates:**

| Quantity | Atomiverse / pin | CP2K twin | Tol / gate | Notes |
| -------- | ---------------: | --------: | ---------- | ----- |
| Relative \(\lvert E-E_0\rvert/\mathrm{scale}\) | ≤ **0.10** | envelope (re-pin) | ≤ 0.10 | soft vs full-BO 0.05 |
| Production OT `iters` | ≤ **5** once ASPC warm | `MAX_SCF 5` sketch | ≤ `corrector_max_iter` | method identity |
| O–H distances | < **1.8 Å** | — | < 1.8 | no bond explode |
| PE after seed | < **−100 eV** | — | bound | Frame eV |
| Seed KE | > 0 | — | non-trivial | stretch-mode seed |

Offline oracle sketch (host with CP2K module; **not** ctest):

```bash
# See tools/cp2k_oracle/README.md — never invoke from ctest
cp2k.psmp -i tools/cp2k_oracle/inputs/second-gen-cpmd-ot/h2o_sgcp_ot_nve.inp \
          -o /tmp/cp2k_oracle_h2o_sgcp_ot_nve.out
# Record version + .ener drift + fair wall when available
```

## Performance

| Twin | Wall | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ---: | ----------------------------------------: | ---- | ------ |
| CP2K SGCP+OT MD | — | **perf_exempt** | n/a | no binary on pin host; NVE drift gate class |
| ATV `reg_second-gen-cpmd-ot` | short traj ≤60 s e2e budget | — | — | not a fair OT single-point twin |

Gate class: **`perf_exempt`** (handoff wave 9 — “NVE drift”), same as sibling
dynamics slugs `cpmd-water`, `second-gen-cpmd-water`, `second-gen-cpmd-uks`,
`xl-bomd-water`. Fair OT-only single-point wall is documented on
[`ot-minimizer-water`](ot-minimizer-water.md) (\(r\sim 9\)); do not ratio the
MD binary against a static SCF twin.

## Limitations

- Closed-shell RKS Γ-point only for OT+SGCP; **UKS OT + SGCP remains residual**
  (`build()` throws; dual natural-orbital seed not productized).
- Short trajectory + soft drift tol — not a long-run microcanonical audit or
  CP2K energy-class match claim.
- ASPC order and corrector budget are explicit members; defaults differ from
  the regression (\(K=3\) default vs \(K=1\) pin).
- Dense OT preconditioner cost (see `ot-minimizer-water`); not sparse OT.
- Mutually exclusive with `cpmd` / `xl_bomd`; requires `compute_forces=true`.
- No MPI multi-rank industrial gate on this slug.
- Offline CP2K re-pin deferred until a binary is available on the worker host.

## References

- Offline pin: `regressions/dynamics/second-gen-cpmd-ot/REFERENCE.md`
- Regression: `regressions/dynamics/second-gen-cpmd-ot/reg_second-gen-cpmd-ot.cpp`
- CP2K oracle sketch: `tools/cp2k_oracle/inputs/second-gen-cpmd-ot/h2o_sgcp_ot_nve.inp`
- Roadmap: `docs/roadmap/molecular-dynamics.md` (SGCP OT corrector product)
- Sibling pages: [ot-minimizer-water](ot-minimizer-water.md),
  [cpmd-water](cpmd-water.md), [xl-bomd-water](xl-bomd-water.md),
  [gpw-bomd-water](gpw-bomd-water.md)
- Kühne, T. D.; Krack, M.; Mohamed, F. R.; Parrinello, M. *Phys. Rev. Lett.*
  **98**, 066401 (2007), doi:10.1103/PhysRevLett.98.066401
- Kolafa, J. *J. Comput. Chem.* **25**, 335 (2004), doi:10.1002/jcc.10385
- VandeVondele, J.; Hutter, J. *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154
