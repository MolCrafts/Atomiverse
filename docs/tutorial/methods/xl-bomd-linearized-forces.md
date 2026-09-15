# Zero-SCF linearized XL-BOMD forces — H₂O NVE (GPW shadow BO)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-linearized-forces` |
| Regression | `regressions/dynamics/xl-bomd-linearized-forces/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{…}, k, cpu::electronic::XlBomdLinearized{}}` bound with `set_evaluator_ref`; propagator `cpu::electronic::XlBomd` |
| Roadmap | [molecular-dynamics.md](../../roadmap/molecular-dynamics.md) XL-BOMD residual closed 2026-07-26 |
| Dual gate | Correctness A1–A4 unit (`test_cpu_gpw_xl_bomd`) + A5 e2e NVE drift; speed **`perf_exempt`** |
| Wave | CP2K capability loop — harness wave 12 (2026-07-26) |
| Oracle | Offline CP2K twin sketch only (`tools/cp2k_oracle/inputs/xl-bomd-linearized-forces/`); never live in ctest |

## Summary

**Zero-SCF linearized-functional XL-BOMD** evaluates nuclear forces of the
Niklasson **shadow Born–Oppenheimer** potential after each ionic step, with
**no iterative SCF** on warm MD steps. The outer loop is the MD preset
(`cpu::md::MD` + NVE `VelocityVerlet`); the force slot binds a
`cpu::electronic::XlBomdForce` built from a `cpu::gpw::GPWEvaluator` with a
dissipation order \(K\) and the **`XlBomdLinearized`** tag that selects the
shadow-force path.

Cold seed (first step): full Γ-RKS SCF seeds the dissipative XL auxiliary
density \(n = P_{\mathrm{aux}}\). Warm steps: build \(H[n]\) at fixed
auxiliary density (`max_iter = 0`), one generalized diagonalization
\(D = q[n]\), write the shadow energy

\[
\mathcal{E}(R,n) = \mathrm{Tr}\!\bigl(H[n]\,D\bigr)
  - \mathrm{Tr}\!\bigl(H[n]\,n\bigr)
  + E_{\mathrm{KS}}[n],
\]

and assemble Hellmann–Feynman forces of \(\mathcal{U}(R,n)\) at
\((D, H[n])\) via `gpw_forces_at_density`. Then `XlBomd::advance(D)` with
Table-I \(K=5\). Public energy remains **eV** on `frame.meta`; internal
Ha/Ry are method-private.

This page closes the residual that SCF-converged XL product paths
([xl-bomd-water](xl-bomd-water.md), Γ/multi-k RKS/UKS) explicitly excluded:
**zero-SCF linearized forces**. The e2e pin is the **literature shadow-XL /
NVE envelope** plus **`last_result().iters == 0`** on warm steps — not a
golden Atomiverse total energy and not a measured CP2K AIMD wall.

## Method

### XL auxiliary density (same recursion as SCF XL)

An auxiliary AO density matrix is a **dynamical variable**, not a Kolafa ASPC
extrapolation (`cpu::electronic::ASPCPredictor`). After each warm electronic step the
propagator advances

\[
P_{n+1}
  = 2\,P_{n}
  - P_{n-1}
  + \kappa\,\bigl(D_{n} - P_{n}\bigr)
  + \alpha \sum_{m=0}^{K} c_m\, P_{n-m},
\]

with \(D_n\) the one-shot diagonalization density (linearized path) or the
SCF density (SCF-XL path). Default \(K=5\) (Niklasson et al., *JCP* **130**,
214109 (2009), Table I):

| Symbol | Value |
| ------ | ----- |
| \(\kappa\) | \(1.82\) |
| \(\alpha\) | \(0.018\) |
| \(c_0\ldots c_5\) | \(\{-6,\,14,\,-8,\,-3,\,4,\,-1\}\) |

Table-I coefficients for \(K\in[3,9]\) are unit-pinned in
`test_cpu_gpw_xl_bomd`. Time-reversible form: Niklasson, *PRL* **100**, 123004
(2008). Dissipation removes the secular energy drift of plain guess reuse.

### Shadow / linearized free-energy functional

When \(n = P_{\mathrm{aux}}\) is **not** self-consistent, the ordinary KS
energy \(E_{\mathrm{KS}}[n]\) and the Hellmann–Feynman force of that density
are inconsistent with nuclear dynamics. The linearized (shadow)
functional

\[
\mathcal{E}(R,n)
  = \mathrm{Tr}\!\bigl(H[n]\,D\bigr)
  - \mathrm{Tr}\!\bigl(H[n]\,n\bigr)
  + E_{\mathrm{KS}}[n],
  \qquad D = q[n]
\]

recovers \(E_{\mathrm{KS}}\) exactly at the fixed point \(D = n\), and
\(\mathrm{Tr}\!\bigl(H[n]\,(D-n)\bigr)\) is the first-order residual heat
when \(D \neq n\). Forces are taken of \(\mathcal{U}(R,n)\) at fixed \(n\)
(density \(D\), KS matrix \(H[n]\)). Literature: Niklasson & Cawkwell, *PRB*
**86**, 174308 (2012); Niklasson, *JCP* **147**, 054103 (2017).

Unit pin A3 (`ShadowEnergyRecoversAtFixedPoint`): when \(D = n\),
\(\mathcal{E} = E_{\mathrm{KS}}\); when \(D \neq n\),
\(\mathcal{E} - E_{\mathrm{KS}} = \mathrm{Tr}(H(D-n))\).

### Warm electronic step (zero SCF)

| Stage | Action |
| ----- | ------ |
| 0. Cold seed | Full SCF → `xl_->advance(D_scf)`; history becomes ready |
| 1. Fixed-\(n\) KS | `solver_->max_iter = 0` with `guess_P = n_aux` → \(H[n]\), \(E_{\mathrm{KS}}[n]\) |
| 2. One diag | Generalized Fock diagonalization → \(D = q[n]\), \(C_{\mathrm{occ}}\) |
| 3. Shadow \(E\) | \(\mathcal{E} = \mathrm{Tr}(H D) - \mathrm{Tr}(H n) + E_{\mathrm{KS}}[n]\) → `kEnergy` [eV] |
| 4. Forces | `gpw_forces_at_density(D, C_occ, H[n])` → eV/Å |
| 5. XL advance | `xl_->advance(D)`; `last_result().iters = 0`, `converged = true` |

No DIIS mixing loop, no residual threshold iteration on warm steps.

### GPW force engine (this pin)

- **GPW (Quickstep family):** Gaussian AO basis, density collocation, FFT
  Poisson Hartree, grid XC, GTH PP — Lippert, Hutter, Parrinello, *Mol. Phys.*
  **92**, 477 (1997); VandeVondele et al., *Comput. Phys. Commun.* **167**, 103
  (2005).
- **XC:** PBE (closed-shell RKS).
- **Boundary:** isolated H₂O, Martyna–Tuckerman Poisson (`nonperiodic = true`).
- **Nuclear MD (NVE):** Velocity Verlet; total
  \(E = E_{\mathrm{pot}} + E_{\mathrm{kin}}\) [eV] conserved up to a bounded
  non-secular envelope (Frenkel & Smit §4.3.4; Marx & Hutter BOMD). Drift
  tolerance is **looser** than full-SCF XL (`0.10` vs `0.05`) because of
  one-shot residual heat.

### Primary observable (envelope)

Let \(E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)\) [eV], with \(E_0\) after
the cold seed:

\[
\delta = \max_t \frac{\lvert E(t) - E_0\rvert}{\max\!\bigl(1,\,\lvert E_0\rvert\bigr)}
  \le 0.10.
\]

Additional gates (`reg_xl-bomd-linearized-forces`):

- Cold seed: `last_result().iters > 0` and `converged`
- Warm steps: **`iters == 0`** and `converged == true`
- PE finite and bound (\(< -100\) eV after seed)
- KE finite, \(\ge 0\), non-trivial after seed
- Soft O–H: both \(r_{\mathrm{OH}} < 1.8\) Å after production
- All coordinates finite

### Exclusions (this slug)

- Not SCF-converged XL only (`xl-bomd-water`, `xl-bomd-uks`, multi-k XL) —
  those keep iterative SCF to `tol`
- Not UKS linearized XL ([xl-bomd-linearized-uks](xl-bomd-linearized-uks.md))
  or multi-k linearized XL
  ([xl-bomd-linearized-kpoints](xl-bomd-linearized-kpoints.md)) — shipped
  as follow-ons; this page stays the Γ-RKS foothold
- Not OT + XL, CPMD, or second-gen CP simultaneous with XL (mutually exclusive)
- Not ASPC predictor as a substitute for Niklasson XL
- Not hybrid / meta-GGA XC
- Not a measured fair-wall twin vs CP2K AIMD (see Performance)
- Not molecular post-HF (PySCF loop) or classical electrochem MD (MetalWalls)

## Implementation

Public surface only (`cpu::Frame`, concrete evaluator members; **no**
`*Config` bag, **no** `make_*` / `build_*` factories):

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

// Isolated H₂O — CP2K getting-started geometry (Å); twin of xl-bomd-water
cpu::Frame frame(3);
// ... Z, masses, coords as in REFERENCE; COM-preserving H stretch vx ...
frame.box = cpu::Box::cubic(10.0);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

cpu::gpw::GPWEvaluator gpw(
    {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
// The third argument is a tag type, not a bool: it selects the zero-SCF
// shadow-BO force path (and turns forces on) at construction.
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5,
                                  cpu::electronic::XlBomdLinearized{}};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/ 0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, 1);   // cold seed: full SCF, iters > 0
// production: iters == 0; |E(t)-E0|/scale ≤ 0.10; soft O–H
```

`build()` **throws** `std::invalid_argument` when:

| Condition | Reason |
| --------- | ------ |
| UKS (`n_alpha + n_beta > 0`) | Γ RKS foothold only |
| multi-k mesh | shipped follow-on ([xl-bomd-linearized-kpoints](xl-bomd-linearized-kpoints.md)) |
| `use_ot`, or a `CarParrinello` / `SecondGenCp` wrapper | Mutually exclusive electronic dynamics |

Two older failure modes are now unreachable from the public surface: the
`XlBomdLinearized` constructor is the only way to select shadow forces, and it
sets both the XL auxiliary density and `compute_forces` itself — so "linearized
without XL" and "linearized without forces" cannot be expressed.

Unit gates A1–A2 in `test_cpu_gpw_xl_bomd`.

### How to run the regression

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake --build build -j$(nproc) --target atomiverse reg_xl-bomd-linearized-forces
  cmake --build build --target run_reg_xl-bomd-linearized-forces
'
```

Relevant sources:

| Layer | Path |
| ----- | ---- |
| Propagator | `include/atomiverse/cpu/electronic/xl_bomd.h`, `src/cpu/electronic/xl_bomd.cpp` |
| XL wrapper | `include/atomiverse/cpu/electronic/xl_bomd_force.h` (`XlBomdForce`, `XlBomdLinearized`) |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` (private XL mode bits) |
| Warm shadow path | `src/cpu/gpw/gpw_evaluator.cpp` (zero-SCF block after XL ready) |
| Forces at density | `src/cpu/gpw/internal/gpw_forces.cpp` (`gpw_forces_at_density`) |
| Unit A1–A4 | `tests/cpu/gpw/test_gpw_xl_bomd.cpp` |
| E2E A5 | `regressions/dynamics/xl-bomd-linearized-forces/` |

## Validation (vs CP2K / literature)

Source: `regressions/dynamics/xl-bomd-linearized-forces/REFERENCE.md`
(offline pin; never live CP2K in ctest).

### System

Identical twin of `xl-bomd-water` (only `xl_bomd_linearized` differs):

| Quantity | Value |
| -------- | ----- |
| Species | H₂O (O + 2 H), isolated / MT |
| Geometry | CP2K getting-started (table below) |
| Masses | O 15.999 amu; H 1.008 amu |
| Cell | cubic \(L = 10\) Å; nonperiodic |
| XC | PBE |
| Basis / PP | DZVP-MOLOPT-GTH; GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 200 Ry |
| SCF (cold only) | `tol = 1e-5`, `max_iter = 60`, DIIS diagonalization |
| XL-BOMD | \(K=5\); `xl_bomd_linearized = true` |
| Ensemble | NVE; \(\Delta t = 0.5\) fs |
| \(v_0\) | COM-preserving H stretch \(v_x(\mathrm{H1/H2}) = \pm 0.0015\) Å/fs |
| Steps | 1 cold seed + 4 production (2 chunks × 2) |

| Atom | x | y | z (Å) |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

### Accuracy table

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.10\) | literature shadow-XL / NVE envelope | 0.10 | looser than SCF-XL 0.05 (residual heat) |
| Warm `last_result().iters` | \(= 0\) | zero iterative SCF | — | product diagnostic |
| Warm `converged` | true | stationary for shadow functional | — | |
| Cold seed `iters` | \(> 0\), converged | full SCF seed | — | history not ready until seed |
| PE after seed | \(< -100\) eV | bound H₂O | — | |
| KE after seed | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(r_{\mathrm{OH}}\) after production | both \(< 1.8\) Å | soft bond window | — | no dissociation |
| Shadow \(\mathcal{E}\) at \(D=n\) | \(= E_{\mathrm{KS}}\) | Niklasson 2012/2017 | unit | A3 |
| Table-I \(\kappa,\alpha,c_k\) (\(K=5\)) | exact | Niklasson 2009 Table I | unit | `test_cpu_gpw_xl_bomd` |
| Illegal combos throw | build-time | no silent degrade | unit | A1–A2 |

### Offline CP2K twin (sketch only)

| Field | Value |
| ----- | ----- |
| Input | `tools/cp2k_oracle/inputs/xl-bomd-linearized-forces/h2o_xl_linearized_nve.inp` |
| Intent | Short NVE Quickstep MD, ASPC extrapolation, **`MAX_SCF 1`** as a low-budget stand-in for one-shot KS; same H₂O / PBE / DZVP-MOLOPT-GTH / 200 Ry / MT cell as ATV |
| Status | **Unexecuted** on pin host 2026-07-26 (no `cp2k` binary / Lmod module) |
| When binary available | Re-run offline; replace pin section with CP2K version + measured drift + fair wall |

CP2K `EXTRAPOLATION ASPC` + low `MAX_SCF` is **not** identical to Niklasson
shadow \(\mathcal{E}\) with true single-diagonalization linearized forces.
The twin is a method-family sketch for inventory and future fair-bench, not a
bit-identical density history.

## Performance

Source: the case `REFERENCE.md` (wave 12),
`REFERENCE.md` Perf gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | -----------------------------------------: | ---- | ------ |
| zero-SCF linearized XL AIMD | n/a | n/a | **perf_exempt** | n/a | No offline CP2K binary for fair wall of multi-step AIMD (2026-07-26 inventory) |

Gate: \(r \le 10\) unless `perf_exempt`. This slug is **exempt** for the same
reason as other XL AIMD rows: correctness is the NVE drift envelope + zero warm
SCF iterations, not a wall-clock ratio.

Note: warm steps are cheaper than full-SCF XL (one \(H\) build + one
diagonalization vs iterative DIIS). That cost saving is intentional product
value; it is not claimed as a measured fair-bench ratio here.

## Limitations

- **Envelope, not absolute energy.** Broken XL recursion or wrong shadow
  formula destroys zero-iter / drift gates; a correct path is not a
  high-precision total-energy twin of CP2K Quickstep.
- **Γ RKS foothold.** UKS and multi-k linearized shadow forces were
  refused at `build()`; productize as separate queue rows if needed.
- **Looser drift window.** \(\delta \le 0.10\) vs SCF-XL \(0.05\) reflects
  one-shot residual heat; do not tighten without re-pinning literature/CP2K.
- **Cold seed still pays full SCF.** Linearized savings apply only after
  `xl_->ready()`.
- **Perf debt.** Fair wall vs CP2K deferred until a CP2K binary is available
  offline; do not invent ratios.
- **Combinatorial gaps out of scope:** linearized XL + OT, + \(k\), + UKS,
  + smearing, + DFT+U — separate rows if ever productized.
- **Not CPMD / SGCP.** Fictitious orbital dynamics and ASPC+partial-SCF
  correctors are different electronic propagators.

## References

- Offline pin: `regressions/dynamics/xl-bomd-linearized-forces/REFERENCE.md`
- Regression: `regressions/dynamics/xl-bomd-linearized-forces/reg_xl-bomd-linearized-forces.cpp`
- CP2K twin sketch: `tools/cp2k_oracle/inputs/xl-bomd-linearized-forces/h2o_xl_linearized_nve.inp`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Prior product paths: [xl-bomd-water](xl-bomd-water.md) (SCF-converged Γ RKS);
  [xl-bomd-uks-kpoints](xl-bomd-uks-kpoints.md) (multi-k UKS SCF XL);
  [gpw-bomd-water](gpw-bomd-water.md) (plain BOMD foothold)
- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012)
- A. M. N. Niklasson, *J. Chem. Phys.* **147**, 054103 (2017)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
