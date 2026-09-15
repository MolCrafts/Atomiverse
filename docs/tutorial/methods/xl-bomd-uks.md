# XL-BOMD — extended-Lagrangian BOMD on Γ-UKS (O₂ triplet NVE, GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-uks` |
| Regression | `regressions/dynamics/xl-bomd-uks/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{…, compute_forces=true, n_alpha, n_beta}, k=5}` bound with `set_evaluator_ref`; propagator class `cpu::electronic::XlBomd` (dual α/β) |
| Roadmap | AIMD residual prose in [dft.md](../../roadmap/dft.md) |
| Dual gate | Correctness A1–A6 unit + A7 e2e envelope; speed **`perf_exempt`** (no CP2K AIMD twin wall) |
| Wave | CP2K capability loop — wave 6 **done** (merged 2026-07-26) |

## Summary

**Extended-Lagrangian Born–Oppenheimer MD** (XL-BOMD; Niklasson, *PRL*
**100**, 123004 (2008); dissipative form Niklasson et al., *JCP* **130**,
214109 (2009)) on **open-shell Γ-UKS** Gaussian-and-plane-waves forces.
Independent auxiliary AO densities \(P^\alpha\) and \(P^\beta\) are
propagated with the same Table-I recursion and serve as SCF initial spin
densities each MD step. This productizes the dual-spin XL path that the
closed-shell foothold `xl-bomd-water` deliberately left out.

Outer loop: `cpu::md::MD` + NVE `VelocityVerlet`. Force/energy slot: a
`cpu::electronic::XlBomdForce` wrapping a `cpu::gpw::GPWEvaluator` with
analytic forces and UKS occupations. SCF still converges to `tol` each step
(**SCF-converged** XL-BOMD). Public energy remains **eV** on `frame.meta`;
Hartree and \(\langle S^2\rangle\) via evaluator accessors only.

Product gate: `reg_xl-bomd-uks` — short O₂ triplet NVE energy-drift
envelope with dual XL spin guesses. Absolute UKS energy / force parity vs
offline CP2K stays on `gpw-uks-o2-gamma` / `gpw-uks-forces`. Speed is
**`perf_exempt`** until a multi-step CP2K AIMD fair wall is recorded.

## Method

**Level.** Classical Born–Oppenheimer MD with symplectic Velocity Verlet on
the nuclei; electrons re-solved self-consistently each step. The XL layer
does **not** replace SCF: it supplies a time-reversible (to \(O(\alpha)\))
initial density guess so long BOMD trajectories do not accumulate the
systematic energy drift of broken time-reversibility from plain guess reuse.

**Dissipative XL recursion (per spin \(\sigma\in\{\alpha,\beta\}\)):**

\[
P^\sigma_{n+1}
  = 2P^\sigma_n - P^\sigma_{n-1}
    + \kappa\,(D^\sigma_n - P^\sigma_n)
    + \alpha \sum_{k=0}^{K} c_k P^\sigma_{n-k},
\]

with \(D^\sigma_n\) the SCF-converged spin density at step \(n\) and
\((\kappa,\alpha,c_k)\) from Table I of Niklasson et al. (2009). Default
\(K=5\):

| \(K\) | \(\kappa\) | \(\alpha\) | \(c_0\ldots c_K\) |
| ----: | ---------: | ---------: | ---------------- |
| 5 | 1.82 | 0.018 | \(\{-6,\,14,\,-8,\,-3,\,4,\,-1\}\) |

Table-I coefficients for all \(K\in[3,9]\) are unit-pinned bitwise in
`test_cpu_gpw_xl_bomd` (shared with the RKS path).

**UKS / LSD electronic structure.** Separate \(\rho_\alpha,\rho_\beta\) and
KS matrices; spin-resolved **PADE (LSD)** XC; dual-spin DIIS on the
diagonalization path. Occupations are explicit integer members
`n_alpha` / `n_beta` (no multiplicity bag). O₂ with GTH-PBE-q6 → 6 valence
e⁻ per O → 12 valence e⁻ total; multiplicity 3 ⇒ \(N_\alpha=7\),
\(N_\beta=5\). Löwdin \(\langle S^2\rangle\) after each SCF
(ideal high-spin \(S(S+1)=2\)).

**GPW (Quickstep family).** Contracted Gaussian AO basis; density on a
uniform grid; FFT Poisson Hartree; XC on the grid; GTH analytic PP.
Isolated electrostatics via Martyna–Tuckerman (`nonperiodic = true` /
CP2K `PERIODIC NONE` + `PSOLVER MT`). Molecule must sit **centered** in
the cell. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477
(1997); Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999).

**Primary observable (envelope, not absolute energy):**

\[
\delta = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)}
  \le 0.05,
\]

with \(E(t)=E_{\mathrm{pot}}(t)+E_{\mathrm{kin}}(t)\) [eV] and \(E_0\) after
one warm-up step (XL histories seeded). A broken dual-spin XL
implementation (α-only history, swapped channels, non-density aux) destroys
SCF convergence or this drift envelope immediately.

**System (regression settings).**

| Quantity | Value |
| -------- | ----- |
| Species | O₂ (2 × O), isolated / MT |
| Geometry | \(R = 1.2075\) Å, centered in cubic cell |
| Masses | 15.999 amu each |
| Cell | cubic \(L = 12\) Å; `PERIODIC NONE` / `nonperiodic = true` |
| Spin | UKS triplet, \(n_\alpha=7\), \(n_\beta=5\) (GTH-q6, 12 valence e⁻) |
| XC | PADE (LSD) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| CUTOFF | **200 Ry** (envelope run; absolute E/F pins use 400 Ry) |
| SCF | `tol = 1e-5`, `max_iter = 80`, `mix = 0.4`, DIIS diagonalization |
| XL-BOMD | `K = 5`; dual α/β `XlBomd` propagators |
| Ensemble | NVE (no thermostat / barostat) |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

| Atom | \(x\) | \(y\) | \(z\) (Å) |
| ---- | ----: | ----: | --------: |
| O1 | 6.0000 | 6.0000 | 5.39625 |
| O2 | 6.0000 | 6.0000 | 6.60375 |

**Why 200 Ry?** Absolute UKS energy / force twin lives at 400 Ry on
`gpw-uks-o2-gamma` / `gpw-uks-forces`. Multi-step UKS AIMD at that cost
exceeds the ≤60 s regression budget. 200 Ry keeps the same method family
while staying inside the time gate; this slug gates **dual-spin XL + MD
binding + NVE conservation**, not absolute E/F parity.

**ASPC ≠ XL.** CP2K MD’s default density-guess propagation is Kolafa
**ASPC extrapolation** (`EXTRAPOLATION ASPC`), not Niklasson XL-BOMD. The
oracle twin input uses ASPC as the closest Quickstep MD analogue for
offline comparison; Atomiverse engages the Niklasson auxiliary-density
dynamics via `xl_bomd`. ASPC also exists in-tree as
`cpu::electronic::ASPCPredictor` — a different object.

**Key exclusions.**

| In | Out |
| -- | --- |
| Γ-UKS diagonalization + dual XL | OT (`use_ot` mutually exclusive with `xl_bomd`) |
| SCF-converged XL (guess only) | Zero-SCF linearized-functional XL force |
| Short O₂ NVE envelope | Thermostatted / multi-ps production AIMD |
| Closed-shell twin `xl-bomd-water` | CPMD / second-gen CP (`cpmd`, `second_gen_cp`) |
| Offline CP2K pin design | Live CP2K in ctest; k-point UKS XL; hybrid / GAPW |

## Implementation

Public surface only: `cpu::Frame`, `cpu::gpw::Kind` / lookup constructors,
`GPWEvaluator` constructor args / members (`n_alpha`, `n_beta`), the
`cpu::electronic::XlBomdForce` wrapper (dissipation order \(K\)), MD preset,
`set_evaluator_ref` binding. **No** public `*Config` bag, **no** `make_*` /
`build_*` factories.

Headers: `include/atomiverse/cpu/dynamics/md.h`, `velocity_verlet.h`,
`include/atomiverse/cpu/gpw/gpw_evaluator.h`,
`include/atomiverse/cpu/electronic/xl_bomd_force.h`,
`include/atomiverse/cpu/electronic/xl_bomd.h`,
`gaussian_basis.h`, `gth_potential.h`. Unit gate (not e2e):
`tests/cpu/gpw/test_gpw_xl_bomd.cpp` (Table I, fixed-point, UKS build
allowed, OT+XL refused).

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

// Envelope: 200 Ry / tol 1e-5. UKS occupations give the dual XL spin
// propagators; the XL wrapper supplies the dissipation order.
cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
    /*mix*/ 0.4, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
gpw.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
gpw.n_beta = 5;
cpu::electronic::XlBomdForce eval{std::move(gpw),
                                  /*k=*/5};  // Niklasson 2009 Table-I row

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, 1);   // warm-up: seed XL histories, populate PE/KE
// E0 = frame.meta.get(kEnergy) + frame.meta.get(kKinetic)  [eV]
// s2 = eval.method().last_s2()  — triplet window [1.5, 2.5]
sim.run(frame, 4);   // production NVE
```

The propagator itself is a public primitive (`cpu/electronic/xl_bomd.h`); UKS
engages two instances internally (`xl_` = α, `xl_beta_` = β):

```cpp
cpu::electronic::XlBomd xl(5);  // throws unless 3 <= K <= 9
xl.advance(D_scf_spin);       // first call seeds; later calls propagate
const auto& guess = xl.aux(); // SCF initial-density guess for next step
```

Rules (architecture):

- Parameters are **constructor arguments / members** of `GPWEvaluator` — no
  public `*Config` bag.
- Force binding is `set_evaluator_ref(eval)` (borrow) or
  `set_evaluator(std::move(eval))` (own); `on(md::Phase::force).add_ref(...)`
  remains available when a specific order is needed.
- `n_alpha` / `n_beta` select UKS; inside an `XlBomdForce` they allocate dual
  propagators (unit-gated: build allowed; OT+XL still throws).
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
  cmake --build build -j"$(nproc)" --target atomiverse \
        test_cpu_gpw_xl_bomd reg_xl-bomd-uks
  ctest --test-dir build -R "^test_cpu_gpw_xl_bomd$" --output-on-failure
  cmake --build build --target run_reg_xl-bomd-uks
'
```

## Validation (CP2K)

Source:
[`regressions/dynamics/xl-bomd-uks/REFERENCE.md`](../../../regressions/dynamics/xl-bomd-uks/REFERENCE.md)
(offline pin; **never** live CP2K in ctest). Oracle tooling:
[`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md).
Twin sketch:
`tools/cp2k_oracle/inputs/xl-bomd-uks/o2_uks_xl_bomd_nve.inp`
(Quickstep UKS NVE; `EXTRAPOLATION ASPC` as CP2K MD density-guess analogue).

### Offline CP2K status

| Field | Value |
| ----- | ----- |
| CP2K binary on pin host | **unavailable** (no `cp2k` / `cp2k.psmp` / `cp2k.sopt`, no CP2K Lmod module on Arrhenius login-class inventory **2026-07-26** — same as `xl-bomd-water`) |
| Twin design (when available) | Short NVE Quickstep UKS MD on the same O₂ cell (PADE / DZVP-MOLOPT-GTH / GTH-PBE-q6 / MT / MULT 3); re-run offline via `tools/cp2k_oracle/`; record version + measured drift + fair wall |
| Current pin basis | Literature XL-BOMD / NVE conservation envelope + dual-spin SCF convergence + triplet \(\langle S^2\rangle\) window + finite/bounded trajectory diagnostics |
| Absolute UKS E/F twin | `gpw-uks-o2-gamma`, `gpw-uks-forces` vs offline CP2K (separate slugs) |
| Table I coefficients | Unit-pinned in `test_cpu_gpw_xl_bomd` (shared RKS/UKS math) |

Do **not** invent absolute total-energy digits or a CP2K trajectory energy
series. When a CP2K AIMD twin is recorded, replace the status block with
version string + input digest + measured \(\delta\).

### Correctness gates (hard-coded in `reg_xl-bomd-uks`)

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| Relative total-energy drift \(\delta\) | \(\le 0.05\) (envelope) | n/a (no binary) | **0.05** | \(\max_t |E-E_0|/\max(1,|E_0|)\); A7 e2e |
| SCF converged each step | dual XL spin guesses | n/a | — | Broken α-only / channel-swap fails immediately |
| PE after warm-up | finite; \(< -100\) eV | n/a | bound | O₂ PADE GPW O(−10²) eV |
| KE | finite; \(\ge 0\); non-trivial after warm-up | n/a | — | Antisymmetric stretch seeds KE |
| \(\langle S^2\rangle\) | \([1.5,\,2.5]\) after warm-up and production | n/a | window | Triplet Löwdin window |
| Soft O–O window | \(0.9 < r_{\mathrm{OO}} < 1.6\) Å | n/a | Å | No dissociation / collapse on short traj |
| Coordinates | finite over traj | n/a | — | Trajectory non-explosion |
| Absolute \(E_{\mathrm{tot}}\) | **not pinned** | — | — | Envelope only; E/F tables are UKS single-point slugs |

### Unit surface (`test_cpu_gpw_xl_bomd`)

| Check | What it pins |
| ----- | ------------ |
| A1–A5 | Table-I coefficients, fixed point on constant density, bounded response under oscillatory driving, RKS `guess_P` plumbing |
| A6 | `xl_bomd` + UKS occupations **allowed** at `build()`; `xl_bomd` + OT **refused**; invalid \(K\) refused |

Geometry, cutoff, SCF tolerances, \(\Delta t\), step count: table in
**Method** / REFERENCE (match the regression binary).

## Performance

Source: REFERENCE.md dual-gate summary; queue/handoff
``,
``.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| O₂ UKS NVE XL envelope | — | (login-class; ≤60 s budget) | **perf_exempt** | n/a | No offline CP2K binary for multi-step UKS AIMD fair wall (2026-07-26) |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step UKS AIMD |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |
| Regression wall budget | ≤ 60 s (login-node class CPU) |

When a CP2K twin becomes available, record wall times for the same step count
/ comparable settings and require
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) or document blocked-speed after
one optimize pass.

Single-point UKS energy wall vs CP2K (not this multi-step twin) is already
measured on `gpw-uks-o2-gamma` (ratio \(\approx 0.98\)) — ENERGY only, not
this XL-BOMD trajectory.

## Limitations

- **No CP2K absolute energy / drift twin** until an offline binary pin is
  recorded (oracle path + `o2_uks_xl_bomd_nve.inp` documented; not invented).
- **Cheaper cutoff than the E/F pins.** 200 Ry / `tol=1e-5` / 80 SCF iters —
  enough for the NVE envelope inside ≤60 s, not production AIMD accuracy or
  absolute E/F vs CP2K at the UKS single-point settings (400 Ry).
- **Short traj only.** 1 warm-up + 4 production steps; not equilibration,
  thermostats, or statistics.
- **Fixed cubic cell NVE.** No barostat; GPW rejects box changes after
  `build()`.
- **Γ-UKS diagonalization only.** No k-point XL, no OT+XL, no hybrid/GAPW,
  no zero-SCF linearized XL force.
- **ASPC twin is not bitwise XL.** CP2K sketch uses ASPC extrapolation; ATV
  uses Niklasson Table-I dynamics. Fair wall when available compares
  **workflow class** (open-shell BOMD with density-guess propagation), not
  coefficient-level identity.
- **Follow-ons:** CP2K AIMD twin wall when binary available; tighter cutoff
  once wall budget allows; multi-ps open-shell production AIMD; mutual
  exclusion with CPMD / second-gen CP remains intentional.

## References

- Offline pin: [`regressions/dynamics/xl-bomd-uks/REFERENCE.md`](../../../regressions/dynamics/xl-bomd-uks/REFERENCE.md)
- Regression source: [`regressions/dynamics/xl-bomd-uks/reg_xl-bomd-uks.cpp`](../../../regressions/dynamics/xl-bomd-uks/reg_xl-bomd-uks.cpp)
- Closed-shell XL twin: [xl-bomd-water](xl-bomd-water.md) /
  `regressions/dynamics/xl-bomd-water/`
- UKS energy / forces absolute twins: [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md),
  `regressions/gpw/gpw-uks-forces/`
- Unit dual-spin / Table-I gate: `tests/cpu/gpw/test_gpw_xl_bomd.cpp`
- Public propagator: `include/atomiverse/cpu/electronic/xl_bomd.h`
- Suggested CP2K input: `tools/cp2k_oracle/inputs/xl-bomd-uks/o2_uks_xl_bomd_nve.inp`
- Oracle: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Narrative GPW / MD: [ch09-gpw.md](../ch09-gpw.md),
  [ch10-classical-md.md](../ch10-classical-md.md)
- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) —
  Table I coefficients
- Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE LSD
- Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- Closed acceptance (git history): `.claude/specs/xl-bomd-uks/` (deleted
  after merge on wave 6)
