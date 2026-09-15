# Zero-SCF linearized XL-BOMD on Γ-UKS — O₂ triplet NVE (GPW shadow BO)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-linearized-uks` |
| Regression | `regressions/dynamics/xl-bomd-linearized-uks/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{…, n_alpha, n_beta}, k=5, cpu::electronic::XlBomdLinearized{}}` bound with `set_evaluator_ref`; dual α/β `cpu::electronic::XlBomd` propagators |
| Roadmap | follow-on of the `xl-bomd-linearized-forces` residual (UKS refused there at `build()`) |
| Dual gate | Correctness A1–A3 unit (`test_cpu_gpw_xl_bomd`) + A4 e2e `reg_xl-bomd-linearized-uks`; speed **`perf_exempt`** |
| Oracle | Offline CP2K twin sketch only (`tools/cp2k_oracle/inputs/xl-bomd-linearized-uks/`); never live in ctest |

## Summary

**Zero-SCF linearized-functional XL-BOMD on open-shell Γ-UKS.** Nuclear
forces of the Niklasson **shadow Born–Oppenheimer** potential are evaluated
with **no iterative SCF** on warm MD steps, on collinear spin-polarized GPW
(dual α/β densities). The outer loop is the MD preset (`cpu::md::MD` + NVE
`VelocityVerlet`); the force slot binds a `cpu::electronic::XlBomdForce`
built from a `cpu::gpw::GPWEvaluator` with UKS occupations
(`n_alpha = 7`, `n_beta = 5`), dissipation order \(K = 5\), and the
**`XlBomdLinearized`** tag that selects the shadow-force path.

Cold seed (first step): full dual-spin UKS SCF seeds **both** spin XL
auxiliary densities \(n^\sigma = P^\sigma_{\mathrm{aux}}\). Warm steps:
build \(H^\sigma[n]\) at fixed auxiliary densities (`max_iter = 0`), one
generalized diagonalization **per spin** \(D^\sigma = q^\sigma[n]\)
(occupancy 1), write the dual-spin shadow energy

\[
\mathcal{E}(R, n^\alpha, n^\beta)
  = \sum_{\sigma}\Bigl[\mathrm{Tr}\!\bigl(H^\sigma[n]\,D^\sigma\bigr)
  - \mathrm{Tr}\!\bigl(H^\sigma[n]\,n^\sigma\bigr)\Bigr]
  + E_{\mathrm{KS}}[n^\alpha, n^\beta],
\]

and assemble Hellmann–Feynman forces of \(\mathcal{U}(R,n^\alpha,n^\beta)\)
at \((D^\sigma, H^\sigma[n])\) via the dual-spin fixed-density overload of
`gpw_forces_at_density`. Then `XlBomd::advance(D^σ)` per spin with Table-I
\(K=5\). Public energy remains **eV** on `frame.meta`; internal Ha/Ry are
method-private.

This page closes the residual that
[xl-bomd-linearized-forces](xl-bomd-linearized-forces.md) (closed-shell Γ-RKS
foothold) explicitly excluded: **UKS linearized shadow forces** — there the
combination refused at `build()`. It is the zero-SCF sibling of the
SCF-converged open-shell path [xl-bomd-uks](xl-bomd-uks.md): the regression
system is an **identical twin** of that slug with only the
`XlBomdLinearized{}` tag added, giving a direct SCF-XL vs linearized-XL A/B
on the same O₂ triplet.

The e2e pin is the **literature shadow-XL / NVE envelope** plus
**`last_result().iters == 0`** on warm steps plus the **Löwdin ⟨S²⟩ triplet
window** at the shadow densities — not a golden Atomiverse total energy and
not a measured CP2K AIMD wall.

## Method

### Dual-spin XL auxiliary densities

Two independent auxiliary AO density matrices \(P^\alpha, P^\beta\) are
**dynamical variables** (not Kolafa ASPC extrapolations). Each spin channel
propagates the same dissipative recursion:

\[
P^\sigma_{n+1}
  = 2\,P^\sigma_{n} - P^\sigma_{n-1}
  + \kappa\,\bigl(D^\sigma_{n} - P^\sigma_{n}\bigr)
  + \alpha \sum_{m=0}^{K} c_m\, P^\sigma_{n-m},
  \qquad \sigma \in \{\alpha, \beta\},
\]

with \(D^\sigma_n\) the one-shot per-spin diagonalization density
(linearized path). Default \(K=5\) (Niklasson et al., *JCP* **130**, 214109
(2009), Table I): \(\kappa = 1.82\), \(\alpha = 0.018\),
\(c_0\ldots c_5 = \{-6, 14, -8, -3, 4, -1\}\) — identical coefficients per
spin, unit-pinned bitwise in `test_cpu_gpw_xl_bomd` for \(K\in[3,9]\).
Time-reversible form: Niklasson, *PRL* **100**, 123004 (2008).

### Shadow / linearized free-energy functional (spin-diagonal)

When \(n^\sigma\) is not self-consistent, the ordinary KS energy and the
Hellmann–Feynman force of that density are inconsistent with the nuclear
dynamics. The collinear-UKS shadow functional above recovers
\(E_{\mathrm{KS}}\) exactly at the dual fixed point \(D^\sigma = n^\sigma\),
and \(\sum_\sigma \mathrm{Tr}\bigl(H^\sigma (D^\sigma - n^\sigma)\bigr)\)
is the first-order residual heat when they differ. Forces are taken of
\(\mathcal{U}\) at fixed \(n^\alpha, n^\beta\). Literature: Niklasson &
Cawkwell, *PRB* **86**, 174308 (2012); Niklasson, *JCP* **147**, 054103
(2017) — spin-diagonal for collinear UKS. The dual-spin shadow-energy
fixed-point identity is unit-pinned in `test_cpu_gpw_xl_bomd`.

### Warm electronic step (zero SCF)

| Stage | Action |
| ----- | ------ |
| 0. Cold seed | Full dual-spin UKS SCF → `advance(D^σ_scf)` per spin; both histories ready |
| 1. Fixed-\(n\) KS | `max_iter = 0` with `guess_P^σ = n^σ` → \(H^\sigma[n]\), \(E_{\mathrm{KS}}[n^\alpha,n^\beta]\) |
| 2. Two diags | One generalized diagonalization per spin → \(D^\sigma = q^\sigma[n]\), \(C^\sigma_{\mathrm{occ}}\) |
| 3. Shadow \(E\) | \(\mathcal{E}\) per the dual-spin formula → `kEnergy` [eV] |
| 4. Forces | `gpw_forces_at_density(D^α, C^α, H^α, D^β, C^β, H^β)` → eV/Å |
| 5. ⟨S²⟩ | Löwdin at the shadow densities (Szabo & Ostlund §2.5) |
| 6. XL advance | `advance(D^σ)` per spin; `last_result().iters = 0`, `converged = true` |

No dual-spin DIIS mixing loop, no residual-threshold iteration on warm steps.

### System and observable

Identical twin of `xl-bomd-uks` (only `XlBomdLinearized{}` differs):

| Quantity | Value |
| -------- | ----- |
| Species | O₂ triplet (2 × O), isolated / Martyna–Tuckerman |
| Multiplicity | 3 — `n_alpha = 7`, `n_beta = 5` (GTH-q6, 12 valence e⁻) |
| Geometry | \(r_{\mathrm{OO}} = 1.2075\) Å, centered in cubic cell |
| Cell | cubic \(L = 12\) Å; `nonperiodic = true` |
| XC | PADE (LSD) |
| Basis / PP | DZVP-MOLOPT-GTH; GTH-PBE-q6 |
| CUTOFF | 200 Ry (envelope run; absolute E/F pins at 400 Ry live on the UKS single-point slugs) |
| SCF (cold seed only) | `tol = 1e-5`, `max_iter = 80`, `mix = 0.4`, DIIS diagonalization; seed converges in ~23 iterations |
| XL-BOMD | \(K = 5\); linearized forces on; dual α/β propagators |
| Ensemble | NVE; \(\Delta t = 0.5\) fs; \(v_z(\mathrm{O1/O2}) = \mp 0.0008\) Å/fs |
| Steps | 1 cold seed + 4 production (2 chunks × 2) |

Primary observable, with \(E(t) = E_{\mathrm{pot}} + E_{\mathrm{kin}}\) [eV]
and \(E_0\) after the cold seed:

\[
\delta = \max_t \frac{\lvert E(t) - E_0\rvert}{\max\!\bigl(1,\,\lvert E_0\rvert\bigr)}
  \le 0.10.
\]

Drift tolerance is **looser** than SCF-converged XL-UKS (`0.10` vs `0.05`)
because of one-shot residual heat — same class as
`xl-bomd-linearized-forces`.

**Why O₂ and not an OH-radical "water-family" twin?** The queue sketched an
OH doublet, but its cold-seed DIIS is pathological in this tree (locks in
only at ~196 iterations at every `mix`/`tol` probed, PBE; PADE worse —
measured 2026-07-31). The O₂ triplet seed converges in ~23 iterations and
pins the same dual-spin linearized-XL physics.

### Exclusions (this slug)

- Not SCF-converged XL (`xl-bomd-uks` keeps iterative SCF to `tol`)
- Not closed-shell linearized XL (`xl-bomd-linearized-forces`)
- Not multi-k linearized XL (shipped follow-on:
  [xl-bomd-linearized-kpoints](xl-bomd-linearized-kpoints.md))
- Not OT + XL, CPMD, or second-gen CP simultaneous with XL (mutually exclusive)
- Not hybrid / meta-GGA XC; not thermostatted / multi-ps production AIMD
- Not a measured fair-wall twin vs CP2K AIMD (see Performance)
- Not molecular post-HF (PySCF loop) or classical electrochem MD (MetalWalls)

## Implementation

Public surface only (`cpu::Frame`, constructor args / members; **no**
`*Config` bag, **no** `make_*` / `build_*` factories):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
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
}
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.span(kAtom_vz)[0] = -0.0008;   // antisymmetric stretch [Å/fs]
frame.span(kAtom_vz)[1] = 0.0008;
frame.box = cpu::Box::cubic(L);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
    /*mix*/ 0.4, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
gpw.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
gpw.n_beta = 5;
// The tag type (not a bool) selects the zero-SCF shadow-BO force path.
// With UKS occupations it allocates dual α/β XL propagators.
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5,
                                  cpu::electronic::XlBomdLinearized{}};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, 1);   // cold seed: full dual-spin SCF, iters > 0
// production: iters == 0; |E(t)-E0|/scale ≤ 0.10; ⟨S²⟩ ∈ [1.5, 2.5]
// s2 = eval.method().last_s2()  — Löwdin at the shadow densities
sim.run(frame, 4);
```

`build()` still **throws** `std::invalid_argument` for multi-k meshes,
`use_ot`, or a `CarParrinello` / `SecondGenCp` wrapper. The former UKS
refusal on the linearized path is **removed** — dual-spin shadow forces are
the delivered feature of this slug.

### How to run the regression

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake --build build -j$(nproc) --target atomiverse reg_xl-bomd-linearized-uks
  cmake --build build --target run_reg_xl-bomd-linearized-uks
'
```

Relevant sources:

| Layer | Path |
| ----- | ---- |
| Propagator (dual instances) | `include/atomiverse/cpu/electronic/xl_bomd.h` |
| XL wrapper / tag | `include/atomiverse/cpu/electronic/xl_bomd_force.h` (`XlBomdForce`, `XlBomdLinearized`) |
| Warm dual-spin shadow path | `src/cpu/gpw/gpw_evaluator.cpp` |
| Forces at density (dual overload) | `src/cpu/gpw/internal/gpw_forces.cpp` (`gpw_forces_at_density`) |
| Unit A1–A3 | `tests/cpu/gpw/test_gpw_xl_bomd.cpp` |
| E2E A4 | `regressions/dynamics/xl-bomd-linearized-uks/` |

## Validation (vs CP2K / literature)

Source: `regressions/dynamics/xl-bomd-linearized-uks/REFERENCE.md`
(offline pin; never live CP2K in ctest).

### Accuracy table

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.10\); measured **1.8e-07** (2026-07-31, 8 threads) | literature shadow-XL / NVE envelope | 0.10 | looser than SCF-XL-UKS 0.05 (residual heat) |
| Warm `last_result().iters` | \(= 0\) | zero iterative SCF | — | product diagnostic |
| Warm `converged` | true | stationary for shadow functional | — | |
| Cold seed `iters` | \(> 0\) (~23), converged | full dual-spin SCF seed | — | both spin histories seeded |
| ⟨S²⟩ (Löwdin, shadow densities) | \([1.5, 2.5]\); measured 2.0047 (seed) → 2.0047 (final) | triplet \(S(S+1)=2\) | window | at seed and every sampled step |
| PE after seed | \(< -100\) eV | bound O₂ | — | |
| KE after seed | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(r_{\mathrm{OO}}\) after production | \(0.9 < r < 1.6\) Å; measured 1.220 Å | soft bond window | Å | no dissociation / collapse |
| Dual-spin shadow \(\mathcal{E}\) at \(D^\sigma = n^\sigma\) | \(= E_{\mathrm{KS}}\) | Niklasson 2012/2017 | unit | fixed-point identity |
| Table-I \(\kappa,\alpha,c_k\) (\(K\in[3,9]\)) | exact per spin | Niklasson 2009 Table I | unit | `test_cpu_gpw_xl_bomd` |
| Absolute \(E_{\mathrm{tot}}\) / forces | **not pinned here** | — | — | UKS E/F twins: `gpw-uks-o2-gamma`, `gpw-uks-forces` |

### Offline CP2K twin (sketch only)

| Field | Value |
| ----- | ----- |
| Input | `tools/cp2k_oracle/inputs/xl-bomd-linearized-uks/o2_uks_xl_linearized_nve.inp` |
| Intent | Short NVE Quickstep UKS MD, low `MAX_SCF` as a stand-in for one-shot per-spin KS; same O₂ / PADE / DZVP-MOLOPT-GTH / GTH-PBE-q6 / 200 Ry / MT / MULT 3 as ATV |
| Status | **Unexecuted** on pin host 2026-07-31 (no `cp2k` binary / Lmod module — same as the whole XL/BOMD family) |
| When binary available | Re-run offline; replace pin section with CP2K version + measured drift + fair wall |

CP2K `EXTRAPOLATION ASPC` + low `MAX_SCF` is **not** identical to Niklasson
dual-spin shadow \(\mathcal{E}\) with true per-spin single-diagonalization
forces. The twin is a method-family sketch for inventory and future
fair-bench, not a bit-identical density history. Do **not** invent absolute
total-energy digits or a CP2K trajectory series.

## Performance

Source: `REFERENCE.md` Perf gate.

### Speed table

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | -----------------------------------------: | ---- | ------ |
| zero-SCF linearized XL-UKS AIMD (O₂, 5 steps) | n/a | 40 (worker host, 8 threads) | **perf_exempt** | n/a | No offline CP2K binary for fair AIMD wall (2026-07-31 inventory) |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair multi-step UKS AIMD wall twin |
| `ratio_atv_over_cp2k` | n/a |
| `within_10x` | n/a (exempt) |
| Regression wall | 40 s measured (≤60 s budget) |

Gate: \(r \le 10\) unless `perf_exempt`. Warm steps are cheaper than
full-SCF XL-UKS (one dual-KS build + two diagonalizations vs iterative
dual-spin DIIS); that saving is intentional product value, not a claimed
measured fair-bench ratio.

## Limitations

- **Envelope, not absolute energy.** Broken dual-spin recursion, α-only
  history, channel swap, or a wrong shadow formula destroys the zero-iter /
  drift / ⟨S²⟩ gates; a passing path is not a high-precision total-energy
  twin of CP2K Quickstep.
- **Γ-UKS diagonalization here.** Multi-k linearized shadow forces shipped
  as [xl-bomd-linearized-kpoints](xl-bomd-linearized-kpoints.md); OT+XL,
  hybrid / GAPW remain refused at `build()`.
- **Looser drift window** (\(\delta \le 0.10\)) than SCF-XL; do not tighten
  without re-pinning literature/CP2K.
- **Cold seed still pays full dual-spin SCF** (~23 DIIS iterations here).
- **200 Ry envelope settings**, not the 400 Ry absolute-E/F pin settings.
- **Short traj** (1 seed + 4 production); no thermostats, no statistics.
- **Perf debt.** Fair wall vs CP2K deferred until an offline binary exists;
  do not invent ratios.

## References

- Offline pin: `regressions/dynamics/xl-bomd-linearized-uks/REFERENCE.md`
- Regression: `regressions/dynamics/xl-bomd-linearized-uks/reg_xl-bomd-linearized-uks.cpp`
- CP2K twin sketch: `tools/cp2k_oracle/inputs/xl-bomd-linearized-uks/o2_uks_xl_linearized_nve.inp`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Sibling paths: [xl-bomd-linearized-forces](xl-bomd-linearized-forces.md)
  (closed-shell zero-SCF foothold);
  [xl-bomd-uks](xl-bomd-uks.md) (SCF-converged open-shell A/B twin);
  [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md) (absolute UKS energy twin)
- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
- A. M. N. Niklasson & M. J. Cawkwell, *Phys. Rev. B* **86**, 174308 (2012)
- A. M. N. Niklasson, *J. Chem. Phys.* **147**, 054103 (2017)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
- Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE LSD
- Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
