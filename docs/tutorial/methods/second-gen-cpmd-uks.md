# Second-generation Car–Parrinello MD — O₂ triplet NVE (Γ-UKS GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `second-gen-cpmd-uks` |
| Regression | `regressions/dynamics/second-gen-cpmd-uks/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{…, n_alpha, n_beta}, aspc_k, corrector_max_iter}` bound with `set_evaluator_ref` |
| Roadmap | AIMD / SGCP UKS residual closed wave 7 |
| Dual gate | Correctness A1–A4 (unit + e2e envelope); speed **`perf_exempt`** (NVE drift; no CP2K twin wall) |
| Wave | CP2K capability loop — harness wave 7 (2026-07-26) |
| Oracle | Offline CP2K sketch only (`tools/cp2k_oracle/`); never live in ctest |

## Summary

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on **open-shell
Γ-UKS** Gaussian-and-plane-waves forces (Kühne et al., *PRL* **98**, 066401
(2007)). Nuclei integrate under ordinary NVE `VelocityVerlet`. Electrons use
**dual Kolafa ASPC** predictors on independent spin densities
\(P^\alpha, P^\beta\), then a **partial** SCF corrector with iteration budget
`corrector_max_iter` ≪ full BOMD. Analytic nuclear forces are taken at the
post-corrector density.

This is **not** BOMD (production SCF is capped), **not** classical CPMD (no
fictitious electronic mass / \(T_e\)), and **not** XL-BOMD (ASPC extrapolation,
not Niklasson auxiliary Verlet). Closed-shell product path:
`second-gen-cpmd-water`. Related dual-spin AIMD: `xl-bomd-uks`, `cpmd-uks`.

Public energy is still **eV** on `frame.meta`; Hartree/Ry only on private
`last_result()` surfaces when cited.

## Method

**SGCP idea.** Classical CPMD carries fictitious electronic kinetic energy with
mass \(\mu\). Second-generation CP drops the fictitious electronic dynamics and
instead couples a high-order density **predictor** (Kolafa ASPC) to a short
**corrector** SCF:

1. Seed \(K+1\) full-SCF MD steps to fill ASPC history.
2. Predict \(P^\sigma_{n+1}\) from prior converged densities (per spin).
3. Run at most `corrector_max_iter` SCF iterations from that guess.
4. Record the converged \(P^\sigma\) into ASPC; evaluate forces at that state.

**ASPC (Kolafa).** Polynomial multistep extrapolation of the density history
with coefficients fixed by order \(K\) (Kolafa, *J. Comput. Chem.* **25**, 335
(2004)). Atomiverse reuses `cpu::electronic::ASPCPredictor` — one instance for RKS,
two for UKS (α and β). CP2K twin: `&QS EXTRAPOLATION ASPC` + limited
`MAX_SCF`.

**UKS / PADE LSD.** Triplet O₂ with GTH-q6 → 12 valence e⁻;
\(n_\alpha=7\), \(n_\beta=5\). Spin-resolved PADE LSD on the GPW grid; dual
DIIS diagonalization corrector (not OT). \(\langle S^2\rangle\) is Löwdin
single-determinant (ideal \(S=1\): \(S(S+1)=2\)).

**Conserved envelope (NVE).** With partial-SCF heat the regression gates
relative drift of \(E = E_{\mathrm{pot}} + E_{\mathrm{kin}}\) [eV] from the
first **warm** production step:

\[
\delta = \max_t \frac{|E(t)-E_0|}{\max(1,|E_0|)} \le 0.10
\]

(softer than full BO 0.05 — same class as closed-shell `second-gen-cpmd-water`).

**Key exclusions.**

- No OT corrector; no classical CPMD / XL simultaneous engagement
- No k-points; Γ + MT / nonperiodic only on this pin
- No thermostat / NVT-SGCP; NVE envelope only
- No CP2K binary energy twin on the pin host (literature + method-identity
  gates; offline input kept)

## Implementation

Public surface only (`cpu::Frame`, concrete members; **no** `*Config` bag,
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
}
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.span(kAtom_vz)[0] = -0.0008; // Å/fs antisymmetric stretch
frame.span(kAtom_vz)[1] =  0.0008;
frame.box = cpu::Box::cubic(L);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

constexpr double dt = 0.5; // fs
cpu::gpw::GPWEvaluator gpw(
    {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
    /*mix*/ 0.4, cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
gpw.n_alpha = 7;
gpw.n_beta = 5;
// SGCP is a wrapper TYPE: ASPC order K and the corrector budget are its
// constructor arguments (the evaluator has no public second_gen_cp* flags).
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/5};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, /*seed*/ 2);   // fill dual ASPC (K+1 densities)
sim.run(frame, /*prod*/ 3);   // warm partial-SCF production
// frame.meta.get(kEnergy), kKinetic [eV]; eval.method().last_s2();
// eval.last_result().iters
```

The wrapper constructor throws for \(K < 0\) or a corrector budget \(< 1\), and
sets `compute_forces` itself. `build()` **throws** if SGCP is combined with
`use_ot` or with a `CarParrinello` / `XlBomdForce` wrapper. UKS alone is
accepted (dual ASPC).

```bash
# Scoped build/test (never bare cmake --build build)
cmake --build build -j$(nproc) --target atomiverse \
      test_cpu_gpw_second_gen_cpmd reg_second-gen-cpmd-uks
ctest --test-dir build -R '^test_cpu_gpw_second_gen_cpmd$' --output-on-failure
cmake --build build --target run_reg_second-gen-cpmd-uks
```

Relevant paths:

- `include/atomiverse/cpu/gpw/gpw_evaluator.h` — members `second_gen_cp*`
- `src/cpu/gpw/gpw_evaluator.cpp` — dual ASPC + partial SCF launch path
- `include/atomiverse/cpu/electronic/aspc_predictor.h` — Kolafa ASPC primitive
- `include/atomiverse/cpu/electronic/second_gen_cp.h` — `SecondGenCp` wrapper
- `tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp` — build guards (incl. UKS accept)
- `regressions/dynamics/second-gen-cpmd-uks/`

## Validation (vs offline CP2K / literature envelope)

Source: `regressions/dynamics/second-gen-cpmd-uks/REFERENCE.md` (offline pin;
never live CP2K in ctest).

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | asserted | literature SGCP envelope | ≤ 0.10 | soft after first warm production step |
| Production SCF iters | ≤ 5 | CP2K `MAX_SCF` sketch | identity | once ASPC warm — not full BOMD |
| \(\langle S^2\rangle\) | \([1.5, 2.5]\) | Löwdin triplet window | hard | after seed and production |
| PE after seed | \(< -100\) eV | bound | hard | finite trajectory |
| KE after seed | \(> 0\), finite | identity | hard | antisymmetric stretch |
| O–O distance | \(0.9 < r < 1.6\) Å | structural | soft | after production |

**System (hard-coded):** O₂, \(R=1.2075\) Å, \(L=12\) Å cubic, MT /
`nonperiodic`, PADE LSD, DZVP-MOLOPT-GTH / GTH-PBE-q6, 200 Ry, `tol=1e-5`,
seed `max_iter=80`, ASPC \(K=1\), corrector 5, `dt=0.5` fs, 2 seed + 1
baseline + 2 drift steps.

**Offline oracle (CP2K):** no `cp2k` binary on the worker pin host
(Arrhenius login-class, **2026-07-26**). Twin sketch:

`tools/cp2k_oracle/inputs/second-gen-cpmd-uks/o2_uks_sgcp_nve.inp`

(`UKS`, `MULTIPLICITY 3`, `EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 1`,
`MAX_SCF 5`, PADE, MT). When a binary is available, re-run offline via
`tools/cp2k_oracle/` and replace the REFERENCE status with version + measured
drift + fair wall.

Static UKS energy / force pins for the same molecule family:
`regressions/gpw/gpw-uks-o2-gamma`, `gpw-uks-forces`.

## Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| O₂ UKS SGCP short NVE | n/a | n/a | n/a | exempt | no CP2K binary for multi-step UKS SGCP AIMD twin wall (2026-07-26) |

Gate: **`perf_exempt=true`** (handoff wave 7). Correctness is the NVE drift +
corrector-budget identity pin, not wall ratio.

## Limitations

- Γ-UKS diagonalization/DIIS corrector only; OT+SGCP refused
- Short-trajectory soft drift tol (partial-SCF heat); not a microcanonical
  long-run conservation claim
- No k-point SGCP; no NVT electronic/nuclear thermostat product on this slug
- Offline CP2K input is an **unexecuted sketch** until a binary is inventoried
- Mutual exclusion with `cpmd` / `xl_bomd` — pick one AIMD electronic path

## References

- Offline pin: `regressions/dynamics/second-gen-cpmd-uks/REFERENCE.md`
- CP2K twin sketch: `tools/cp2k_oracle/inputs/second-gen-cpmd-uks/o2_uks_sgcp_nve.inp`
- Closed-shell sibling: `regressions/dynamics/second-gen-cpmd-water`
- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007)
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT
