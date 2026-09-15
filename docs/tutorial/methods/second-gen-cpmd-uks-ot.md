# Second-generation CP + OT corrector — O₂ triplet NVE (Γ-UKS GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `second-gen-cpmd-uks-ot` |
| Regression | `regressions/dynamics/second-gen-cpmd-uks-ot/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{…, n_alpha, n_beta, use_ot=true}, aspc_k, corrector_max_iter}` bound with `set_evaluator_ref` |
| Roadmap | `docs/roadmap/molecular-dynamics.md` §7 — SGCP Γ-UKS OT |
| Dual gate | Correctness: NVE drift envelope + OT corrector-budget identity + triplet ⟨S²⟩; speed **`perf_exempt`** (no CP2K AIMD twin wall on pin host) |
| Wave | CP2K capability loop — wave 10 (merged 2026-07-26) |
| Oracle | Offline CP2K sketch only (`tools/cp2k_oracle/`); **never** live in ctest |

## Summary

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on **open-shell
Γ-UKS** Gaussian-and-plane-waves (GPW / Quickstep) forces, with the **Orbital
Transformation (OT)** SCF as the **partial-SCF corrector**. Nuclei integrate
with ordinary NVE `VelocityVerlet`. Each force step (once warm) runs:

1. **Dual Kolafa ASPC** predictors on independent spin densities
   \(P^\alpha\), \(P^\beta\);
2. Purification of the predicted densities to **spin-resolved natural orbitals**
   that seed the OT manifolds;
3. A **capped OT** corrector (`corrector_max_iter`, not full BOMD);
4. Analytic nuclear forces at the post-corrector density.

This is **not** Born–Oppenheimer MD (production OT is capped), **not**
classical CPMD (no fictitious electronic mass / \(T_e\)), **not** XL-BOMD
(ASPC extrapolation, not Niklasson auxiliary Verlet), **not** DIIS SGCP-UKS
(`second-gen-cpmd-uks`, `use_ot=false`), and **not** closed-shell SGCP-OT
(`second-gen-cpmd-ot`). Product gate:
`reg_second-gen-cpmd-uks-ot`. Public Frame energy is **eV**; method channel is
Hartree on `last_energy_hartree()` / `last_result()`.

## Method

**Setting.** Collinear unrestricted KS at the Γ point: separate α/β densities
and KS matrices; PADE LSD XC; DZVP-MOLOPT-GTH Gaussians; GTH-PBE-q6
pseudopotentials; Martyna–Tuckerman isolated Poisson (`nonperiodic = true`).
Valence bookkeeping for O₂ with GTH-q6: 12 valence e⁻; triplet multiplicity
⇒ \(N_\alpha=7\), \(N_\beta=5\).

**Second-generation CP** (Kühne, Krack, Mohamed, Parrinello, *Phys. Rev. Lett.*
**98**, 066401 (2007)). Instead of fictitious electronic DOF, each MD step
extrapolates the density with Kolafa’s always-stable predictor–corrector
(ASPC; *J. Comput. Chem.* **25**, 335 (2004)) and applies a **limited** SCF
corrector. With ASPC order \(K\) and history of SCF densities \(D_{n-k}\),

\[
P_{n+1}^{\mathrm{pred}}
  = \sum_{k=0}^{K} B_k\, D_{n-k},
\]

then a partial SCF (here **OT**) produces the corrected \(D_n\) used for
energy and forces. Production steps therefore spend \(O(\texttt{max\_iter})\)
OT iterations with \(\texttt{max\_iter} \ll\) full BOMD budget — the SGCP
signature vs BO.

**UKS dual ASPC.** Open-shell product path keeps **independent** histories for
\(P^\alpha\) and \(P^\beta\). A broken spin guess (α-only, swapped channels,
ignored density) fails production-iter, ⟨S²⟩, or drift gates immediately.

**OT corrector** (VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003);
product RKS OT: `ot-minimizer-water`). Occupied MO coefficients live on the
Stiefel-type manifold \(C^\top S C = I\). For SGCP, the ASPC density is
purified to natural orbitals that warm-start the OT manifold; the corrector
then runs at most `corrector_max_iter` outer OT steps (5 in the
pin). Same gapped fixed-point family as full OT when the budget is raised;
product identity is the **cap**, not milli-Ha energy parity vs full BO.

**Nuclear NVE.** Fixed-step Velocity Verlet; no thermostat/barostat. Conserved
quantity for the pin is the classical total energy

\[
E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)
\quad[\mathrm{eV}],
\]

with \(E_{\mathrm{pot}}\) from the post-corrector KS evaluation on Frame
(`kEnergy`) and \(E_{\mathrm{kin}}\) from nuclear KE (`kKinetic`). There is
**no** electronic \(T_e\) term (contrast classical CPMD).

**Primary observable (envelope, not absolute energy):**

\[
\delta = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)}
  \le 0.10,
\]

with \(E_0\) after the first **warm** production step (ASPC history filled).
Soft relative drift (0.10 vs full BO 0.05) absorbs partial-OT heat on a short
trajectory — same class as `second-gen-cpmd-ot` / `second-gen-cpmd-uks`.

**Method-identity gates (binding):**

- Production OT **iters ≤ `corrector_max_iter` (5)** once ASPC is warm
  (must not silently fall back to full BOMD OT);
- PE finite and bound (\(< -100\) eV after seed);
- KE finite, \(\ge 0\), non-trivial after seed;
- ⟨S²⟩ in \([1.5, 2.5]\) (triplet window) after seed and production
  (Löwdin single-determinant metric; ideal \(S=1 \Rightarrow S(S+1)=2\));
- Coordinates finite; soft O–O window \(0.9 < r_{\mathrm{OO}} < 1.6\) Å.

**Key exclusions.**

- Multi-k / smearing / DFT+U with SGCP-OT (separate OT×k / OT×U / OT×smear
  product rows; not this slug).
- Classical CPMD / XL-BOMD concurrent with `second_gen_cp` (`build()` mutual
  exclusion).
- Absolute energy twin vs live CP2K AIMD (host inventory: no binary on pin
  host; offline sketch only).
- Electronic thermostat / NVT-CP / cell dynamics.
- CUDA / MPI multi-rank AIMD scaling claim.

## Implementation

Public surface only (`cpu::Frame` + evaluator **members** / ctor args; **no**
`*Config` bag, **no** `make_*` / `build_*` factories). Match the regression:

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

// O₂ triplet, centered in 12 Å cubic MT cell
cpu::Frame frame(2);
constexpr double L = 12.0;
constexpr double R = 1.2075; // Å
const double mass = 15.999;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 8;
    frame.span(kAtom_mass)[i] = mass;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_x)[i] = 0.5 * L;
    frame.span(kAtom_y)[i] = 0.5 * L;
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

constexpr double dt = 0.5; // fs — BOMD-class step (not classical CP dt)
constexpr int corrector_budget = 5;
cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {O}, /*cutoff_ry*/ 200.0, /*max_iter*/ 80, /*tol*/ 1e-5,
    /*mix*/ 0.4, cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
gpw.n_alpha = 7; // multiplicity 3, 12 valence e⁻
gpw.n_beta = 5;
gpw.use_ot = true;                     // OT partial-SCF corrector (this slug)
// SGCP is the wrapper type; Kolafa ASPC order K (needs K+1 seeds) and the
// corrector budget are its constructor arguments.
cpu::electronic::SecondGenCp eval{std::move(gpw), /*aspc_k=*/1,
                                  /*corrector_max_iter=*/corrector_budget};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});
sim.set_evaluator_ref(eval);
sim.build();

// Seed: full-budget OT fills dual ASPC history (K+1 = 2 for K=1)
sim.run(frame, /*n_seed=*/2);
// Production: dual ASPC + capped OT; assert last_result().iters <= budget
sim.run(frame, /*n_prod=*/1 + 2);
const double e = frame.meta.get(kEnergy) + frame.meta.get(kKinetic); // eV
const double s2 = eval.method().last_s2();
const int iters = eval.last_result().iters;
```

Build and run the regression (scoped target only):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_second-gen-cpmd-uks-ot
  ./build/regressions/reg_second-gen-cpmd-uks-ot
  # or: cmake --build build --target run_reg_second-gen-cpmd-uks-ot
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/electronic/second_gen_cp.h` | `SecondGenCp` wrapper (ASPC order + corrector budget) |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Public members `use_ot`, `n_alpha`/`n_beta`; private SGCP mode bits |
| `include/atomiverse/cpu/gpw/internal/ot.h` | OT parameterization / preconds (internal) |
| `include/atomiverse/cpu/electronic/aspc_predictor.h` | Kolafa ASPC (shared with XL / BOMD warm-start) |
| `src/cpu/gpw/` | Wire dual-spin ASPC + OT corrector into force path |
| `tests/cpu/gpw/` (`test_gpw_second_gen_cpmd`, related) | Unit: method identity / mutual exclusion |
| `regressions/dynamics/second-gen-cpmd-uks-ot/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/second-gen-cpmd-uks-ot/` | Offline CP2K twin sketch (not ctest) |

## Validation (vs CP2K)

Source: [`regressions/dynamics/second-gen-cpmd-uks-ot/REFERENCE.md`](../../../regressions/dynamics/second-gen-cpmd-uks-ot/REFERENCE.md).
Oracle: offline **CP2K** sketch under `tools/cp2k_oracle/inputs/second-gen-cpmd-uks-ot/`;
**never** live in ctest. Pin host inventory (2026-07-26): **no** `cp2k`
binary / CP2K Lmod module — same class as sibling SGCP dynamics slugs
(`second-gen-cpmd-uks`, `second-gen-cpmd-ot`, `xl-bomd-uks`). When a binary is
available, re-run offline and replace the oracle section with version +
measured drift + fair wall.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Isolated O₂, charge 0, UKS triplet |
| Geometry [Å] | \(R = 1.2075\); O at \(z = L/2 \pm R/2\), \(x=y=L/2\), \(L=12\) |
| Masses | 15.999 amu each |
| Cell | 12 Å cubic, `PERIODIC NONE`, Martyna–Tuckerman |
| Spin | \(n_\alpha=7\), \(n_\beta=5\) (GTH-q6, 12 valence e⁻) |
| Basis / PP | DZVP-MOLOPT-GTH; GTH-PBE-q6 |
| XC | PADE (LSD) |
| Cutoff (pin) | **200 Ry** |
| SCF seed | `max_iter=80`, `tol=1e-5`, **OT** |
| SGCP | ASPC \(K=1\); corrector budget `corrector_max_iter=5` once warm |
| Ensemble | NVE |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_z = \pm 0.0008\) Å/fs |
| Steps | 2 seed (warm ASPC) + 1 baseline production + 2 drift steps |
| CP2K twin sketch | `EXTRAPOLATION ASPC`, `EXTRAPOLATION_ORDER 1`, `&OT` + limited `MAX_SCF`, `UKS` / `MULTIPLICITY 3` |

**Pinned observables (literature envelope + method identity):**

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| Relative NVE drift \(\delta\) | ≤ **0.10** | (sketch; unexecuted) | **0.10** | \(E_0\) after first warm production step |
| Production OT iters | ≤ **5** | limited `MAX_SCF` | = budget | identity: partial OT, not full BOMD |
| PE after seed [eV] | \(< -100\) | — | bound | finite + physical |
| KE after seed [eV] | \(> 0\), finite | — | — | stretch velocity non-trivial |
| ⟨S²⟩ after seed / prod | \([1.5, 2.5]\) | LSD triplet class | window | Löwdin single-det |
| \(r_{\mathrm{OO}}\) after prod [Å] | \((0.9, 1.6)\) | — | soft | no explode/collapse |

This is **not** a golden snapshot of Atomiverse total energy and **not** a
milli-Ha absolute energy twin. Broken dual-spin ASPC or ignored OT budget
fails the table immediately.

Offline oracle sketch (host with CP2K module; **not** ctest):

```bash
module load ... CP2K/2026.1   # when available
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/second-gen-cpmd-uks-ot/o2_uks_sgcp_ot_nve.inp \
          -o o2_uks_sgcp_ot_nve.out
# Record: ENERGY| / .ener drift, fair wall vs reg_second-gen-cpmd-uks-ot
```

Related absolute energy / force offline pins (static O₂ UKS, not this MD slug):
`regressions/gpw/gpw-uks-o2-gamma`, `gpw-uks-forces`.

## Performance

Source: `regressions/dynamics/second-gen-cpmd-uks-ot/REFERENCE.md` and
Campaign record: **`perf_exempt`** (NVE drift / method-identity gate; no fair AIMD
wall twin on pin host 2026-07-26).

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K UKS SGCP-OT AIMD | n/a | n/a | n/a (`perf_exempt`) | no binary on pin host |
| ATV `reg_second-gen-cpmd-uks-ot` | short NVE (≤60 s class) | n/a | exempt | e2e budget, not fair twin |

Gate: when a CP2K binary is available, fair-bench multi-step UKS SGCP-OT
AIMD with matched ASPC order / OT corrector budget / cutoff / OMP and record
\(r\le 10\) or keep explicit exemption with reason.

## Limitations

- Γ-point collinear UKS only; no multi-k SGCP-OT, no metallic smear on this
  slug (see `ot-uks-kpoints`, `ot-uks-smearing`, `ot-uks-dftu-smearing` for
  static OT composition rows).
- Soft drift tol (0.10) is appropriate for **short** partial-OT trajectories;
  production long-run stability vs full BO / CP2K AIMD is not claimed.
- No absolute energy / force re-pin vs CP2K on the MD trajectory until the
  offline oracle is executed on a host with CP2K.
- Mutually exclusive with classical `cpmd` and `xl_bomd` on the same
  evaluator (`build()` throws).
- Dense OT preconditioner cost; no sparse / linear-scaling OT claim.
- Electronic thermostat, NVT-CP, and cell dynamics remain residual AIMD
  productization (roadmap MD §7).
- CUDA port out of scope for this loop instance.

## Sibling product matrix

| Slug | Spin | Corrector | Role |
| ---- | ---- | --------- | ---- |
| `second-gen-cpmd-water` | RKS | DIIS / diag partial SCF | closed-shell SGCP foothold |
| `second-gen-cpmd-uks` | UKS | DIIS / diag partial SCF | dual ASPC without OT |
| `second-gen-cpmd-ot` | RKS | **OT** partial SCF | closed-shell OT corrector |
| **`second-gen-cpmd-uks-ot`** | **UKS** | **OT** partial SCF | **this page** |
| `ot-minimizer-water` | RKS | full OT SCF (static) | OT energy product |
| `gpw-uks-o2-gamma` | UKS | full DIIS SCF (static) | O₂ energy / ⟨S²⟩ pin |
| `cpmd-uks` | UKS | classical CP (fictitious \(\mu\)) | not SGCP |
| `xl-bomd-uks` | UKS | Niklasson XL + full SCF | not ASPC-SGCP |

## References

- Offline pin + dual gate: `regressions/dynamics/second-gen-cpmd-uks-ot/REFERENCE.md`
- Regression binary: `regressions/dynamics/second-gen-cpmd-uks-ot/reg_second-gen-cpmd-uks-ot.cpp`
- CP2K oracle sketch: `tools/cp2k_oracle/inputs/second-gen-cpmd-uks-ot/o2_uks_sgcp_ot_nve.inp`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline only; never ctest)
- Roadmap: `docs/roadmap/molecular-dynamics.md` §7 (SGCP UKS OT ✓)
- Closed-shell OT static: `docs/tutorial/methods/ot-minimizer-water.md`
- Γ-UKS static O₂: `docs/tutorial/methods/gpw-uks-o2-gamma.md`
- Kühne, T. D.; Krack, M.; Mohamed, F. R.; Parrinello, M. *Phys. Rev. Lett.*
  **98**, 066401 (2007), doi:10.1103/PhysRevLett.98.066401 (second-gen CP)
- Kolafa, J. *J. Comput. Chem.* **25**, 335 (2004), doi:10.1002/jcc.10385 (ASPC)
- VandeVondele, J.; Hutter, J. *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154 (OT)
- Kühne, T. D. et al. *J. Chem. Phys.* **152**, 194103 (2020) §VII C,
  doi:10.1063/5.0007045 (CP2K / OT formulation followed in-tree)
