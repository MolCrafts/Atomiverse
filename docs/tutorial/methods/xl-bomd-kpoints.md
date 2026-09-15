# XL-BOMD multi-k — extended-Lagrangian BOMD (H₂ NVE, GPW + k-mesh)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-kpoints` |
| Regression | `regressions/dynamics/xl-bomd-kpoints/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{…, compute_forces=true, k_mesh, k_time_reversal}, k=5}` bound with `set_evaluator_ref`; complex propagator `cpu::electronic::XlBomdCplx` (one per sampled \(k\)) |
| Roadmap | MD roadmap XL-BOMD multi-k RKS |
| Dual gate | Correctness: NVE drift envelope + multi-k \(P_k\) payload; speed **`perf_exempt`** (no offline CP2K multi-k AIMD twin wall) |
| Wave | CP2K capability loop — harness wave 10 (2026-07-26) |
| Oracle | Offline CP2K pins / twin input only (`tools/cp2k_oracle/`); never live in ctest |

## Summary

**Extended-Lagrangian Born–Oppenheimer MD** (XL-BOMD; Niklasson, *PRL*
**100**, 123004 (2008); dissipative form Niklasson et al., *JCP* **130**,
214109 (2009)) on **multi-k closed-shell RKS** Gaussian-and-plane-waves
(GPW / Quickstep) forces. The outer loop is the MD preset
(`cpu::md::MD` + NVE `VelocityVerlet`); the force slot binds a
`cpu::electronic::XlBomdForce` wrapping a `cpu::gpw::GPWEvaluator` with
analytic forces and a non-Γ `k_mesh`.

Unlike Γ-only XL (`xl-bomd-water`, `xl-bomd-uks`), this product path
propagates **one complex auxiliary density \(P_k\) per sampled Bloch
point** via `cpu::electronic::XlBomdCplx` and feeds the vector of guesses into
the multi-k SCF. SCF still converges to `tol` each ionic step
(SCF-converged XL-BOMD). Public units remain eV, Å, fs, eV/Å on Frame;
internal KS energies may be cited in Ha/Ry on method channels only.

This page is the **multi-k XL product pin**: periodic H₂ (twin geometry
family of `gpw-kpoint-forces`), PADE / DZV-GTH-PADE / GTH-PADE-q1,
\(1\times1\times2\) unreduced mesh (\(n_k=2\)), short NVE trajectory. The
pin is **not** a golden Atomiverse total energy — it is the literature
NVE conservation envelope plus multi-k payload diagnostics. Absolute
multi-k force quality is already offline-pinned in `gpw-kpoint-forces`;
Γ XL product paths are `xl-bomd-water` / `xl-bomd-uks`.

## Method

**Level.** Classical BOMD nuclei with Velocity Verlet; electrons
re-solved self-consistently each step, but the SCF **initial density**
is an extended-Lagrangian dynamical variable rather than a cold guess
or Kolafa ASPC extrapolation (`cpu::electronic::ASPCPredictor`). Not
Car–Parrinello fictitious-mass MD (`cpu::electronic::CarParrinello`), not
second-generation CP (`cpu::electronic::SecondGenCp`).

**Auxiliary density recursion (per \(k\)).** With \(D^k_n\) the SCF
density at step \(n\) and Table-I coefficients \((\kappa,\alpha,c_m)\)
for order \(K\):

\[
P^{k}_{n+1}
  = 2 P^{k}_n - P^{k}_{n-1}
    + \kappa\,(D^{k}_n - P^{k}_n)
    + \alpha \sum_{m=0}^{K} c_m P^{k}_{n-m}.
\]

Default \(K=5\): \(\kappa=1.82\), \(\alpha=0.018\),
\(c=\{-6,14,-8,-3,4,-1\}\). Real \((\kappa,\alpha,c_m)\) act
element-wise on complex Hermitian \(P_k\) (Hermitian history remains
Hermitian up to floating-point roundoff). First MD step seeds history
with the cold SCF density (stationary start); later steps propagate.

**Why XL vs plain warm-start.** Plain reuse of the previous density (or
MOs) breaks time reversibility and produces secular total-energy drift
in long BOMD. The XL recursion is time-reversible up to the \(O(\alpha)\)
dissipation, which removes that systematic drift class while still
providing a high-quality guess.

**Electronic structure.** GPW Kohn–Sham (VandeVondele et al., *Comput.
Phys. Commun.* **167**, 103 (2005)): density matrix in contracted
Gaussians; Hartree / XC / local short-range GTH PP on a uniform grid;
periodic Poisson (`nonperiodic = false`). Multi-k SCF uses complex
Bloch matrices per \(k\) and k-summed density/forces. Analytic forces
write `frame.span(kAtom_f*)` [eV/Å] when `compute_forces = true`.

**Nuclear dynamics (NVE).** Fixed-step symplectic Velocity Verlet
(\(\Delta t\) in fs). Total energy
\(E(t)=E_{\mathrm{pot}}(t)+E_{\mathrm{kin}}(t)\) [eV] is conserved up to
bounded non-secular error when \(\mathbf{F}\approx -\nabla E\).

**Primary observable (envelope, not absolute energy):**

\[
\delta
  = \max_t \frac{\lvert E(t)-E_0\rvert}{\max\!\big(1,\,\lvert E_0\rvert\big)}
  \le 0.05,
\]

with \(E_0\) after one warm-up step (XL histories seeded). Additional
gates: SCF converged every sampled step with multi-k XL \(P_k\) guesses;
`n_kpts >= 2` and `P_k.size() == n_kpts` after warm-up; PE finite and
bound (\(< -5\) eV for this H₂ PADE setup); KE finite, \(\ge 0\), and
non-trivial after warm-up; all coordinates finite; soft H–H window
\(0.4 < r_{\mathrm{HH}} < 1.5\) Å after production.

**System (regression settings).**

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic |
| Spin | closed-shell RKS, \(n_{\mathrm{occ}}=1\) |
| XC | PADE (LDA) |
| Basis | DZV-GTH-PADE |
| PP | GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run; force quality pinned elsewhere) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF | `tol = 1e-5`, `max_iter = 60`, DIIS diagonalization |
| XL-BOMD | `K = 5`; one `XlBomdCplx` per k |
| Ensemble | NVE (no thermostat / barostat) |
| \(\Delta t\) | 0.5 fs |
| Initial velocity | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

**Key exclusions (binding product surface).**

- Multi-k **UKS** + XL — `build()` throws (`multi-k UKS xl_bomd is not productized`); Γ-UKS XL is separate (`xl-bomd-uks`).
- XL + OT (`use_ot`) — mutually exclusive; diagonalization path only.
- XL + CPMD / second-gen CP — mutually exclusive.
- Zero-SCF linearized-functional XL force — not productized (SCF still to `tol`).
- Fair multi-step multi-k AIMD wall vs CP2K — **not** gated (`perf_exempt`).

## Implementation

Public surface only (`cpu::Frame` + `GPWEvaluator` members / ctor args;
no `*Config` bags, no `make_*` / `build_*` factories). Match
`reg_xl-bomd-kpoints`:

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

cpu::Frame frame(2);
constexpr double bohr2ang = 0.52917720859;
constexpr double L_bohr = 12.0;
constexpr double L_ang = L_bohr * bohr2ang;
constexpr double r0_bohr = 1.0;
const double x0[2] = {(L_bohr / 2.0 - 0.5 * r0_bohr) * bohr2ang,
                      (L_bohr / 2.0 + 0.5 * r0_bohr) * bohr2ang};
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_mass)[i] = 1.008;
    frame.span(kAtom_tag)[i] = i + 1;
    frame.span(kAtom_x)[i] = x0[i];
    frame.span(kAtom_y)[i] = L_ang / 2.0;
    frame.span(kAtom_z)[i] = L_ang / 2.0;
}
frame.span(kAtom_vx)[0] = -0.0010;   // Å/fs antisymmetric stretch
frame.span(kAtom_vx)[1] =  0.0010;
frame.box = cpu::Box::cubic(L_ang);

const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};

cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {H}, /*cutoff_ry*/ 100.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
    /*compute_forces*/ true);
gpw.k_mesh[0] = 1;
gpw.k_mesh[1] = 1;
gpw.k_mesh[2] = 2;
gpw.k_time_reversal = false;   // two sampled k (unreduced)
// XL is the wrapper type; K = 5 is the Table-I dissipation order.
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt fs=*/0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, 1);              // warm-up seeds XL histories
// frame.meta.get(kEnergy) [eV]; eval.last_result().n_kpts, P_k, converged
sim.run(frame, 4);              // production; gate |ΔE|/scale ≤ 0.05
```

Complex propagator primitive (`include/atomiverse/cpu/electronic/xl_bomd.h`) —
consumed by the evaluator when multi-k XL is active; callable for
unit/protocol tests:

```cpp
cpu::electronic::XlBomdCplx xl(5);  // throws unless 3 <= K <= 9
xl.advance(D_k_scf);            // first call seeds; later propagates
const auto& guess = xl.aux();   // complex P_{n+1} for next SCF
```

Relevant paths:

- Public: `include/atomiverse/cpu/gpw/gpw_evaluator.h`,
  `include/atomiverse/cpu/electronic/xl_bomd_force.h`,
  `include/atomiverse/cpu/electronic/xl_bomd.h`
- Impl: `src/cpu/gpw/gpw_evaluator.cpp` (multi-k XL advance / `guess_P_k`)
- Unit: `tests/cpu/gpw/test_gpw_xl_bomd.cpp` (Table-I coefficients, real + complex)
- Regression: `regressions/dynamics/xl-bomd-kpoints/`
- Oracle twin sketch: `tools/cp2k_oracle/inputs/xl-bomd-kpoints/h2_xl_bomd_kpoints_nve.inp`

### Build and run (scoped)

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake --build build -j$(nproc) --target atomiverse reg_xl-bomd-kpoints
  cmake --build build --target run_reg_xl-bomd-kpoints
'
```

Unit Table-I / complex path (when developing the propagator only):

```bash
cmake --build build -j$(nproc) --target test_cpu_gpw_xl_bomd
ctest --test-dir build -R '^test_cpu_gpw_xl_bomd$' --output-on-failure
```

## Validation (vs offline CP2K / literature)

Source: `regressions/dynamics/xl-bomd-kpoints/REFERENCE.md`. Offline pins
only — **CP2K is never invoked from ctest**. Multi-k force absolute
quality lives in `gpw-kpoint-forces`; this slug gates **MD binding +
multi-k XL guess plumbing + NVE envelope**.

| Quantity | Atomiverse / pin | Oracle | Tol / gate | Notes |
| -------- | ---------------: | ------ | ---------- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.05\) | Literature XL-BOMD/NVE envelope | hard | \(E_0\) after warm-up; scale \(\max(1,\|E_0\|)\) |
| SCF converged each chunk | true | product requirement | hard | multi-k XL \(P_k\) guesses |
| \(n_{\mathrm{kpts}}\) | \(\ge 2\) | mesh \(1\times1\times2\) unreduced | hard | `last_result().n_kpts` |
| \(P_k\) payload | `P_k.size() == n_kpts` | multi-k SCF contract | hard | after warm-up |
| PE after warm-up | \(< -5\) eV | H₂ PADE bound | hard | not a golden energy |
| KE after warm-up | \(\ge 0\), \(\gt 10^{-12}\) eV | kinetic non-trivial | hard | stretch seed |
| \(r_{\mathrm{HH}}\) after prod | \((0.4, 1.5)\) Å | soft bond window | hard | no collapse / fly-off |
| Table-I \((\kappa,\alpha,c)\) \(K=5\) | exact | Niklasson *JCP* **130**, 214109 (2009) | unit | `test_cpu_gpw_xl_bomd` |

**Offline CP2K inventory (2026-07-26):** no `cp2k` binary / CP2K Lmod
module on the worker host used for this pin (same as `xl-bomd-water` /
`xl-bomd-uks`). Twin sketch (short NVE Quickstep MD with ASPC-class
extrapolation + `&KPOINTS`) is kept at
`tools/cp2k_oracle/inputs/xl-bomd-kpoints/`. When a binary is available,
re-run offline and replace REFERENCE with version + measured drift +
fair wall; do **not** wire live CP2K into ctest.

Related pins:

| Related slug | Role |
| ------------ | ---- |
| `gpw-kpoint-forces` | multi-k analytic ionic forces (absolute quality) |
| `xl-bomd-water` | Γ closed-shell RKS XL NVE (H₂O / PBE) |
| `xl-bomd-uks` | Γ UKS XL NVE |
| `gpw-bomd-water` | plain BOMD NVE (no XL) |

## Performance

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No offline CP2K binary on host for a fair wall twin of multi-step multi-k AIMD (2026-07-26 inventory); correctness is NVE drift / payload, not wall |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | n/a |

Gate policy for CP2K-loop AIMD envelopes: wall ratio ≤10× only when a
fair offline twin exists; otherwise `perf_exempt` with cited reason (this
slug).

## Limitations

- **Closed-shell multi-k only** for product XL; multi-k UKS + XL remains
  residual (dual-spin complex \(P_k^\sigma\) not productized).
- **Diagonalization/DIIS only** — OT multi-k XL is not a product combo
  (OT multi-k energy is a separate OT product path without XL).
- **SCF still fully converged** each step; linearized zero-SCF XL forces
  are out of scope for this foothold.
- Envelope CUTOFF is cheap (100 Ry); do not treat this regression as a
  multi-k force accuracy pin — use `gpw-kpoint-forces`.
- Short trajectory (1+4 steps) is a **plumbing / conservation smoke**,
  not a production sampling length.
- No fair multi-k AIMD wall ratio until offline CP2K twin is re-run on
  a host with a binary.

## References

- Offline pin: `regressions/dynamics/xl-bomd-kpoints/REFERENCE.md`
- Regression: `regressions/dynamics/xl-bomd-kpoints/reg_xl-bomd-kpoints.cpp`
- CP2K twin sketch: `tools/cp2k_oracle/inputs/xl-bomd-kpoints/`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline only; never live ctest)
- Γ XL tutorial: [xl-bomd-water.md](xl-bomd-water.md)
- Multi-k forces family: [gpw-kpoint-forces](../../roadmap/dft.md) product row /
  `regressions/gpw/gpw-kpoint-forces`
- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW
