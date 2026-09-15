# Multi-k UKS XL-BOMD — H₂ NVE (GPW, dual-spin \(P_k^\sigma\))

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-uks-kpoints` |
| Regression | `regressions/dynamics/xl-bomd-uks-kpoints/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{…, n_alpha, n_beta, k_mesh}, k}` bound with `set_evaluator_ref`; propagator class `cpu::electronic::XlBomdCplx` |
| Roadmap | MD AIMD XL multi-k UKS residual closed 2026-07-26 |
| Dual gate | Correctness A6 (e2e NVE drift) + unit A1–A5 (`test_cpu_gpw_xl_bomd`); speed **`perf_exempt`** |
| Wave | CP2K capability loop — harness wave 11 (2026-07-26) |
| Oracle | Offline CP2K twin sketch only (`tools/cp2k_oracle/inputs/xl-bomd-uks-kpoints/`); never live in ctest |

## Summary

**Extended-Lagrangian Born–Oppenheimer MD (XL-BOMD)** in the dissipative
Niklasson form, on **multi-k collinear UKS** Gaussian-and-plane-waves (GPW)
forces. The outer loop is the MD preset (`cpu::md::MD` + NVE
`VelocityVerlet`); the force slot binds a `cpu::electronic::XlBomdForce`
wrapping a `cpu::gpw::GPWEvaluator` with `compute_forces = true`, integer
occupations `n_alpha` / `n_beta`, and a
non-Γ `k_mesh`. Independent auxiliary **complex**
densities \(P_k^\alpha\) and \(P_k^\beta\) are propagated per sampled \(k\)
and used as the multi-k UKS SCF initial densities each ionic step.

This page is the **product path** that closes the remaining XL axis after
Γ-RKS (`xl-bomd-water`), Γ-UKS (`xl-bomd-uks`), and multi-k RKS
(`xl-bomd-kpoints`): dual-spin multi-k XL. The e2e pin is **not** a golden
Atomiverse total energy and **not** a measured CP2K AIMD wall (no `cp2k`
binary on the pin host 2026-07-26). It is the **literature NVE conservation
envelope** plus multi-k dual-spin payload diagnostics. Public energy stays
**eV** on `frame.meta`; internal Ha/Ry are method private.

Related: [xl-bomd-water](xl-bomd-water.md) (Γ RKS foothold),
[gpw-uks-o2-gamma](gpw-uks-o2-gamma.md) (Γ UKS energy),
`regressions/gpw/gpw-kpoint-uks-forces` (multi-k UKS force math).

## Method

### XL-BOMD recursion (per \(k\), per spin)

An auxiliary AO density matrix is a **dynamical variable**, not a Kolafa
ASPC extrapolation of past SCF densities
(`cpu::electronic::ASPCPredictor`). After each SCF, the multi-k UKS path advances
one `XlBomdCplx` per \((k,\sigma)\):

\[
P^{k\sigma}_{n+1}
  = 2\,P^{k\sigma}_{n}
  - P^{k\sigma}_{n-1}
  + \kappa\,\bigl(D^{k\sigma}_{n} - P^{k\sigma}_{n}\bigr)
  + \alpha \sum_{m=0}^{K} c_m\, P^{k\sigma}_{n-m},
\]

with \(D^{k\sigma}_{n}\) the SCF-converged spin density at step \(n\).
Default dissipation order \(K=5\) (Niklasson et al., *JCP* **130**, 214109
(2009), Table I):

| Symbol | Value |
| ------ | ----- |
| \(\kappa\) | \(1.82\) |
| \(\alpha\) | \(0.018\) |
| \(c_0\ldots c_5\) | \(\{-6,\,14,\,-8,\,-3,\,4,\,-1\}\) |

Table-I coefficients for \(K\in[3,9]\) are unit-pinned in
`test_cpu_gpw_xl_bomd`. The recursion is time-reversible up to the
\(O(\alpha)\) dissipation, which removes the systematic energy drift that
plain guess reuse (broken time-reversibility) causes on long BOMD runs.
Time-reversible form: Niklasson, *PRL* **100**, 123004 (2008).

This product path is **SCF-converged** XL-BOMD: SCF still runs to `tol`
each step; XL only improves the multi-k dual-spin density guess (and the
force-pass warm-start). The zero-SCF linearized-functional XL force is
**out of scope**.

### Multi-k UKS GPW forces

- **GPW (Quickstep family):** Gaussian AO basis, density collocation on a
  real-space grid, FFT Poisson Hartree, grid XC, GTH pseudopotentials.
  Citation: Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997).
- **Collinear UKS:** separate \(\alpha/\beta\) densities and KS matrices;
  occupations are explicit members `n_alpha` / `n_beta` (not a multiplicity
  bag). Multi-k UKS solves \(H_k^\sigma\) per sampled \(k\).
- **k-mesh:** `k_mesh = {1,1,2}` with `k_time_reversal = false` → two
  sampled \(k\) (\(n_k = 2\)). Complex Bloch densities \(P_k^\sigma\) are
  Hermitian; real \((\kappa,\alpha,c)\) act element-wise on
  `linalg::CplxMatrix`.
- **XC (this pin):** PADE (LSD) — Goedecker–Teter–Hutter Padé LDA for spin
  densities (*PRB* **54**, 1703 (1996)).
- **Nuclear MD (NVE):** fixed-step symplectic Velocity Verlet; total
  \(E = E_{\mathrm{pot}} + E_{\mathrm{kin}}\) [eV] is conserved up to a
  bounded non-secular envelope when forces are consistent with the
  electronic energy surface (Frenkel & Smit §4.3.4; Marx & Hutter BOMD).

### Why closed-shell fillings on the UKS path?

High-spin H₂ (\(N_\alpha=2\), \(N_\beta=0\)) is unbound at this level and
cannot pin a short NVE drift window. The regression uses **closed-shell
fillings on the collinear UKS path** (\(N_\alpha = N_\beta = 1\)): both
spin channels are non-empty (dual \(P_k^\alpha / P_k^\beta\) XL engages)
while the molecule stays bound. Multi-k UKS force quality is pinned
separately by `gpw-kpoint-uks-forces` / `gpw-kpoints-uks`.

### Primary observable (envelope)

Let \(E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)\) [eV], with \(E_0\)
after one warm-up step (XL histories seeded):

\[
\delta = \max_t \frac{\lvert E(t) - E_0\rvert}{\max\!\bigl(1,\,\lvert E_0\rvert\bigr)}
  \le 0.05.
\]

Additional gates (regression `reg_xl-bomd-uks-kpoints`):

- SCF **converged** at every sampled step with dual multi-k XL guesses
- \(n_k \ge 2\) and `P_k.size() == P_k_beta.size() == n_kpts` after warm-up
- PE finite and bound (\(< -5\) eV after warm-up)
- KE finite, \(\ge 0\), non-trivial after warm-up
- \(\langle S^2\rangle\) finite and \(\ge 0\) (multi-k UKS stores the
  integer seed formula \(s_z(s_z+1)+N_\beta\), not full Löwdin)
- All coordinates finite; soft H–H window \(0.4 < r_{\mathrm{HH}} < 1.5\) Å

### Exclusions (this slug)

- Not Γ-only UKS XL (`xl-bomd-uks`) or multi-k RKS XL (`xl-bomd-kpoints`)
- Not ASPC / second-gen Car–Parrinello (`second-gen-cpmd-*`)
- Not OT + XL, smearing + XL, or DFT+U + XL combinations
- Not zero-SCF / linearized XL forces; not hybrid / meta-GGA XC
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

// Periodic H₂ — cell L = 12 Bohr, compressed R = 1.0 Bohr along x
cpu::Frame frame(2);
constexpr double bohr2ang = 0.52917720859;
constexpr double L_ang = 12.0 * bohr2ang;
// ... place two H at cell centre ± 0.5 R; masses 1.008 amu ...
frame.span(kAtom_vx)[0] = -0.0010;   // antisymmetric stretch [Å/fs]
frame.span(kAtom_vx)[1] =  0.0010;
frame.box = cpu::Box::cubic(L_ang);

const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};

cpu::gpw::GPWEvaluator gpw(
    {H}, /*cutoff_ry*/ 100.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
    /*compute_forces*/ true);
gpw.n_alpha = 1;               // closed-shell fillings on UKS path
gpw.n_beta  = 1;
gpw.k_mesh[0] = 1;
gpw.k_mesh[1] = 1;
gpw.k_mesh[2] = 2;
gpw.k_time_reversal = false;   // n_k = 2
// The wrapper engages the dual XlBomdCplx per k (α + β); K = 5 is the
// Niklasson 2009 Table-I order.
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/ 0.5});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, 1);              // warm-up: seed XL histories
// production: assert |E(t)-E0|/scale ≤ 0.05, SCF converged, dual P_k payloads
```

The complex multi-k propagator is also a public primitive
(`include/atomiverse/cpu/electronic/xl_bomd.h`):

```cpp
cpu::electronic::XlBomdCplx xl(5);   // throws unless 3 <= K <= 9
xl.advance(D_k_scf);          // first call seeds; later calls propagate
const auto& guess = xl.aux(); // complex P_{n+1} for next SCF at this k
```

### How to run the regression

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake --build build -j$(nproc) --target atomiverse reg_xl-bomd-uks-kpoints
  cmake --build build --target run_reg_xl-bomd-uks-kpoints
'
```

Relevant sources:

| Layer | Path |
| ----- | ---- |
| Propagator | `include/atomiverse/cpu/electronic/xl_bomd.h`, `src/cpu/electronic/xl_bomd.cpp` |
| XL wrapper | `include/atomiverse/cpu/electronic/xl_bomd_force.h` (dissipation order \(K\)) |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` (`k_mesh`, `n_alpha`/`n_beta`; private XL mode bits) |
| Multi-k UKS SCF + XL payload | `src/cpu/gpw/gpw_solver.cpp` (`run_gpw_scf_kpoints_uks`) |
| Unit Table-I / dual-spin | `tests/cpu/gpw/` (`test_cpu_gpw_xl_bomd`) |
| E2E | `regressions/dynamics/xl-bomd-uks-kpoints/` |

## Validation (vs CP2K / literature)

Source: `regressions/dynamics/xl-bomd-uks-kpoints/REFERENCE.md` (offline pin;
never live CP2K in ctest).

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic XYZ |
| Spin | collinear UKS, \(n_\alpha=1\), \(n_\beta=1\) (closed-shell fillings) |
| XC | PADE (LSD) |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| SCF | `tol = 1e-5`, `max_iter = 60`, DIIS diagonalization |
| XL-BOMD | \(K=5\); dual `XlBomdCplx` per \(k\) (\(\alpha+\beta\)) |
| Ensemble | NVE; \(\Delta t = 0.5\) fs |
| \(v_0\) | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 1 warm-up + 4 production (chunked) |

### Accuracy table

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.05\) | literature XL-BOMD/NVE envelope | 0.05 | \(E_0\) after warm-up |
| SCF converged every step | true | multi-k UKS + dual XL guesses | — | wrong recursion fails immediately |
| \(n_kpts\) | \(\ge 2\) | mesh \(1\times1\times2\) unreduced | — | payload diagnostic |
| \(\lvert P_k\rvert\), \(\lvert P_k^\beta\rvert\) | \(= n_kpts\) | dual-spin multi-k XL | — | α-only / RKS leak fails |
| PE after warm-up | \(< -5\) eV | bound closed-shell H₂ | — | high-spin H₂ excluded |
| KE after warm-up | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(\langle S^2\rangle\) | finite, \(\ge 0\) | integer seed formula | — | not full Löwdin on multi-k UKS |
| \(r_{\mathrm{HH}}\) after production | \((0.4, 1.5)\) Å | soft bond window | — | no dissociation / collapse |
| Table-I \(\kappa,\alpha,c_k\) (\(K=5\)) | exact | Niklasson 2009 Table I | unit | `test_cpu_gpw_xl_bomd` |

### Offline CP2K twin (sketch only)

| Field | Value |
| ----- | ----- |
| Input | `tools/cp2k_oracle/inputs/xl-bomd-uks-kpoints/h2_uks_xl_bomd_kpoints_nve.inp` |
| Intent | Short NVE Quickstep UKS MD with `&KPOINTS` MONKHORST-PACK \(1\,1\,2\), ASPC density-guess extrapolation (CP2K MD default analogue of ATV multi-k UKS XL) |
| Status | **Unexecuted** on pin host 2026-07-26 (no `cp2k` binary / Lmod module) |
| When binary available | Re-run offline; replace pin section with CP2K version + measured drift + fair wall |

CP2K `EXTRAPOLATION ASPC` is **not** identical to Niklasson XL-BOMD; the twin
is a method-family sketch for inventory and future fair-bench, not a
bit-identical density history.

## Performance

Source: the case `REFERENCE.md` (wave 11),
`REFERENCE.md` Perf gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | -----------------------------------------: | ---- | ------ |
| multi-k UKS XL AIMD | n/a | n/a | **perf_exempt** | n/a | No offline CP2K binary for fair wall of multi-step multi-k UKS AIMD (2026-07-26 inventory) |

Gate: \(r \le 10\) unless `perf_exempt`. This slug is **exempt** for the same
reason as `xl-bomd-water` / `xl-bomd-uks` / `xl-bomd-kpoints`: correctness is
the NVE drift envelope, not a wall-clock ratio.

## Limitations

- **Envelope, not absolute energy.** A broken dual-spin multi-k XL guess
  destroys SCF convergence or the drift window; a correct path is not a
  high-precision total-energy twin of CP2K Quickstep.
- **Closed-shell UKS fillings only on this pin.** Open-shell multi-k UKS XL
  on a bound system (e.g. O₂ mesh) is a natural follow-on, not claimed here.
- **SCF-converged XL only.** No zero-SCF linearized force; no OT-only XL.
- **Spin metric.** Multi-k UKS \(\langle S^2\rangle\) is the integer seed
  formula, not the full Löwdin diagnostic used on Γ UKS.
- **Cutoff / basis cheap for envelope.** Absolute multi-k UKS forces live in
  `gpw-kpoint-uks-forces`; this regression does not re-pin force vs CP2K.
- **Perf debt.** Fair wall vs CP2K deferred until a CP2K binary is available
  offline; do not invent ratios.
- **Combinatorial gaps out of scope:** XL + OT + \(k\), XL + smearing, XL +
  DFT+U + \(k\), hybrid XC XL — separate queue rows if ever productized.

## References

- Offline pin: `regressions/dynamics/xl-bomd-uks-kpoints/REFERENCE.md`
- Regression: `regressions/dynamics/xl-bomd-uks-kpoints/reg_xl-bomd-uks-kpoints.cpp`
- CP2K twin sketch: `tools/cp2k_oracle/inputs/xl-bomd-uks-kpoints/h2_uks_xl_bomd_kpoints_nve.inp`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Prior product paths: [xl-bomd-water](xl-bomd-water.md);
  `regressions/dynamics/xl-bomd-uks`; `regressions/dynamics/xl-bomd-kpoints`;
  multi-k UKS forces `regressions/gpw/gpw-kpoint-uks-forces`
- A. M. N. Niklasson, *Phys. Rev. Lett.* **100**, 123004 (2008)
- A. M. N. Niklasson et al., *J. Chem. Phys.* **130**, 214109 (2009) — Table I
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997)
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996)
