# Multi-k second-generation CP (SGCP) — H₂ NVE (GPW, ASPC on \(P_k\))

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `second-gen-cpmd-kpoints` |
| Regression | `regressions/dynamics/second-gen-cpmd-kpoints/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::SecondGenCp{cpu::gpw::GPWEvaluator{…, k_mesh}, aspc_k, corrector_max_iter}` bound with `set_evaluator_ref`; predictor `cpu::electronic::ASPCPredictorCplx` |
| Roadmap | MD AIMD multi-k SGCP residual closed 2026-07-26 |
| Dual gate | Correctness (NVE drift + corrector budget + multi-k payload); speed **`perf_exempt`** |
| Wave | CP2K capability loop — harness wave 12 (2026-07-26) |
| Oracle | Offline CP2K twin sketch only (`tools/cp2k_oracle/inputs/second-gen-cpmd-kpoints/`); never live in ctest |

## Summary

**Second-generation Car–Parrinello MD (SGCP / 2nd-gen CP)** on **multi-k
closed-shell RKS** Gaussian-and-plane-waves (GPW / Quickstep) forces. The
outer loop is the MD preset (`cpu::md::MD` + NVE `VelocityVerlet`); the force
slot binds `cpu::gpw::GPWEvaluator` with `compute_forces = true`, a non-Γ
`k_mesh`, and `second_gen_cp = true`. Per sampled \(k\), a Kolafa **ASPC**
predictor (`cpu::electronic::ASPCPredictorCplx`) extrapolates the complex Bloch
density \(P_k\) from prior ionic steps; a **partial SCF corrector**
(iteration budget ≪ full BOMD) refines that guess; analytic multi-k forces
are taken at the post-corrector density.

This page is the **product multi-k SGCP path** that closes the remaining
ASPC axis after Γ-RKS (`second-gen-cpmd-water`), Γ-UKS (`second-gen-cpmd-uks`),
and Γ OT correctors (`second-gen-cpmd-ot` / `second-gen-cpmd-uks-ot`): ASPC on
complex \(P_k\) with a multi-k corrector budget. The e2e pin is **not** a
golden Atomiverse total energy and **not** a measured CP2K AIMD wall (no
`cp2k` binary on the pin host 2026-07-26). It is the **literature SGCP/NVE
conservation envelope** plus multi-k payload diagnostics and the production
corrector-budget gate. Public energy stays **eV** on `frame.meta`; internal
Ha/Ry are method private.

Related: multi-k density-guess socket twin [xl-bomd-uks-kpoints](xl-bomd-uks-kpoints.md)
(Niklasson XL, not ASPC); multi-k force math `regressions/gpw/gpw-kpoint-forces`;
classical CPMD [cpmd-water](cpmd-water.md) (fictitious \(\mu\), not SGCP);
Γ BOMD foothold [gpw-bomd-water](gpw-bomd-water.md).

## Method

### Second-generation CP (Kühne SGCP)

Classical CPMD treats occupied KS orbitals as fictitious classical DOF with
mass \(\mu\) and integrates them every step without SCF (Car & Parrinello,
*PRL* **55**, 2471 (1985)). **Second-generation CP** (Kühne, Krack, Mohamed,
Parrinello, *PRL* **98**, 066401 (2007)) instead keeps a **Born–Oppenheimer
nuclear integrator** and replaces full SCF each step by:

1. **ASPC density predictor** (Kolafa, *J. Comput. Chem.* **25**, 335 (2004))
2. A **few corrector SCF iterations** at the new ionic geometry
3. Analytic forces at the post-corrector density

There is **no** fictitious electronic mass and **no** \(T_e\) term on the
conserved energy. The nuclear NVE conserved quantity is the ordinary

\[
E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)
\quad\text{[eV on Frame]},
\]

not the classical CP \(E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} +
E_{\mathrm{KS}}\).

**Not BOMD:** production steps cap SCF at `corrector_max_iter` once ASPC
is warm — they must not run a full `max_iter` BOMD SCF.
**Not classical CPMD:** `cpmd = false`; no electronic RATTLE / \(\mu\).
**Not XL-BOMD:** ASPC is Lagrange history extrapolation + Kolafa blend, not
Niklasson auxiliary Verlet on a dynamical \(P\).

### ASPC on complex Bloch densities

For multi-k RKS the product path keeps **one** `ASPCPredictorCplx` per
sampled \(k\). Real Lagrange coefficients \(\omega_m\) act **element-wise**
on Hermitian \(P_k \in \mathbb{C}^{n_{\mathrm{ao}}\times n_{\mathrm{ao}}}\):

\[
P_k^{\mathrm{pred}}
  = \sum_{m=0}^{K} \omega_m\, P_k^{(n-m)},
\]

requiring \(K+1\) history densities before prediction is available (this pin:
\(K=1\) → 2 seed SCF steps). After a partial SCF at the new geometry yields
\(P_k^{\mathrm{scf}}\), the Kolafa corrector blend

\[
P_k^{\mathrm{corr}}
  = \frac{K+2}{2K+3}\, P_k^{\mathrm{scf}}
  + \frac{K+1}{2K+3}\, P_k^{\mathrm{pred}}
\]

is recorded for the next step (`ASPCPredictor` / `ASPCPredictorCplx::correct`).

| Symbol | This pin | Role |
| ------ | -------- | ---- |
| \(K\) | `aspc_k = 1` | ASPC Lagrange order |
| Corrector budget | `corrector_max_iter = 5` | SCF iters after ASPC warm |
| Cold SCF | `max_iter = 60`, `tol = 1e-5` | Seed steps / insufficient history |

Unit coverage of multi-k ASPC wiring and mutual exclusion with `cpmd` /
`xl_bomd` / multi-k+`use_ot` lives in `test_cpu_gpw_second_gen_cpmd`
(`tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp`).

### Multi-k RKS GPW forces

- **GPW (Quickstep family):** Gaussian AO basis, density collocation on a
  real-space grid, FFT Poisson Hartree, grid XC, GTH pseudopotentials.
  Citation: Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997);
  VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005).
- **Closed-shell RKS:** one occupied band for H₂; complex Bloch densities
  \(P_k\) per sampled \(k\).
- **k-mesh:** `k_mesh = {1,1,2}` with `k_time_reversal = false` → two
  sampled \(k\) (\(n_k = 2\)).
- **XC (this pin):** PADE (LDA) — Goedecker–Teter–Hutter Padé LDA
  (*PRB* **54**, 1703 (1996)).
- **Nuclear MD (NVE):** fixed-step symplectic Velocity Verlet; total
  \(E = E_{\mathrm{pot}} + E_{\mathrm{kin}}\) [eV] is conserved up to a
  bounded non-secular envelope when forces are consistent with the
  electronic energy surface (Frenkel & Smit §4.3.4; Marx & Hutter AIMD).

### Primary observable (envelope)

Let \(E(t) = E_{\mathrm{pot}}(t) + E_{\mathrm{kin}}(t)\) [eV], with \(E_0\)
after seed + first warm ASPC production step:

\[
\delta = \max_t \frac{\lvert E(t) - E_0\rvert}{\max\!\bigl(1,\,\lvert E_0\rvert\bigr)}
  \le 0.10.
\]

The soft 0.10 window (vs 0.05 on full-SCF XL/BOMD pins) is intentional:
partial-SCF heat on a short multi-k traj is larger than fully converged
BOMD; the gate still fails a broken ASPC or a path that silently falls
back to full BOMD.

Additional gates (`reg_second-gen-cpmd-kpoints`):

- Production SCF **iters ≤ `corrector_max_iter`** once ASPC is warm
  (SGCP signature vs full BOMD)
- \(n_k \ge 2\) and `P_k.size() == n_kpts` after seed
- PE finite and bound (\(< -5\) eV after seed)
- KE finite, \(\ge 0\), non-trivial after seed
- All coordinates finite; soft H–H window \(0.4 < r_{\mathrm{HH}} < 1.5\) Å

### Exclusions (this slug)

- Not Γ-only SGCP (`second-gen-cpmd-water` / UKS / OT)
- Not classical CPMD (`cpmd-*`) or XL-BOMD (`xl-bomd-*`)
- Not multi-k + OT corrector (build throws; no \(P_k\) OT warm-start)
- Not multi-k UKS dual ASPC e2e (build accepts dual \(P_k^\sigma\); not
  pinned here — closed-shell multi-k RKS only)
- Not hybrid / meta-GGA XC; not DFT+U + SGCP
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
#include "atomiverse/cpu/electronic/second_gen_cp.h"
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

constexpr double dt = 0.5;          // fs
constexpr int corrector_budget = 5; // SCF iters per warm production step
constexpr int n_seed = 2;           // ASPC K=1 needs K+1 densities

cpu::gpw::GPWEvaluator gpw(
    /*kinds*/ {H}, /*cutoff_ry*/ 100.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
    /*compute_forces*/ true);
gpw.k_mesh[0] = 1;
gpw.k_mesh[1] = 1;
gpw.k_mesh[2] = 2;
gpw.k_time_reversal = false;        // n_k = 2
// SGCP (ASPC on the complex P_k + partial SCF) is the wrapper TYPE: the
// Lagrange order K (needs K+1 seeds) and the production corrector budget are
// its constructor arguments — the evaluator has no public mode flags.
cpu::electronic::SecondGenCp eval{std::move(gpw), 1, corrector_budget};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, n_seed);   // seed: fill ASPC histories (full SCF OK)
// production: eval.last_result().iters <= corrector_budget,
//             |E(t)-E0|/scale <= 0.10, last_result().n_kpts >= 2
```

The complex ASPC primitive is also public (`include/atomiverse/cpu/electronic/aspc_predictor.h`):

```cpp
cpu::electronic::ASPCPredictorCplx aspc(/*order K=*/1);
// after cold multi-k SCF: aspc.record(P_k);
// when history_size() >= K+1:
//   aspc.predict(P_pred);  // partial SCF → P_scf
//   aspc.correct(P_scf, P_pred, P_corr);
//   aspc.record(P_corr);
```

### How to run the regression

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake --build build -j$(nproc) --target atomiverse reg_second-gen-cpmd-kpoints
  cmake --build build --target run_reg_second-gen-cpmd-kpoints
'
```

Relevant sources:

| Layer | Path |
| ----- | ---- |
| SGCP wrapper | `include/atomiverse/cpu/electronic/second_gen_cp.h` (`aspc_k`, `corrector_max_iter`) |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` (`k_mesh`, `k_time_reversal`; private SGCP mode bits) |
| ASPC real / complex | `include/atomiverse/cpu/electronic/aspc_predictor.h` |
| Multi-k RKS SCF + SGCP payload | `src/cpu/gpw/` (GPW multi-k force path + ASPC seed/correct) |
| Unit multi-k build / OT refuse | `tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp` |
| E2E | `regressions/dynamics/second-gen-cpmd-kpoints/` |

## Validation (vs CP2K / literature)

Source: `regressions/dynamics/second-gen-cpmd-kpoints/REFERENCE.md` (offline
pin; never live CP2K in ctest).

### System

| Quantity | Value |
| -------- | ----- |
| Species | H₂ (2 × H), periodic |
| Geometry | \(R = 1.0\) Bohr compressed bond along \(x\), cell-centred |
| Masses | 1.008 amu each |
| Cell | cubic \(L = 12\) Bohr ≈ 6.350 Å; periodic XYZ |
| Spin | closed-shell RKS, \(n_{\mathrm{occ}}=1\) |
| XC | PADE (LDA) |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| CUTOFF | 100 Ry (envelope run) |
| k-mesh | \(1\times1\times2\), `k_time_reversal=false` → \(n_k=2\) |
| ASPC \(K\) | 1 (needs \(K+1=2\) seed densities) |
| Corrector SCF | `corrector_max_iter = 5` after ASPC warm |
| Cold SCF | `tol = 1e-5`, `max_iter = 60`, DIIS diagonalization |
| Ensemble | NVE; \(\Delta t = 0.5\) fs |
| \(v_0\) | antisymmetric stretch \(v_x = \pm 0.001\) Å/fs |
| Steps | 2 seed + 1 baseline warm ASPC + 2 production |

### Accuracy table

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| Relative NVE drift \(\delta\) | \(\le 0.10\) | literature SGCP/NVE envelope | 0.10 | \(E_0\) after seed + first warm ASPC step |
| Production SCF iters | \(\le 5\) | SGCP partial corrector | — | full BOMD `max_iter` fails gate |
| \(n_kpts\) | \(\ge 2\) | mesh \(1\times1\times2\) unreduced | — | payload diagnostic |
| \(\lvert P_k\rvert\) | \(= n_kpts\) | multi-k ASPC seed | — | Γ-only / missing \(P_k\) fails |
| PE after seed | \(< -5\) eV | bound closed-shell H₂ | — | |
| KE after seed | \(\ge 0\), non-trivial | NVE stretch | — | |
| \(r_{\mathrm{HH}}\) after production | \((0.4, 1.5)\) Å | soft bond window | — | no dissociation / collapse |
| Multi-k build + OT refuse | unit | `test_cpu_gpw_second_gen_cpmd` | unit | multi-k+OT throws |

### Offline CP2K twin (sketch only)

| Field | Value |
| ----- | ----- |
| Input | `tools/cp2k_oracle/inputs/second-gen-cpmd-kpoints/h2_sgcp_kpoints_nve.inp` |
| Intent | Short NVE Quickstep RKS MD with `&KPOINTS` MONKHORST-PACK \(1\,1\,2\), `EXTRAPOLATION ASPC` order 1, `MAX_SCF 5` (SGCP signature) |
| Status | **Unexecuted** on pin host 2026-07-26 (no `cp2k` binary / Lmod module) |
| When binary available | Re-run offline; replace pin section with CP2K version + measured drift + fair wall |

CP2K `EXTRAPOLATION ASPC` + limited `MAX_SCF` is the method-family twin of
ATV multi-k SGCP; the sketch is for inventory and future fair-bench, not a
bit-identical density history or energy.

## Performance

Source: the case `REFERENCE.md` (wave 12),
`REFERENCE.md` Perf gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | -----------------------------------------: | ---- | ------ |
| multi-k SGCP AIMD | n/a | n/a | **perf_exempt** | n/a | No offline CP2K binary for fair wall of multi-step multi-k SGCP (2026-07-26 inventory) |

Gate: \(r \le 10\) unless `perf_exempt`. This slug is **exempt** for the same
reason as `xl-bomd-kpoints` / `second-gen-cpmd-water`: correctness is the
NVE drift + corrector-budget envelope, not a wall-clock ratio.

## Limitations

- **Envelope, not absolute energy.** A broken multi-k ASPC guess or a path
  that ignores the corrector cap fails the drift / iters gates; a correct
  path is not a high-precision total-energy twin of CP2K Quickstep.
- **Closed-shell multi-k RKS only on this pin.** Multi-k UKS dual ASPC is
  accepted at `build()` but has no dedicated e2e envelope here.
- **Diagonalization corrector only.** Multi-k + OT is refused; Γ OT SGCP
  is a separate product path (`second-gen-cpmd-ot`).
- **Soft drift tol.** Partial-SCF heat justifies \(\delta \le 0.10\) on a
  short traj; do not tighten without a longer literature-aligned window.
- **Cutoff / basis cheap for envelope.** Absolute multi-k forces live in
  `gpw-kpoint-forces`; this regression does not re-pin force vs CP2K.
- **Perf debt.** Fair wall vs CP2K deferred until a CP2K binary is available
  offline; do not invent ratios.
- **Combinatorial gaps out of scope:** SGCP + smearing, SGCP + DFT+U + \(k\),
  hybrid XC SGCP, multi-k classical CPMD (`cpmd-kpoints` incomplete) —
  separate queue rows.

## References

- Offline pin: `regressions/dynamics/second-gen-cpmd-kpoints/REFERENCE.md`
- Regression: `regressions/dynamics/second-gen-cpmd-kpoints/reg_second-gen-cpmd-kpoints.cpp`
- CP2K twin sketch: `tools/cp2k_oracle/inputs/second-gen-cpmd-kpoints/h2_sgcp_kpoints_nve.inp`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Prior product paths: `regressions/dynamics/second-gen-cpmd-water`;
  multi-k force pin `regressions/gpw/gpw-kpoint-forces`;
  multi-k XL twin [xl-bomd-uks-kpoints](xl-bomd-uks-kpoints.md);
  classical CP [cpmd-water](cpmd-water.md)
- T. D. Kühne, M. Krack, F. R. Mohamed, M. Parrinello, *Phys. Rev. Lett.*
  **98**, 066401 (2007) — second-generation CP
- J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) — ASPC
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997)
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
