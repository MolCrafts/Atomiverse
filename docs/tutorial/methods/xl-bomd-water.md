# XL-BOMD — extended-Lagrangian BOMD (H₂O NVE, GPW forces)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-water` |
| Regression | `regressions/dynamics/xl-bomd-water/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdForce{cpu::gpw::GPWEvaluator{…, compute_forces=true}, k=5}` bound with `set_evaluator_ref`; propagator class `cpu::electronic::XlBomd` |
| Roadmap | MD roadmap XL-BOMD (`ASPC ≠ XL`) |
| Dual gate | Correctness A1–A9 of the spec acceptance; speed **`perf_exempt`** (no CP2K AIMD twin wall) |
| Wave | CP2K capability loop — wave 3 |

## Summary

**Extended-Lagrangian Born–Oppenheimer MD** (XL-BOMD; Niklasson, *PRL*
**100**, 123004 (2008); dissipative form Niklasson et al., *JCP* **130**,
214109 (2009)) propagates an auxiliary AO density matrix \(P\) alongside the
nuclei and uses it as the SCF initial guess each MD step:

\[
P_{n+1} = 2P_n - P_{n-1} + \kappa\,(D_n - P_n)
          + \alpha \sum_{k=0}^{K} c_k P_{n-k},
\]

with \(D_n\) the SCF density at step \(n\) and \((\kappa, \alpha, c_k)\) from
Table I of the 2009 paper (\(K \in [3,9]\); default \(K=5\):
\(\kappa=1.82\), \(\alpha=0.018\), \(c=\{-6,14,-8,-3,4,-1\}\)). Unlike the
Kolafa **ASPC extrapolation** (`cpu::electronic::ASPCPredictor`), the auxiliary
density is a *dynamical variable*: the recursion is time-reversible up to
the \(O(\alpha)\) dissipation, which removes the systematic energy drift
that plain guess reuse (broken time reversibility) causes in long BOMD
trajectories. This foothold is the **SCF-converged** form: SCF still runs
to `tol` each step; the XL machinery improves the guess (and in XL mode the
force-pass SCF also warm-starts from the just-converged MOs).

Scope: closed-shell RKS Γ-point diagonalization/DIIS GPW path. UKS, OT, and
the zero-SCF linearized-functional XL force are refused loudly
(`std::invalid_argument`), never silently degraded.

## Usage

```cpp
cpu::gpw::GPWEvaluator gpw(
    {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
// XL is a wrapper TYPE around the pure evaluator, not an `xl_bomd = true`
// switch: the Niklasson 2009 Table-I dissipation order K is a constructor
// argument, and the mode bits it sets are private on GPWEvaluator.
cpu::electronic::XlBomdForce eval{std::move(gpw), /*k=*/5};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.5});   // fs
sim.set_evaluator_ref(eval);   // borrow: readbacks stay live after run()
sim.build();
sim.run(frame, n_steps);   // frame.meta.get(kEnergy) [eV]
```

The propagator itself is a public primitive (`cpu/electronic/xl_bomd.h`):

```cpp
cpu::electronic::XlBomd xl(5);      // throws unless 3 <= K <= 9
xl.advance(D_scf);           // first call seeds; later calls propagate
const auto& guess = xl.aux(); // SCF initial-density guess for the next step
```

## Pin

- **Unit:** exact Table-I coefficients (`test_cpu_gpw_xl_bomd`), fixed point
  on a constant density, bounded response under oscillatory driving,
  `GpwDft::run` `guess_P` plumbing (converged-density guess reaches the cold
  fixed point in ≤ cold iterations).
- **E2E:** `reg_xl-bomd-water` — the `gpw-bomd-water` H₂O NVE twin (PBE /
  DZVP-MOLOPT-GTH / GTH-PBE / MT, 200 Ry, dt 0.5 fs, 1 + 4 steps) with
  `xl_bomd = true`: relative total-energy drift ≤ 0.05, SCF converged at
  every sampled step, bounded O–H, finite trajectory. See
  `regressions/dynamics/xl-bomd-water/REFERENCE.md` (no CP2K binary on the
  worker host — literature envelope pin; CP2K twin input kept at
  `tools/cp2k_oracle/inputs/xl_bomd_water_nve.inp`).
