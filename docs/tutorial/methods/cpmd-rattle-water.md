# Car–Parrinello MD + electronic RATTLE — H₂O NVE (GPW)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-rattle-water` |
| Regression | `regressions/dynamics/cpmd-rattle-water/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{…, compute_forces=true}, mu_au}` bound with `set_evaluator_ref`; propagator `cpu::electronic::Cpmd` (RATTLE). CP mode is the **wrapper type**, not a flag. The nuclear dt is pushed from the integrator into the wrapper by `MD` (`TimestepProvider` → `TimestepConsumer`) — never set on the evaluator |
| Dual gate | Correctness A1–A16 of the spec acceptance; speed **`perf_exempt`** |
| Wave | CP2K capability loop — wave 5 |

## Summary

Upgrade of the **Löwdin foothold** (`cpmd-water`) to **electronic RATTLE**
(Andersen 1983) on the occupied MO coefficients. The Car–Parrinello extended
Lagrangian still carries fictitious electronic kinetic energy \(T_e\), but
the holonomic constraint \(C^\top S C = I\) is enforced by Lagrange-multiplier
position and velocity stages rather than post-hoc Löwdin clean-up.

After one cold SCF seed, production force steps remain **fixed-density** KS
evaluations (`iters==0`) — not BOMD, not XL-BOMD, not classical nuclear
`FixRattle`.

Conserved quantity (eV):

\[
E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}}.
\]

Regression gate: relative drift \(\le 0.05\) (tighter than the Löwdin 0.10
envelope). Offline CP2K twin input:
`tools/cp2k_oracle/inputs/cpmd_rattle_water_nve.inp`.

## Usage

```cpp
cpu::gpw::GPWEvaluator gpw({O, H}, /*cutoff_ry*/200.0, /*max_iter*/60,
                           /*tol*/1e-5, /*mix*/0.3, cpu::gpw::XcKind::PBE,
                           /*nonperiodic*/true, /*compute_forces*/true);
// mu [a.u. electron mass] is a constructor argument of the CP wrapper; the
// mode bits it flips are private on GPWEvaluator.
cpu::electronic::CarParrinello eval{std::move(gpw), /*mu_au=*/2000.0};
// No electronic dt: MD pushes the integrator's dt into the wrapper at build.

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.1});  // fs — the ONE owner of dt
sim.set_evaluator_ref(eval);
sim.build();
sim.run(frame, nsteps);

const double Te = eval.last_electronic_ke_ev(); // [eV]
const double Econs = frame.meta.get(kEnergy) + frame.meta.get(kKinetic) + Te;
```

## Scope / non-goals

| In | Out |
| -- | --- |
| Closed-shell RKS Γ GPW | UKS, OT, k-points |
| Electronic RATTLE (pos + vel) | Nuclear RATTLE (`FixRattle`) |
| Short H₂O NVE envelope | Live CP2K in ctest; multi-ps production |

## Dual gate

- **Unit:** `test_cpu_gpw_cpmd` — orthonormality residual, velocity
  antisymmetry, existing evaluator guards.
- **E2E:** `reg_cpmd-rattle-water` — H₂O NVE twin (PBE / DZVP-MOLOPT-GTH /
  GTH-PBE, 200 Ry, μ=2000 a.u., dt=0.1 fs). See REFERENCE.md.
