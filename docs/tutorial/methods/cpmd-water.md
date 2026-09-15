# Car–Parrinello MD — H₂O NVE (GPW fictitious electronic DOF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `cpmd-water` |
| Regression | `regressions/dynamics/cpmd-water/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::CarParrinello{cpu::gpw::GPWEvaluator{…, compute_forces=true}, mu_au}` bound with `set_evaluator_ref`; propagator `cpu::electronic::Cpmd`. CP mode is the **wrapper type**, not a flag — `cpmd`/`cpmd_mu` are private on `GPWEvaluator`. The nuclear dt is pushed from the integrator into the wrapper by `MD` (`TimestepProvider` → `TimestepConsumer`) — never set on the evaluator |
| Roadmap | MD roadmap Car–Parrinello |
| Dual gate | Correctness A1–A15 of the spec acceptance; speed **`perf_exempt`** |
| Wave | CP2K capability loop — wave 3 |

## Summary

**Car–Parrinello molecular dynamics** (Car & Parrinello, *PRL* **55**, 2471
(1985)) treats the occupied Kohn–Sham orbitals as classical fictitious
degrees of freedom with mass \(\mu\). The extended Lagrangian is

\[
\mathcal{L} = \sum_I \tfrac12 M_I \dot R_I^2
            + \sum_i \tfrac12 \mu \lVert \dot c_i\rVert^2
            - E_{\mathrm{KS}}[C,R]
            - \text{constraints }(C^\top S C = I).
\]

Nuclei integrate with the ordinary MD `VelocityVerlet`. Electrons integrate
with velocity-Verlet + electronic RATTLE on the occupied MO matrix \(C\)
(\(\mu\,\ddot C = -4 H C\) for closed-shell \(P = 2 C C^\top\); product path
is RATTLE — this document retains the historical Löwdin foothold narrative).
After one cold SCF seed,
**production steps do not re-converge SCF** — they evaluate the KS energy and
matrix once at the current density and take analytic nuclear forces at that
fixed electronic state.

This is **not** Born–Oppenheimer MD (BOMD: SCF every step) and **not**
XL-BOMD (auxiliary density + SCF). The literature conserved quantity is

\[
E_{\mathrm{cons}} = T_{\mathrm{n}} + T_{\mathrm{e}} + E_{\mathrm{KS}},
\]

with \(T_{\mathrm{e}} = \tfrac12\mu\sum_i\lVert\dot c_i\rVert^2\) published on
`frame.meta[kEnergyElectronKinetic]` [eV] (also readable as
`last_electronic_ke_ev()`) and \(E_{\mathrm{KS}}\) on
`frame.meta.get(kEnergy)`. The sum itself is assembled by
`cpu::md::CpmdConserved` — see *Watching the conserved energy* below.

Scope: closed-shell RKS Γ-point GPW. UKS, OT, and simultaneous `xl_bomd` are
refused loudly at `build()`.

## Usage

```cpp
constexpr double dt = 0.1; // fs — nuclear and electronic share one step
cpu::gpw::GPWEvaluator gpw(
    {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
    /*mix*/ 0.3, cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);
// Car–Parrinello is a wrapper TYPE around the pure GPW evaluator, not a
// `method.cpmd = true` switch: the fictitious mass mu [a.u. electron mass] is a
// constructor argument, and the mode bits it sets are private on GPWEvaluator.
// mu = 2000 here because the AO residual is stiffer than the plane-wave
// textbook mu ~ 400. No electronic dt: MD pushes the integrator's dt into the
// wrapper at build.
cpu::electronic::CarParrinello eval{std::move(gpw), /*mu_au=*/2000.0};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});  // the ONE owner of dt
sim.set_evaluator_ref(eval);        // borrow: readbacks stay live after run()
sim.set_observer(cpu::md::CpmdConserved{});       // E_cons -> meta[kConserved]
sim.build();
sim.run(frame, n_steps);

// CP conserved energy [eV] — one channel, assembled by the engine:
const double e_cons = frame.meta.get(kConserved);
```

### Watching the conserved energy

`cpu::md::CpmdConserved` (`cpu/dynamics/cpmd_conserved.h`) is the single owner
of \(E_{\mathrm{cons}}\): bound at `md::Phase::end_of_step` (default order −90,
i.e. after `KineticEnergy`), it sums `kEnergy + kKinetic +
kEnergyElectronKinetic + kEnergyElectronThermostat` onto
`frame.meta[kConserved]` [eV], so Thermo's `Conserved` column reports CP drift
instead of a nuclear-only quantity — and it throws (naming the missing key)
rather than substituting 0 when no CPMD evaluator ran. Do **not** combine it
with another `kConserved` owner (`NoseHooverNPT`, `NphIntegrator`,
`ParrinelloRahmanNPT`, `FixCSVR`): two components would fight over one channel,
and CPMD is an NVE-nuclei path anyway.

Because the value lives on the frame, a drift watchdog is ordinary user code —
a `Criterion` written outside the library (the LAMMPS `fix halt` shape):

```cpp
class StopWhenDrift : public cpu::Criterion {
  public:
    explicit StopWhenDrift(double max_rel) : max_rel_(max_rel) {}
    std::string name() const override { return "stop_when_drift"; }
    void launch(...) override {}  // verdict-only — no per-step work
    ATV::Verdict verdict(const cpu::LoopState& s) override {
        const double e = s.frame.meta.get(ATV::kConserved);
        if (!seeded_) { e0_ = e; seeded_ = true; return ATV::Verdict::Continue; }
        return (std::abs(e - e0_) > max_rel_ * std::abs(e0_))
                   ? ATV::Verdict::Stop : ATV::Verdict::Continue;
    }
  private:
    double max_rel_, e0_ = 0.0; bool seeded_ = false;
};

sim.add_criterion(StopWhenDrift{1e-3});  // halt past 0.1 % relative drift
```

Seed on the first poll, not at construction: the baseline must be the step that
engaged \(T_e\) (the cold-SCF seed step has \(T_e = 0\), so anchoring there
would count the electronic turn-on as drift).

The propagator primitive (`cpu/electronic/cpmd.h`):

```cpp
cpu::electronic::Cpmd cp(2000.0);  // throws if mu <= 0
cp.seed(C_occ);             // after cold SCF
cp.align_to_overlap(S);     // after nuclei move
cp.advance(H, S, dt_fs);    // projected Verlet + Löwdin
double te_ha = cp.kinetic_ha();
```

## Pin

- **Unit:** mass/seed/advance guards; `GPWEvaluator` refuses UKS / OT /
  `xl_bomd` / non-positive \(\mu,\Delta t\) (`test_cpu_gpw_cpmd`).
- **E2E:** `reg_cpmd-water` — H₂O NVE twin family (PBE / DZVP-MOLOPT-GTH /
  GTH-PBE / MT, 200 Ry, dt 0.1 fs, \(\mu=2000\)): relative \(E_{\mathrm{cons}}\)
  drift ~8×10⁻⁷ (tol 0.10) after Te is engaged, read from
  `frame.meta[kConserved]`; production `iters==0`; bounded O–H; wall ~32 s. See
  `regressions/dynamics/cpmd-water/REFERENCE.md`.

## Honest non-goals

- Full RATTLE constraint forces (Löwdin reorth only)
- Electronic thermostat / NVT-CP
- UKS, OT seed, k-points
- CP2K binary energy twin (literature envelope + structural gates;
  `perf_exempt`)
- Second-generation CP (`second-gen-cpmd-water`) — separate queue row
