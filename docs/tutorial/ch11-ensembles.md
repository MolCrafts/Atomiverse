# Chapter 11. Ensembles

Thermostats and barostats: `NoseHoover`, `FixLangevin`, Berendsen `Barostat`,
`NoseHooverNPT`.

NVE must be solid before NVT/NPT. From the live NVE run (Chapter 10):

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true}
data:
  values:
    - {step: 0, E: -9.622484}
    - {step: 200, E: -9.622481}
    - {step: 400, E: -9.622482}
    - {step: 600, E: -9.622480}
    - {step: 800, E: -9.622481}
    - {step: 1000, E: -9.622481}
encoding:
  x: {field: step, type: quantitative, title: Step}
  y: {field: E, type: quantitative, title: "E_total (eV)", scale: {zero: false}}
```

Relative drift \(\sim 3\times 10^{-7}\) over 1000 steps. Thermostats **exchange**
energy with a bath (total energy is no longer conserved); acceptance is then
mean temperature and pressure vs set points — exercised in unit tests and
benchmarks, not in the 1000-step tutorial binary.

[Chapter 10](ch10-classical-md.md) built a complete classical MD run and, at
the end, added a thermostat and a barostat almost in passing. This chapter
takes ensembles seriously. An MD trajectory does not just "run" — it samples
a *statistical ensemble*, and which ensemble you sample (microcanonical NVE,
canonical NVT, isothermal-isobaric NPT) is decided entirely by which
temperature- and pressure-control components you attach to the driver. The
loop, the frame, and the force evaluator are identical across all three.

The six questions:

1. **The concept.** What NVE / NVT / NPT are and why you would want each.
2. **The Atomiverse representation.** Thermostats and barostats as
   role-tagged components, not driver modes.
3. **The C++ API.** `set_integrator`, `set_thermostat`, `set_barostat`,
   `set_nvt`, and the five control components.
4. **The expected result.** ⟨T⟩ and ⟨P⟩ that converge to their setpoints
   with the correct fluctuations.
5. **The validation.** Ensemble means against LAMMPS; conserved-quantity
   drift for the deterministic thermostats.
6. **The recurring pattern.** The same ensemble machinery drives *ab
   initio* MD by swapping only the force provider.

This chapter assumes the FCC-argon setup and the `cpu::md::MD` categorical
setters from [Chapter 10](ch10-classical-md.md); code below reuses its
`make_fcc_argon(n_cells, a)` helper.

## The Concept: A Trajectory Samples an Ensemble

Statistical mechanics computes observables as averages over an *ensemble* —
a distribution of microstates consistent with a set of fixed macroscopic
constraints. MD generates that ensemble as a time series, and the constraints
you impose choose which ensemble:

- **NVE (microcanonical).** Fixed particle number, volume, and total energy.
  No thermostat, no barostat — just Newton's equations. Energy is conserved
  (up to integrator error), temperature fluctuates freely. This is the
  ensemble to certify your *integrator*: a flat total-energy trace is the
  canary for time-step and force-consistency bugs.
- **NVT (canonical).** Fixed N, V, and *temperature*. A thermostat exchanges
  energy with a fictitious heat bath so the system samples the Boltzmann
  distribution at temperature $T$. This is the everyday equilibrium ensemble
  — you want a target temperature, not a target energy.
- **NPT (isothermal-isobaric).** Fixed N, *pressure*, and temperature. A
  barostat rescales the box to hold pressure at $P$ while a thermostat holds
  temperature. This is what you use to find equilibrium densities, model
  phase behavior, or match experimental conditions.

The subtlety that trips up newcomers: a thermostat is *not* just "rescale
velocities to the target every step." A correct thermostat reproduces the
right *fluctuations*, not merely the right mean — a system pinned exactly at
⟨T⟩ every step is not sampling the canonical ensemble at all. That is why
Atomiverse ships several thermostats with different fluctuation properties,
not one.

## The Representation: Ensemble Control Is a Set of Components

There is no `NVTDriver` or `NPTDriver`. The ensemble is assembled from
components attached to the one `cpu::md::MD` driver, each tagged by its
structural `role()`:

| component | role | what it does |
|---|---|---|
| `VelocityVerlet` | Integrator | symplectic time stepping (the NVE core) |
| `NoseHoover` | Thermostat | deterministic single-chain canonical thermostat |
| `FixLangevin` | Thermostat | stochastic friction + random kick (also writes Force) |
| `FixCSVR` | Thermostat | Bussi stochastic velocity rescaling (canonical, with a conserved quantity) |
| `Barostat` | Barostat | Berendsen isotropic pressure coupling |
| `NoseHooverNPT` | Integrator | MTK integrator that *is* thermostat + barostat + integration in one |

The role tag is what the driver validates against: `cpu::md::MD` requires
*something* that produces forces and *an* integrator, and it schedules each
component at the LAMMPS-style phase its physics belongs to (thermostat
half-steps interleave with the velocity-Verlet kicks; the Berendsen barostat
runs at `end_of_step` after the global reductions). You compose the ensemble;
the driver sequences it.

One consequence worth stating up front: **most thermostats need a separate
integrator.** `NoseHoover`, `FixLangevin`, and `FixCSVR` each thermostat an
existing NVE trajectory, so you pair them with `VelocityVerlet`.
`NoseHooverNPT` is the exception — it subsumes integration, thermostat, and
barostat into one Trotter splitting, so it *replaces* `VelocityVerlet` and is
registered as the integrator.

## The C++ API: Building Each Ensemble

The categorical setters on `cpu::md::MD` are the primary API:
`set_integrator` (singular — one integrator), `set_thermostat` and
`set_barostat` (plural — each call appends), plus the shared `set_evaluator`,
`set_neighbor`, `set_thermo`. A component with no categorical setter of its
own — `InitVelocity` below, for instance — is bound with the explicit
`sim.on(phase).add(component, order)`. There is no bare `sim.add(...)`: the
phase is always either implied by a setter or written out.

### NVE — the baseline

No thermostat, no barostat. This is [Chapter 10](ch10-classical-md.md)'s run,
restated as the reference point:

```cpp
using namespace ATV;

auto frame = make_fcc_argon(4, /*a=*/5.32);   // 256 Ar atoms, Å

cpu::md::MD sim;
sim.on(md::Phase::initial_integrate)
   .add(cpu::md::InitVelocity{100.0, /*seed=*/42u}, /*order=*/-999);
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0104, /*sigma=*/3.4, /*cutoff=*/6.8});
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt=*/4.0});
sim.set_thermo(100);
sim.build();
sim.run(frame, 10000);
```

### NVT with Nosé-Hoover — canonical, deterministic

Add a `NoseHoover` thermostat via `set_thermostat`. Its constructor is
`(T_start, tau_T, dt)`: the target temperature (K), the coupling time
constant (fs) (LAMMPS `Tdamp` — how quickly the bath responds), and the time
step (fs).

```cpp
sim.set_integrator(cpu::md::VelocityVerlet{4.0});
sim.set_thermostat(cpu::md::NoseHoover{/*T=*/100.0, /*tau_T=*/200.0, /*dt=*/4.0});
```

Nosé-Hoover couples the velocities to a single deterministic thermostat
variable $\xi$ that itself relaxes toward the target; it carries a
**conserved quantity** (the extended Hamiltonian), which the driver writes to
`frame.meta.get(kConserved)` and `Thermo` logs. A flat `conserved` trace is the
correctness check that the thermostat is sampling correctly. Use Nosé-Hoover
when you want a smooth, reproducible time series — a clean vibrational power
spectrum, or a trajectory you can rerun bit-for-bit. (It is a single-chain
thermostat; for the full LAMMPS M-chain with `drag`, use `NoseHooverNPT`,
which carries `mtchain`/`drag` fields even at zero pressure coupling.)

A linear temperature ramp — anneal from `T_start` to `T_stop` over a run — is
opt-in through chained setters, mirroring LAMMPS `fix nvt temp Tstart Tstop`:

```cpp
auto nh = cpu::md::NoseHoover{300.0, 200.0, 4.0};
nh.set_T_stop(100.0).set_run_span(/*nfirst=*/0, /*nlast=*/10000);
sim.set_thermostat(std::move(nh));
```

### NVT with CSVR — canonical, stochastic, with a conserved quantity

The Bussi canonical-sampling-through-velocity-rescaling thermostat is
`FixCSVR(T_start, tau, dt, seed)`. It rescales all velocities once per step by
a single stochastic factor chosen to drive the kinetic energy toward its
canonical target *with the correct fluctuations*, and — unlike Berendsen
rescaling — it carries a conserved quantity (accumulated in
`frame.meta.get(kConserved)`), so it genuinely samples the canonical ensemble.

```cpp
sim.set_integrator(cpu::md::VelocityVerlet{4.0});
sim.set_thermostat(cpu::md::FixCSVR{/*T=*/100.0, /*tau=*/100.0,
                                    /*dt=*/4.0, /*seed=*/12345u});
```

CSVR is a good default when you want canonical fluctuations, gentle coupling,
and reproducibility across MPI rank counts (its randoms are seeded only from
`(seed, step)`, so the trajectory is identical regardless of decomposition).

### NVT with Langevin — canonical, stochastic, force-based

`FixLangevin(T_start, T_stop, damp, dt, seed)` applies a per-atom Stokes drag
plus a fluctuating random kick that together satisfy the
fluctuation-dissipation theorem. Note the **five** arguments: pass
`T_stop == T_start` for a constant-temperature run (the ramp form uses a
distinct `T_stop` with `set_run_span`).

```cpp
sim.set_integrator(cpu::md::VelocityVerlet{4.0});
sim.set_thermostat(cpu::md::FixLangevin{/*T_start=*/100.0, /*T_stop=*/100.0,
                                        /*damp=*/100.0, /*dt=*/4.0,
                                        /*seed=*/0xDEADBEEFu});
```

Langevin's `role()` is Thermostat, but because it *writes additive forces*
into `fx/fy/fz` it also advertises `Capability::Force` — which means a *pure*
Langevin run (free diffusion in a box, no pair potential at all) still
satisfies the driver's "something must produce forces" check. Pick Langevin
when you want trajectories from different seeds to be *independent* canonical
samples — useful for free-energy estimators or for hopping out of metastable
wells. The three stochastic-vs-deterministic thermostats are mutually
exclusive; attach exactly one.

### NPT — thermostat plus barostat

For NPT, combine a thermostat with the Berendsen `Barostat(P_target, tau_P,
dt, bulk_modulus=1.0)`. Pressure is in eV/Å³ (LAMMPS-standard sign, positive
⇔ compressive); **1 bar ≈ 6.2415e-7 eV/Å³**.

```cpp
sim.set_integrator(cpu::md::VelocityVerlet{4.0});
sim.set_thermostat(cpu::md::NoseHoover{100.0, 200.0, 4.0});
sim.set_barostat  (cpu::md::Barostat{/*P=*/6.2415e-7, /*tau_P=*/2000.0, /*dt=*/4.0});
```

`Barostat` reads `frame.meta.get(kStress)` to compute the internal pressure,
which is why it declares `reads() == Capability::Stress` — and that
declaration makes `cpu::md::MD` **auto-inject a `Virial` stress observer** at
build time so the pressure is actually available (you do not add `Virial`
yourself). It rescales the box edges and all owned atoms by the Berendsen
dilation factor each `end_of_step`. Berendsen relaxes pressure smoothly but
does *not* produce a rigorous NPT distribution — for production NPT
statistics, prefer the MTK integrator below.

### Rigorous NPT — `NoseHooverNPT`

`NoseHooverNPT` is a Martyna-Tobias-Klein integrator that fuses velocity
integration, an M-chain Nosé-Hoover thermostat, and an NHC barostat into one
symplectic Trotter step. It **replaces** `VelocityVerlet` / `NoseHoover` /
`Barostat` and is set as the *integrator*. Its parameters are plain
constructor arguments — there is no `Config` bag:

```cpp
sim.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.8});
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
sim.set_integrator(cpu::md::NoseHooverNPT{
    /*T_target*/ 100.0,
    /*tau_T*/ 200.0,
    /*P_target*/ 6.2415e-7, // eV/Å³
    /*tau_P*/ 2000.0,
    /*dt*/ 4.0});
sim.set_thermo(100);
```

The first five constructor arguments (`T_target`, `tau_T`, `P_target`, `tau_P`,
`dt`) are required; the rest default sensibly (`drag = 0`, `mtchain = mpchain =
3`, `couple_axis = {true,true,true}` for isotropic coupling, `T_stop` folds to
`T_target` for no ramp). Like the Berendsen barostat it declares
`reads() == Stress`, so `Virial` is auto-injected. It writes the full MTK
extended Hamiltonian to `frame.meta.get(kConserved)` — the flat trace of *that*
quantity is the rigorous correctness check for an NPT run. Pass `couple_axis`
to barostat individual axes (e.g. `{false,false,true}` for uniaxial), and use
`set_run_span` with a
non-default `T_stop` for a temperature ramp.

### The `set_nvt` shortcut

For the common canonical case, `set_nvt` bundles integrator + thermostat into
one call, mirroring LAMMPS `fix nvt`:

```cpp
sim.set_nvt(cpu::md::NVT{/*dt=*/4.0, /*T=*/100.0, /*tau=*/100.0,
                         cpu::md::NVT::Thermostat::CSVR, /*seed=*/12345u});
```

It expands into `VelocityVerlet{dt}` plus the chosen thermostat
(`NVT::Thermostat::NoseHoover` or `CSVR`), each with its correct phase
bindings — exactly the two explicit calls, so it composes with `set_evaluator` /
`set_neighbor` and even with a DFT evaluator on the force phase the same way.

## The Expected Result: Means at the Setpoint, Correct Fluctuations

A correctly thermostatted run has two signatures, and both matter:

- **The mean converges to the setpoint.** After equilibration, the running
  average ⟨T⟩ from `Thermo` should sit at the target temperature (within a
  couple of percent for a stochastic thermostat over a finite run), and for
  NPT the average box volume should settle so that ⟨P⟩ matches the target.
- **The fluctuations are physical.** Instantaneous $T$ swings around ⟨T⟩ with
  a spread that *shrinks* as $1/\sqrt{N}$ — for 256 argon atoms the swings
  are modest; for a 3-atom molecule they are enormous and *correct* (a small
  system genuinely has large relative energy fluctuations; see
  [Chapter 15](ch15-cross-method.md)). A thermostat that pins $T$ dead-flat is a red
  flag, not a success.

For the deterministic thermostats there is a third, sharper check: the
`frame.meta.get(kConserved)` trace (Nosé-Hoover extended Hamiltonian, or the MTK
Hamiltonian for `NoseHooverNPT`) should be *flat* over the run. Drift in the
conserved quantity means the time step is too large or the coupling time too
short.

Reading the class of each claim: ⟨T⟩ and ⟨P⟩ matching their setpoints is a
**stochastic-variation** result — you validate it with a fixed seed and an
ensemble mean, with a tolerance (e.g. Langevin ⟨T⟩ vs LAMMPS ≤ 2%), never as
a single-step exact value. The conserved-quantity flatness is closer to a
**numerical-tolerance** check on the integrator.

## The Validation: Ensemble Means Against LAMMPS

Because every Atomiverse thermostat and barostat is a deliberate port of its
LAMMPS `fix`, the validation is ensemble-level agreement with LAMMPS under
matched parameters: the same target, coupling time, time step, and (for the
stochastic thermostats) a controlled seed, comparing the *equilibrium mean*
temperature and pressure — not individual trajectories, which necessarily
differ because the RNG streams differ. The tolerance is a small percentage on
the means (stochastic-variation class), and for the deterministic thermostats
the conserved-quantity drift bound is tighter. The component-level ports are
each audited against the verbatim LAMMPS source and regression-tested in the
CPU MD suite; the ensemble-mean-vs-LAMMPS benchmark is the manual's
designated ensemble validation case.

## The Recurring Pattern: Ensembles Are Force-Provider-Agnostic

Every ensemble in this chapter was built on classical `PairLJCut` forces, but
nothing in the thermostat or barostat touches the force *source*. Replace the
pair potential with an `RHF` on the force phase and every line about
temperature and pressure control stays exactly as written — that is **ab
initio molecular dynamics in the canonical or isothermal-isobaric ensemble**:

```cpp
cpu::md::MD aimd;
aimd.set_integrator(cpu::md::VelocityVerlet{0.5});
aimd.set_thermostat(cpu::md::NoseHoover{300.0, 100.0, 0.5});   // NVT, unchanged
aimd.on(md::Phase::force)
    .add(cpu::scf::RHF{}.predictor(3));            // quantum forces
aimd.build();
aimd.run(frame, n_steps);
```

The thermostat does not know, and does not care, whether the forces it is
regulating came from a Lennard-Jones cutoff or a converged Kohn-Sham
calculation. This is the same "swap the evaluator, keep the loop" invariant
that unifies the whole engine — [Chapter 15](ch15-cross-method.md) runs it end-to-end,
and [Chapter 15](ch15-cross-method.md) makes the pattern explicit as a design
principle. Learn the ensembles once, on cheap classical forces; apply them
unchanged to any level of theory.

## Validation Reference

The thermostats and barostats are exercised end-to-end by the ensemble
regressions — [`regressions/md/lj_argon`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/md/lj_argon) (NVE
drift), [`regressions/md/npt_lj_mtk`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/md/npt_lj_mtk) (diagonal
MTK NPT), [`regressions/md/pr_npt_lj`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/md/pr_npt_lj)
(full-cell Parrinello-Rahman) and
[`regressions/md/nph_lj`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/md/nph_lj) (thermostat-free NPH) and the categorical
setter wiring — including the automatic `Virial` injection triggered by a
barostat's `reads() == Stress` — in
[`tests/cpu/dynamics/test_md_set_api.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/dynamics/test_md_set_api.cpp).
Multi-rank NPT correctness (box consistency across ranks) is covered by
[`tests/mpi/test_nose_hoover_npt_multirank.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/mpi/test_nose_hoover_npt_multirank.cpp).
The ensemble-means-vs-LAMMPS benchmark is the designated external validation
case, alongside the LJ and polyethylene cases under
[`benchmarks/md`](https://github.com/MolCrafts/Atomiverse/blob/dev/benchmarks/md).
