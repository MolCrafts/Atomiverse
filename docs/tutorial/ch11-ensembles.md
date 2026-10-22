# Statistical Ensembles

Status tables: [MD roadmap](../roadmap/molecular-dynamics.md).

A thermally isolated sample and one immersed in a heat bath are different
experiments. Ensembles describe these conditions as probability distributions
over possible microscopic states. A thermostat is useful when it samples the
intended distribution, not merely when a plotted temperature looks smooth.

**Learning goals.** Distinguish NVE, NVT, and NPT; interpret fluctuations;
separate equilibration from statistical measurement.

**Before you begin.** Read [Molecular Dynamics](ch10-classical-md.md) and the
discussion of probability in [Mathematical Tools](mathematical-tools.md).

## Microstates and ensembles

A microstate specifies positions and momenta. An observable, such as
potential energy or pressure, assigns a value to that microstate. An
ensemble specifies a probability distribution over possible microstates
under stated macroscopic conditions. It describes how observations are
weighted, not a single configuration that must look representative.

For **NVE**, particle number $N$, volume $V$, and total energy $E$ are fixed.
The microcanonical distribution is supported on the allowed constant-energy
surface. A Hamiltonian trajectory remains on that surface in exact dynamics,
but conservation alone does not show that it explores all relevant regions.
Using time averages as ensemble averages requires adequate sampling of the
accessible states, an assumption that should be tested rather than presumed.

For **NVT**, the system exchanges energy with a much larger heat reservoir
at temperature $T$, while $N$ and $V$ remain fixed. The temperature is a
reservoir parameter; the system's instantaneous kinetic energy fluctuates.

## The canonical distribution

The probability of a system state is proportional to the number of reservoir
states compatible with it. Write reservoir entropy as $S_{\mathrm{res}}=k_{\mathrm B}\ln\Omega_{\mathrm{res}}$. If the system has
energy $H$ and the combined energy is $E_{\mathrm{tot}}$, a first-order
expansion for a sufficiently large reservoir gives

$$S_{\mathrm{res}}(E_{\mathrm{tot}}-H)
\simeq S_{\mathrm{res}}(E_{\mathrm{tot}})-\frac{H}{T}.$$

Here $\partial S_{\mathrm{res}}/\partial E=1/T$ defines temperature.
Exponentiating the entropy relation yields the Boltzmann weight. With
$H(\mathbf R,\mathbf p)=K+U$,

$$p(\mathbf R,\mathbf p)=Z^{-1}e^{-\beta H},\qquad
\beta=\frac{1}{k_{\mathrm B}T},\qquad
Z=\int e^{-\beta H}\,d\mathbf R\,d\mathbf p.$$

The partition function $Z$ normalizes the density. Constant factors in the
classical phase-space measure cancel from normalized averages here; they
must be treated consistently when comparing absolute free energies or
systems with different particle numbers. At fixed $N,V,T$, an observable
is averaged with this probability, not by setting its value to a target
at every time step.

## Kinetic temperature

For an unconstrained Cartesian momentum component of mass $m$, the canonical
density is Gaussian: $p(p_\alpha)\propto
\exp[-p_\alpha^2/(2m k_{\mathrm B}T)]$. Its second moment is
$\langle p_\alpha^2\rangle=m k_{\mathrm B}T$, so the corresponding mean
kinetic energy is $k_{\mathrm B}T/2$. This is classical equipartition for
independent quadratic momentum degrees of freedom.

If $f$ such degrees of freedom remain, define the kinetic-temperature
estimator $T_{\mathrm{inst}}=2K/(f k_{\mathrm B})$. In the corresponding
canonical momentum distribution,

$$\langle K\rangle=\frac{f}{2}k_{\mathrm B}T,\qquad
\operatorname{Var}(K)=\frac{f}{2}(k_{\mathrm B}T)^2,\qquad
\frac{\sqrt{\operatorname{Var}(T_{\mathrm{inst}})}}{T}
=\sqrt{\frac{2}{f}}.$$

For an unconstrained three-dimensional atomic sample with only its total
center-of-mass momentum fixed, $f=3N-3$. Additional independent constraints
change that count. Use the degrees of freedom and any streaming-velocity
subtraction associated with the actual temperature estimator. These kinetic
relations are classical; they are not quantum equipartition statements.

Fluctuations are therefore expected, especially in a small system. Forcing
the instantaneous estimator to equal the reservoir temperature removes part
of the canonical kinetic-energy distribution rather than demonstrating
better sampling.

## Pressure and volume

In **NPT**, the reservoir also imposes pressure $p_{\mathrm{ext}}$, and the
volume can fluctuate. For isotropic volume changes, the joint weight in
physical coordinates contains $e^{-\beta(H+p_{\mathrm{ext}}V)}$. If positions
are instead represented by scaled coordinates $\mathbf R_A=L\mathbf s_A$
in a cubic cell with $V=L^3$, their integration measure contributes $V^N$.

That Jacobian is why a barostat is more than a rule for driving mean pressure
toward a number. The equations, cell variables, measure, and constraints
must together produce the intended distribution. Instantaneous pressure can
be noisy; the reservoir fixes neither the instantaneous pressure nor the
instantaneous volume. General cell-shape sampling requires additional care.

## Choosing a control method

Deterministic Nosé–Hoover dynamics uses auxiliary variables to exchange
energy with the system. It can sample poorly for small, nearly harmonic
systems, so a plausible mean is not a sufficient check. Langevin dynamics
balances friction with stochastic forcing; their relation is needed for
the intended equilibrium distribution. Stochastic velocity rescaling uses
a random global rescaling chosen to preserve canonical kinetic statistics.

Weak-coupling methods can be useful for preparing a state, but matching a
mean temperature or pressure does not establish canonical or NPT
fluctuations. If a different method is used for production, allow that
production distribution to equilibrate before measuring it.

The system's ordinary $K+U$ need not be constant under a thermostat. Some
methods provide an extended conserved quantity or an energy-exchange
diagnostic; its meaning is method-specific. None by itself proves that all
slow configurations have been visited. Controls can also alter time
correlations, which matters when the target is a transport coefficient.

## From a trajectory to an estimate

For stationary samples $A_j$, the joint statistics do not change when the
sample indices are shifted together. Let $C_\ell$ be the covariance between
samples separated by $\ell$ saved intervals. In terms of deviations
$\delta A_j=A_j-\langle A\rangle$, this means
$C_\ell=\langle\delta A_j\delta A_{j+\ell}\rangle$, and $C_0$ is the variance.
Positive covariance means that deviations tend to persist with the same
sign across that interval. Expanding the variance of the sample mean gives
the finite-sample identity

$$\operatorname{Var}(\overline A)
=\frac{1}{M}\left[C_0+2\sum_{\ell=1}^{M-1}
\left(1-\frac{\ell}{M}\right)C_\ell\right].$$

Independence removes the nonzero-lag terms. Positive correlations usually
increase uncertainty relative to that independent-sample estimate. The
correlation time depends on the observable: temperature may settle long
before density, molecular orientation, or a phase change is sampled.

A practical check divides an equilibrated trajectory into $B$ equal blocks,
each long compared with the relevant correlation time. If their means are
approximately independent and have sample standard deviation $s_b$, estimate
the standard error of the overall mean as $s_b/\sqrt B$. Increase block
length to look for a stable estimate while retaining enough blocks to judge
its variability. If neither condition can be met, more simulation is needed.
A smooth running mean alone does not establish a small uncertainty.

## Practice plan

Use one argon parameterization and density for the initial NVE and NVT
comparison. Changing the model or density at the same time would obscure
what the thermostat changes.

1. Establish the time-step and force checks of
   [Molecular Dynamics](ch10-classical-md.md) without heat or pressure control.
2. Prepare an NVT state. Record target temperature, coupling parameters,
   degrees of freedom, initialization, and any random seeds. Inspect the
   kinetic-energy mean and distribution, not only the last temperature.
3. Choose and justify an equilibration interval. Estimate uncertainty from
   the remaining trajectory using sufficiently long blocks, and repeat from
   independent initial conditions when feasible.
4. For NPT, additionally record the pressure convention, cell degrees of
   freedom, mean volume, and volume fluctuations. Check the supported stress
   evaluation and input pressure units before starting.
5. Compare means within statistical uncertainties and separately study
   time-step and coupling-parameter effects. For transport, check whether the
   control scheme perturbs the time correlation being measured.

The reference fragments below illustrate several controls, not instructions
to combine them into one run. Choose the ensemble from the physical question
before choosing the corresponding software components.

## Setup fragments

These original fragments and build notes are retained for reference.
They may omit initialization; use the complete model example when running code.

```molplot preset="molplot" theme="auto" width="100%" aspect="16:10"
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

## Control components

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

## Ensemble configurations

The categorical setters on `cpu::md::MD` are the primary API:
`set_integrator` (singular — one integrator), `set_thermostat` and
`set_barostat` (plural — each call appends), plus the shared `set_evaluator`,
`set_neighbor`, `set_thermo`. A component with no categorical setter of its
own — `InitVelocity` below, for instance — is bound with the explicit
`sim.on(phase).add(component, order)`. There is no bare `sim.add(...)`: the
phase is always either implied by a setter or written out.

### NVE — the baseline

No thermostat, no barostat. This is [Molecular Dynamics](ch10-classical-md.md)'s run,
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

Nosé–Hoover couples motion to a deterministic auxiliary variable whose
evolution responds to a kinetic-energy deviation from the target. The
inherited interface describes an extended-energy diagnostic through
`frame.meta.get(kConserved)`. Check its definition and time-step dependence
for the selected implementation. A small drift is not proof of canonical
sampling or adequate exploration of configuration space.

Deterministic control avoids explicit random forcing, but it does not
guarantee an unperturbed vibrational spectrum or bitwise agreement across
hardware and parallel decompositions. Check the observable and the sampling
behavior before choosing a coupling scheme for production.

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

CSVR provides a canonical velocity-rescaling scheme. Even if the rescaling
random numbers depend only on `(seed, step)`, trajectories need not be
bitwise identical across MPI decompositions: force reductions and other
floating-point operations can change their rounding order. Test ensemble
statistics separately from random-stream and bitwise reproducibility.

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

### Extended-system NPT

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

## Controls and force models

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
    .add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G}.predictor(3));            // quantum forces
aimd.build();
aimd.run(frame, n_steps);
```

The thermostat does not know, and does not care, whether the forces it is
regulating came from a Lennard-Jones cutoff or a converged Kohn-Sham
calculation. This is the same "swap the evaluator, keep the loop" invariant
that unifies the whole engine — [Comparing Models](ch15-cross-method.md) runs it end-to-end,
and [Comparing Models](ch15-cross-method.md) makes the pattern explicit as a design
principle. Learn the ensembles once, on cheap classical forces; apply them
unchanged to any level of theory.

The following numerical material is retained from the source snapshot.
It was not rerun for this editorial revision; match its input and provenance
before using it as a validation target.

## Interpreting the diagnostics

After equilibration, compare the mean kinetic temperature with the reservoir
temperature using its estimated standard error. For NPT, inspect pressure
and volume together. A fixed percentage is not a universal statistical
tolerance: the expected uncertainty depends on correlation, run length,
system size, and the estimator.

Check fluctuations as well as means. The classical kinetic prediction
$\sqrt{\operatorname{Var}(T_{\mathrm{inst}})}/T=\sqrt{2/f}$ applies to the
canonical momentum distribution with the specified degrees of freedom.
It does not state that every observable's fluctuations scale the same way,
particularly near a phase transition.

When a component reports an extended conserved quantity, examine its
time-step dependence rather than demanding an exactly flat discrete trace.
Drift can indicate inadequate resolution, force or stress inconsistency,
or an implementation problem; it does not identify a unique cause. The
availability and definition of `kConserved` must be checked for the selected
method and source revision.

Separate three questions in the report: are the mean and fluctuations
consistent with the intended distribution, is numerical error controlled,
and has the trajectory explored the configurations relevant to the observable?

## Ensemble validation

An external comparison should match the potential, cutoff treatment, state,
constraints, estimator definitions, and numerical settings. Agreement of
individual stochastic trajectories is neither expected nor required. Compare
equilibrated distributions and means with quantified uncertainty, and use
independent initial conditions where practical.

The inherited material names LAMMPS comparisons and component tests as
reference routes. Their presence is not a new execution of those tests, nor
does a shared algorithm name guarantee identical scheduling or conventions.
Treat any quoted percentage threshold as belonging to its documented case,
not as a general definition of correct ensemble sampling.

## Reference provenance

The thermostats and barostats are exercised end-to-end by the ensemble
regressions — `lj_argon` (NVE
drift), `npt_lj_mtk` (diagonal
MTK NPT), `pr_npt_lj`
(full-cell Parrinello-Rahman) and
`nph_lj` (thermostat-free NPH) and the categorical
setter wiring — including the automatic `Virial` injection triggered by a
barostat's `reads() == Stress` — in
[`tests/cpu/dynamics/test_md_set_api.cpp`](https://github.com/MolCrafts/Atomiverse/blob/master/tests/cpu/dynamics/test_md_set_api.cpp).
Multi-rank NPT correctness (box consistency across ranks) is covered by
[`tests/mpi/test_nose_hoover_npt_multirank.cpp`](https://github.com/MolCrafts/Atomiverse/blob/master/tests/mpi/test_nose_hoover_npt_multirank.cpp).
The ensemble-means-vs-LAMMPS benchmark is the designated external validation
case, alongside the LJ and polyethylene cases under
[`benchmarks/md`](https://github.com/MolCrafts/Atomiverse/blob/master/benchmarks/md).

## Further reading

- G. Bussi, D. Donadio, and M. Parrinello, *Canonical sampling through velocity
  rescaling*, J. Chem. Phys. 126, 014101 (2007), DOI: 10.1063/1.2408420.
  The [authors' manuscript](https://arxiv.org/abs/0803.4060) explains the
  stochastic rescaling method and its diagnostic quantity.
- The official [LAMMPS thermostat guide](https://docs.lammps.org/Howto_thermostat.html)
  describes temperature estimators and control choices. Use its conventions
  as a comparison reference, not as an Atomiverse API specification.

## Questions for reflection

Why does a mean of 300 K not prove canonical sampling? Why can adding more saved frames leave the uncertainty of the mean almost unchanged?
