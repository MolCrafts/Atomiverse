# Molecular Dynamics

Status tables: [MD roadmap](../roadmap/molecular-dynamics.md) · [Force Fields](../roadmap/force-fields.md).

How does a collection of atoms move when each responds to its neighbors?
Classical molecular dynamics turns that question into a sequence of force
evaluations and small time steps. Argon provides a manageable first model:
its approximately spherical neutral atoms let us begin without molecular topology.

**Learning goals.** Relate pair energy to force, interpret velocity Verlet,
and test whether a chosen time step resolves the modeled motion.

**Before you begin.** Use the derivative and Taylor expansion in
[Mathematical Tools](mathematical-tools.md), and the frame conventions in
[Systems and Units](ch01-foundations.md).

## The classical model

Classical molecular dynamics treats nuclei as particles with positions and
momenta. An energy function represents their interactions, and Newton's
equations determine their motion. This is a model choice, not a property
of the integration algorithm. Nuclear quantum effects and the adequacy of
the chosen interaction model need separate consideration.

Argon is a useful starting point because an isotropic interaction between
neutral atoms avoids bonds and angles in the input topology. The
Lennard–Jones pair energy is

$$u(r)=4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6],\qquad
U=\sum_{A<B}u(r_{AB}).$$

The sum counts each distinct pair once. The energy scale $\epsilon>0$ is
the well depth, and the length $\sigma>0$ is the zero-crossing distance,
not the equilibrium separation. The attractive inverse-sixth-power term
models dispersion in a simplified form; the inverse-twelfth-power repulsion
is a convenient empirical choice, not a derivation of electronic overlap.

Define $\mathbf R_{AB}=\mathbf R_A-\mathbf R_B$. Applying the chain rule gives
the force on $A$ due to $B$:

$$\mathbf F_{A\leftarrow B}
=\frac{24\epsilon}{r_{AB}}
\left[2\left(\frac{\sigma}{r_{AB}}\right)^{12}
-\left(\frac{\sigma}{r_{AB}}\right)^6\right]
\frac{\mathbf R_{AB}}{r_{AB}}.$$

The force is repulsive at short distance and attractive at sufficiently
large distance before truncation. It vanishes at $r_0=2^{1/6}\sigma$, where
$u(r_0)=-\epsilon$. Check these identities and the opposite force on $B$
before constructing a many-atom sample. They test the chosen formula, not
whether it describes every property of real argon.

## What is conserved

For fixed masses in coherent units, Newton's equation is
$m_A\ddot{\mathbf R}_A=\mathbf F_A$. Kinetic energy is
$K=\sum_A m_A|\mathbf v_A|^2/2$. If the potential has no explicit time
dependence and all forces are its negative gradient, the chain rule gives

$$\frac{d(K+U)}{dt}
=\sum_A\mathbf v_A\cdot
\left(m_A\dot{\mathbf v}_A+\nabla_A U\right)=0.$$

This is a statement about the continuous equations for an isolated
conservative model. A thermostat, external driving, or moving boundary
can exchange energy with the system. A finite-step algorithm also has
numerical error. These are different reasons for seeing a changing energy.

For an isolated pair model, opposite internal forces imply conservation of
total momentum. Net center-of-mass motion is not the same as internal
thermal motion. Record whether it was removed when initializing velocities.

## Advancing one time step

Velocity Verlet divides the acceleration update into two halves. For an
atom of mass $m$, omitting its label for readability, the update is

$$\begin{aligned}
\mathbf v^{n+1/2}&=\mathbf v^n+\frac{\Delta t}{2m}\mathbf F(\mathbf R^n),\\
\mathbf R^{n+1}&=\mathbf R^n+\Delta t\,\mathbf v^{n+1/2},\\
\mathbf v^{n+1}&=\mathbf v^{n+1/2}
+\frac{\Delta t}{2m}\mathbf F(\mathbf R^{n+1}).
\end{aligned}$$

The superscript labels a numerical time level. Forces at the initial
geometry must already be available for the first half update. After the
position update, evaluate forces again before completing the velocity.
These equations use coherent units; a library using eV, Å, fs, and amu
must include the appropriate conversion in force divided by mass.

For smooth conservative forces and a fixed step, this method is
time-reversible and symplectic: it preserves the canonical phase-space
structure, not the exact original energy at each discrete step. Small
bounded energy oscillations are therefore not automatically a failure.
Long-time accuracy still depends on the model, regularity, and step size.

A harmonic oscillator makes the limitation visible. With angular frequency
$\omega$, the Verlet position recurrence is

$$x_{n+1}-2x_n+x_{n-1}=-(\omega\Delta t)^2x_n.$$

An oscillatory trial solution gives
$\cos(\widetilde\omega\Delta t)=1-(\omega\Delta t)^2/2$.
Bounded harmonic motion requires $0<\omega\Delta t<2$ away from the
marginal endpoints. This is a stability condition, not an accuracy target:
the numerical frequency $\widetilde\omega$ can be wrong well before the
trajectory becomes unstable. Refine the step for the observable being measured.

## Boundaries and cutoffs

In a cubic periodic cell of edge $L$, the usual single-image treatment of a
short-range pair interaction requires a cutoff $r_c<L/2$. Other cell shapes
require the corresponding geometric condition. The neighbor-list skin is
an additional search buffer that permits list reuse; it does not extend
the physical interaction range. The list must be rebuilt before motion
can bring an unlisted pair into that range.

Three cutoff conventions should not be confused. Simple truncation sets
the interaction to zero beyond $r_c$. An energy shift uses
$u_{\mathrm{ES}}(r)=u(r)-u(r_c)$ below $r_c$ and zero above it. This makes
energy continuous but generally leaves a force jump. A force shift uses

$$u_{\mathrm{FS}}(r)=u(r)-u(r_c)-(r-r_c)u'(r_c),\qquad r<r_c,$$

and zero beyond the cutoff. It changes the force throughout the retained
range while making it vanish continuously at $r_c$. Match the convention
when comparing energies, forces, and pressure between implementations.
Smooth-force arguments about the integrator do not automatically apply at
a discontinuous cutoff.

## Choosing a state

For a face-centered-cubic starting lattice, $n_c$ conventional cells per
direction contain $N=4n_c^3$ atoms, with box edge $L=n_c a$ and nearest
separation $a/\sqrt2$. The number density is $N/L^3=4/a^3$; mass density
also includes the atomic mass and a unit conversion. Thus choosing a
lattice constant is already choosing a density.

The reduced quantities $T^*=k_{\mathrm B}T/\epsilon$ and
$\rho^*=(N/V)\sigma^3$ make the state easier to compare across parameter
sets. A useful time scale is $t_0=\sigma\sqrt{m/\epsilon}$ in coherent
units. An initial crystal does not prove that the equilibrium phase at
the chosen density and temperature is crystalline.

## Practice plan

1. Verify the isolated pair energy and force at a compressed separation,
   at $r_0$, and at a stretched separation below the cutoff. Compare a
   nonzero force with finite differences before using a many-atom geometry.
2. Build the argon sample below. Record density, pair parameters, cutoff
   convention, neighbor settings, boundaries, and velocity initialization.
3. Prepare the state, then run a short trajectory without heat or pressure
   control. Record potential energy, kinetic energy, total momentum, and
   elapsed physical time at consistently completed time steps.
4. Repeat at half the time step for the same physical duration, rather
   than the same number of steps. Compare energy drift and oscillation
   amplitude; investigate close contacts or cutoff crossings when necessary.
5. Only after these checks add the ensemble controls developed in
   [Statistical Ensembles](ch11-ensembles.md). A thermostat can remove
   integration-generated heat and thereby conceal a poor baseline.

Keep the first successful test small. It gives a reference you can return
to when a larger sample behaves unexpectedly.

## Checking a run

- **Geometry and cell.** Initialize all coordinates and a valid cell before
  neighbor construction. Inspect separations; a severe overlap can produce
  enormous forces even when the program is behaving as specified.
- **Search range.** Ensure the neighbor scheme covers every interacting pair
  and respects its rebuild criterion. Do not infer these settings from the
  potential cutoff alone.
- **Velocities.** Explicitly initialize or load them. Zero velocities are a
  legitimate chosen state, but they are not a thermal distribution at an
  arbitrary requested temperature.
- **Topology.** For bonded models, document exclusions or scaled nonbonded
  terms. Otherwise the intended bonded interaction can coexist with an
  unintended short-range repulsion.
- **Controls.** Begin with one validated thermostat or barostat arrangement
  for each controlled group. Do not stack controls on the same degrees of
  freedom without a defined sampling scheme.

## Changing the evaluator

An electronic evaluator can occupy the force stage of an MD workflow, but
the substitution is not a complete physical setup. It requires an electronic
state, model data, converged energy and derivatives, and consistent units.
Stress availability must also be checked before allowing the cell to change.
The cost and tolerable time step can change substantially.

A validated classical test can help isolate a scheduling problem from an
electronic-convergence problem. It is not an alternative physical model of
the same accuracy simply because the software interface is shared.
[Comparing Models](ch15-cross-method.md) develops this distinction.

## Force evaluation in the driver

The execution design places force evaluation between position and final
velocity updates. The inherited phase-order sketch is

```text
initial_integrate → pre_force → force → post_force → final_integrate → end_of_step
```

works for both regimes. In a classical run, the components bound at the
`force` phase are pair potentials and bonded-energy terms; in AIMD the
same phase may hold a concrete electronic method. The force evaluator
writes forces and energy; the integrator uses forces to update motion, and
observers read the quantities they report. Sharing the frame does not mean
that every component writes the same fields.

The following inherited table relates component names to their physical
contributions. Its capability column describes reported outputs; it should
not be read as a specification of a current `capabilities()` C++ member.
Check registration and derivative support against the selected revision.

| Component | Functional form | Capability |
|---|---|---|
| `cpu::PairLJCut` | $4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6]$, shifted | Energy &#124; Force &#124; Stress |
| `cpu::BondHarmonic` | $K(r - r_0)^2$ | Energy &#124; Force &#124; Stress |
| `cpu::AngleHarmonic` | $K(\theta - \theta_0)^2$ | Energy &#124; Force |
| `cpu::DihedralMultiHarmonic` | $\sum_n a_n \cos^n\phi$ | Energy &#124; Force |

They share the same lifecycle, the same chunk-owned OpenMP force
buffers (deterministic at a fixed thread count), and the same cell-list-driven neighbor-list
hand-off. None of them know about the integrator. That decoupling is
what makes the rest of the chapter short.

## Building an argon sample

A small but realistic test problem is a face-centred-cubic crystal of
Argon at the experimental nearest-neighbour spacing. Four atoms per
conventional cell, $4^3$ cells, gives 256 atoms in an isotropic box
that fits comfortably in cache:

```cpp
cpu::Frame make_fcc_argon(int n_cells, double a) {
    constexpr double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };
    cpu::Frame f(n_cells * n_cells * n_cells * 4);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto mass = f.span(kAtom_mass);
    auto type = f.span(kAtom_type);
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
      for (int iy = 0; iy < n_cells; ++iy)
        for (int iz = 0; iz < n_cells; ++iz)
          for (int b = 0; b < 4; ++b) {
              x[idx] = (ix + basis[b][0]) * a;
              y[idx] = (iy + basis[b][1]) * a;
              z[idx] = (iz + basis[b][2]) * a;
              mass[idx] = 39.948;                  // amu
              type[idx] = 18;                      // Ar
              ++idx;
          }
    f.box = cpu::Box::cubic(n_cells * a);
    return f;
}
```

The `cpu::Box` is constructed last from the side length. A frame is a
plain value object — there is no "open" or "init" call to make, no
implicit registration. Hand it to the driver and you are done.

## Integrating the pair model

The simplest possible classical MD is an NVE run with a single pair
potential. The full driver setup is:

```cpp
auto frame = make_fcc_argon(4, /*a=*/5.32);  // Å, FCC Ar lattice constant

cpu::md::MD sim;
sim.on(md::Phase::initial_integrate)
   .add(cpu::md::InitVelocity{/*T_K=*/100.0, /*seed=*/42u}, /*order=*/-999);
sim.set_integrator(cpu::md::VelocityVerlet{/*dt_fs=*/4.0});
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/6.8});
sim.on(md::Phase::force).add(
    cpu::PairLJCut{/*eps=*/0.0104, /*sig=*/3.4, /*cutoff=*/6.8});
sim.build();
sim.run(frame, 10000);
```

Five lines describe a complete simulation. Read them in the order they
appear:

1. **`InitVelocity{T, seed}`** has no categorical setter, so it is bound
   explicitly at `initial_integrate @ -999` — the phase and order its own
   `default_bindings()` declare. Its
   `launch()` is a no-op; its `build()` draws a Maxwell-Boltzmann sample
   at temperature $T$, scales velocities by $\sqrt{k_B T / m}$ per atom,
   and removes the centre-of-mass velocity. By the time the first MD
   step runs, the `kAtom_vx/vy/vz` columns already carry physical thermal
   velocities.
2. **`VelocityVerlet{dt}`** is the integrator, so it goes through the
   `set_integrator` categorical setter (singular — a second call throws).
   It binds at `initial_integrate @ 0` (first half-kick + position drift)
   and `final_integrate @ 0` (second half-kick). The factor between
   $f$ in eV/Å, $m$ in amu, $v$ in Å/fs is the metal-units conversion
   constant `units::force_to_accel` — handled internally; the user sees physical
   units on every public field.
3. **`NeighborList`** goes through `set_neighbor`, which places it at the
   driver's `neighbor_phase @ -10` — for MD the dedicated `neighbor` phase,
   after the ghost build and before `force`. It owns a cell-list strategy
   with a Verlet skin; the rebuild cadence follows LAMMPS's `neigh_modify
   every 1 delay 0 check yes` discipline. On non-rebuild steps the cost
   collapses to a single $O(N)$ skin scan.
4. **`PairLJCut`** binds explicitly at `force`. The constructor takes
   $\varepsilon$ in eV, $\sigma$ in Å, and a cutoff in Å. The cutoff
   choice has to match — or be larger than — the neighbor-list cutoff;
   the test suite uses $6.8 \approx 2\sigma$ for Argon. Energy is
   shifted so that $V(r_{\mathrm{cut}}) = 0$.
5. **`build()`** compiles the plan: the bindings contributed by the
   categorical setters are merged with the explicit `on(phase).add()`
   bindings, sorted by `(phase, order, insertion)`, and validated.

There is no bare `sim.add(component)`. Registration is either a categorical
setter (`set_integrator`, `set_thermostat`, `set_barostat`, `set_evaluator`,
`set_neighbor`, `set_observer`, `set_record`, `set_thermo`) or the explicit
`sim.on(phase).add(component, order)` — the phase is never guessed. The
rvalue form transfers ownership; pass an lvalue you own with
`add_ref` / `set_*_ref` instead (plain `add(lvalue)` is `= delete`d).

The driver's auto-added infrastructure — `ZeroForces`, `ZeroEnergy`,
`ZeroStress` at `pre_force @ -1000/-999/-998`, `KineticEnergy` at
`end_of_step @ -100`, `ZeroKinetic` at `end_of_step @ -1000`,
`SnapshotPublisher` at `end_of_step @ +10` — is wired by `cpu::md::MD`'s
constructor before the user code runs. You almost never have to think
about them, but if you ever wonder why `frame.meta.get(kEnergy)` is zero before the
first force component runs each step, that is why.

## Temperature control

The fragments in this and the next section are interface references. Read
[Statistical Ensembles](ch11-ensembles.md) for the distribution being sampled
and the checks needed before interpreting the resulting averages.

Two thermostats ship today. They differ in *how* they enforce
temperature, not in the wiring you write:

**Nose-Hoover** is a deterministic single-chain thermostat (Martyna et
al., 1992). It rescales velocities in `initial_integrate` and
`final_integrate` by an integrator variable $\xi$ that itself relaxes
toward the target on a timescale $\tau$. Use it when you want a smooth,
deterministic time series — the textbook canonical ensemble.

```cpp
sim.set_thermostat(cpu::md::NoseHoover{/*T_target=*/100.0, /*tau=*/200.0,
                                       /*dt=*/4.0});
```

**Langevin** is a stochastic thermostat applied as additive friction
plus a fluctuating random kick (Schneider–Stoll form). It binds at
`post_force @ +500`, after every physical-force evaluator has finished,
so the friction and noise see the total force already in the array. The
combined drag + kick satisfies the fluctuation–dissipation theorem;
velocity-Verlet integration of the augmented force gives canonical NVT.

```cpp
sim.set_thermostat(cpu::md::FixLangevin{/*T_start=*/100.0, /*T_stop=*/100.0,
                                        /*damp=*/100.0, /*dt=*/4.0,
                                        /*seed=*/0xDEADBEEFu});
```

A Langevin run is the right choice when you want trajectories from
different seeds to be *independent* samples of the canonical ensemble
(useful for free-energy estimators or for hopping out of metastable
wells). A Nose-Hoover run is cheaper per step, has no random forces, and
is the right choice when you need a smooth power spectrum or want to
reproduce a single trajectory exactly across runs. The two cannot be
stacked — pick one.

A bookkeeping note on Langevin: its role is `Thermostat`, but because
its `launch()` writes additive forces into `fx/fy/fz`, its capability
set advertises `Capability::Force`. That lets a *pure* Langevin run
(diffusion in a featureless box, no LJ at all) still satisfy
`cpu::md::MD`'s "something must produce forces" build-time check.
Roles describe structure; capabilities describe data flow.

## Pressure control

For NPT, combine a thermostat with the Berendsen barostat:

```cpp
sim.set_thermostat(cpu::md::NoseHoover{T_target, tau_T, dt});
sim.set_barostat  (cpu::md::Barostat{P_target, tau_P, dt});
```

`Barostat` needs the instantaneous pressure, and it builds that from *two*
stress accumulators on the Frame. The configurational part is the pair
virial $\sigma$ in `frame.meta.get(kStress)`, accumulated by the force
evaluators as $\sigma = \sum_{i<j} \mathbf{r}_{ij}\otimes\mathbf{f}_{ij}$
with $\mathbf{r}_{ij}$ the pair separation and $\mathbf{f}_{ij}$ the force
it carries. The kinetic part is $\sigma^{\mathrm{kin}}$ in
`frame.meta.get(kStressKinetic)`, written by the `Virial` observer as
$\sum_i m_i \mathbf{v}_i\otimes\mathbf{v}_i$, so its trace is twice the
kinetic energy. Neither is divided by volume when stored, so the barostat
does that division itself, with $V$ the cell volume:

$$
P \;=\; \frac{\mathrm{tr}\,\sigma + \mathrm{tr}\,\sigma^{\mathrm{kin}}}{3V}
  \;=\; \frac{2\,\mathrm{KE} + \sum_{i<j}\mathbf{r}_{ij}\cdot\mathbf{f}_{ij}}{3V}.
$$

That plus sign matters: it is LAMMPS's `compute pressure` convention, in
which a positive $P$ is a system pushing outwards on its box, so holding it
requires compression. Every pressure quantity in Atomiverse — `Virial`,
`Barostat`, `NoseHooverNPT` — uses this one convention, and a `P_target`
you pass anywhere carries the same sign.

The box edges and every atom position (owned rows *and* their ghost copies,
so neighbour separations stay consistent until the next halo refresh) are
then multiplied by the Berendsen dilation factor

$$
\mu \;=\; \left[\,1 - \frac{\Delta t}{\tau_P\,\beta}
                 \bigl(P_{\mathrm{target}} - P\bigr)\right]^{1/3},
$$

where $\beta$ is the `bulk_modulus` constructor argument (LAMMPS `modulus`)
— a tunable stiffness that divides the correction, so a larger $\beta$ gives
a gentler per-step rescale. Atomiverse defaults it to $1.0$, deliberately
*not* LAMMPS's $10.0$; pass `10.0` explicitly for LAMMPS-identical
relaxation. Read the sign off the formula: if the system is over-pressurised
($P > P_{\mathrm{target}}$) the bracket exceeds 1, $\mu > 1$, and the box
expands — which is what relieves the pressure. The bracket is clamped to
$[0.5, 2]$ before the cube root so one pathological step (a force spike at
step 0, say) cannot collapse or explode the cell.

It is bound at `end_of_step @ -45`, just after the global reduction at
`-50`, so the stresses it sees are already summed across MPI ranks: every
rank computes the same $\mu$ from the same $P$ and $V$, and the box update
stays consistent without any extra communication.

For pressure target conversions: the project's stress is in
$\mathrm{eV}/\textrm{Å}^3$. Using $1\,\mathrm{bar} \approx 6.24 \times 10^{-7}\,\mathrm{eV}/\textrm{Å}^3$,
1 atm and ambient room pressure both round to about $6 \times 10^{-7}$
in code units.

This Berendsen barostat is *isotropic*: one scalar $\mu$ multiplies the
whole cell matrix, so the cell keeps its shape and only changes size. It is
also not a rigorous NPT method — it relaxes the pressure towards the target
without sampling the correct constant-pressure distribution, which makes it
a good equilibrator and a poor production ensemble. The CPU backend does
have the rigorous alternatives: `cpu::md::NoseHooverNPT`
(`nose_hoover_npt.h`), which can also
barostat individual axes, and `cpu::md::ParrinelloRahmanNPT`
(`parrinello_rahman_npt.h`), whose full
cell matrix lets the box shear as well as breathe. Both are integrators
that *replace* `VelocityVerlet` + thermostat + barostat rather than joining
them. [Statistical Ensembles](ch11-ensembles.md) is where the NPT ensemble is worked
through properly — when Berendsen is enough, how `NoseHooverNPT` is
configured, and the conserved-quantity trace that tells you an NPT run is
actually correct; `ParrinelloRahmanNPT` takes the same arguments in the
order `(T_target, P_target, tau_T, tau_P, dt)` and is documented in its
header.

The `Barostat::set_enabled` toggle and `set_P_target` setter are intended
for multi-stage protocols that mix NVT and NPT segments under a single
driver instance.

## Bonded interactions

Pair potentials alone describe noble gases and not much else. Most
soft-matter force fields are built from harmonic-bond, harmonic-angle,
and dihedral terms; the bonded directory implements all three as
straightforward `Evaluator` components.

A typical OPLS-style snippet looks like:

```cpp
sim.on(md::Phase::force).add(cpu::PairLJCut{eps, sigma, cutoff});
sim.on(md::Phase::force).add(
    cpu::BondHarmonic{ {bond_K},  {bond_r0} });
sim.on(md::Phase::force).add(
    cpu::AngleHarmonic{{angle_K}, {angle_t0}});
```

Each constructor takes a `std::vector` indexed by *bond type* (or angle
type, or dihedral type) — the type indices are what you store in
the `kAtom_bond_type`, `kAtom_angle_type`, and `kAtom_dihedral_type`
columns when you assemble the topology.

A subtle detail that is easy to miss: by default, `PairLJCut` does *not* know
about your topology. If you want LAMMPS-style "special bonds" — i.e.
weight the LJ pair on 1–2 (bonded), 1–3 (across an angle vertex), or 1–4
(across a dihedral) atom pairs — you must opt in after
`Frame::build_special_list()`:

```cpp
// Water-style full exclusions:
auto lj = cpu::PairLJCut{eps, sigma, cutoff}
            .set_special_bonds(/*s12=*/0.0, /*s13=*/0.0, /*s14=*/0.0);

// Amber / GAFF: 1–2 and 1–3 off, 1–4 at half weight (scales energy *and* force):
auto lj_gaff = cpu::PairLJCut{eps, sigma, cutoff}
            .set_special_bonds(/*s12=*/0.0, /*s13=*/0.0, /*s14=*/0.5);
// Mirror the same triple on PairCoulCut / CoulLong for charged molecules.
```

Scales are true weights: `0` drops the pair, values in `(0,1]` multiply
pair energy and force. Charged GAFF stacks also need `load_lammps_data`
with **atom_style full** (partial charges) — see
`gaff_peo_fragment` and `cpu/io/lammps_real.h` for
kcal/mol → eV helpers. If you forget specials when topology is present,
bonded atoms clash through the nonbonded wall and the trajectory will
explode.

**Still future work for a full GAFF product path:** multi-term
`dihedral_style fourier` (single-term maps to `DihedralCharmm` today), a
complete `.ff` text parser, CUDA Amber specials parity, and paper-length
polymer equilibration protocols.

## Measurements and output

The `end_of_step` phase is where observers and recorders live. It runs
once per MD step after the integrator has finished, the thermostat has
had its second half-kick, and the kinetic energy has been recomputed.

The auto-added components in `cpu::md::MD` are, in order:

1. `ZeroKinetic` (-1000) — clear `frame.meta.get(kKinetic)` so KE is reaccumulated
   from current velocities each step.
2. `KineticEnergy` (-100) — sum $\frac{1}{2} \sum_i m_i v_i^2$ with OpenMP
   reduction.
3. `ReduceEnergy` (-50) — single-rank no-op; multi-rank
   `MPI_Allreduce` for `kEnergy`, `kKinetic`, and the stress
   components so every rank sees the same global totals.
4. `SnapshotPublisher` (+10) — publishes a `FrameSnapshot` to async
   observers (zero-cost early return when no observer is subscribed).

User-side, the canonical first additions are:

```cpp
sim.set_thermo(/*every=*/100);
sim.set_observer(ATV::RDF{/*r_max=*/5.5, /*n_bins=*/200,
                          /*freq=*/100, "rdf.dat"});
```

`set_thermo(N)` binds `cpu::Thermo{N}` at `end_of_step @ +20`; it prints the
LAMMPS-style "step PE KE T" line synchronously on rank 0. `ATV::RDF` is an
async observer — it inherits `cpu::md::AsyncObserver`, runs on a background
thread, and accumulates a half-list pair-distance histogram from the frames the
driver captures for it (lossless: a slow analysis makes the run wait, it
never skips a frame). At the end of the run its `finalize()` writes the normalized
$g(r)$ to disk. Note that `set_observer` takes an **rvalue**; to keep the
observer around and read `rdf.gr()` after the run, declare it first and borrow
it with `set_observer_ref(rdf)`.

**Trajectory output is a Recorder.** A *Recorder* is the Component kind whose
only job is writing outside the process — the LAMMPS `dump` slot. For
trajectories that component is `cpu::XyzRecorder`, and it goes in through
`set_record`:

```cpp
sim.set_record(cpu::XyzRecorder{"traj.xyz", /*freq=*/100});
```

It binds itself at `end_of_step @ +10` and follows the LAMMPS `dump` step
convention: a frame's step is the number of completed MD steps and the frame
is the state after exactly that many steps. Every `run()` first writes the
current state when its step is a multiple of `freq` (LAMMPS `Output::setup`),
and the end of every step whose completed count is a multiple of `freq` writes
another; no step is written twice. The count keeps going across consecutive
`run(frame, n)` calls on the same `cpu::Frame`, so a 1000-step run at
`freq=100` writes frames for steps 0, 100, …, 1000 — eleven of them, the
initial and the final state included. The recorder truncates the file on its first write
and appends afterwards, so one recorder owns one trajectory file; MPI runs
gather to rank 0, which writes the complete tag-ordered geometry.

`set_record` takes an rvalue and owns the recorder; use `set_record_ref` to
borrow one you declared yourself when you want to read `n_frames_written()`
after the run. Underneath it, `cpu::XYZ` is still a plain writer (`write` /
`append`) you can drive by hand for one-off geometries, and
`cpu::minimizer::TrajectoryWriter` is the sibling Recorder bound to the
minimizer phases.

The following numerical material is retained from the source snapshot.
It was not rerun for this editorial revision; match its input and provenance
before using it as a validation target.

## Reference energy conservation

Thermo every 200 steps from the binary (total energy in eV):

| step | \(E_\mathrm{pot}\) | \(E_\mathrm{kin}\) | \(E_\mathrm{total}\) |
|-----:|-------------------:|-------------------:|---------------------:|
| 0 | −10.0451 | 0.4226 | **−9.622484** |
| 200 | −9.8187 | 0.1962 | −9.622481 |
| 400 | −9.8734 | 0.2509 | −9.622482 |
| 600 | −9.9145 | 0.2920 | −9.622480 |
| 800 | −10.0050 | 0.3825 | −9.622481 |
| 1000 | −10.2620 | 0.6395 | **−9.622481** |

Drift over 1000 steps: **\(2.93\times 10^{-6}\) eV** (relative \(\sim 3\times 10^{-7}\)).

```molplot preset="molplot" theme="auto" width="100%" aspect="16:10"
mark: {type: line, point: true, tooltip: true}
data:
  values:
    - {step: 0, series: E_total, E: -9.622484}
    - {step: 200, series: E_total, E: -9.622481}
    - {step: 400, series: E_total, E: -9.622482}
    - {step: 600, series: E_total, E: -9.622480}
    - {step: 800, series: E_total, E: -9.622481}
    - {step: 1000, series: E_total, E: -9.622481}
    - {step: 0, series: E_pot, E: -10.0451}
    - {step: 200, series: E_pot, E: -9.8187}
    - {step: 400, series: E_pot, E: -9.8734}
    - {step: 600, series: E_pot, E: -9.9145}
    - {step: 800, series: E_pot, E: -10.0050}
    - {step: 1000, series: E_pot, E: -10.2620}
encoding:
  x: {field: step, type: quantitative, title: Step}
  y: {field: E, type: quantitative, title: "Energy (eV)", scale: {zero: false}}
  color: {field: series, type: nominal, title: null}
```

Potential and kinetic trade off; **total** is flat — the NVE signature of a
correct Velocity-Verlet + LJ force pipeline. The CUDA twin
should match within floating-point noise.

## Scope and next steps

Three pieces of MD machinery exist on the CPU side and are not
discussed in detail here:

- **`Deform`** — affine box deformation for NVT-deformation studies.
  Implemented in `cpu/dynamics/deform.h`.
- **`NoseHooverNPT`** — a single integrator that does both thermostat
  and barostat work, implemented as a Martyna–Tobias–Klein chain. Used
  by `tests/cpu/dynamics/test_md.cpp::NoseHooverNPT`. Prefer it over the
  Berendsen barostat when you need rigorous NPT statistics.
- **MPI sub-namespace under `cpu::md::mpi`** — domain decomposition,
  halo exchange, atom migration. Single-rank runs ship every MPI
  component as a no-op via the `cpu::md::MD` constructor; the public
  multi-rank API is described in `.claude/specs/parallel-md-engine.md`.

The next chapter
([Geometry Minimization](ch12-geometry-optimization.md)) reuses the same
force evaluators under a different driver — `cpu::minimizer::Minimize`
— and shows how the SCF pattern that anchored the AIMD chapter
also anchors *ab initio* geometry optimization. From a wiring
perspective, the two are the same recipe with the integrator swapped
for an L-BFGS step.

## Questions for reflection

Why can the pair force vanish where its energy is negative? Which cutoff errors would not disappear merely by halving the time step?
