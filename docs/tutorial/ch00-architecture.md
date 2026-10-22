# Simulation Workflows

A physical question should determine the calculation. For the same pair of
atoms, we might ask for the energy at a chosen separation, a nearby resting
geometry, or the motion after release. These questions can use one potential
energy function while requiring different mathematical procedures.

**Learning goals.** Distinguish evaluation, minimization, and dynamics;
connect state, evaluator, and driver to their physical roles; identify the
evidence required before accepting an outcome.

**Before you begin.** Read [Systems and Units](ch01-foundations.md) and the
derivative discussion in [Mathematical Tools](mathematical-tools.md). You do
not need the class hierarchy to understand the three physical questions.

## Energy, structure, and motion

An evaluation holds the coordinates fixed and computes $U(\mathbf R)$ and,
when supported, the forces $-\nabla U$. It answers a question about the chosen
configuration; it neither locates an equilibrium nor measures a temperature.

A geometry optimization changes coordinates to seek a stationary point.
For a harmonic bond, this means approaching the minimum of
$U(r)=k(r-r_0)^2/2$. A small gradient is necessary for an unconstrained
minimum, but it does not distinguish a minimum from every other stationary
point. Curvature and the constraints of the problem matter as well.

Dynamics instead advances positions and velocities with equations of motion.
A stretched harmonic bond released from rest accelerates toward equilibrium
and can pass through it. In an isolated conservative model, potential energy
becomes kinetic energy; motion need not settle at the minimum. An optimization
step is therefore not a physical time step, even if both produce pictures
of atoms moving toward one another.

These distinctions determine what to inspect: the energy and forces for an
evaluation, stationary-point criteria for optimization, and time resolution
and sampling for a trajectory. A successfully completed process is not by
itself evidence for any of these physical conclusions.

## State and operations

The frame holds coordinates, velocities when needed, atom information,
the cell, and calculated quantities. An evaluator applies a particular
energy model to that state. A driver schedules operations and controls
their repetition. Components perform the individual operations needed by
the driver, such as an integration update or a measurement.

The tutorial's CPU interface uses `cpu::Frame` for this shared state. The
important question is not how many classes to remember, but which fields an
operation reads and which results it makes available. A force measurement
must follow the evaluation at the geometry it is intended to describe.
Stored numbers may otherwise belong to an earlier state.

For velocity Verlet, the order is a half velocity update, a position update,
a force evaluation at the new position, and the remaining half velocity
update. The detailed equations appear in
[Molecular Dynamics](ch10-classical-md.md). The same scheduling idea is useful
for optimization, but the optimizer's search direction has a different
meaning from a velocity.

## Inner and outer iterations

An electronic energy evaluator introduces another calculation at fixed
nuclear coordinates: the electronic state must become self-consistent.
Only then can the outer optimization or dynamics step use the resulting
energy and supported derivatives with the intended accuracy.

There are therefore two distinct questions. Did the inner electronic solve
meet its criteria? Did the outer calculation reach its requested endpoint?
A small outer force cannot repair an inadequately converged electronic
state. Likewise, many converged electronic calculations do not prove that a
trajectory has sampled equilibrium.

In the inherited C++ design, concrete methods such as `cpu::scf::RHF` can
run their own electronic iteration and be composed with an outer driver.
This is a software arrangement, not a new physical approximation. Before
using it, check that the selected binding actually supplies the forces and,
if required, stresses used by the outer procedure.

## Stopping and accepting a result

A work budget limits cost. A convergence criterion describes an endpoint
within a numerical model. A validation check supports the interpretation of
that endpoint. Keep all three in the calculation record.

- For a fixed-geometry evaluation, check finite values, units, and any inner
  convergence requirement. Confirm that an energy-only request has not been
  mistaken for a force calculation.
- For optimization, inspect the stopping reason and the requested force and
  displacement criteria. Exhausting the iteration budget does not establish
  a stationary geometry; a stationary geometry still needs interpretation.
- For dynamics, record the elapsed physical time and time step, check the
  relevant conservation or ensemble diagnostics, and estimate uncertainty
  in the observable. Completing the requested steps is not a sampling test.

When an inner solve fails, record that failure and follow an explicit retry
or stopping policy. Do not silently reuse a last available force and present
the resulting outer step as a validated calculation.

## Practice plan

Use the two-argon model introduced in
[Molecular Dynamics](ch10-classical-md.md). Before running a full example,
trace one force evaluation and one complete time step on paper:

1. List the initial coordinates, masses, velocities, cell, and pair parameters.
2. Identify where the initial forces are obtained. The first half velocity
   update already needs forces at the initial geometry.
3. Record which coordinates the next force evaluation uses and when both
   halves of the velocity update are complete.
4. Decide when to record potential and kinetic energy so that they describe
   the same completed state. Then identify a separate test of time-step error.

Only after these classical checks should an electronic evaluator be composed
with the loop. That substitution additionally requires elements, charge and
spin conventions, electronic model data, convergence controls, and supported
derivatives. A shared interface does not make an argon pair model and a
molecular electronic model physically interchangeable.

## Setup fragments

These original fragments and build notes are retained for reference.
They may omit initialization; use the complete model example when running code.

| Pillar | Owns | Example types |
|--------|------|---------------|
| **Runtime** (`cpu::Backend`) | frame binding, plan storage, MPI, timing | one per run |
| **Driver** (`cpu::Driver<P>`) | stages + plan + criteria | `MD`, `Minimize`, `MC`, `Evaluate`, … |
| **Component** | one step of physics | `PairLJCut`, `LBFGS`, `Thermo` |
| **Frame** | host-side CPU state | `blocks` + `meta` + `box` |

```cpp
// Same MD loop — different physics by swapping the force component
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.8});       // classical
// sim.on(md::Phase::force).add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G});       // AIMD
sim.build();
sim.run(frame, 1000);
```

## Reading paths

- [Molecular Dynamics](ch10-classical-md.md) develops the argon potential and
  a first trajectory; [Statistical Ensembles](ch11-ensembles.md) then introduces
  heat and pressure control.
- [Basis Sets](ch04-basis-sets.md) and [Hartree–Fock Theory](ch02-rhf.md) introduce
  the representation and physics of a molecular electronic evaluator.
- [Geometry Optimization](ch12-geometry-optimization.md) develops stationary
  structures, while [Comparing Models](ch15-cross-method.md) explains controlled
  comparisons between different physical approximations.
- [CPU Workflow Reference](../developer/cpu-workflow-reference.md) retains
  lifecycle, phase, ownership, and stopping fragments for readers extending
  the implementation. It is also included in the print appendix.

## Questions for reflection

Why can a stretched bond pass through its equilibrium separation in dynamics
but approach it in minimization? Which force is needed before the first
velocity update? What should an outer driver do with a failed electronic
evaluation, and what evidence is still needed after a successful run?
