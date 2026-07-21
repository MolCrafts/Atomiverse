# Chapter 0: A Tour of the CPU Backend

This is the orientation chapter — no physics yet. By the end you know the
three pillars (**Driver**, **Component**, **Frame**), how termination works
(**Criterion** vs **OutcomePolicy**), and why swapping an evaluator changes
the method without rewriting the loop.

| Pillar | Owns | Example types |
|--------|------|---------------|
| **Runtime** (`cpu::Backend`) | frame binding, plan storage, MPI, timing | one per run |
| **Driver** (`cpu::CPUDriver<P>`) | stages + plan + criteria | `MD`, `Minimize`, `MC`, `Evaluate`, … |
| **Component** | one step of physics | `PairLJCut`, `SCF`, `LBFGS`, `Thermo` |
| **Frame** | device-resident state | `blocks` + `meta` + `simbox` |

```cpp
// Same MD loop — different physics by swapping the force component
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.add_potential(cpu::PairLJCut{0.0104, 3.4, 6.8});       // classical
// sim.on(md::Phase::force).add(cpu::scf::RHF{});       // AIMD
sim.build();
sim.run(frame, 1000);
```

Runnable proofs of the architecture (not full e2e science):

```bash
# classical MD energy conservation — proves the loop + potential pipeline
cmake --build build --target 07_md_lj && ./build/examples/07_md_lj
# RHF single point — proves SCFSolver as an inner solver, not a Driver
cmake --build build --target 02_rhf && ./build/examples/02_rhf
```


If you arrived at Atomiverse looking for "how do I run a simulation?" the
honest answer is: it depends on what you mean by *run*. Three different
loops live inside the project, and the first thing a beginner needs is a
mental map of which loop to reach for and why.

This chapter draws that map. It does not run any code. It explains the
three pillars of the CPU backend — **Frame**, **Component**, and
**Driver** — and the three loops that share them: classical molecular
dynamics, geometry minimization, and electronic-structure self-consistent
field. Each later chapter then drills into one of those loops with a
concrete example.

A shorter, task-oriented version of this vocabulary is in
[User Guide → Core concepts](../user-guide/concepts.md). If you have not built
the project yet, see [Getting Started](../user-guide/getting-started.md) first.
Example programs under `examples/` map one-to-one to the loops described here.

## The One Truth: `cpu::Frame`

Every CPU simulation in Atomiverse begins and ends with a single host-side
object,
[`cpu::Frame`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/frame.h). A frame holds:

- **per-atom arrays** — positions $x, y, z$, velocities $v_x, v_y, v_z$,
  forces $f_x, f_y, f_z$, masses, atom types, and global tags;
- **topology blocks** — bonds, angles, dihedrals (each is an SoA block
  matching the `molpy` "Block" model);
- **system scalars** — potential energy, kinetic energy, and a six-component
  stress tensor;
- **a simulation cell** — `cpu::Box`, the canonical 3×3 cell with cached
  inverse, half-edge, and minimum-image helpers.

You never create a separate "system" object alongside the frame. The frame
*is* the system. You fill it once with starting coordinates, hand it to a
driver, and read final coordinates and energies out of the same fields
when the run finishes. Units are eV, Å, fs, amu throughout.

The same frame is reused across MD, minimization, and SCF. That is why we
can compose, for example, `cpu::md::MD` with `cpu::scf::SCF` to do *ab
initio* molecular dynamics: the velocity-Verlet integrator and the SCF
component agree to write to the same `kAtom_fx/fy/fz` columns, and the
driver routes them in the right order.

## What a Component Is, and What It Is Not

A **component** is one well-scoped piece of work that runs at a specific
phase of one step of one driver. The base class
[`cpu::Component`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/component/component.h)
defines exactly three lifecycle hooks:

```cpp
void build  (const BuildContext&);   // one-time setup
void launch (const LaunchContext&);  // per-step compute
void destroy(const DestroyContext&); // teardown
```

Every concrete piece of physics — Lennard-Jones forces, harmonic bonds,
the Nose-Hoover thermostat, the L-BFGS optimizer step, the SCF force
component — is a class that inherits a thin **kind** (`Potential`,
`Integrator`, `HF`, …) which itself inherits `Component`. One kind per
header under `cpu/component/`; depth ≤ 2.

The contract is strict, and the strictness is the point:

1. **`build` may allocate; `launch` may not.** Per-step allocation
   destroys cache locality and parallel scaling, so the project enforces
   it as a structural rule, not a guideline.
2. **Kind is the C++ type; Role is a bitset.** Inheritance says *what
   class of work* this is (`Potential`, `HF`, `Optimizer`, …). `role()`
   returns `RoleFlag`s (`FrameImmutable`, `Deferred`, `ExternalIO`) —
   how the loop may treat it. There is no `Capability` write-set; outputs
   are Frame keys (`kEnergy`, `kAtom_fx`, …).
3. **A component does not own a loop.** The driver does. A component
   advances state by exactly one step's worth of work and returns.

This last rule is what lets you swap, say, L-BFGS for conjugate-gradient
inside `cpu::minimizer::Minimize` without touching the minimizer driver
at all. Both inherit `Optimizer`; the driver's run loop is identical.

## What a Driver Is

A **driver** owns the outer loop. It collects components, validates that
the registered set is sensible (does an MD driver have an integrator? a
minimizer have an optimizer? both have something that produces forces?),
compiles a sorted **plan** of `(phase, order, component)` triples, and
then iterates that plan with the right termination condition for the
job at hand.

There is exactly **one** driver class in each backend — the *loop node*,
`cpu::CPUDriver<PhaseEnum>` — and the classes you actually construct are
**presets** over it. A preset is a class whose constructor attaches the
default policy and infrastructure a given kind of run wants. It is a
convenience, not a layer of the architecture.

| Preset | Loop semantics | Phase enum |
|---|---|---|
| `cpu::Evaluate` | exactly one pass | `evaluate::Phase` |
| `cpu::md::MD` | the step budget you pass to `run()` | `md::Phase` |
| `cpu::minimizer::Minimize` | stop early when converged | `minimizer::Phase` |
| `cpu::vib::Hessian` | a fixed 6N-displacement schedule, then finalize | `vib::Phase` |
| `cpu::mc::MC` | a trial-move budget | `sampling::Phase` |
| `cpu::PathDriver` | stop when every replica of a chain has converged | `path::Phase` |

The three you will meet in this tutorial are the first three; the rest
appear in the later chapters.

*How the loop decides to stop* is not a property of the class at all.
It is a **criterion** — a Component (`cpu::Criterion`) the
driver polls once per iteration, which answers `Continue`, `Converged`,
or `Stop`. `Minimize` attaches a `ForceConverged` criterion in its
constructor (via `add_criterion`); `MD` attaches none and simply runs its budget. You can
attach your own, and you can attach several — their verdicts compose by
severity (`Stop` beats `Converged` beats `Continue`). We come back to
this at the end of the chapter.

Algorithm choice (velocity-Verlet vs leapfrog, L-BFGS vs CG vs FIRE,
RHF vs UKS) lives in components, not in driver subclasses. There will
never be a `cpu::minimizer::LBFGS` driver class — there is a *component*
called `LBFGS` that plugs into the `Minimize` driver. The same goes for
integrators and thermostats.

The reason for this layout is that algorithm choice is something you
want to swap mid-experiment, and so is the stopping rule. Neither is
worth a class hierarchy.

## A Single Step, Phase by Phase

Each preset compiles its plan against a fixed, named **phase order**
([`facade/stage.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/facade/stage.h)).

For `cpu::md::MD` the phase order matches LAMMPS:

```text
initial_integrate → pre_force → force → post_force → final_integrate → end_of_step
```

For `cpu::minimizer::Minimize` the order is:

```text
search_direction → line_search → force → convergence_check → end_of_step
```

For `cpu::Evaluate` (single-point work) the order is:

```text
prepare → evaluate → finalize
```

When you call `driver.add(Component{...})`, the driver owns the component and
its `default_bindings()` say which phase it belongs to and at what order within
that phase. An existing component is borrowed only with the explicit
`driver.add_ref(component)` spelling. Explicit placement follows the same rule:
`driver.on(phase).add(Component{...}, order)` owns, while
`driver.on(phase).add_ref(component, order)` borrows. Plain `add(lvalue)` is
deleted. Negative orders run before positive orders, so user observers (default
`0`) sit naturally after infrastructure (negative orders).

You will see this concretely in every later tutorial. The takeaway here
is just that **the loop has names**. You are never expected to read
through a thousand-line scheduler — you ask "what runs at the *force*
phase?" and the answer is the list of components bound there.

## The Three Loops in One Picture

Here is the same picture for all three CPU drivers, with the bits that
differ called out.

### Classical molecular dynamics — `cpu::md::MD`

```text
loop for n_steps:
    initial_integrate  ← VelocityVerlet first half-kick + drift
    pre_force          ← NeighborList rebuild
    force              ← PairLJCut / BondHarmonic / AngleHarmonic / …
    post_force         ← (constraints, long-range corrections)
    final_integrate    ← VelocityVerlet second half-kick
    end_of_step        ← KineticEnergy, Thermo, Recorder, RDF observer
```

`MD::run(frame, n_steps)` exits after `n_steps` iterations. There is no
convergence concept — time evolution does not "succeed" and "fail", it
just runs — so `MD` attaches no criterion by default. If you want it to
bail out early (energy has gone to NaN, the box has melted), attach a
criterion: `sim.add_criterion(cpu::StopOnNaN{})`.

### Geometry minimization — `cpu::minimizer::Minimize`

```text
loop until converged or max_iter:
    search_direction   ← LBFGS / CG computes a descent direction
    line_search        ← (optional Armijo backtracking)
    force              ← PairLJCut / SCF / … (same evaluators as MD!)
    convergence_check  ← LBFGS / CG *measure*: max force, displacement, ΔE
    end_of_step        ← OptLog, FinalGeometry, TrajectoryWriter
```

`Minimize::run(frame, max_iter)` exits as soon as its criterion says
`Converged`. Note the division of labour, because it is the one thing
newcomers get backwards: at the `convergence_check` phase the optimizer
**measures** (it publishes max force, RMS force, displacement and ΔE
into the loop's `ConvergenceMetrics`), and the **criterion decides** whether
those numbers are good enough. `Minimize` attaches `cpu::ForceConverged`
— the five-criteria test — for you, so a single `LBFGS{...}` instance is
still all you have to add. But the tolerances live on the criterion, and
you ask the *driver* for the verdict afterwards:

```cpp
opt.run(frame, 100);
if (opt.converged()) { /* … */ }
std::cout << opt.stop_reason() << "\n";   // which criterion ended the loop
```

### Single-point evaluation — `cpu::Evaluate`

```text
exactly once:
    prepare    ← ZeroForces, ZeroEnergy
    evaluate   ← SCF / PairLJCut / BondHarmonic / …
    finalize   ← (recorders that read the final state)
```

`Evaluate::run(frame)` runs the registered evaluators once and stops.
This is the tool for "compute the energy and forces at this geometry, do
not change the geometry". `cpu::single_point(frame, evaluator)` is the
tiny convenience helper around it.

## Where SCF Sits

A common point of confusion is whether SCF is its own driver. It is not.

`cpu::scf::SCFSolver` is an **inner solver** — a self-contained class
that runs the Hamiltonian-build → diagonalize → density → DIIS →
convergence-check cycle to converge a single-point electronic structure
calculation. It has its own internal loop, its own internal helpers
(under `cpu::scf::internal::`), and its own convergence channel. None of
those internals are registered into any outer Plan, and none inherit
from `cpu::Component`.

What plugs into a driver is a wrapper:
[`cpu::scf::SCF`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/scf_evaluator.h).
The evaluator is a `cpu::Component` with kind `HF` (or `Potential`) and
capabilities `Energy | DensityMatrix | Orbitals` (and `Force` when
gradient computation is enabled). Inside its `launch()` it builds the
integrals at the current geometry, calls into a pre-built `SCFSolver`,
optionally computes the analytic nuclear gradient, and writes
`frame.meta.get(kEnergy)` and (when applicable) the force columns.

That is why one and the same component is the right answer for three
very different workflows:

- **Single-point energy.** `cpu::Evaluate + SCF{cfg}.with_gradient(false)`
- **Geometry optimization.** `cpu::minimizer::Minimize + LBFGS + SCF{cfg}` (the example in
  [Chapter 12](ch12-geometry-optimization.md))
- **Ab initio molecular dynamics.** `cpu::md::MD + VelocityVerlet + NoseHoover + SCF{cfg}.predictor(3)`
  (the example in [Chapter 15](ch15-cross-method.md))

Same evaluator. Three different drivers. Same `cpu::Frame`.

## How the Subsystems Compose

The thing that is genuinely powerful about Atomiverse, once you have the
mental model, is that the three loops compose by *swapping force
providers*, not by writing a new driver.

```text
cpu::md::MD        + PairLJCut                          = classical MD
cpu::md::MD        + SCF                        = AIMD
cpu::minimizer::Minimize + LBFGS + PairLJCut             = classical relaxation
cpu::minimizer::Minimize + LBFGS + SCF          = ab initio geometry opt
cpu::Evaluate      + PairLJCut                           = classical single-point
cpu::Evaluate      + SCF                        = quantum single-point
```

The driver code does not know whether the force is empirical or
quantum-mechanical; it only requires that *something* registered on the
`force` phase produces `a force-writing energy kind (Potential / HF / …)`. That single architectural
choice is what unifies the rest of the documentation.

## Two Types You Will Meet Often

You can read the rest of the tutorials with just three names in your
head — `cpu::Frame`, the driver class for your loop, and the components
you add to it. There are two more types that show up so often it is
worth naming them now.

[`cpu::Box`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/box.h) is the simulation cell.
It holds the cell matrix, plus a per-step-immutable cache of inverses
and minimum-image helpers (`min_image`, `wrap_axis`, `length`,
`volume`). For an orthorhombic box the helpers are branch-light scalar
math; an `xsimd` batched overload (`min_image_batch<B>`) lives in
[`cpu/box_simd.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/box_simd.h) for hot pair
loops. You will rarely call these directly; force kernels and
thermostats use them on your behalf.

[`ConvergenceMetrics`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/facade/loop.h)
is how a component tells the loop what it measured: energy, max force,
RMS force, the displacement since the previous iteration, |ΔE|. The
optimizer writes it; the criterion reads it; the driver records the
resulting verdict in an
[`IterationStatus`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/facade/iteration_status.h)
(`converged`, `stop_requested`, and a human-readable `reason`) that it
**owns**. A component may *request* a stop; only a criterion can declare
convergence. And because each loop owns its own status rather than
sharing one on the Context, a loop nested inside another cannot
accidentally stop its parent. SCF internals manage their own internal
convergence and never touch any of this.

That is the whole termination story, and it is why writing your own
stopping rule takes about ten lines:

```cpp
class StopWhenTooHot : public cpu::Criterion {
  public:
    explicit StopWhenTooHot(double t_max) : t_max_(t_max) {}
    std::string name() const override { return "stop_when_too_hot"; }
    void launch(const cpu::LaunchContext&) override {}  // verdict-only
    ATV::Verdict verdict(const cpu::LoopState& s) override {
        return (temperature(s.frame) > t_max_) ? ATV::Verdict::Stop
                                                      : ATV::Verdict::Continue;
    }
  private:
    double t_max_;
};

sim.add_criterion(StopWhenTooHot{2000.0});   // LAMMPS's `fix halt`, from outside the library
```

## What Each Later Chapter Adds

You can now read the rest of the manual without losing your bearings.

- **Chapter 1 — SCF From First Principles.** Strips chemistry away and
  shows the SCF feedback loop on a 2×2 toy system. Best read first if
  the very idea of "self-consistent" is new to you.
- **Chapter 2 — Restricted Hartree-Fock (RHF).** Same loop, real
  molecule. Introduces `IntegralPack`, `SCFConfig`, and the
  `SCF`-with-`cpu::Evaluate` recipe.
- **Chapter 3 — Unrestricted Hartree-Fock (UHF).** Adds spin
  polarization. Same driver, different `SCFConfig`.
- **Chapter 4 — Unrestricted Kohn-Sham DFT (UKS).** Adds the XC grid.
  Same driver, same evaluator class, different `SCFConfig`.
- **Chapter 6 — *Ab Initio* Molecular Dynamics.** Switches the driver to
  `cpu::md::MD`, adds an integrator and a thermostat, keeps the
  `SCF` at the force phase.
- **Chapter 7 — Classical Molecular Dynamics.** Replaces the
  `SCF` with `PairLJCut` and bonded force-field components. Shows the
  full thermo / trajectory / RDF observer stack.
- **Chapter 8 — Geometry Minimization.** Switches the driver to
  `cpu::minimizer::Minimize`, swaps `LBFGS` and `CG` as the optimizer
  component, keeps the `SCF` at the force phase.

If you only have time for one example to start with, build and run
`examples/01_scf.cpp` (the SCF toy) and read Chapter 1 next.
Everything that follows reuses the same loop with progressively more
realistic physics.
