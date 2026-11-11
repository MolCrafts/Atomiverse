# CPU Workflow Reference

This appendix contains the detailed execution material formerly embedded in
[Simulation Workflows](../tutorial/ch00-architecture.md). Read the tutorial
for physical meaning and the [CPU internals guide](cpu-internals.md) when
implementing or changing a component. Keeping this reference in the developer
manual lets the introductory chapter focus on a reader's first calculation.

## Scope and evidence

The class relationships and code fragments below are inherited implementation
descriptions, not a newly tested API specification. Fragments containing
ellipses or omitted initialization are schematic. Confirm them against a
complete example and the exact source revision used for a calculation.

The upstream `cpu/frame/frame.h` inspected on 16 September 2026 declares
`blocks`, typed `meta`, and `box` as the frame's physical representation.
That narrow source inspection does not validate every lifecycle, registration,
or stopping contract below. In particular, distinguish a loop's status from
the application's policy for responding to a failed inner calculation.

The print edition includes this same Markdown file as an appendix. Do not
copy these implementation details back into a second tutorial chapter.

## Electronic iteration

A common point of confusion is whether SCF is its own driver. It is —
but not a *category* of driver, and there is no method-agnostic class
standing in front of it.

The real thing is the **concrete method**: `cpu::scf::RHF`, `UHF`, `RKS`,
`UKS`, … Each derives
[`MeanFieldMethod<D>cpu::Driver<ATV::scf::Phase>` with the typed stages
`fock → mix → diagonalize → density → check`. Each stage is a Component
the method owns as a direct member; the algorithms those stages call
(`DIISMixer`, `EigenSolver`) stay `cpu::scf::internal::` helpers and are
never registered into an outer Plan. Integrals and AO workspaces are
private members of the method, rebuilt from the `Frame`.

There is no `cpu::scf::SCFSolver` bag, no `SCFConfig`, and no generic
`cpu::scf::SCF` — parameters are constructor arguments on the concrete
class, and `Frame` is the only I/O currency.

Because a Driver is an executable node, the *same object* is the right
answer for three very different workflows — standalone it runs its own
loop, nested at an outer force phase it runs via `run_nested`:

- **Single-point energy.** `cpu::scf::RHF rhf{BasisSet::STO3G}; rhf.run(frame);`
- **Geometry optimization.** `cpu::minimizer::Minimize + LBFGS`, with
  `opt.on(minimizer::Phase::force).add(cpu::scf::RHF{basis})` (the example in
  [Geometry Optimization](../tutorial/ch12-geometry-optimization.md))
- **Ab initio molecular dynamics.** `cpu::md::MD + VelocityVerlet + NoseHoover`,
  with `sim.on(md::Phase::force).add(cpu::scf::RHF{basis}.predictor(3))` —
  the ASPC predictor warm-starts the density across steps (the example in
  [Comparing Models](../tutorial/ch15-cross-method.md))

Same method. Three different drivers. Same `cpu::Frame`.

## Components

A **component** is one well-scoped piece of work that runs at a specific
phase of one step of one driver. The base class
[`cpu::Component``cpp
void build  (...);   // one-time setup (allocate scratch)
void launch (...);   // per-step compute (no allocation)
void destroy(...);   // teardown
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

## Drivers

A **driver** owns the outer loop. It collects components, validates that
the registered set is sensible (does an MD driver have an integrator? a
minimizer have an optimizer? both have something that produces forces?),
compiles a sorted **plan** of `(phase, order, component)` triples, and
then iterates that plan with the right termination condition for the
job at hand.

There is exactly **one** driver class in each backend — the *loop node*,
`cpu::Driver<PhaseEnum>` — and the classes you actually construct are
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

## Step ordering

Each preset compiles its plan against a fixed, named **phase order**
([`core/stage.hcpu::md::MD` the phase order matches LAMMPS:

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

There is **no `driver.add(component)`**. Every registration goes one of two
ways:

- A **categorical setter** — `set_evaluator`, `set_neighbor`, `set_integrator`,
  `set_thermostat`, `set_barostat`, `set_optimizer`, `set_observer`,
  `set_record`, `set_thermo`. The setter consults the component's
  `default_bindings()` (or a fixed phase of its own) to decide where it lands,
  so you never name the phase.
- **Explicit placement** — `driver.on(phase).add(Component{...}, order)`. The
  `PhaseBuilder` returned by `on(phase)` is the only public carrier of `add`,
  and the phase is always written out. This is the road for anything without a
  categorical setter of its own.

Both roads have the same ownership rule: the `add` / `set_*` form takes an
**rvalue** and transfers ownership into the driver; `add_ref` / `set_*_ref`
**borrows** a component you own (declare it before the driver so the driver
dies first). Plain `add(lvalue)` is `= delete`d — there is no implicit borrow.

Negative orders run before positive orders, so user observers (default `0`) sit
naturally after infrastructure (negative orders).

(Python's `sim.add(...)` *is* a single dispatching convenience on the binding
layer. It has no C++ counterpart; do not translate it back.)

You will see this concretely in every later tutorial. The takeaway here
is just that **the loop has names**. You are never expected to read
through a thousand-line scheduler — you ask "what runs at the *force*
phase?" and the answer is the list of components bound there.

## Evaluation, optimization, and dynamics

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
convergence concept — the step budget specifies elapsed integration work, not statistical convergence — so `MD` attaches no criterion by default. If you want it to
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
`Converged`. The distinction is between measurement and decision: at the `convergence_check` phase the optimizer
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

## Subsystem composition

The thing that is genuinely powerful about Atomiverse, once you have the
mental model, is that the three loops compose by *swapping force
providers*, not by writing a new driver.

```text
cpu::md::MD              + PairLJCut         = classical MD
cpu::md::MD              + scf::RHF          = AIMD
cpu::minimizer::Minimize + LBFGS + PairLJCut = classical relaxation
cpu::minimizer::Minimize + LBFGS + scf::RHF  = ab initio geometry opt
cpu::Evaluate            + PairLJCut         = classical single-point
scf::RHF{basis}.run(frame)                   = quantum single-point
```

The driver code does not know whether the force is empirical or
quantum-mechanical; it only requires that *something* registered on the
`force` phase produces `a force-writing energy kind (Potential / HF / …)`. That single architectural
choice is what unifies the rest of the documentation.

## Stopping and outcomes

You can read the rest of the tutorials with just three names in your
head — `cpu::Frame`, the driver class for your loop, and the components
you add to it. There are two more types that show up so often it is
worth naming them now.

[`cpu::Boxmin_image`, `wrap_axis`, `length`,
`volume`). For an orthorhombic box the helpers are branch-light scalar
math; an `xsimd` batched overload (`min_image_batch<B>`) lives in
[`cpu/box_simd.hConvergenceMetricsIterationStatusconverged`, `stop_requested`, and a human-readable `reason`) that it
**owns**. A component may *request* a stop; only a criterion can declare
convergence. And because each loop owns its own status rather than
sharing one on the Context, a loop nested inside another cannot
accidentally stop its parent. Concrete electronic methods can also expose method-specific convergence
information. Check how an inner failure is communicated to the enclosing
workflow; separation of loop status does not by itself validate an outer step.

The following schematic criterion illustrates an early stop. Its ellipses
and unresolved temperature helper make it a design sketch, not compilable code:

```cpp
class StopWhenTooHot : public cpu::Criterion {
  public:
    explicit StopWhenTooHot(double t_max) : t_max_(t_max) {}
    std::string name() const override { return "stop_when_too_hot"; }
    void launch(...) override {}  // verdict-only — no per-step work
    ATV::Verdict verdict(const cpu::LoopState& s) override {
        return (temperature(s.frame) > t_max_) ? ATV::Verdict::Stop
                                                      : ATV::Verdict::Continue;
    }
  private:
    double t_max_;
};

sim.add_criterion(StopWhenTooHot{2000.0});   // LAMMPS's `fix halt`, from outside the library
```
