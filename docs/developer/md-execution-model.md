# MD Execution Model

This page explains how one molecular-dynamics timestep runs in the CPU
backend: the fixed sequence of **phases**, what each phase is responsible for
(**even when it is empty**), and how a piece of work — an integrator, a force
kernel, a thermostat, a communication step — attaches itself to that sequence.
It is the Atomiverse counterpart of the LAMMPS
[*Developer flow*](https://docs.lammps.org/Developer_flow.html) page: the goal
is that you can understand the whole timestep by reading one enum and one
table, without chasing call sites.

It assumes the user-level mental model from the
[CPU overview tutorial](../tutorial/ch00-architecture.md) — `Frame` (the system
state), `Component` (one unit of work), `Driver` (the loop) — and the
[CPU Backend Internals](cpu-internals.md) chapter (why there is one loop node
and a handful of presets, and why termination is a criterion). Here we zoom into
*one step of the MD driver*.

The phase enum (`ATV::md::Phase`) is backend-agnostic, so this skeleton is
shared: the CUDA single-GPU MD driver (`cuda::md::MD`) walks the same phases
with the same `(phase, order)` placements. It binds the same `neighbor →
pre_force → force → post_force → integrate → end_of_step` work and simply
leaves the multi-rank communication phases (`pre_exchange`, `pre_neighbor`)
empty — a concrete instance of the empty-phase principle below.

## The three-stage driver lifetime

Every driver runs the same shape, composed from the loop primitives
`Driver` provides:

```
run(frame, n_steps):
    setup_pass(frame)                # once, on the first run() of a trajectory
        backend.init()               #   allocate backend + timer
        backend.load(plan)           #   the compiled Plan (phase-ordered slots)
        backend.set_frame(frame)     #   bind the frame; partition atoms (multi-rank)
        ctx.loop = {&status_, &metrics_}   # the Context BORROWS this loop's status
        backend.build_components()   #   each component's build(): one-time allocation
    for s in [0, n_steps):           # the run loop
        status_.reset()
        backend.step(s)              #   one full timestep (walks every phase)
        poll_criteria_(s)            #   ask every Criterion; Stop > Converged > Continue
        # MD attaches no criterion by default, so a plain MD run simply exhausts
        # its budget. Attach one (StopOnNaN, a `fix halt` analogue) and the same
        # loop gains an early exit with no change to the driver.
    backend.drain_analysis()         # flush async observers
    # teardown happens on driver destruction or when a different frame is run
```

This maps onto LAMMPS as `setup_pass` ↔ `init()`+`setup()`, the loop ↔
`Verlet::run()`, and `drain_analysis`/teardown ↔ `post_run()`. The loop is
**continuable**: consecutive `run()` calls extend a single trajectory, and
`build()` runs only once so stateful components (Nosé–Hoover chains, RNG
streams, velocity initialisers) keep their state across calls.

## The timestep: phases in order

`backend.step()` walks a compiled **Plan** — a list of *slots*, one per phase,
in phase order. Each slot holds the components bound to that phase, launched in
ascending *order* (see below). The phases are the `ATV::md::Phase` enum
(`include/atomiverse/facade/stage.h`), modelled on the LAMMPS Verlet timestep so
the whole step is legible from the enum:

| Phase | Responsibility | LAMMPS equivalent |
|---|---|---|
| `initial_integrate` | First half-kick `v += ½·dt·f/m`, drift `x += dt·v`; barostat box push | `initial_integrate` |
| `post_integrate` | Hook immediately after the kick/drift (usually empty) | `post_integrate` |
| `pre_exchange` | Re-home atoms that crossed a rank boundary (`ctx.md.comm->exchange`) | `pre_exchange` |
| `pre_neighbor` | Build ghost atoms — send owned positions to neighbour ranks (`ctx.md.comm->borders`) | `pre_neighbor` |
| `neighbor` | (Re)build the cell / pair neighbour list (`NeighborList`) | neighbor build |
| `pre_force` | Zero the force / energy / stress accumulators before evaluation | `force_clear` + `pre_force` |
| `force` | Accumulate forces, energy, and virial (pair + bonded kernels) | `pair/bond/...->compute()` |
| `post_force` | Reverse ghost-force comm (`ctx.md.comm->reverse_comm`), virial finalize, external/constraint forces | reverse comm + `post_force` |
| `final_integrate` | Second half-kick `v += ½·dt·f/m`; barostat velocity scale | `final_integrate` |
| `end_of_step` | Global reductions (energy, kinetic, virial), thermostat/barostat bookkeeping, observers, I/O | `end_of_step` + output |

### Empty phases are intentional

Every phase always exists and is always walked, even when no component is bound
to it. A bare NVE run binds nothing to `post_integrate`, `pre_exchange`,
`pre_neighbor`, or `post_force` — and that is the point. "Nothing happens in
`post_force` for this run" is then an explicit, documented fact you can read off
the table, not an absence you have to infer from the code. When you add a
feature that needs one of these slots (a constraint in `post_force`, a particle
insertion in `pre_exchange`), the phase is already there waiting.

## How a component attaches: (phase, order)

A component declares where it runs with `default_bindings()` returning a list of
`(phase, order)` pairs, or a caller pins an owned temporary explicitly with
`driver.on(phase).add(Component{...}, order)`. Existing instances use
`driver.on(phase).add_ref(component, order)` and must outlive the driver; implicit
lvalue borrowing is deleted. Within a phase, components run in ascending
`order`. The convention is a small signed integer:

- **negative** — infrastructure that must run *first* (zeroing buffers, ghost
  setup),
- **0** — the user-facing operation (the pair kernel, the integrator),
- **positive** — consumers that run *after* (reductions, publishers, recorders).

This two-level scheme — named phases for the coarse structure, integer order for
fine placement within a phase — is deliberate: a new operation usually slots in
at an existing phase with a new order, without needing a new phase. You only add
a phase when a genuinely new point in the timestep is required.

The current binding map (what runs where) for a typical MD run:

```
initial_integrate :  integrator(0)            velocity_verlet | nve_limit | nose_hoover_npt
                     deform(+200)              (uniaxial-tension box deform)
pre_exchange      :  migrate(0)
pre_neighbor      :  halo_send(0)
neighbor          :  neighbor_list(-10)
pre_force         :  zero_forces(-1000), zero_energy(-999), zero_stress(-998)
force             :  pair(0)                  lj | morse
                     bonded(0)                harmonic_bond | harmonic_angle | multi_harmonic_dihedral
post_force        :  halo_recv(-1000), virial(-50)
final_integrate   :  integrator(0)            (same component, second half)
end_of_step       :  zero_kinetic(-1000), kinetic_energy(-100), reduce(-50),
                     barostat(-45), nose_hoover_npt(-40), publish(+10),
                     observers(+20), rebalance(+1000)
```

The categorical setters (`set_evaluator`, `set_neighbor`, `set_integrator`,
`set_thermostat`, `set_barostat`, …) are thin wrappers that call
`on(phase).add(...)` at the documented phase/order, so a plan built with the
setters is identical to one built with raw `add()` calls. `set_neighbor` binds
at the driver's `neighbor_phase` — the dedicated `neighbor` phase for MD; the
minimiser/evaluate drivers, which have no separate neighbour phase, fold it into
their force phase just ahead of the kernels.

## The component contract

A component (`cpu::Component`) is the unit of scheduled work — the rough
analogue of a LAMMPS `Fix` or `Pair`:

```cpp
std::string name() const;                        // identity / timing key, snake_case
Role role() const;                               // RoleFlag bitset (default 0)
std::vector<DefaultBinding> default_bindings();   // {(phase, order), ...}
void build(const BuildContext&);                 // one-time allocation (NOT per step)
void launch(const LaunchContext&);               // per-step work (NO allocation)
void destroy(const DestroyContext&);             // teardown
```

Kinds (`Integrator`, `Potential`, `HF`, …) are C++ types — one header
each under `cpu/component/`. There is no `Capability` write-set.

Two invariants are load-bearing and enforced by tests:

- **No allocation in `launch()`.** Buffers are sized in `build()` (or grown only
  on rare capacity events such as a ghost-region expansion). The MD inner loop
  is allocation-free at steady state.
- **Owned vs ghost rows.** Owned atoms live in the first `frame.nlocal()` rows;
  ghosts in `[n_local, n_local + n_ghost)`. Force kernels accumulate into the
  shared `kAtom_fx/fy/fz` columns; integrators and reductions iterate owned rows only
  (see `owned_range.h` / `ResolveView`).

## Adding a component or a phase

To add a component, inherit the matching kind, give it a `name()`, optional
`role()` flags, a `default_bindings()` at the right `(phase, order)`, and a
`build()`/`launch()` pair that respects the two invariants above. See
[Extending CPU Subsystems](extending-cpu-subsystems.md) for worked examples
(adding an MD integrator, a minimiser, an SCF helper).

To add a **phase**, insert a value into `md::Phase` at the correct point in the
timestep and extend `phase_names()` in the same order; the Plan resizes
automatically (it is sized from `Phase::COUNT`) and the backend walks the new
slot in enum order. Adding a phase is rarely necessary — prefer a new `order`
within an existing phase. Reach for a new phase only when a feature needs a
genuinely new point in the step that no existing phase models.

## Naming convention

Component class names are one to two words, `UpperCamelCase`, with the `name()`
string the `snake_case` echo (`NeighborList` → `"neighbor_list"`). Names avoid redundant
filler ("Exchange", "Stress") and prefer the direct verb or noun (
`Reduce`, `Publish`, `Rebalance`, `Virial`). Force-kernel names keep the
functional form they implement (`BondHarmonic`, `DihedralMultiHarmonic`) because
the form — not just "a bond" — is what distinguishes one style from the next.
