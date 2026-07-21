# Core Concepts

Atomiverse is a **staged loop** with pluggable physics. Three names cover
almost every public surface:

| Pillar | Owns | You write |
|--------|------|-----------|
| **Frame** | System state (positions, box, meta, topology) | Fill once, read results back |
| **Component** | One step of work (force, integrate, observe, write) | Construct and register |
| **Driver** | The loop (stages + plan + when to stop) | Pick a preset, call `run` |

A fourth idea — **Runtime** (`cpu::Backend`) — is shared infrastructure
(frame binding, MPI, timing). One Runtime per run; nested drivers borrow it.

## Frame

`cpu::Frame` is the system. Positions in Å, energies on
`frame.meta.get(kEnergy)` in eV (SCF/PW also report Hartree/Ry on their
result objects). There is no separate "system" object beside the frame.

```cpp
cpu::Frame frame(n);
auto x = frame.span(kAtom_x);
// … fill x, type, mass …
frame.simbox = cpu::Box::cubic(L);
```

CUDA workflows use an analogous device frame; transfers are explicit
(`cpu::to_cuda` / `cuda::to_cpu`). See [Backend switching](backend-switching.md).

## Component

A component does **one** thing at a **typed stage** of the plan:

- force / energy sources: `PairLJCut`, `cpu::scf::SCF`, `PWEvaluator`, …
- dynamics: `VelocityVerlet`, thermostats, barostats
- optimizers / movers: `LBFGS`, `TranslateMove`
- observers / recorders: RDF, Thermo, XYZ

Lifecycle: `build()` once → `launch()` every iteration → `destroy()`.
No allocation in `launch()`.

Registration is ownership-explicit:

```cpp
sim.add_potential(cpu::PairLJCut{eps, sigma, rcut});  // owns a copy
sim.on(md::Phase::force).add(cpu::scf::RHF{});     // owns a copy
sim.on(md::Phase::force).add_ref(my_component);       // borrows
```

## Driver (presets)

You do not subclass `Driver<>` for ordinary work. You construct a **preset**
— a loop node whose constructor attaches the defaults that job wants:

| Preset | Job | Stops when |
|--------|-----|------------|
| `cpu::Evaluate` | single point | one pass |
| `cpu::md::MD` | molecular dynamics | step budget |
| `cpu::minimizer::Minimize` | geometry optimization | `ForceConverged` (or budget) |
| `cpu::mc::MC` | Monte Carlo | trial budget |
| `cpu::vib::Hessian` | vibrations | fixed 6N displacements |
| `cpu::PathDriver` | CI-NEB / path | all replicas converged |

```cpp
cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.add_potential(cpu::PairLJCut{0.0104, 3.4, 6.8});
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
sim.build();
sim.run(frame, 1000);
```

The **method** is not a driver class. SCF, PW, GPW, and AM1 are **inner
solvers** wrapped by **evaluator components** you bind at the force/energy
stage. Same MD loop, different physics:

```cpp
// classical
sim.add_potential(cpu::PairLJCut{…});
// ab initio
sim.on(md::Phase::force).add(cpu::scf::RHF{});
```

## When the loop stops

Termination is a **policy**, not a hard-coded `if` inside a driver class.

- **`Criterion`** (plural) — *should we stop?* A Component the driver polls
  once per iteration. Compose with
  `add_criterion(...)`. Severity: **Stop > Converged > Continue**.
  Built-ins: `ForceConverged`, `StopOnNaN`, `AllReplicasConverged`.
  You can write your own outside the library (LAMMPS-style `fix halt`).
- **`OutcomePolicy`** (singular) — *does this iteration count?*
  `Commit` / `Rollback`. MC attaches `Metropolis`; reject undoes the trial.

Algorithms publish metrics (`max_force`, energy, …); **criteria judge them**.
`driver.converged()` and `driver.stop_reason()` are on the loop, not on L-BFGS.

## Mental model for a new calculation

1. Build a **Frame** with geometry (and topology if needed).
2. Pick a **preset** for the job shape (MD / Minimize / Evaluate / …).
3. Attach **components** for forces, integration, I/O.
4. `build()` then `run(frame, budget)`.
5. Read results from the frame (and optional result objects on evaluators).

For which preset + evaluator to choose, see [Methods](methods.md).
For the scientific story behind each method, open the [Tutorial](../tutorial/).
