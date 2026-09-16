# Architecture

This chapter explains how Atomiverse is organized as a software system. The
goal is to give you a mental model that is simple enough to remember, but still
accurate enough that you can add features without guessing.

The shortest correct summary is this:

Atomiverse presents one staged simulation API to the user, but it implements
that API with two different execution backends. The CPU backend operates
directly on host memory. The CUDA backend materializes a separate device-side
representation, runs the staged workflow there, and then copies results back.

## Why The Project Is Split This Way

The repository tries to solve two different engineering problems at once.

The first problem is usability. A new user should be able to describe a system
once, in normal C++ data structures, without learning CUDA memory management
before they can run an example.

The second problem is performance and correctness. CPU code and CUDA code have
different storage models, different libraries, and different constraints. The
project therefore does not pretend that host and device data are the same
object. Instead, data residency is made explicit in the type system.

That is why Atomiverse puts one CUDA-free contract layer over two separate
implementations, instead of a single class full of `if (is_cuda)` branches.

## The Three Layers

The easiest way to understand the codebase is as three layers.

### 1. Core — the shared contract

The files under [`include/atomiverse/core`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core)
define the smallest shared concepts:

- role-specific numeric tiers such as `StorageReal`, `AccReal`, and `TagInt`
- small value types such as `Mat3`
- execution traits such as `Scheduling`
- device tags used for compile-time backend selection
- default binding metadata used by components

The same directory also holds the public execution skeleton:

- `driver_template.h` defines `Driver<PhaseEnum, DeviceTag>`, which owns
  user-registered components; `PhaseBuilder` supports the fluent
  `.on(phase).add(...)` syntax, and `PlanBuilder` compiles unordered
  registrations into a sorted execution plan
- `stage.h` defines the phase order for each workflow (MD, minimization,
  evaluation, vibrational analysis, Monte Carlo sampling, and coupled-replica
  paths) plus the `kind_of(Phase)` overloads behind `make_binding`
- `verdict.h` defines `Verdict` and `ConvergenceMetrics` — the vocabulary a
  loop's termination is decided in. A `Criterion` judges the metrics and
  returns a verdict; the driver reduces the verdicts across criteria and across
  MPI ranks, so a criterion author never writes MPI code.

These types are intentionally small, and the whole layer is deliberately
CUDA-free. Its job is to let the CPU and CUDA backends talk about the same
ideas without forcing them to share a data layout — to collect intent, not to
run the algorithm itself.

### 2. CPU backend and CPU analysis layer

The CPU side lives under [`include/atomiverse/cpu`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu)
and [`src/cpu`](https://github.com/MolCrafts/Atomiverse/blob/dev/src/cpu). It contains three distinct roles.

First, it defines the user-facing host data model through
[`cpu::Frame`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/frame.h). This is the structure that
examples and users fill directly.

Second, it provides a true CPU execution backend through
[`cpu::Backend`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/backend.h), which runs components
directly on that host frame with no transfer step.

Third, it provides CPU-only utilities that are not part of the staged CUDA
component model, such as integral generation, the Gaussian SCF, plane-wave, GPW and
semiempirical methods, Mulliken analysis, density grids, trajectory analysis,
and trajectory writing.

### 3. CUDA backend and staged components

The CUDA side lives under [`include/atomiverse/cuda`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda)
and [`src/cuda`](https://github.com/MolCrafts/Atomiverse/blob/dev/src/cuda). This layer owns:

- the device-side frame representation
- transfer between host and device
- the CUDA execution backend
- staged MD, minimizer, and SCF components
- async observer and recorder bridges

The key idea is that once a workflow has been materialized onto the GPU, the
main computation remains on device until the backend retrieves final results.

## The Most Important Runtime Story

Almost every run in Atomiverse follows the same sequence.

```text
user fills cpu::Frame
        |
        v
Driver collects components and explicit bindings
        |
        v
Driver::build() compiles a Plan
        |
        v
Backend initializes execution context
        |
        +--> CPU path: use cpu::Frame directly
        |
        +--> CUDA path: cpu::Frame -> cuda::Frame
        |
        v
components run through build -> launch -> destroy
        |
        v
CUDA path copies final state back into cpu::Frame
```

This is the main architectural invariant of the project. Users do not directly
call component `build()` or `launch()` methods. The backend owns lifecycle
dispatch, and the driver owns plan compilation.

## `cpu::Frame` And `cuda::Frame`

The project uses two separate frame types on purpose.

[`cpu::Frame`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/frame.h) stores host-resident data in
xtensor arrays. It is the natural type for examples, input preparation, CPU
solvers, and final results.

[`cuda::Frame`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/frame/frame.cuh) stores device pointers,
device-only work buffers, and host-cached metadata such as atom counts. It is
not meant to be edited directly by users.

This split solves a practical problem. Host code wants safe, indexable arrays.
CUDA kernels want raw device pointers and a layout that avoids repeated
host-device queries. Trying to force both worlds into one type would make both
implementations worse.

The transfer boundary is explicit:

- [`cpu::to_cuda`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/transfer.h) allocates and uploads
  the device frame
- [`cuda::to_cpu`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/frame/transfer.cuh) copies results back
- [`cuda::destroy`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/frame/transfer.cuh) frees device data

## How Plans Are Built

The plan system is the reason examples can read like a script instead of a
handwritten event loop.

Each component can either declare `default_bindings()` or be attached manually.
`driver.add(Component{...})` and `driver.on(phase).add(Component{...}, order)`
transfer ownership to the driver. Borrowing requires the explicit `add_ref`
spelling on the corresponding path; plain `add(lvalue)` is deleted. The driver
stores those requests as pending bindings. When `build()` is called,
`PlanBuilder` resolves them into one `StageSlot` per phase, with operations
sorted by `order`.

That means two things follow automatically.

First, users can write code in a human order and let the plan compiler enforce
the execution order. Second, the backend can execute the same compiled plan many
times without re-resolving bindings on every step.

## The Component Lifecycle

Every staged component follows the same lifecycle:

1. `build()`
2. `launch()` repeated for each step or iteration
3. `destroy()`

This rule is strict because it makes performance and ownership predictable.

`build()` is the place for one-time setup. On CUDA that can include `cudaMalloc`,
device uploads, and handle creation. On CPU it can include workspace allocation
or preprocessing.

`launch()` is the hot path. It should do the per-step computation only. For CUDA
components that means no hidden host-device copies, no synchronization unless
the component explicitly exists to bridge data, and no repeated allocation.

`destroy()` is the cleanup phase. The backend runs it in reverse plan order so
that dependent resources are torn down safely.

## Scheduling And Async Bridges

Most components are `Scheduling::blocking`, which means they run on the main
execution stream. A smaller set of components are `Scheduling::deferrable`. On
the CUDA backend, those components run on a defer stream that can wait on the
main stream and then perform side work.

This is the purpose of
[`AsyncBridge`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/asynchronous/async_bridge.cuh). It is the base
class for CUDA observers and recorders such as:

- `Thermo`
- `Recorder`
- `RDF`
- `MullikenObs`

The pattern is always the same. The main GPU computation runs first. Then the
bridge records an event, waits on the defer stream, performs the needed
device-to-host copy into pinned memory, synchronizes that small side path, and
finally calls a CPU helper that owns the actual formatting or analysis logic.

This separation keeps file I/O and host-side post-processing out of the hot GPU
components while still letting simulations produce usable output.

## How The Driver Families Fit Together

Atomiverse uses one execution model for several workflow families, each
distinguished by its *termination rule* rather than its algorithm.

### Molecular dynamics

`md::MD` defines a LAMMPS-like phase order. The full `ATV::md::Phase` enum has
ten phases:

`initial_integrate -> post_integrate -> pre_exchange -> pre_neighbor -> neighbor -> pre_force -> force -> post_force -> final_integrate -> end_of_step`

It auto-adds zeroing and kinetic-energy infrastructure, while components such as
`VelocityVerlet`, `NeighborList`, `PairLJCut`, `NoseHoover`, and `Barostat` fill
in the actual physics.

### Minimization

The minimizer driver reuses the same staged idea with a different phase order:

`search_direction -> line_search -> force -> convergence_check -> end_of_step`

That lets the project share force-evaluation components with MD while changing
only the orchestration logic.

### Single-point evaluation and electronic structure

`cpu::Evaluate` runs a `prepare -> evaluate -> finalize` pass exactly once — the
home of single-point energy and force calculations.

**SCF is not a driver *family* — it is a driver *node*.** There is no
"electronic structure" category beside MD and Minimize. Gaussian SCF is a
concrete method (`cpu::scf::RHF` / `UHF` / `RKS` / `UKS`), and the CRTP base
they share, `cpu::scf::MeanFieldMethod<D>`, **is** a `cpu::Driver<scf::Phase>`
— the same loop-node shape as MD and Minimize. Plane-wave DFT has not been
re-hosted yet: `cpu::pw::PW` behind `cpu::pw::PWEvaluator` (`PlaneWave`) is
still a plain class with its own loop, which is scheduled debt rather than a
pattern to copy. There is no `Capability` write-set — components write Frame
keys and plug into the *same* Evaluate, MD, and Minimize drivers ("swap the
energy source, keep the driver").

The SCF cycle is a typed stage sequence, `fock → mix → diagonalize → density
→ check`. Each stage under
[`cpu/scf/stages/`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/stages/)
derives `cpu::Component` and is owned by the method as a direct member; the
algorithms they call (`DIISMixer`, `EigenSolver`, density and Hamiltonian
builders) stay `internal::` helpers that never register into a foreign Plan.
`ATV::scf::Phase` exists **because** there is a Driver node to carry it — the
banned shape is a free-floating phase enum with no node.

So the SCF loop is neither a monolithic function nor a user-pluggable phase
order: the method fixes the stage sequence, and members on the method
(`mixer`, `tol`, `max_iter`, guess) select the algorithm inside each stage.

## Why CPU And CUDA SCF Both Exist

A common question is why the repository has both a CPU and a CUDA SCF stack
when they now present the same public shape (`cpu::scf::RHF{basis}.run(frame)`
vs `cuda::scf::RHF{basis}.run(frame)`).

They solve different engineering problems. The CPU stack is the clearest place
to express reference algorithms with xtensor and dense linear algebra: it is
where validation, deep-dive tutorials, and physical-correctness tests live, and
it covers every method and basis. The CUDA stack is organized around device
residency — for the shells its kernels cover it keeps integrals, the SCF
iteration, and forces on the GPU with no scientific data returning to the host
mid-solve — and falls back to the host path outside that coverage.

So the duplication is not accidental: one emphasizes breadth and
reference-quality clarity, the other device-resident execution. Their
agreement is what the dual-device regressions in `regressions/scf/` pin.

## Where To Make Changes

When you need to modify the project, use this decision rule.

If you are changing public workflow structure, phase ordering, or component
registration semantics, start in `core`.

If you are changing user-visible system data, start with `cpu::Frame` and then
audit the transfer path and the device frame.

If you are adding new host-only analysis or a reference chemistry algorithm,
start in `cpu`.

If you are adding a new staged GPU component, start in `include/atomiverse/cuda`
and `src/cuda`, then add a matching example or test.
