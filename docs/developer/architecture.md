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

The core contracts live under [`core`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core)
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

Backend trait specializations own the concrete Frame type, nested DriverNode
adaptation, and execution lifecycle. The shared Driver compiles registrations
into a Plan and delegates through those traits; CPU Runtime and CUDA transfer
code stay in their respective backend directories.

### 2. CPU backend and CPU analysis layer

The CPU side lives under [`cpu`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu) (headers) and [`src/cpu`](https://github.com/MolCrafts/Atomiverse/blob/dev/src/cpu). It contains three distinct roles.

First, it defines the user-facing host data model through
[`cpu::Frame`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/frame.h). This is the structure that
examples and users fill directly.

Second, it provides a true CPU execution backend through
[`cpu::Runtime`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/runtime.h), which runs components
directly on that host frame with no transfer step.

Third, it provides CPU-only utilities that are not part of the staged CUDA
component model, such as integral generation, the Gaussian SCF, plane-wave, GPW and
semiempirical methods, Mulliken analysis, density grids, trajectory analysis,
and trajectory writing.

### 3. CUDA backend and staged components

The CUDA side lives under [`cuda`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda)
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

CPU MD initializes forces after communication and neighbor setup, before the
first integration. `MD.run(frame, 0)` therefore computes the initial energy and
forces without advancing time. Force-stage evaluators run with
`ctx.runtime.exec.initializing_forces`; they must evaluate the current electronic
or auxiliary state without advancing its history. Components in pre-force,
post-force or scalar-finalization stages opt in through `initialize_forces()`.
Integrators and ordinary observers do not run during this initialization.

CPU MD has one rebuild boundary: wrap owned coordinates, exchange owners,
sort owned rows, create borders, then build neighbors. Runtime schedules it;
`DomainDecomposition` owns fractional partitions and topology, `CommBrick`
owns transport and explicit periodic images, `NeighborList` owns rebuild
cadence and owned-position history, and `CellList` owns spatial indexing.
`Frame` alone records owned, ghost and allocated row counts. `PairList` stores
row indices; pair kernels use raw Cartesian differences for every rank count
and cell shape. Integrators advance owned rows only. Owned rows may be
spatially reordered at a rebuild (`atom_modify sort`), on one rank as on
many: an atom's identity is its `tag`, which the write-back and every gather
carry, and `gather_to_root` returns owners in tag order at any rank count.

Initial and live-mounted physical components share pre-build timestep
negotiation and unique-owner validation. The same integrator object may bind
two phases; distinct integrator or neighbor-provider objects cannot silently
compete for one physical pipeline. Sampling energy evaluations use the same
communication boundary, rebuilding for each proposal and rollback score.

Neighbor coverage has two separate distances: the supported interaction cutoff
and the enumeration radius including skin. Contract compilation rejects a
consumer whose interaction cutoff exceeds the provider's supported cutoff,
even when it fits inside the skin band. Communication covers the widest
consumer halo plus the provider's actual skin, and at least the enumeration
radius. Initial registration, live changes and borrowed nested consumers use
the same coverage check.

Nested MD uses the same Verlet scheduler and retains its own cumulative step
counter across invocations. A scoped active Plan restores the parent's services
on return, including exceptions. A physical child can run in a generic outer
workflow or at MD `end_of_step`; placing it inside an active parent force or
integration pipeline is rejected because it can migrate atoms and overwrite
partially accumulated forces. After a physical child, the parent rebuilds its
communication state and initializes its forces before resuming integration.

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

## Asynchronous Consumers And The Capture Buffer

Recorders and online-analysis observers whose real work is host-side (file
I/O, molpy analysis) run off the stepping thread. Each derives from one base
per backend —
[`cuda::AsyncConsumer`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/asynchronous/async_consumer.cuh)
or `cpu::AsyncConsumer` — and reads its steps from the driver's ONE shared
capture buffer (`cuda::CaptureBuffer` / `cpu::CaptureBuffer`, both on the
host-side `ATV::CaptureLedger`). CUDA consumers include:

- `Thermo`
- `XyzRecorder`, `MrecRecorder`, `md::PressureSeries`, `DensityRecorder`
- `RDF`
- `MullikenObs`

The pattern is always the same. A consumer declares the columns it reads and
its cadence. On a step it is due, the host admits the capture on the ledger
(host counters only; it waits only when the driver's byte budget is full),
the step's state is frozen once into a buffer slot shared by every consumer
due on that step (CUDA: graph-capturable kernels on the compute stream writing
pinned host-mapped memory), and each consumer's worker thread reads the slot,
calls the CPU helper that owns the formatting or analysis, and releases it.
Nothing is dropped; a consumer failure stops the run with an error naming it.

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

Component lifecycle is per instance, rather than per phase binding. The shared
Plan enumerates unique instances, and the runtime ledger reference-counts nested
plans and rolls back failed builds. Every binding still launches in its phase.

The initial-force schedule is shared by CPU and CUDA. Force-slot evaluators run;
other operations opt in through `initialize_forces`. Integration, recorder
launches and algorithm-history advancement are excluded. NPT seeds kinetic
history from the prepared initial velocities.

CUDA NPT keeps its chains and scalar propagation on the device. The NHC scalar
half-step, pressure convention and temperature ramps use the common mathematical
rules in `core/dynamics_math.h`. Box writers declare `changes_box()` so the
backend publishes the derived device cell before subsequent consumers. Trajectory
recorders freeze the cell and coordinates together.

### Minimization

The minimizer driver reuses the same staged idea with a different phase order:

`search_direction -> line_search -> force -> convergence_check -> end_of_step`

That lets the project share force-evaluation components with MD while changing
only the orchestration logic. MD and minimization have independent build flags;
CUDA accumulator initializers (`cuda/component/zero.cuh`) belong to the shared
component layer. They reset fields before every force evaluation, including
line-search trials, rather than only at one-time setup.

Minimizer budgets, completed counts, and diagnostic record steps use 64-bit
`BigInt`; Python and task transport preserve that width. Optimization records
are declared in `core/optimization_record.h`, shared by CPU and CUDA.

Both minimizers ask their execution backend to prepare initial forces before
searching. A zero iteration budget performs preparation without advancing the
optimizer or ordinary output; a negative budget is rejected before setup.
On both backends a Criterion is a Component: its phase bindings measure, while
its `verdict()` is polled by the loop. Unscheduled criteria still participate in
Plan build/destroy; repeated bindings build one instance.
Convergence requires all declared requirements to succeed. Stop-only guards
are neutral unless they request a stop, which takes precedence over success.
Minimizers with only guards still install the default force criterion. The
first measured energy has no previous sample and cannot satisfy the energy-change
tolerance. Each `run()` remains a fresh optimization with fresh optimizer history.

### Single-point evaluation and electronic structure

`cpu::Evaluate` runs a `prepare -> evaluate -> finalize` pass exactly once — the
home of single-point energy and force calculations.

**SCF is not a driver *family* — it is a driver *node*.** There is no
"electronic structure" category beside MD and Minimize. Gaussian SCF is a
concrete method (`cpu::scf::RHF` / `UHF` / `RKS` / `UKS`), and the CRTP base
they share, `cpu::scf::MeanFieldMethod<D>`, **is** a `cpu::Driver<scf::Phase>`
— the same loop-node shape as MD and Minimize. CPU plane-wave DFT also runs
through `Driver<pw::Phase>`; both `cpu::pw::PW` and `cuda::pw::PW` mount
directly as nested nodes on a force slot. CPU PW publishes actual iteration
energy, energy change and density residual before user checks. Standalone
stages receive an owned geometry snapshot, while nested stages and criteria
use the parent's actual Frame and restore its services afterwards. A private,
call-scoped final-band state owns all local C/f/k weights and orbital densities,
retaining reference metadata separately. CPU Gamma semilocal fixed/FD/Gaussian RKS/populated UKS,
semilocal fixed/FD/Gaussian general-k derivatives with populated spins share one unstrained sweep and
closed-form tensor. Separately qualified MP1/PBE Gamma RKS/populated-UKS slices
retain their signed occupations and generalized entropy. Gamma fixed-LSDA with Na>0/Nb=0 additionally carries a
producer declaration and canonical zero-row beta blocks. Collective admission
checks the declaration, counts, layout and exactly zero beta densities,
including ranks without local k points; reference result metadata stays
independent. Actual auxiliary residual/normalization checks reject an
incomplete state within the existing budget; local auxiliary failures are
synchronized before later spectrum/density collectives. CUDA PW uses one typed-stage SCF
loop for `run`, `forces` and `stress`. Nested calls borrow the parent's
Backend, device Frame and stream, then restore the parent's loop services.
The CUDA Gamma RKS fixed/FD LSDA/PBE-family epilogue builds a separate certified orbital
density, preserves the retained reference density/result and shares its raw
full-D projection between requested force and virial terms. Its per-call
workspace owns device scratch and drains the borrowed stream before release;
it never owns or destroys that stream. Finite/reference/antisymmetry checks
precede one nested E/F/W publication kernel. Energy-only calls construct no
derivative workspace. Force-only requests skip XC tensor quadrature; PBE-family
virials reuse the existing XC point law with distinct raw-gradient and flux
buffers. Spin SCF mixes concatenated alpha/beta histories through the same
Pulay engine as RKS, retaining the unmixed joint stopping norm. Fixed-size host
coefficient upload storage lives through its stream fences; cold upload arrays
also outlive normal and exceptional stream draining. FD refills the final spectrum at one chemical potential and retains
every nonzero final occupation. The fixed nonempty UKS owner packs independent
spin band sums and checks each spin's completion and charge. It reuses the
positive-spin XC component with distinct six-gradient/six-flux storage. A
typed per-call finish result distinguishes completion from a finite XC domain
outside the supported interior. The latter stages the inherited stress and
force candidates from the same accepted state before one publication; actual
auxiliary, reference and nonfinite failures still throw. The nested caller
uses the actual per-call publication result. No persistent solver flag or
second static selector decides whether to publish. This owner has
scoped native GH200 qualification
(2026-10-28)
with the current joint mixer for Gamma fixed nonempty UKS LSDA/PBE-family;
UKS FD, multiple k points and empty endpoints remain separate gaps.
Cold radial/Ewald setup, host scalar filling
and its small occupation upload, and requested host-output boundaries remain explicit; other stress domains retain their numerical paths.
Both PW backends
preserve user stages and criteria when rebuilding their internal callbacks,
and compile the current mount table before an SCF execution. There is no
`Capability` write-set — components write Frame keys and plug into the
*same* Evaluate, MD, and Minimize drivers ("swap the
energy source, keep the driver").

CPU and CUDA GPW likewise own `Driver<gpw::Phase>` execution. CPU DIIS and
all six supported OT outer paths use that loop; bounded OT line searches
remain numerical helpers inside a stage. CUDA Γ RKS/UKS, RKS multigrid and
multi-k RKS use one stage loop. Derivative calls consume one accepted SCF
state. Standalone calls materialize the input Frame; nested calls borrow the
parent execution services and restore them on success or exception. Internal
callback refresh preserves user stages and criteria, and nested budgets also
bound OT polish. Live-bound methods reject moves, mutation and recursive
execution. Public geometry is expressed through `Frame` and `Kind`.
Driver conformance has separate device-residency checks. Γ RKS single-grid
forces now contract accepted device P, physical raw-H W and resident grid
potentials using immutable derivative tables prepared before SCF. Final payload
retrieval follows force assembly. Other GPW force branches, cold setup and
numerical diagonal cell stress retain open residency work in the DFT roadmap.

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
vs `cuda::scf::RHF{basis}.run(frame)`, and
`UHF{basis, n_alpha, n_beta, charge}.run(frame)` / `.forces(frame)` on both).

They solve different engineering problems. The CPU stack is the clearest place
to express reference algorithms with xtensor and dense linear algebra: it is
where validation, deep-dive tutorials, and physical-correctness tests live, and
it covers every method and basis. The CUDA stack is organized around device
residency — for the shells its kernels cover it keeps integrals, the SCF
iteration, and forces on the GPU with no scientific data returning to the host
mid-solve. `cuda::scf::RHF` does not host-fill ERIs: Cartesian l ≤ 3 runs
`launch_eri_lgt1` (STO-3G s/p stays on `launch_eri_sp`); spherical l ≤ 3
sandwiches that tensor with CartToSph. l ≥ 4 throws `launch_eri_g`. Forces
outside STO-3G l ≤ 1 throw `launch_rhf_forces_lgt1`. `cuda::scf::UHF` throws past
STO-3G l ≤ 1 (`launch_eri_sp`, `launch_uhf_forces_sp`).

So the duplication is not accidental: one emphasizes breadth and
reference-quality clarity, the other device-resident execution. Their
agreement is what the dual-device SCF end-to-end cases pin.

Shared physical contracts and immutable model data live under `electronic/`.
The PBE family uses one `PbeParameters` definition for exchange μ/κ,
correlation β and the RPBE enhancement form. The CPU XC evaluator and CUDA
grid kernels consume these same values; PW and GPW pass the functional
identity through to the kernels. Backend admission uses the explicit pure
PBE-family predicate, since a generic GGA predicate would also admit models
without device implementations. Numerical kernels stay under `cpu/xc` and
`cuda/xc`; each method retains its own grid, workspace and iteration history.
Sharing a physical model does not require sharing execution state or adding
another SCF loop.

## Where To Make Changes

When you need to modify the project, use this decision rule.

If you are changing public workflow structure, phase ordering, or component
registration semantics, start in `core`.

If you are changing user-visible system data, start with `cpu::Frame` and then
audit the transfer path and the device frame.

If you are adding new host-only analysis or a reference chemistry algorithm,
start in `cpu`.

If you are adding a new staged GPU component, start in the CUDA public headers
and `src/cuda`, then add a matching example or test.
