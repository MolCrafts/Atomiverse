# Atomiverse project contract

Project architecture, build recipes and conventions. The former agent harness
and capability-loop orchestration have been removed. Regression case records
are archived evidence, not runnable checks.

## What this repo is

- Engine for multiscale, multiphysics atomistic simulation: classical MD, MC,
  geometry optimization, vibration, electronic structure (Gaussian SCF,
  plane-wave DFT, GPW, NDDO/AM1, DFTB). C++20, namespace `ATV::`.
- Two backends (CPU, CUDA) behind one Driver/Component/Frame API; where they
  still diverge is listed under **Known debt**.
- Develop on **`dev`**; worktrees branch off `dev` explicitly. `master` is the
  released snapshot, advanced only by merging `dev` — never commit to it. When
  they diverge, `dev` describes current behaviour.

## License & LAMMPS porting

Atomiverse is **GPL-2.0-only** like LAMMPS (`LICENSE` is LAMMPS's GPLv2 text),
so parallel machinery is **ported from LAMMPS, not independently derived**.

- **Scope:** domain decomposition, halo/forward/reverse comm, pack/unpack,
  k-space parallel FFT decomposition and remap, load balancing, neighbor-build
  parallel structure. Read the LAMMPS implementation first and transplant its
  algorithm and communication structure; do not invent an alternative and A/B it.
- **Reference checkout:** `../../lammps-src` (shallow sparse `src/` at
  `970d4ba`). Deepen or re-clone rather than trust memory; stamp the SHA you
  read into every header you touch. Headers naming `78a7ddf` came from an
  earlier checkout — do not rewrite them wholesale.
- **Provenance headers are mandatory:** name the LAMMPS source file
  (e.g. `src/comm_brick.cpp`) and keep the Sandia copyright notice.
- ATV's laws still govern the *shape* of ported code; public API stays
  ATV-native. Physics/serial kernels (SCF, GPW, TB, pair math) are outside the
  mandate and retain their scientific oracle provenance.

## Where things live

| Path | Contents |
| --- | --- |
| `include/atomiverse/` + `src/` | Mirrored header/impl trees |
| `tests/` | Unit tests only |
| `regressions/` | Archived scientific cases and oracle records; no runner |
| `benchmarks/` | Standing performance benchmarks; never registered with ctest |
| `docs/` | The manual. `docs/roadmap/` is the **only** place that states what is implemented on which backend |
| `.claude/notes/` | Passive project knowledge (current understanding only) |
| `.claude/specs/` | Live specs; deleted at `done` |
| `app/`, `python/` | Installable apps, nanobind wheel |
| `tools/` | Data generators, oracle pin scripts, debug harnesses, `cpu_job.sh` / `gpu_job.sh` batch templates |

## Build & test

Run in a **login shell** (`bash -lc`) so Lmod is defined. Nothing auto-loads.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich          # CPU (x86_64 login node)
  export OPENBLAS_ROOT="$OPENBLASDIR"             # module sets OPENBLASDIR only
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu   # default is wrongly aarch64
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"   # module sets link path only
  export ATV_MOLPY_PATH="$(pwd)/../molrs"       # merged local molpy checkout
  cmake -B build -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target atv_tests
  ctest --test-dir build --output-on-failure   # unit tests
'
```

- Skipping the `LD_LIBRARY_PATH` mirror is the #1 cause of `libopenblas.so.0` /
  `GLIBCXX_3.4.32 not found`. `-march=native` is required for the SCF ERI kernels.
- **CUDA builds and runs only on the aarch64 GH200 node.** Agents submit and
  inspect compute jobs through the public `molq` CLI, using its installed help
  and the authorized resource budget. Reuse a running job's reference and
  observer; never duplicate an uncertain submission or bypass its scheduler
  interface. `tools/gpu_job.sh` supplies the environment recipe (knobs
  `ATV_CMAKE_FLAGS`, `ATV_TARGETS`, `ATV_CTEST_RE`). The script is
  `#!/bin/bash -l`, loads `GPU/buildenv-gcccuda/2026.03-cu13.0`, and keeps the
  default aarch64 Rust. `tools/gpu_job.sh` / `tools/cpu_job.sh` are
  parameterised templates, not one script per slug. Never pipe `module load`.
- **molpy** is consumed from the merged local checkout via `ATV_MOLPY_PATH`
  (default `../molrs`). Use its `molcxx/` bridge and `molpy/` wheel together;
  no revision pin is imposed during migration.
- Source transport into a reused build tree must invalidate changed files;
  do not preserve older timestamps over newer compiled objects. After class
  layout changes, rebuild every consuming translation unit, including tests
  and regressions. Mixed-layout binaries are not numerical acceptance evidence.
- Never kill a `git commit` / `git push` mid-hook (pre-commit restores stashed
  unstaged changes only on a normal exit).
- Machines and workbenches: [arrhenius-workbench.md](../../.claude/notes/arrhenius-workbench.md).
  Never set `OMP_PROC_BIND`/`OMP_PLACES` when timing BLAS-bound paths:
  [openblas-thread-binding.md](../../.claude/notes/openblas-thread-binding.md).

| CMake option | Default | Enables |
| --- | --- | --- |
| `ATV_BUILD_CUDA` | ON | CUDA backend (`ATV_HAS_CUDA`) |
| `ATV_BUILD_HF` | OFF | Gaussian SCF (`ATV_HAS_HF`) |
| `ATV_BUILD_PW` | OFF | Plane-wave DFT (FFTW3) |
| `ATV_BUILD_GPW` | OFF | GPW (needs HF + PW) |
| `ATV_BUILD_KSPACE` | OFF | PME reciprocal-space electrostatics (FFTW3) |
| `ATV_BUILD_TESTS` / `_EXAMPLES` | OFF | target groups |
| `ATV_REAL_STORAGE` / `ATV_REAL_WORK` | `double` | CPU per-atom storage / kernel-work precision |
| `ATV_ID_SIZE` | `bigbig` | 64-bit `TagInt` (`smallbig` = 32-bit) |

## Iron laws

Binding; enforced by **this file and code review**, never by grep gates
registered as tests.

### 1. Tests are unit tests — nothing else

`tests/` holds **only** single-function / single-method unit tests mirroring
the source layout, linked into one gtest executable (`atv_tests`, plus
`atv_tests_cuda` / `atv_tests_mpi`; a dedicated executable only for a test
that needs its own link line); the suite runs in seconds. Forbidden there:

- **End-to-end of any kind** — no full SCF/MD/vib/MC solve, convergence loop,
  timing, or external-engine comparison.
- **Reading outside the repo** at build or test time. In-tree parameter data
  (`basis_data/`, a future `data/`) is first-party code with provenance.
- **Style / surface / architecture gates** — no grep scrubs, no scripts via
  `add_test`, no test reading the source tree for files or tokens.

No opt-in switch: if it is registered, it runs.

### 1a. Archived end-to-end evidence

`regressions/` preserves scientific case sources and external oracle records.
Its runner and CMake registration have been removed; these cases are not
current executable acceptance gates. See `regressions/README.md`.

### 2. Frame is the only public token

The sole public I/O currency is **`Frame`** = `blocks` + `meta` + `box`.

| Lands where | What |
| --- | --- |
| **Frame** | System state: positions, species, charges, masks/regions, topology, energies, forces, box, meta |
| **Owning type** (ctor args / public members) | Algorithm parameters (`tol`, cutoffs, `dt`, basis) |
| **Categorical slots** | Other components (`set_evaluator`, `set_engine`, `set_integrator`) |
| **Engine-private only** | `cpu::Context` / `cuda::Context`, `LaunchContext`, `SolverContext`, service slices |

Users never define, fill, subclass or pass a Context, and never hand-drive
Component lifecycle (`.launch(...)`, `build({ctx, frame})`, `setup_pass`).
Writing a Component = overriding `launch(const LaunchContext&)` (author side).
No parallel `*System`, no public `AtomList`, no `SinglePoint` class.

### 3. No public Config bags, no factories

- Parameters are **constructor arguments** (`RHF{BasisSet::STO3G, charge, tol}`,
  `NoseHooverNPT(T, tau_T, P, tau_P, dt)`) or public members.
- Forbidden as public API: `*Config` / nested `::Config` / any bag stuffed into
  a constructor; `obj.param = …` after default construction as the parameter
  story; `make_*` / `build_*` / `create_*` / `*_config()` returning objects.
- Narrow exceptions: true free math with no owner; lookup-table constructors
  (`GthPotential(Element, name)`); subsystem-private logging/async infra;
  `ATV::make_binding`; the comm `make_*_plan` makers.

### 4. Region and selection are Frame fields

- Which atoms, bonds or region a method acts on is declared on the **Frame**
  (named columns / blocks / topology), never via `set_region` / `set_mask` /
  `set_links` / constructor atom-index lists.
- A missing or invalid required field **throws** with a how-to — no silent
  "treat everything as MM".
- Derived membership is fine when it is a pure function of Frame data (QM/MM
  link cuts = bonds whose endpoints differ in the QM mask; no bonds ⇒ no links).
- Algorithm parameters and slots stay on the object. Named exception: Gaussian
  SCF ghost indices (`MeanFieldMethod::ghost_atoms`).

### 5. Frame columns are `span`, meta is `get`

C++ columns are `frame.span(ColKey)` (non-owning `Span<T>`); typed scalars are
`frame.meta.get(MetaKey)`. No `get_column`; pythonic names belong on nanobind.

### 6. Composition over inheritance

- Workflows are **composed** from presets + Components. AIMD = `MD` + an
  electronic evaluator. QM/MM = the same + MM evaluators + a connection-layer
  evaluator that `set_engine`s the QM method — no `QmmmDriver`, no second MD.
- A new preset needs a genuinely new **loop topology**; a different force
  recipe is a Component.
- Inheritance depth ≤ 2 (`Concrete → Kind → Component`); no category driver bases.

### 7. No freestanding integral bag

Integral state is **private fields** on the concrete method (`ints_S_`,
`ints_Hcore_`, shells), rebuilt from Frame; engines take a non-owning
`internal::IntegralView`. Public API is only `RHF{basis, …}.run(frame)` /
`.forces(frame)`, same shape on CUDA.

### 8. Every method domain has its own build flag

- Each method domain is gated by `ATV_BUILD_<DOMAIN>` in
  `cmake/AtvProductModules.cmake`; **OFF excludes its sources, headers, tests
  and regressions**. An unconditionally compiled domain is a defect; a new
  domain adds its flag in the same change.
- Dependencies live in the module DAG (`atv_force_on`). Work on one domain with
  its closure only (e.g. SE-only: `SEMIEMPIRICAL` + `EVALUATE` ON; everything
  else, CUDA included, OFF).
- molrs is **derived**: products force `ATV_BUILD_MOLPY=ON`, never the inverse
  (`FATAL_ERROR "X requires MOLRS"` is banned). A slice needing a specific molrs
  pin (WHAM ⇒ `ATV_MOLPY_HAS_WHAM`) gates sources on it.

### 9. No silent debt

On finding an anti-pattern, failing test, broken invariant or clear bug in the
surface you touch or depend on: **fix it now** if local and stage-allowed,
otherwise **stop** and report `path:line`. Never "pre-existing", never
skip-mark, never weaken an assert, never land on known rot. Name it in the
summary (found / fixed / blocking). Outranks "stay in scope" and "minimal diff".

## Design defaults

- **OOP by default** — domain concepts are types with methods
  (`NeighborList.build`); free functions only for true free operations.
- **Primitive, single-responsibility public APIs** — construct → configure →
  one concern → read result. No `run_everything` / `compute_all` / `pipeline`
  façade; composition is the caller's job.
- **Inline until the second real use** (or until a unit test must target it).
- **No fallback values** — lookup tables throw on unsupported input.
- **Hot-path data carry only fundamental physical quantities** (positions,
  velocities, integrals, shell descriptors), never indices, caches or derived
  maps; consumers derive their own views.
- **Before adding a public symbol:** owning type → method; >1 user-visible step
  → primitives; one call site → do not extract; context-bag field → Frame
  field, ctor arg or member instead.

## Architecture

Pillars: **Driver** (owns the loop), **Component** (one single-step
operation), **Frame** (device-resident state). Backends `cpu/` and `cuda/` sit
behind the CUDA-free contract layer `core/`. **No backward compatibility** — no
shims, alias headers, deprecated forwarding or `#error` tombstones.

### Directory layering

Each backend root holds **directories only**, full English names, mirrored
across backends (a CPU-only domain is absent on `cuda`, never renamed); a
feature lives in the same directory on every backend.

```
<backend>/
  frame/ driver/ component/ evaluate/       pillars (+ Evaluate preset)
  device/ communication/ neighbor/          Device/Domain runtimes, PairList
  asynchronous/                             AnalysisRuntime, snapshot ring, D2H bridge
  dynamics/ minimize/ scf/ planewave/ gpw/  method domains on both backends
  longrange/ potentials/ observe/ record/   (CUDA gpw = method over internal stages)
  io/ linalg/ electronic/ montecarlo/ replica/ path/ semiempirical/ xc/
  vibration/ bias/ multiscale/ electrodes/ task/                 CPU only
```

- Shared code goes in the matching pillar dir, never inside a domain subdir.
- Electrostatics splits LAMMPS-style: real-space pair styles in `potentials/`;
  reciprocal solvers **and their math** in each backend's `longrange/`.
- `core/` holds only scalar laws both backends call: `longrange_math.h` (Ewald
  self energy, neutralizing background), `dynamics_math.h` (NHC half-step,
  thermal DOF, pressure/Berendsen, ramps). Parallel kernels stay per backend.
- `core/` also holds precision tiers, `Mat3`, units, Frame schema, device tags,
  backend traits, `Driver<PhaseEnum, DeviceTag>`, `Plan`, `Stage`, `role.h`,
  `verdict.h`, `iteration_status.h`, `ghost_contract.h`. `molpy.hpp` is the only
  include for CXX-generated code.

### Box and decomposition

- **Box** stores lattice vectors in columns (`r = H s`, molrs/LAMMPS H) on both
  backends; CellList, comm, PW, Ewald/PME/PPPM, DFTB lattice sums and
  vibrational supercells use that convention.
- CPU Box: inverse/face-height/reduced-lattice caches, exact general-skew
  nearest image, host `revision` bumped on mutation. CUDA Box: FP64 cell,
  narrowed `as<force_t>()` copy for kernels, lattice-vector `wrap_position`,
  `revision` keying cell-derived tables (PME/Ewald).
- Neighbor-listed CPU kernels use raw differences between explicit atom/image
  rows and never min-image; bonded styles and the list-less pair fallback still
  apply `Box::min_image`.
- CPU `DomainDecomposition` owns topology and fractional bounds; **`Frame` alone
  owns `nlocal`, `nghost` and capacity.** `CommBrick` creates explicit periodic
  ghost rows on every axis, single-rank self images included; reach may span
  several images. No non-periodic brick path.

### Driver — a staged loop, not a category

A Driver is **stages** (typed phase enum) + **iteration** + **termination**
(a policy object). No category bases — `EvaluateDriver` / `DynamicsDriver` /
`OptimizeDriver` are never the answer.

| | What it is | How many |
| --- | --- | --- |
| **Runtime** (CPU) | Composes Domain/Device/Analysis runtimes; owns frame binding, the rebuild boundary, initial forces, the component ledger, the dispatch lock, nested `ScopedPlan`s | one per run |
| **Backend** (CUDA) | Streams, Frame, Plan, `detail::ComponentLifecycle` | one per run |
| **Driver<P>** | **One loop node**: stages + plan + criteria | any number, nestable |
| **Criterion** | A Component the driver polls once per iteration | any number |

```
Driver<PhaseEnum, DeviceTag>               plan / dispatch primitive (core)
  └─ cpu::Driver<P> / cuda::Driver<P>      THE loop node
       └─ {MD, Minimize, MC, Evaluate, Hessian, PathDriver}   presets
```

A **preset** is not a layer: its constructor attaches default criteria and
infrastructure — defaults, not semantics.

| Preset | Phase enum | Default policy |
| --- | --- | --- |
| `cpu::Evaluate` / `cuda::Evaluate` | `evaluate::Phase` | — (one pass) |
| `cpu::md::MD` / `cuda::md::MD` | `md::Phase` | — (step budget) |
| `cpu::minimizer::Minimize` / `cuda::…` | `minimizer::Phase` | criterion `ForceConverged` |
| `cpu::vib::Hessian` | `vib::Phase` | — (fixed 6N + finalize) |
| `cpu::mc::MC` | `sampling::Phase` | **outcome** `Metropolis` |
| `cpu::PathDriver` | `path::Phase` | criterion `AllReplicasConverged` |
| Gaussian SCF (`MeanFieldMethod<D>`, `cuda::scf::RHF`) | `scf::Phase` | residual / ΔE criteria |

- **`Criterion`** — *should the loop stop?* `add_criterion(…)`, plural; sees a
  `LoopState`, never a phase, so one criterion serves every preset. CPU
  criteria declare `contributes_to_convergence()`: success = every declared
  requirement `Converged` on **every rank of the driver domain** (one MPI
  reduction); stop-only guards are neutral; an undeclared `Converged` is an
  error; any `Stop` aborts. Minimize with only guards still installs
  `ForceConverged`.
- **`OutcomePolicy`** — *does this iteration count?* `set_outcome(…)`,
  singular, `Commit`/`Rollback`. MC accept/reject is an outcome.
- **Algorithms never judge themselves.** `LBFGS`/`CG` publish
  `ConvergenceMetrics`; a Criterion judges. A component may *request* a stop,
  never *declare* convergence.
- **Status belongs to the loop.** Each node owns `IterationStatus` +
  `ConvergenceMetrics`; `ctx.loop` borrows the active loop's; `run_nested`
  re-points and restores, so an inner loop cannot stop its parent.
- **A Driver is an executable node, not a Component** (`launch` = one step, no
  allocation; a nested driver runs its own loop). `on(phase).add` of a
  `DriverNode` mounts a registration-private slot (core
  `PhaseBuilder::mount_nested`, `nested_driver`) copying `provides_energy` /
  `provides_force`. Type erasure happens at the **node** boundary, never at the
  stage boundary.
- **Nesting** runs the child's Plan on the parent's Runtime (no second
  Runtime/DD/communicator). A physical child (MD, Evaluate, Minimize, Vib, MC)
  rebuilds halo/list for its plan, invalidates the parent's halo/list/forces on
  exit, and may run only in an outer workflow or at MD `end_of_step`;
  electronic children borrow the parent's communication.
- **Replicas are a nested loop**: replica loop + budget criterion + a `couple`
  component; replicas come from a `ReplicaExpansion` (owning per-replica
  parameters); `run()` takes one Frame and writes one back.
- **Known ceiling:** components cannot be cloned, so one inner Driver body over
  N replicas (N L-BFGS histories) is not expressible. Undecided.

### Stage typing

Phase enums all start at 0, so an `int` phase is ambiguous. Typed entry points:
`ATV::slot_of(phase)` (the one enum→slot conversion),
`ATV::slot_count<PhaseEnum>()`, `backend.run_stage(phase, step)`,
`ctx.at(phase)`, `ATV::make_binding(phase, order)`. Hand-written
`static_cast<int>` over a Phase enum is banned; `DefaultBinding`
(`{PhaseKind, int phase, int order}`) is built only by `make_binding`.

### CPU MD execution

Binding summary; mechanism and LAMMPS source mapping in
[cpu-lammps-alignment.md](../../.claude/notes/cpu-lammps-alignment.md).

- **One rebuild boundary** (LAMMPS `Verlet::setup`/`run`): wrap → exchange →
  sort → borders → neighbor. Runtime does migration, borders and force reverse
  comm; `pre_exchange` / `pre_neighbor` / `post_force` are hooks.
- Only MD applies the provider's delay/every/check; linear workflows rebuild at
  their force slot with LAMMPS `Min::init` cadence; a plan without a
  NeighborProvider migrates every step. Minimize, MC and Hessian/Phonon forbid
  row reordering; elsewhere rows move and consumers resolve atoms **by tag**.
- **Neighbor requests:** consumers declare Half/Full and ghost-centred rows;
  Runtime aggregates them for exactly one NeighborProvider per plan.
  `ctx.md.pair_list` is always half; `full_pair_list` is separate. Consumers fit
  the provider cutoff (excluding skin).
- Initial, live-mounted and nested physical components share identity checks
  and timestep negotiation before build; one integrator object may bind both
  integration phases, two distinct objects may not own the same integration.
- **Mutation lock:** structural edits (`unfix`, live `on(phase).add`) throw while
  `Runtime::dispatch_active()`. Every ghost-contract refold marks the
  communicator stale for the next rebuild.
- **Ledger:** unique-instance lifecycle with alive token and separate nested
  DriverNode counting; a failed Driver-owned run destroys built components and
  rebuilds on retry.
- **Execution declarations:** `requires_full_frame()` throws on a spatial
  partition unless `set_serial_on_replica()`; stress output = slot
  `stress_requested` AND `exec.stress_requested`. Resolve Frame columns once per
  launch (per-launch resolution budget).
- **Initial forces:** MD evaluates forces before the first half kick, including
  `run(frame, 0)`; the same Frame continues without re-preparing. Force slots
  run; other bindings opt in via `initialize_forces()`; integrators and ordinary
  observers do not run. CPU walks `pre_force`..`end_of_step` and re-prepares
  when force-assembly bindings change or the box revision moved since
  `end_of_step`. CUDA mechanism: see Known debt.
- **History flags:** stateful evaluators (CPMD/XL/ASPC, metadynamics,
  Langevin/DPD RNG, constant-potential, auxiliary dipoles) never advance history
  while `exec.initializing_forces` (setup) or `exec.trial_evaluation` (MC trial)
  is set.
- Step counters are `BigInt`.

### A loop is a Driver; a method is concrete

- A Driver is the execution/iteration framework, not a base required by every
  named physical method. A genuine SCF/SCC or workflow loop may inherit or own
  a Driver; its stages publish residuals and a Criterion judges convergence.
  Single-pass methods (DFTB0, GFN0), XC functionals, basis/PP parameter data,
  Hamiltonian terms and force kernels do not gain artificial Drivers.
  Parameter/model variants with the same topology reuse one loop implementation,
  with independent state per calculation. Numerical library iterations do not
  require an extra framework node. See `.claude/specs/es-extensibility.md`.
- No generic "SCFSolver", no electronic-structure *category*. The concrete
  method — `RHF`, `UHF`, `RKS`, `UKS`, `AM1`, `RM1`, `MNDO`, `PM3`, `PW`, `GpwDft` — owns its
  Hamiltonian, density build and parameters as members.
- Gaussian SCF is a Driver node on both backends (`MeanFieldMethod<D> :
  cpu::Driver<scf::Phase>`, `cuda::scf::RHF : cuda::Driver<scf::Phase>`),
  stages `fock → mix → diagonalize → density → check`; CPU stages under
  `cpu/scf/stages/`, CUDA stages owned by `RHF` (`DIISMixer`, `EigenSolver`
  stay `internal::`). `scf::Phase` is legal **because** a node carries it.
  `cpu::pw::PW` is `Driver<pw::Phase>`; `GpwDft` DIIS paths are Driver stages.
  DFTB2 owns `Driver<scc::Phase>` (`shift → diagonalize → charges → mix →
  check`) and its persistent stages. Other SCC DFTB/xTB methods compose a
  helper Driver; consolidate matching orchestration and lifecycle contracts,
  without requiring their public method classes to inherit Driver.

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10};
rhf.run(frame);        // Frame in / Frame out; workspace private
```

| Member | Does | Frame writes (overwrite) |
| --- | --- | --- |
| `run(Frame&)` | SCF | `kEnergy` [eV] |
| `run(const Frame&)` | SCF, geometry read-only | nothing |
| `forces(Frame&)` | SCF + forces | `kAtom_f*` [eV/Å], `kEnergy` |
| `stress(Frame&)` (periodic) | SCF + analytic stress; returns σ [eV/Å³] | `kStress` = σΩ [eV], `kEnergy` |

- Knobs of those members (FD step, strain step, axis mask) are members. FD
  oracles stay free functions (`*_forces_fd`, `*_stress_fd`) in public units;
  analytic assemblies are `internal`. Hartree/Ry live only on result objects.
- CUDA presents the same shape. An outer MD/Minimize/Evaluate binds the method
  (or a thin force-slot component owning it) that accumulates (`+=`) onto the
  Frame through the non-writing entries, never duplicating its parameters.
- *Properties* (charges, dipoles, Mulliken, AM1-BCC) are component **outputs
  written to Frame**, never Driver categories.

### Component

- **Role is a bitset**: `FrameImmutable` (does not modify x/v/f/box/topology),
  `Deferred` (optional; unset ⇒ main-loop required), `ExternalIO`. Outputs are
  Frame keys (no `Capability` set). Validation: `Evaluate` ≥1 energy source,
  `MD` ≥1 `Integrator`, `Minimize` ≥1 `Optimizer`, `MC` ≥1 `Mover`.
- **Kinds**: one class per header under `component/`, depth ≤ 2. Pipeline glue
  (`NeighborList`, `Zero*`, `Reduce`) inherits `Component` directly; no
  `Infrastructure` kind.

| Kind | Default Role | Examples |
| --- | --- | --- |
| `Potential` | required | PairLJCut, BondHarmonic, CoulLong, KSpace |
| `HF` (tag on the Driver node) / `Semiempirical` / `PlaneWave` / `GPW` | required | concrete methods |
| `Bias` / `PathCoupling` | required | HarmonicBias, NebCoupling |
| `Observer` | FrameImmutable\|Deferred | RDF, MSD, Virial, KineticEnergy |
| `Recorder` | +ExternalIO | Thermo, XYZ |
| `Integrator` / `Thermostat` / `Barostat` / `Optimizer` / `Constraint` / `Mover` | required | VelocityVerlet, LBFGS |

- **Opt-in protocols** the engine probes for (no `dynamic_cast`):
  `NeighborConsumer`, `NeighborProvider`, `LongRangeParticipant` (exactly one
  matched real/reciprocal pair), `PairEvaluator`, `ConstraintProvider`,
  `TimestepProvider` / `TimestepConsumer`. `AsyncComponentBase` is a class.
- `Component` carries default-false **declared-requirement** virtuals
  (`requires_full_frame`, `requires_tag_index`, `requires/provides_system_stress`,
  `updates_owned_charges`, `provides_hessian`, `initialize_forces`,
  `neighbor_provider()`, …); each exists because Runtime/Driver scans for it.
  A protocol or accessor earns a place only when something scans for it.
- `ForwardPacker` / `ReversePacker` (`component/comm.h`) are self-initiated
  (`ctx.md.comm->forward_comm(*this)`), so `Component` carries none of their
  virtuals. Packers: `ChargePacker`, `FixSettle`, `FixRattle`, `EAM`.
- **Lifecycle:** `build()` (one-time alloc) → initial forces when required →
  `launch()` × N → `destroy()`. No allocation in `launch()`.
- **Ownership is explicit:** `on(phase).add(rvalue, order)` transfers;
  `on(phase).add_ref(lvalue, order)` borrows; `add(lvalue)` is deleted.
  `set_xxx` owns, `set_xxx_ref` borrows.
- `on(phase)` is the **only** public C++ registration path; `bind_defaults` /
  `bind_defaults_ref` are protected (`core/driver_template.h`). Python
  `sim.add(...)` is a binding convenience.
- **Observer computes; Recorder writes I/O.** Never both.

### Context tiers

Flat core (threading/MPI topology, timing registry) + per-workflow slices,
reached via `LaunchContext::runtime` (`ctx.runtime.md.pair_list`).

| Slice | Struct | Key fields |
| --- | --- | --- |
| `md` | `DynamicsServices` | `dd`, `comm` (CommBrick), `ghost`, `pair_list` (half) / `full_pair_list`, `atom_sort`, `reneighbor`, `n_dof_global`, `is_last_step` |
| `mc` | `SamplingServices` | `proposal` (`mc::Transaction`), `rng` |
| `opt` | `OptimizeServices` | `reeval_forces`, `line_search_trial` |
| `path` | `PathServices` | replica coupling state |
| `loop` | `LoopServices` | borrowed `status` + `metrics` of the active loop |
| `exec` | `ExecServices` | `runtime`; vetoes `forces_requested` / `stress_requested`; history flags `initializing_forces` / `trial_evaluation`; `serial_on_replica` |
| `analysis` | `AnalysisServices` | `executor` (snapshot-ring analysis) |

Slices are plain aggregates; Context stays non-polymorphic (static-asserted).
`cuda::Context` carries streams, `host_box`, `loop`, `md` and execution requests.
Nested electronic Driver nodes borrow the parent's Backend and device Frame;
they must not create a second execution owner. Internal Gaussian SCF stages
may use `SolverContext{stream, frame, step}`; this is not the public method API.

### Frame

Boundary vocabulary: typed **blocks**, exact-dtype **meta**, **box**.

| Frame | Representation | Topology |
| --- | --- | --- |
| `molpy.Frame` | flat typed Block columns + typed `MetaValue` map + box | flat bonds/angles/dihedrals |
| `cpu::Frame` | exactly `BlockMap blocks; MetaMap meta; Box box;` | per-atom LAMMPS columns (`num_bond`/`bond_atom`, global-tag partners) |
| `cuda::Frame` | raw device pointers + host-cached counts | single-GPU flat lists; multi-GPU per-atom SoA |

- `from_molpy()` / `to_molpy()` — strict schema boundary, once per `run()`.
  Unknown blocks/columns, wrong dtypes and row counts fail immediately.
  `to_molpy` writes x/v/f, charge, image, tag (checked i32 narrowing), box and
  serializable meta; type, mass, topology are static inputs. `cpu::to_cuda()` /
  `cuda::to_cpu()` move host ⇄ device.
- **CPU precision:** `StorageReal` / `WorkReal` / `AccReal` (accumulators
  double).
- **CUDA classical MD precision:** one type per role (`cuda/frame/types.cuh`,
  OpenMM mixed strategy, knob `ATV_CUDA_PREC_MODE`): `force_t` force math,
  `pos_t` coordinates (+ FP32 correction column in mixed mode), `mixed_t`
  integration state, `geom_t` cell (FP64), `acc_t` reductions (FP64),
  `fixed_t` 64-bit fixed-point forces. Integrators reconstruct/split
  coordinates (`precision.cuh`) and refuse a faulted force (`ForceFault`;
  `cuda::to_cpu` reports tag and step). Every per-atom move (sort, migration,
  snapshot, transfer) carries correction and velocity columns untruncated.
  Gaussian SCF CUDA tiers `real_t` / `work_t` / `prec_t` are FP64 in every mode.
- Integer IDs by LAMMPS role: `LocalInt`, `TagInt`, `ImageInt`, `BigInt`
  (always 64-bit).
- **Frame purity:** no method scratch (integrals, AO maps, grids, DIIS history
  belong to the method); the pair list stays on `NeighborList`. No implicit
  transfer in low-level operators; CUDA stays on device between explicit
  transfers.

### molrs integration

The engine *measures*; molrs *analyzes and stores*. Bridges are raw-array FFI
wrappers (`cpu/io/frame_writer.h`, `cpu/observe/trajectory_metrics.h` PIMPL),
no `rust/cxx.h` in public headers. **No analysis math in C++ on either
backend**; RDF observers stream positions + cell into the molrs accumulator;
the one g(r) emitter is `cpu::rdf_write_curve`.

### Ghost data is a lifetime contract, not a wire

A component returns a `static constexpr` table from `declare_ghosts()`
(`core/ghost_contract.h`); each `GhostField` states **freshness** (owner →
ghost: `read_stable` / `read_each_step`) or **data flow** (ghost → owner:
`reduce_to_owner`).

- **A style that reads a ghost column without declaring it gets garbage,
  silently** — symmetric garbage passes every force-balance check, for self
  images and remote ghosts alike. A new ghost-reading style adds its table in
  the same change.
- Presence is implied by the read. The same field from two components merges to
  the **stricter** freshness (order-independent); opposite directions on one
  column conflict. Named constructors are the only way to build a field.
- The Driver folds every table onto `base_fields::kEngine` into one
  `GhostContract` (`ctx.md.ghost`). The contract→pass mapping lives once, in
  `make_wire_plan` (`cpu/communication/atom_pack.cpp`): `Border` = every read,
  `Comm` = `step` reads, `Reverse` = every reduction. Nothing above it names a
  wire; nothing below it names a freshness.
- `Block::resolve_column(GhostField)` = the column a component asked for;
  `resolve_columns(flags)` = which columns belong on wire X (migration only).
  No third selector.
- **CPU comm is LAMMPS `CommBrick`:** one swap at a time on tag 0 (standing
  `Irecv` → pack → blocking `Send` → `Wait`), `sendproc != recvproc` swap table,
  the round only as forwarding unit. Detail: [md-comm.md](../../.claude/notes/md-comm.md).
- **Only halo and migration compile a `PackPlan`** (`make_wire_plan`,
  `make_exchange_plan` on `kColExchange`). `gather_to_root` / `bcast_frame` move
  columns column-major via `Block::resolve_columns` +
  `comm::column_schema_fingerprint`; `mc::Transaction` uses
  `Frame::copy_row_from` on a `clone_structure` patch. Never borrow the
  migration wire for anything else.

## Conventions

- **Namespace:** `ATV`; `ATV::cpu`, `ATV::cuda`, `ATV::md`, `ATV::minimizer`,
  `ATV::sampling`, `ATV::path`, `ATV::scf`, `ATV::pw`, `ATV::evaluate`. Code
  under an `internal/` directory lives in the matching `internal` namespace.
- **Extensions:** CPU `.h`/`.cpp`, CUDA `.cuh`/`.cu`, core `.h`; CUDA-free
  bridge headers under `cuda/` keep `.h`.
- **Phases:** Evaluate `prepare, evaluate, finalize`; MD `initial_integrate,
  post_integrate, pre_exchange, pre_neighbor, neighbor, pre_force, force,
  post_force, final_integrate, end_of_step`; Minimizer `search_direction,
  line_search, force, convergence_check, end_of_step`; Vib `setup, displace,
  force, accumulate, diagonalize, thermochem, end_of_step`; Sampling `propose,
  energy, end_of_step`; SCF `fock, mix, diagonalize, density, check`; SCC
  `shift, diagonalize, charges, mix, check`.
- **Units:** eV, Å, fs, amu, eV/Å, K on the public surface (`core/units.h`);
  SCF/PW compute in Hartree/Ry and write eV to `kEnergy`.
- **Precision:** state it or state the integer role — `Real`/`Int` do not exist.
- **libxc:** when enabled, native XC must match libxc to 1e-12.
- **SIMD = portable xsimd only** (11.2.0): single source, raw pointers/spans,
  scalar tail. Banned: arch intrinsics, per-arch kernels + dispatch, global
  `XTENSOR_USE_XSIMD` (breaks `reshape_view` in `xc_grid_engine.h`). Idioms:
  `native_integrals.cpp` ERI ket-loop, `multigrid.cpp` `axis_dot`/`axis_axpy`.
  A vectorized reduction lands only with a stated tolerance vs the pins; any
  energy-path change needs fresh force validation; archived cases are reference material. Never put a SIMD reduction
  behind a loop-invariant branch; validate virials at the shipped flags.
- **Doxygen:** every public symbol has `@brief`, `@param` (with units), `@return`.
- **API naming law:** `declare_*` = structure, `require_*` = dependency,
  `on_*` = event; never mix (`on_every_step(field)` is wrong). `on(phase).add`
  is legal — a phase is an event. `declare_*` is a Component-author surface.

## CUDA classical MD is an OpenMM engine

- Neighbor rebuild and nonbonded force are an **OpenMM cluster/tile engine**:
  spatial reorder → clusters (target 32) → AABB → cluster CSR + real
  interaction masks → tile/shared-memory force for LJ and real-space Coulomb.
  LAMMPS names on the surface do not license per-atom Verlet. Forbidden: a
  `NonbondedStyle` dual architecture, per-atom Verlet as default force path,
  production `memset` all-ones masks.
- PME (cuFFT) and Ewald stay in `longrange/` and follow a changing cell (mesh /
  mode set fixed at build, cell tables refreshed on `Box::revision`).
- Generic algorithms use NVIDIA libraries directly — CUB, Cooperative Groups,
  Thrust, cuRAND Philox, cuFFT — no handwritten twin, no forwarding wrapper.
  Physics, spatial predicates, fused topology/force packing stay domain kernels.
  Reserve hot-path scratch in build, keep stream order, key noise by seed,
  global tag, step and component domain. Detail:
  [cuda-md-nvidia-libraries.md](../../.claude/notes/cuda-md-nvidia-libraries.md).
- A separate track: not blocked by ES CPU priority, not frozen by ES CUDA freezes.

## Electronic structure optimization order

Architecture and shared algorithm shape before micro-perf; CPU before CUDA ES.

1. **CPU architecture** — Driver/stage semantics, kill forked SCF copies,
   extract only at a second real call site.
2. **CPU performance** — xsimd hot paths, OT economics, grid constants, MPI;
   fair wall vs the external twin (CP2K for GPW, …).
3. **CUDA ES by method slice** — device-only, tiling, minimal H2D;
   **same-settings CPU is the correctness oracle** (plus external pins); no
   giant dual-backend kernel source; mid-path device↔host exchange is a defect
   ([cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)).
4. **Large-system CUDA perf** — only after the slice is correct; small
   molecules are correctness-only, never the "CUDA wins" story.

**CUDA ES validation** (per slice): current CPU checks and external pins
green; public `run`/`forces` stable; CUDA↔CPU tolerances stated; perf claims
only on large systems. Wave-local freezes live in `.claude/notes/`.

**CUDA ES implementation ownership:** Atomiverse owns the complete scientific
algorithm, data flow, execution scheduling and differentiation. After setup,
scientific computation stays on CUDA, including HF-to-post-HF handoffs. Reuse
mature primitives directly when they fit the end-to-end performance and
resource budget; implement missing or unsuitable capabilities locally. Choose
between them using correctness and measured complete-call performance and
resource use. Neither library calls nor independent implementation justify
unnecessary splitting, transfers, intermediate storage or duplicate primitives.

### Semiempirical method acceptance

- Track each Hamiltonian and backend separately in
  [the semiempirical roadmap](../roadmap/semiempirical.md). Energy/charges,
  analytic forces, stress, spin, periodicity and language bindings have separate
  evidence. A passing primitive or energy-only solve does not complete a method.
- CPU/CUDA NDDO methods consume the unique parameter tables and physical result
  types in `include/atomiverse/semiempirical/` and `src/semiempirical/`.
  Keep numerical kernels and scheduling backend-owned; call suitable LAPACK,
  cuBLAS, cuSOLVER and CUB primitives directly without forwarding adapters.
  New public methods expose physical results, not internal system indexing or
  SCF workspace accessors copied from older method interfaces.
- Accepted electronic history changes only after successful normal evaluation.
  Initial-force preparation, trials and failed solves must not commit guesses.
  A nested solve stages a candidate until the whole physical parent step
  succeeds; failure in a later component discards it. The last accepted
  line-search step commits before returning, without requiring another step.
  Test retry with the same outer Driver and method, not only a fresh parent.
  A failed device evaluation must not retrieve partial physical results.
- At a chunk boundary, compare host input with the last successful write-back
  before any device retrieval. Coordinate/velocity/cell edits preserve the
  integrator's state while invalidating affected forces and guesses; identity
  edits rebuild mappings. Test host edits with the same Driver and Frame.
  `supports_input_changes` preflights the entire mounted plan before mutation;
  `invalidate_inputs` refreshes a component's input-dependent state. Unknown
  identity changes reject by default. A failed refresh remains pending until
  a successful checkpoint, even if the caller restores the previous values.
  Refresh callbacks update local caches; the CPU backend collectively agrees
  on changes and errors over its spatial communicator before resuming MD.
- Python methods mount the same native method instance through the existing
  borrowed-node/lifetime mechanism. Keep controls and electronic state on that
  instance; do not create an evaluator parameter copy. Concrete NDDO typed
  setters perform the C++ base conversion: do not register their nonzero-offset
  `DriverNode` base as a Python inheritance cast. Test mounting, not just
  construction. Expose only validated backend combinations and reject
  unsupported launcher requests explicitly.
- Pin method-specific external energies, charges and full force vectors with
  source/version/input/output and independently justified tolerances. CPU-twin
  comparisons supplement those pins; finite differences supplement analytic
  forces. Preserve unexplained SQM/MOPAC convention differences explicitly.
- Run the SE-only dependency closure as well as affected composed workflows.
  GPU job success, reviewed numerical output, sanitizer checks and measured
  residency are distinct evidence; retain the job reference and raw outputs.
  Do not repeat an uncertain or running submission.
- NVE acceptance compares equal physical durations and sampling times when
  halving the time step. Report absolute total-energy error with explicit units
  and sampling cadence; do not normalize by the large binding energy or treat
  short-trajectory convergence as long-time stability. Frame and borrowed
  methods must outlive the Driver that uses them, including test teardown.
- Performance harnesses live in `benchmarks/semiempirical/`, outside CTest.
  Report complete-call boundaries, cold/warm state, SCF controls/iterations,
  thread count and observed allocation/transfer/memory costs. A matrix-storage
  estimate or source inspection does not establish device residency.
  Warm the actual measured kernels, including moving-geometry components.
  Capture allocation lifetimes over the whole process and mark synchronized
  steady intervals; count device-static initialization as well as allocator
  APIs. Distinguish observed logical allocation peaks from driver reservations,
  whole-device free-memory snapshots and cumulative host allocation traffic.

## Known debt (do not extend)

- **CPU/CUDA execution split.** CPU: `cpu::Runtime` (ledger with alive token,
  hand-written initial-force walk, box-revision invalidation, own
  `cpu::Driver::unfix`, MPI-reduced declared-convergence criteria). CUDA:
  `cuda::Backend` + core `Plan::prepare_forces` (from `post_integrate`, flag on
  `LaunchContext`) + core `detail::ComponentLifecycle` + core `Driver::unmount_`
  + single-rank declared-convergence criteria. The `core/plan.h` helpers have one backend
  user — re-unify CPU onto them or move them to `cuda/driver/`; never call them
  shared.
- **Input refresh coverage:** direct CUDA MD checks host input changes at chunk
  boundaries and refreshes the mounted plan. Components must explicitly support
  the changed fields; PME/Ewald charge and identity changes currently reject.
  QM/MM private-frame forwarding and arbitrary mid-chunk thermostat/RNG rollback
  are not covered by the direct NDDO input-refresh contract.
- **CUDA multi-GPU** has no Runtime/DomainRuntime: `cuda::MultiGpu`
  (partition/halo/`run_nve*`) is a second MD stack outside the Plan.
- **Nested electronic methods:** electronic propagators (CPMD, second-gen CP,
  XL-BOMD) on GPW run only through `GPWEvaluator`, which also duplicates the
  BO write-back of a mounted `GpwDft`; QM/MM drives its engine on a private
  Context. Plan: `.claude/specs/es-unification.md`.
- **Residual `int` step counters:** `Runtime::run(int)`, `Driver::run_nested`
  / `run_nested_erased` budgets, PhaseBuilder budgets,
  `Runtime::comm_refresh_(int slot, int step)` (truncates).
- **ES public shapes** diverge across families:
  [es-architecture-debt.md](../../.claude/notes/es-architecture-debt.md).
- Namespaces are not aligned to directories; aligning them is a deliberate
  follow-up, not a drive-by.
