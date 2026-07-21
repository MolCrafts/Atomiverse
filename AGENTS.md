---
mol_project:
  name: atomiverse
  language: cpp
  build:
    install: "cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON"
    check: "cmake --build build -j$(nproc)"
    test: "ctest --test-dir build --output-on-failure"
    test_single: "ctest --test-dir build -R {path} --output-on-failure"
    coverage: "ctest --test-dir build --output-on-failure"
  arch:
    style: backend-pillars
    rules_section: "## Architecture"
  doc:
    style: doxygen
  science:
    required: true
  ci:
    config: .github/workflows/ci.yml
  notes_path: .claude/notes/notes.md
  specs_path: .claude/specs/
---

# AGENTS.md

Atomiverse is a dual-backend (CPU + CUDA) molecular-simulation engine with one
shared Driver/Component API spanning classical molecular dynamics, Monte Carlo
sampling, geometry optimization, vibrational analysis, and electronic
structure (Gaussian-basis SCF, plane-wave DFT, Gaussian-and-plane-waves DFT,
and NDDO/AM1 semiempirics). C++17, `ATV::` namespace.

> **Branch note:** active development is on **`dev`**. `master` is a single
> squashed snapshot with the *old* pre-2026 layout (`cpu/frame.h`, `cpu/md/lj.h`,
> no `pw/`/`potentials/`); do not read it for current behaviour. New git
> worktrees default-base from `origin/master`, so branch a worktree off `dev`
> explicitly (`git worktree add -b <name> <path> dev`).

## Build & Test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
./build/tests/test_md             # single test binary
```

Requires: C++17, CMake ≥ 3.18, a Rust toolchain (for molrs), xtensor
(auto-fetched). CUDA toolkit only for the CUDA backend; FFTW3 only for
plane-wave / k-space.

### Toolchain via Lmod modules — never hardcode `/software/...` paths

Run everything in a **login shell** (`bash -lc '...'`) so Lmod's `module`
function is defined. The build env does **not** auto-load — load it explicitly:

- **CPU (x86_64 epyc login node):** `module load buildenv-gcc/2026.03-mpich`
  (GCC 14.3 + OpenBLAS/LAPACK/ScaLAPACK + FFTW + MPICH). The older
  `buildenv-gcc/2026.03` is **deprecated** — use the `-mpich` variant.
- **GPU (aarch64 GH200 node):** `module load GPU/buildenv-gcccuda/2026.03-cu13.0`
  (GCC 14.3, CUDA 13.0, GPU-aware MPICH, NCCL, cuFFT, FFTW, OpenBLAS).

A correct CPU configure+build+test from an automated agent, copy-pasteable:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"                       # see BLAS note below
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu   # see Rust note below
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"   # see runtime note below
  git submodule update --init --recursive molrs             # once per clone
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc)
  ctest --test-dir build --output-on-failure
'
```

**BLAS:** the module exports `OPENBLASDIR` and adds OpenBLAS to
`CMAKE_PREFIX_PATH` — it does **not** set `OPENBLAS_ROOT`. Mirror it
(`export OPENBLAS_ROOT="$OPENBLASDIR"`) and pass `-DBLA_VENDOR=OpenBLAS`.

**Rust:** on the x86_64 login node the *default* rustup toolchain is wrongly
`aarch64`, so corrosion's `cargo metadata` aborts ("no such file or
directory"). Force `RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu`. On the
GPU node the default (aarch64) is correct — do **not** set the override there.

**molrs (git submodule):** molrs is a hard dependency built via corrosion,
vendored at `./molrs` (`.gitmodules` → `MolCrafts/molrs`, pin `v0.8.2`).
CMake defaults `ATV_MOLRS_PATH` to `${CMAKE_SOURCE_DIR}/molrs`. After clone
or worktree create, run `git submodule update --init --recursive molrs`.
Override with `-DATV_MOLRS_PATH=...` only for local experiments — the CXX
API version gate still applies. Bump the pin by checking out a new tag
inside `molrs/` and committing the submodule pointer (also update
`ATV_MOLRS_GIT_REF` in `cmake/MolrsContract.cmake`).

**Runtime gotcha (#1 cause of `libopenblas.so.0` / `GLIBCXX_3.4.32 not found`):**
the module sets `LIBRARY_PATH` (link time) but not `LD_LIBRARY_PATH` (run
time). Mirror it before `ctest` or before running any binary:
`export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"`.

### GPU node (CUDA build & test on Arrhenius)

The **login nodes are x86_64 with no CUDA toolkit** — the CUDA backend
(`ATV_BUILD_CUDA=ON`, any `.cu`) builds and runs only on the **aarch64
Grace/GH200 GPU node**. The CPU backend (incl. RHF and plane-wave) builds and
runs fine on the login node with `-DATV_BUILD_CUDA=OFF`.

- **Interactive:** `hpc gpu` (NAISS `interactive` srun wrapper; defaults from
  `~/.hpc/config.sh`: partition `gpu`, account `naiss2026-4-715-gpu`, 1 GPU, 3 h).
- **Batch** (use from an automated agent — `hpc gpu` needs a TTY):
  ```bash
  sbatch -p gpu -A naiss2026-4-715-gpu --gpus 1 -t 01:00:00 -c 8 -o out.log job.sh
  ```
- **Inside the job script** (`#!/bin/bash -l` so Lmod is defined; never pipe
  `module load` — a subshell loses the env):
  ```bash
  #!/bin/bash -l
  module load GPU/buildenv-gcccuda/2026.03-cu13.0
  # molrs comes from the ./molrs submodule (default ATV_MOLRS_PATH)
  cmake -B build-cuda -DATV_BUILD_CUDA=ON -DATV_BUILD_TESTS=ON
  cmake --build build-cuda -j$(nproc)
  ctest --test-dir build-cuda --output-on-failure
  ```

### CMake options

| Option | Default | Enables / sets |
|---|---|---|
| `ATV_BUILD_CUDA` | ON | CUDA backend; defines `ATV_HAS_CUDA` |
| `ATV_BUILD_HF` | OFF | Gaussian SCF (RHF/UHF/UKS); defines `ATV_HAS_HF` |
| `ATV_BUILD_PW` | OFF | Plane-wave DFT (needs FFTW3) |
| `ATV_BUILD_GPW` | OFF | Gaussian-and-plane-waves DFT (needs `ATV_BUILD_HF` + `ATV_BUILD_PW`) |
| `ATV_BUILD_KSPACE` | OFF | Reciprocal-space electrostatics PME (needs FFTW3) |
| `ATV_BUILD_TORCH` | OFF | LibTorch backend; defines `ATV_HAS_TORCH` |
| `ATV_BUILD_TESTS` / `ATV_BUILD_BENCHMARKS` / `ATV_BUILD_EXAMPLES` | OFF | test / benchmark / example targets |
| `ATV_USE_MATPLOT` | OFF | plot generation in selected examples (needs gnuplot) |
| `ATV_REAL_STORAGE` / `ATV_REAL_WORK` | `double` / `double` | LAMMPS-style real tiers (`float`\|`double`): per-atom storage / kernel-work precision. Accumulators (`AccReal`) are always `double`. |
| `ATV_ID_SIZE` | `bigbig` | integer-ID width: `bigbig` (64-bit `TagInt`, default — preserves legacy int64 tags) or `smallbig` (32-bit, LAMMPS/molrs-aligned memory opt-in). `BigInt` global counts always 64-bit. |

Performance: `-march=native` is required for the CPU SCF ERI kernels.

## Repository layout

```
include/atomiverse/ + src/   mirrored header/impl trees (architecture below)
tests/                       unit tests ONLY — each exercises one function or
                             method, and the WHOLE SUITE runs in seconds (it is
                             the fast-iteration loop). E2E IS STRICTLY
                             FORBIDDEN here: no full SCF/MD/vib solve, no MC
                             equilibration to convergence, no timing, no
                             external-engine comparison, no third-party package
                             read at build or test time. There is no opt-in
                             switch to run e2e from ctest: if it is registered,
                             it runs.
regressions/                 <domain>/<case>/ — the ONLY home of end-to-end.
                             One minimal textbook case per delivered feature,
                             written as an OUT-OF-LIBRARY consumer (public
                             headers + link atomiverse, nothing else), checked
                             against a cited LITERATURE value, <= 60 s. Built
                             behind ATV_BUILD_REGRESSIONS; run via the
                             `regressions` target, never by default ctest.
                             Every feature ships one. See regressions/README.md.
benchmarks/                  (removed — rebuild later; not part of ctest)
examples/                    flat numbered tutorial sequence — see examples/README.md
docs/                        narrative docs (user-guide / tutorial / developer / api)
tools/                       data generators + debug harnesses
app/                         installable executables (one subdir each; only
                             consumers of public headers + link atomiverse;
                             app/atomiverse = the MPI launcher, the ONLY
                             target that links libpython)
python/                      scikit-build-core + nanobind wheel (drives root CMake)
cmake/                       CMake helper modules
```

## Architecture

Three pillars: **Driver** (owns the loop), **Component** (one single-step
operation), **Frame** (device-resident state). Three execution backends
(`cpu/`, `cuda/`, `torch/`) sit behind one CUDA-free facade.

### Backend directory layering

Each backend root contains **directories only — no loose files**. Every
directory is a pillar layer, an infrastructure cluster, or a method domain:

```
<backend>/
  frame/       Frame pillar: device-resident state + residency ops
  driver/      Driver pillar: the loop node (CPUDriver), LoopHandler, OutcomePolicy,
               SubDriver (nesting), concrete Evaluate, single_point,
               step executor, backend_traits, step-loop instrumentation
  component/   Component pillar: Component, Launch/Context, generic + shared bases
  io/          external formats / molrs bridge (extxyz, frame_writer, lammps_data; CPU only)
  device/      CUDA/torch runtime utilities (device_manager, error, launch, timer)
  comm/        infra: domain decomposition, halo exchange, migration, reductions
  neighbor/    infra: cell/Verlet neighbor structures + the published PairList
  async/       infra: async analysis pipeline (snapshot ring, D2H bridge, executors)
  linalg/      infra: shared dense linear algebra (eigensolve, density build; CPU only)
  md/          domain: integrators, thermostats, barostats, deform, velocity init
  mc/          domain: Monte Carlo — the MC preset, movers,
               trial transaction (CPU only)
  minimizer/   domain: SD, CG, L-BFGS + opt recorders (LbfgsHistory: the curvature
               history + two-loop recursion, over a plain vector — shared with path/)
  replica/     infra: ReplicaSet + ReplicaExpansion (one Frame -> N coupled replicas)
  path/        domain: coupled-replica methods — CI-NEB (tangent/spring/climb as
               `couple`-slot components); PIMD/REMD would live here too
  scf/         domain: Gaussian-basis SCF (solver, integrals, gradients, evaluator)
  semiempirical/ domain: NDDO/AM1 semiempirical SCF (NDDOSolver, AM1Evaluator,
               AM1-BCC charge assigner; CPU only)
  pw/          domain: plane-wave DFT (PWSolver, PWEvaluator)
  gpw/         domain: Gaussian-and-plane-waves DFT (GPWSolver, GPWEvaluator; CPU only)
  xc/          domain: shared exchange-correlation pieces (Padé LDA parameterization)
  vib/         domain: finite-difference Hessian, thermochemistry (CPU only)
  longrange/   domain: reciprocal-space electrostatics (Ewald, PME)
  potentials/  domain: real-space force-field evaluators (pair + bonded styles)
  observer/    domain: derived quantities (RDF, virial stress, Mulliken, density)
  recorder/    domain: I/O (XYZ, cube, thermo, timing report)
```

**Rules:**
- **No loose files at a backend root.** Shared code goes in the matching pillar
  dir (`frame/`, `driver/`, `component/`, `io/`, `device/`); a new cohesive
  multi-file infra cluster gets its own root dir (like `comm/`, `neighbor/`,
  `async/`). Never park shared code inside a domain subdir.
- **Orthogonality: the same feature lives in the same directory on every
  backend.** Virial stress is `observer/`, neighbor structures `neighbor/`,
  decomposition `comm/` — on CPU and CUDA alike.
- Electrostatics splits LAMMPS-style: real-space pair styles in `potentials/`
  (`coul_long`), reciprocal-space solvers in `longrange/`; shared operator
  math is written once in top-level `longrange/` (never duplicated in SCF).
- Namespaces are not yet aligned to directories (`cpu::md::mpi::Reduce` lives in
  `cpu/comm/`, `cpu::Frame` in `cpu/frame/`); aligning them is a deliberate
  follow-up, not a drive-by.

**Box** (`cpu::Box` in `cpu/frame/box.h`, `cuda::Box` in `cuda/frame/box.cuh`)
is the canonical cell type in each backend's `frame/` layer: owns the cell
matrix plus a per-step-immutable cache (`inv_cell`, `inv_L`, `half_L`, `ortho`)
and exposes `length(d)`, `volume()`, `min_image(...)`, `wrap_axis(...)`,
`scale_*`, `set_cell`. CPU adds an xsimd `min_image_batch<B>` in
`cpu/frame/box_simd.h`. LJ keeps an inlined branchy min-image in its xsimd
ket-loop because member-load-through-`this` defeats hoisting there (~7%/step).

### Facade (CUDA-free)

```
core/        precision tiers, Mat3, units, Frame schema, device tags, backend_traits
facade/      Driver<PhaseEnum, DeviceTag>, Plan, Stage (phase enums)
potentials/  backend-agnostic pair force/energy kernels shared by CPU/CUDA
longrange/   backend-agnostic Ewald/PME operator math (no Frame/Component/MPI/FFT)
molrs.hpp    stable CXX FFI header
```

### Runtime / Driver / Handler — there is no Driver "category"

**A Driver is a staged loop. Termination is a Handler.** There are no category base
classes: `EvaluateDriver` / `DynamicsDriver` / `OptimizeDriver` / `ResponseDriver` /
`SamplingDriver` are **deleted** (CPU and CUDA alike). They were six classes that
differed from one another by one `if`, every new method demanded a seventh, and no user
could write an eighth — "when do we stop" was a private `if` inside a driver class.

Stripped to the bone a Driver is three things: **stages** (a typed phase enum),
**iteration** (walk them repeatedly), **termination** (when to stop). Only the third was
welded into the hierarchy. Hand it to a policy object and no category remains:

| | what it is | how many |
|---|---|---|
| **Runtime** (`cpu::Backend`) | frame binding, node storage, timing, MPI, domain decomposition | **one per run**, shared by the whole driver tree |
| **Driver** (`cpu::CPUDriver<P>` / `cuda::CUDADriver<P>`) | **one loop node**: stages + plan + handlers | any number, **nestable** |
| **Handler** (`cpu::LoopHandler`) | asked once per iteration | any number, composable, **user-definable** |

```
Driver<PhaseEnum, DeviceTag>      plan / dispatch primitive
  └─ cpu::CPUDriver<PhaseEnum>    THE loop node
       └─ {MD, Minimize, MC, Evaluate, Hessian, PathDriver}    presets
```

Two levels, not four. A **preset** is *not* an architectural layer: it is a class whose
**constructor** attaches the default handler + infrastructure a given kind of run wants.
It contributes defaults, not semantics, and the user overrides any of them. (A class with
a constructor — never a `make_md()` factory; see the OOP rule below.)

| Preset | Phase enum | Default handler | Termination |
|---|---|---|---|
| `cpu::Evaluate` / `cuda::Evaluate` | `evaluate::Phase` | — | one pass |
| `cpu::md::MD` / `cuda::md::MD` | `md::Phase` | — | the step budget |
| `cpu::minimizer::Minimize` / `cuda::…::Minimize` | `minimizer::Phase` | `ForceConverged` | convergence, or the budget |
| `cpu::vib::Hessian` | `vib::Phase` | — | fixed 6N displacements + finalize |
| `cpu::mc::MC` | `sampling::Phase` | — (`Metropolis` **outcome**) | the trial budget |
| `cpu::PathDriver` | `path::Phase` | `AllReplicasConverged` | every replica converged |
| **SCF iteration (target)** — hosted by concrete `RHF`/`UHF`/`RKS`/`UKS` | SCF stages | residual / energy Δ handlers | SCF convergence, or budget |

SCF is **not** a seventh "electronic structure category" bolted beside MD: it is
the same Driver node shape. The **method** (RHF vs UHF vs UKS) is which
components fill the stages — not an enum on a bag named `SCFSolver`.

**Two orthogonal policy slots.** Every loop has both:

- **`LoopHandler`** — *should the loop stop?* `add_handler(…)`, **plural**, composed by
  severity: **Stop > Converged > Continue**. Deliberately **not** a template: it sees a
  `LoopState` and *never a phase*, which is exactly what lets ONE handler serve MD,
  Minimize, MC and Path alike. Ships `ForceConverged`, `AllReplicasConverged`,
  `StopOnNaN`; a user writes `StopWhenTooHot{2000}` (LAMMPS `fix halt`) from **outside**
  the library. Its verdict is all-reduced across MPI ranks **by the driver**, once —
  handler authors never think about it.
- **`OutcomePolicy`** — *does this iteration count?* `set_outcome(…)`, **singular**,
  `Commit`/`Rollback`. MC's accept/reject was never a termination: rejecting a trial move
  discards one iteration's work and keeps going. `Metropolis` is one implementation. The
  policy **decides**; whoever recorded the changeset **undoes**.

**Algorithms never judge themselves.** `LBFGS`/`CG` **publish metrics** (`max_force`,
`rms_force`, displacement, ΔE) into `LoopMetrics`; a **handler judges** them.
`LBFGS::converged()` is deleted — `Driver::converged()` replaces it, because the *loop*
knows and the algorithm does not. Changing the convergence criteria does not touch the
optimizer. A component may **request** a stop; it **cannot declare convergence** — there
is no field left for it to write.

**Status belongs to the loop, not the Context.** Each loop node **owns** its
`IterationStatus` + `LoopMetrics`; `Context::loop` (a `LoopServices` slice) only *borrows*
the active loop's. `run_nested` re-points them at the child and restores the parent's on
exit, so **an inner loop cannot stop its enclosing loop** — a pointer that points at the
current loop cannot reach the loop above it. This bug shipped twice (LBFGS's unconditional
`converged` write; the first converged NEB image stopping the whole band) and is now
unwritable rather than merely forbidden. Pinned by `tests/cpu/driver/test_sub_driver.cpp`.

**A Driver is an executable node.** Same lifecycle as a component (build → launch × N →
destroy), so a Plan slot holds a component **or** a sub-driver (`cpu::SubDriver`). The
child runs its own Plan on the **parent's Runtime** (`ctx.exec.runtime`): no second
Backend, no rebuilt decomposition, no re-bound frame, no split communicator. **Type
erasure happens at the NODE boundary, never at the stage boundary** — the child keeps its
own phase enum, and stages stay typed enums all the way down (see *Stage typing*, below).

**Replica count was never an axis of a taxonomy — it is a nested loop.** So REMD/PIMD need
**no new category**: they are "a replica loop + a budget handler + a different `couple`
component". The replica machinery is infrastructure (`cpu/replica/`: `ReplicaSet`,
`ReplicaExpansion`), never inside a driver.

**Replicas are generated by an expansion strategy, never supplied by the user.**
`ReplicaSet` is built from `(Frame, ReplicaExpansion&)`, and `run()` takes ONE Frame and
writes ONE Frame back (CI-NEB: the climbing image = the saddle), so Frame stays the sole
I/O currency. Per-replica parameters (temperature ladder, bias centre, bead index, path
coordinate) belong to the *expansion* — putting them on the driver is what welded a
category to one method.

**Known ceiling: components cannot be cloned.** Running one inner Driver body over N
replicas would need N independent component states (N L-BFGS curvature histories), and
there is no `clone()` on `Component`. Nesting makes this **sharper**, not softer — a
`SubDriver` holds ONE inner driver with ONE component set. Undecided; see `.claude/notes/notes.md`.

**Adding a new preset is cheap and adding a new "category" is meaningless.** If you catch
yourself wanting a new driver class, ask what *terminates* differently — and write a
`LoopHandler` instead.

### Stage typing — `slot_of` is the only cast

The six Phase enums all start at 0, so an `int` phase **cannot** be told apart across
workflows (`stage == 1` is simultaneously `md::pre_force`, `minimizer::line_search` and
`evaluate::evaluate`). `Backend::run_phase(int)` is gone. Everything a human writes goes
through a typed entry point:

| Use | Typed entry point |
|---|---|
| the ONE enum→slot conversion | `ATV::slot_of(phase)` (`constexpr`, zero cost) |
| the `COUNT` sentinel as a slot count | `ATV::slot_count<PhaseEnum>()` |
| replay one phase | `backend.run_stage(phase, step)` |
| "is this my stage?" (component) | `ctx.at(phase)` — cpu **and** cuda `LaunchContext` |
| declare a default binding | `ATV::make_binding(phase, order)` |

`test_stage_typing_scrub` (pure grep, every configuration) fails the build on any
hand-written `static_cast<int>` over a Phase enum anywhere in the tree.

`DefaultBinding` still carries `{PhaseKind, int phase, int order}` — that pair **is** the
erasure, not a leak: `Component` is not a template, so `default_bindings()` must
return one concrete type for every workflow. The `PhaseKind` beside the int is what makes
it safe, `make_binding` is the only way to build the pair, the cross-workflow misbind gate
throws at registration, and no caller ever types the int.

**A loop is a Driver. A method is concrete. There is no generic "SCFSolver".**

- **Driver = staged loop + termination.** MD, Minimize, MC, Path, and the
  **self-consistent-field iteration** are all the same shape: stages, iterate,
  stop by Handler. SCF *does not solve anything* — it only walks
  Fock → diagonalize → density → mix → residual until a handler says
  Converged/Stop. Calling that bag `SCFSolver` was a category error.
- **The thing that is real is the concrete method:** `RHF`, `UHF`, `RKS`,
  `UKS`, `AM1`, plane-wave DFT, GPW, … Each owns its Hamiltonian / density
  build / parameters **as members**. There is no
  `SCFSolver{config.method = HF}` bag that morphs via enum.
- **Wrong:**
  ```cpp
  SCFSolver solver; solver.method = HF; solver.spin = Restricted;
  solver.run(frame);
  // or: SCFConfig cfg{RHF{}, ints}; SCFSolver{cfg}.run(ints);
  ```
- **Right (target shape):**
  ```cpp
  cpu::scf::RHF rhf;          // concrete method — parameters are members
  rhf.tol = 1e-10;
  rhf.basis = BasisSet::STO3G;
  // SCF iteration is the Driver loop; RHF supplies the energy/force stages
  rhf.run(frame);             // Frame in / Frame out — no IntegralPack, no Config
  ```
- When an outer MD/Minimize/Evaluate needs forces, it binds the **concrete
  method** (or a thin force-slot component that owns that method) at its
  force/energy phase — not a polymorphic "SCF" that pretends to be every
  mean-field theory at once.

| Concrete method (target) | What it is | Loop |
|---|---|---|
| `RHF` / `UHF` / `RKS` / `UKS` | Gaussian-basis mean-field **method** | SCF iteration = **Driver** |
| plane-wave DFT / GPW / AM1 | same: concrete methods | their SCF/OT loops = **Driver** shape |

*Properties* (charges, dipoles, Mulliken, AM1-BCC, …) are component **outputs
written to Frame** — never Driver-level categories: no "charge driver".

**Frame is the only system.** There is no parallel `*System`, public
`AtomList`, user-facing `IntegralPack`, or `*Config` bag:

| Banned as public API | Why |
|---|---|
| `AtomList` | Frame already is the atoms (Å, Z). Bohr conversion is private scratch. |
| `IntegralPack` | Integrals are method/Driver **private workspace**, rebuilt from Frame. |
| `SCFConfig` / `*Config` bags | Parameters are **members of the concrete method/Driver**. |
| `NDDOSystem` / any `*System` | Frame is the sole I/O currency. AO index is private scratch. |
| `SCFSolver` as a bag | Not a thing — methods are `RHF`/`UHF`/…; the loop is a Driver. |
| free-function factories | `build_*` / `*_config` / `make_*` — constructors or members only. |

Single-point energy in user code is `Evaluate + <concrete method>` (or
`single_point(frame, method)`). There is **no `SinglePoint` class**. Internal
step helpers stay owned by the concrete method/Driver, never inherit
`Component`, never register into a foreign Plan; facade-public headers must
not include them (tests may).

> **Debt (current tree):** still ships `cpu::scf::SCFSolver` + `SCFConfig` +
> public `IntegralPack` / `AtomList` / `NDDOSystem`. That surface is
> **transitional and scheduled for deletion** toward the table above. New
> code and new APIs must not extend the debt. See `.claude/notes/notes.md`
> `[2026-07-17] SCF is a Driver; methods are concrete`.

```
Preset::run(frame, budget)              # the preset composes the loop primitives
  setup_pass(frame)                     # binds ctx.loop at THIS loop's status+metrics
  for each iteration:
    status.reset()
    backend.step(idx)                   # components measure into LoopMetrics
    outcome = outcome_policy?           # Commit | Rollback  (MC only)
    verdict = poll_handlers(idx)        # Stop > Converged > Continue, MPI-allreduced
    if verdict != Continue: break
  teardown_pass()

Driver::run_nested(ctx, frame, budget)  # the SAME loop, on a BORROWED Runtime
  save ctx.loop; point it at MY status+metrics
  ... loop, via runtime->step_plan(my_plan, idx) ...
  restore ctx.loop                      # the child's verdict stays on the child
```

User code does not subclass `Driver<>` directly — it uses a **preset**, or subclasses the
loop node (`CPUDriver<PhaseEnum>`) when it needs a genuinely new loop shape. A preset adds
default handlers + infrastructure in its **constructor**; no runtime behaviour beyond the
loop node.

### Component tagging

**Role is a bitset of `RoleFlag`s**, not a mutually exclusive enum. A Component
returns `Role role()` = combination of:

| RoleFlag | Meaning |
|---|---|
| `FrameImmutable` | Does not modify x/v/f/box/topology |
| `Deferred` | Optional; may run off the hot path / async. **Unset** ⇒ main-loop required |
| `ExternalIO` | Writes files / external I/O |

There is **no `Capability` write-set**. Outputs are Frame keys
(`kEnergy`, `kAtom_fx`, …). Validation uses **kinds** (types) and Role flags:

- `Evaluate` ≥1 energy source (`Potential` / `Bias` / `HF` / `Semiempirical` / `PlaneWave` / `GPW` / …)
- `MD` ≥1 `Integrator` + energy/force kind
- `Minimize` ≥1 `Optimizer` + energy/force kind
- `MC` ≥1 `Mover` + energy kind

**Kinds** (one class per header under `cpu/component/`; depth ≤ 2:
`Concrete → Kind → Component`). `component.h` holds only the base.
Pipeline glue (NeighborList, Halo, Zero*, Reduce) inherits `Component`
directly — no `Infrastructure` kind.

| Kind | Default Role | Examples |
|---|---|---|
| `Potential` | required | PairLJCut, BondHarmonic, CoulLong, KSpace |
| `HF` | required | `cpu::scf::SCF` |
| `Semiempirical` | required | `AM1Evaluator` |
| `PlaneWave` | required | `PWEvaluator` |
| `GPW` | required | `GPWEvaluator` |
| `Bias` | required | HarmonicBias |
| `PathCoupling` | required | NebCoupling |
| `Observer` | FrameImmutable\|Deferred | RDF, MSD, Virial, KineticEnergy |
| `Recorder` | +ExternalIO | Thermo, XYZ |
| `Integrator` / `Thermostat` / `Barostat` / `Optimizer` / `Constraint` / `Mover` | required | VelocityVerlet, LBFGS, … |

**Opt-in protocols** — `Component` stays minimal; cross-cutting
infrastructure needs are declared by implementing a protocol the engine
probes for (no `dynamic_cast`, no dead default virtuals):

| Protocol | Declares | Engine probe |
|---|---|---|
| `HaloParticipant` (`component/halo_participant.h`) | declarative halo `CommField`s + endpoints; engine-side packing in `comm::HaloSend/HaloRecv` | `halo_participant()` |
| `NeighborConsumer` (`component/neighbor_protocol.h`) | `neighbor_cutoff()` the neighbor stage must cover | `as_neighbor_consumer()` |
| `NeighborProvider` (same header) | this component builds/publishes the pair list | `as_neighbor_provider()` |
| `LongRangeParticipant` (`component/long_range_protocol.h`) | real/reciprocal role, shared Ewald `alpha`, and cold metadata; exactly one matched pair is required | `as_long_range_participant()` |
| `PairEvaluator` (`component/pair_evaluator.h`) | shared non-bonded pair-style base (cutoff, special-bonds) | — |
| `AsyncComponent` (`async/async_component_base.h`; CUDA analogue `AsyncBridge`) | deferred snapshot-ring analysis off the hot path | `scheduling() == deferrable` |

**Iteration status channel** (`facade/iteration_status.h`):
`IterationStatus` is `{converged, stop_requested, reason}`, **owned by the loop node**
(`CPUDriver::status_`) and reset before each iteration. The Context only *borrows* the
active loop's, through the non-owning `ctx.loop` (`LoopServices{status, metrics}`) — so a
nested loop physically cannot reach its parent's. A component **measures** into
`ctx.loop.metrics` and may **request** a stop; only a `LoopHandler` declares convergence,
and only the driver writes it. SCF/PW internals never touch any of it.

**Rules:**
- **Driver owns the loop.** Evaluators, integrators, thermostats are
  Components — never standalone classes with their own loops.
- **Termination is a Handler, never a class.** Want a new stopping rule? Write a
  `LoopHandler`. A new driver class is almost never the answer.
- **Energy/force sources are kinds** (`Potential`, `HF`, `Semiempirical`,
  `PlaneWave`, `GPW`, `Bias`, `PathCoupling`) with required (`!Deferred`) Role;
  presets type-check via `dynamic_cast` to those kinds (no free-function classifier).
- **No duplication across methods.** Shared skeletons (gradient assembly, FD
  validation) are single parameterized functions.
- **Component lifecycle:** `build()` (one-time alloc) → `launch()` × N (per-step
  compute) → `destroy()`. No allocation in `launch()`.
- **Ownership is explicit at every registration path:** `add(rvalue)` and
  `on(phase).add(rvalue, order)` transfer ownership; `add_ref(lvalue)` and
  `on(phase).add_ref(lvalue, order)` borrow. Plain `add(lvalue)` is deleted.
  Categorical setters follow the same rule: `set_xxx(rvalue)` owns and
  `set_xxx_ref(lvalue)` borrows.
- **Observer vs Recorder:** Observer computes derived quantities; Recorder
  writes I/O. Never both.
- **Hot-path data structures carry only fundamental physical quantities** —
  positions, velocities, integrals, shell descriptors; never precomputed
  indices, caches, or derived maps. Consumers derive their own views.

### Context tiers

`Context` (cpu `component/context.h`, cuda `component/context.cuh`) is a flat
**core** plus per-workflow **service slices**. Core fields serve every
workflow: threading/MPI topology (`n_threads`, `mpi_rank`, `mpi_size`), the
timing registry, and `outer_status` (owned by the category driver). Workflow
state lives in its slice — a component reaches only the slice of the workflow
it serves:

| Slice | Struct | Serves | Key fields (CPU) |
|---|---|---|---|
| `ctx.md` | `DynamicsServices` | Dynamics force pipeline | `dd` (domain decomposition), `pair_list`, halo/migration hooks |
| `ctx.mc` | `SamplingServices` | Sampling | `proposal` (`mc::Transaction` — Mover writes, driver rolls back on reject) |
| `ctx.opt` | `OptimizeServices` | Optimize | `reeval_forces` line-search callback, `line_search_trial` flag |
| `ctx.analysis` | `AnalysisServices` | async observers/recorders | `executor` (snapshot-ring analysis) |

Slices are plain aggregates and the Context stays non-polymorphic
(static-asserted). Per-backend: `cuda::Context` carries streams + the `md`
slice only (`cuda::DynamicsServices`: `pair_list`, `flat_pairs_requested`,
exclusion CSR, …); the cuda SCF solver deliberately does **not** take the
driver Context — it receives a narrow `SolverContext{stream, frame, step}`
(`cuda/scf/internal/solver_context.cuh`, isolation grep-gated by
`tests/cuda/scf_solver_isolation.sh`).

### Frame — multi-backend design

A Frame is the device-resident system state. Its boundary vocabulary is exactly
three regions: named typed **blocks**, exact-dtype **meta**, and **simbox**.
The CPU hard cut is complete; CUDA and Torch representation work is deferred
and their current layouts are not compatibility surfaces for CPU code.

| Frame | Representation | Topology layout |
|---|---|---|
| `molrs.Frame` | flat typed Block columns + typed `MetaValue` map + `simbox` — exchange schema v2 | flat bonds/angles/dihedrals blocks |
| `cpu::Frame` | exactly `BlockMap blocks; MetaMap meta; Box simbox;`, with native typed host columns and zero-copy spans in Driver | per-atom LAMMPS columns in the `atoms` block (`num_bond`/`bond_atom` with global-tag partners) |
| `cuda::Frame` | raw device pointers + host-cached counts | **dual**: single-GPU flat `BondList`/`AngleList`/`DihedralList` (kernels not yet ported to per-atom, so `cpu::to_cuda` re-flattens on every transfer); multi-GPU per-atom SoA with global-tag partners for migration |
| `torch::Frame` | `at::Tensor` blocks (autograd-capable) | flat bonds/angles/dihedrals sub-structs |

Conversion points, each owning exactly one boundary:

- `materialize()` / `write_back()` (`cpu/frame/materialize.h`) — strict schema-v2
  molrs.Frame ⇄ cpu::Frame boundary, run once per `run()`, never in a step loop.
  Unknown/synthetic blocks, unknown columns, wrong dtypes, and wrong row counts
  fail immediately; there is no old decoder. `write_back` follows a pinned contract
  (golden-tested in
  `tests/cpu/test_materialize.cpp`): x/v/f, charge, image, and tag are
  written back (tag with checked i32 narrowing — overflow throws, no silent
  truncation); type, mass, topology, and simbox are per-run static inputs and
  **not** written; every serializable typed-meta value is written back exactly.
- `cpu::to_cuda()` / `cuda::to_cpu()` — host ⇄ device; `cuda::write_back`
  inherits the cpu contract via `to_cpu`.
- `torch::transfer` — cpu ⇄ torch tensors.

Precision tiers map 1:1 across backends: `core/precision.h`
`StorageReal`/`WorkReal`/`AccReal` (accumulators always double) ⇄
`cuda/frame/types.cuh` `real_t`/`work_t`/`prec_t`; integer IDs by LAMMPS
lmptype role (`LocalInt`/`TagInt`/`ImageInt`/`BigInt`).

Per the Frame-purity rule the frame holds **no** method-specific scratch: the
Method/Driver **private** workspace (integrals, AO maps, grids, DIIS history)
is owned by the concrete method object — never parked on Frame and never a
public type. Neighbor pair list stays on `*::NeighborList` (published as a
non-owning `ctx.md.pair_list`). No implicit transfer in low-level operators;
CUDA pipelines stay on device between explicit transfers.

### CUDA execution model

Components classify as **compute** (device-only launch), **CPU bridge** (D2H +
host callback via AsyncBridge), or **host-dependent compute** (documented
exception). See `.claude/notes/notes.md` for the classification tables.

### molrs integration

molrs (Rust) is a hard dependency for I/O and trajectory analysis. The engine
*measures*; molrs *analyzes/stores*: bridge calls are raw-array FFI wrappers —
per-frame writers in `cpu/io/frame_writer.h`, batch computes + streaming
accumulators (`cpu::compute::*`) in `cpu/observer/trajectory_metrics.h`
(PIMPL) — with no `rust/cxx.h` in public headers. `molrs.hpp` is the only
include for CXX-generated code. **No analysis math lives in C++ on either
backend**: both the CPU and CUDA RDF observers stream positions + per-frame
cell into the molrs accumulator (binning + normalization are molrs-only,
grep-gated by `test_cuda_rdf_molrs_migration`); the one g(r) file emitter is
`cpu::rdf_write_curve` (`cpu/observer/rdf_curve.h`, CUDA-safe).

## Conventions

- **Namespace:** `ATV`. Sub: `ATV::cpu`, `ATV::cuda`, `ATV::md`,
  `ATV::minimizer`, `ATV::sampling`, `ATV::path`, `ATV::scf`, `ATV::pw`,
  `ATV::evaluate`.
  Concrete methods (target): `RHF` / `UHF` / `RKS` / `UKS`, PW/GPW/AM1.
  Debt names still in tree: `SCFSolver`, `PWSolver`, `GPWSolver`, `NDDOSolver`.
  Internals: `ATV::cpu::scf::internal` etc. — not public.
- **File extensions:** CPU `.h`/`.cpp`, CUDA `.cuh`/`.cu`, facade/core `.h`.
  Documented exceptions: deliberately CUDA-free bridge headers under `cuda/`
  keep `.h` (`cuda/log_callback.h`, `cuda/timer.h`).
- **Driver template:** use a **preset** (`cpu::md::MD`, `cpu::minimizer::Minimize`,
  `cpu::mc::MC`, `cpu::Evaluate`, `cpu::vib::Hessian`, `cpu::PathDriver`), or subclass the
  loop node `cpu::CPUDriver<PhaseEnum>` / `cuda::CUDADriver<PhaseEnum>` for a genuinely
  new loop shape. Never `Driver<>` directly. There are no category bases.
  Driver subclass headers include the matching `backend_traits.h`.
- **OOP, not free-function soup.** Atomiverse is object-oriented: behaviour lives
  on classes, state is constructed, invariants are established in constructors.
  - **No factory functions — ever.** Never `make_X()` / `builtin_X()` /
    `resolve_X()` / `parse_X()` / `build_X()` / `*_config()` free functions that
    return a constructed object. Grep-gated by
    `tests/architecture/no_factory_scrub.sh`.
  - **No public intermediate types for "building" a calculation.** Do **not**
    invent `AtomList` / `IntegralPack` / `*Config` / `*System` as user-facing
    steps. Frame is the molecule. Parameters are **members of the concrete
    method or Driver**. Integrals, AO maps, grids are **private workspace**
    inside that object — rebuilt from Frame when geometry moves.
    **Wrong (factory era):**
    ```cpp
    auto ints = build_integrals(frame, BasisSet::STO3G);
    auto cfg  = uhf_config(ints, n_alpha, n_beta, tol);
    SCFSolver{cfg}.run(ints);
    ```
    **Wrong (constructor lipstick on the same layering):**
    ```cpp
    IntegralPack ints{frame, BasisSet::STO3G};
    SCFConfig cfg{SCFConfig::UHF{}, n_alpha, n_beta, tol};
    SCFSolver{cfg}.run(ints);
    ```
    **Right:**
    ```cpp
    cpu::scf::UHF uhf;                 // concrete method
    uhf.n_alpha = n_alpha;
    uhf.n_beta  = n_beta;
    uhf.tol     = tol;
    uhf.basis   = BasisSet::STO3G;
    uhf.run(frame);                    // Frame in — private workspace inside
    ```
    A **lookup** (basis/PP table row) is still a constructor:
    `GthPotential(molrs::Element, name)`, `EAMPotential(molrs::Element, name)`.
    That is not a factory for a parallel world-model type.
  - **Do not extract single-use code into a function.** If a block runs exactly
    once, from exactly one call site, it stays inline where it runs. A function
    exists to be called from more than one place, or to be tested on its own —
    not to give a paragraph a name. Premature extraction is what turns a solver
    into a maze of one-shot helpers that must each be read before the algorithm
    is visible. Extract on the *second* caller, not the first.
- **No fallback values** — lookup tables throw on unsupported input.
- **Three test tiers, and they never mix** (layout above):
  - `tests/` — unit only. One function or method per test. The whole suite runs
    in **seconds**; it is the fast-iteration loop. **E2E is strictly forbidden.**
  - `regressions/` — **every delivered feature ships exactly one** minimal
    textbook end-to-end case here, as an out-of-library consumer of the public
    API, checked against a **cited literature value** (not a golden snapshot of
    our own output), `<= 60 s`. Never in `tests/`.
  - ~~`benchmarks/`~~ — removed for now; will be rebuilt later (never part of ctest).
- **Phase naming:** Evaluate `prepare, evaluate, finalize`; MD LAMMPS-style
  (`initial_integrate, pre_force, force, post_force, final_integrate,
  end_of_step`); Minimizer `search_direction, line_search, force,
  convergence_check, end_of_step`; Vib `setup, displace, force, accumulate,
  diagonalize, thermochem, end_of_step`; Sampling `propose, energy,
  end_of_step`. SCF (the **Driver** loop) gets its own typed phase enum when
  the concrete method is re-hosted as a preset — stages describe the
  iteration (Fock / diagonalize / density / mix / …), not a fake top-level
  "electronic structure category". The deleted free-floating `ATV::scf::Phase`
  without a Driver node stays banned (`test_scf_phase_scrub`).
- **Units:** eV, Å, fs, amu, eV/Å, K. Conversions in `core/units.h`. SCF/PW
  report internally in Hartree/Ry and write eV to
  `frame.meta.get(kEnergy)`.
- **Precision:** LAMMPS-style tiered, in `core/precision.h`. Reals:
  `StorageReal` (per-atom state) / `WorkReal` (kernel math) — lowerable to
  `float` via `-DATV_REAL_STORAGE` / `-DATV_REAL_WORK`; `AccReal` (accumulators,
  energy, virial, SCF) is **always** `double`. Integers by role (LAMMPS
  `lmptype.h`): `LocalInt` (32-bit counts), `TagInt`/`ImageInt` (IDs/image flags,
  32-bit under `-DATV_ID_SIZE=smallbig`, 64-bit under `bigbig`), `BigInt`
  (global totals, always 64-bit). The catch-all `Real` and `Int` names are
  deleted; every use must state its precision or integer role.
  CUDA mirror: `real_t`/`work_t`/`prec_t` in `cuda/frame/types.cuh`.
- **libxc:** when enabled, native XC must match libxc to 1e-12.
- **ERI performance:** `-march=native` required. xsimd 11.2.0 linked directly to
  `atomiverse` for `native_integrals.cpp`. Do **not** set `XTENSOR_USE_XSIMD`
  globally (it activates SIMD-backed xtensor storage which breaks
  `reshape_view` in `xc_grid_engine.h`). `eri_block()` dispatches by Ltot to a
  compile-time-Ltot template; ss|ss uses an xsimd ket-loop.
- **Doxygen:** every public symbol needs `@brief`, `@param` (with units),
  `@return`.

## Key user-facing APIs (current, verified)

- **Frame:** `cpu::Frame f(n);` then resolve typed spans such as
  `auto x = f.span(kAtom_x)` / `auto type = f.span(kAtom_type)` (type = atomic
  number Z for SCF); set `f.simbox = cpu::Box::cubic(L)`; scalar results are
  exact typed entries such as `f.meta.get(kEnergy)` and
  `f.meta.get(kKinetic)`. The public physical representation is exactly
  `blocks + meta + simbox`; there are no `atoms`, `scalars`, or `box` aliases.
  Header `atomiverse/cpu/frame/frame.h`.
- **Electronic structure (target API):** construct a **concrete method**
  (`RHF` / `UHF` / `RKS` / `UKS` / AM1 / PW / GPW), set parameters as
  **members**, `run(frame)`. Integrals and AO workspaces are private. Do not
  document `IntegralPack` / `SCFConfig` / `SCFSolver` as the user path (debt
  only). `BasisSet`: `STO3G, Pople631G, Pople631Gs, Def2SVP, Def2TZVP`.
  `XCType`: `LSDA, PBE, B3LYP, PBE0`.
- **Outer drivers bind concrete methods (no categorical energy setter):**
  `eval.on(evaluate::Phase::evaluate).add(cpu::pw::PWEvaluator{…})`;
  `aimd.on(md::Phase::force).add(/* concrete RHF/… force component */)`.
  Or `cpu::single_point(frame, method)`.
- **Plane wave:** `cpu::pw::PWConfig{upf_path, ecut_ry, k_mesh, tol, max_iter,
  mix_beta, pulay_depth}` → `PWSolver{cfg}.build(frame); run(frame)` →
  `PWResult{energy [Ry], energy_ha, converged, iterations}`. Empty `upf_path` = toy `V_loc`
  (runs, unphysical). A real Si UPF for PW unit tests ships at
  `tests/data/pw/Si_lda_nc.upf`.
- **Categorical setters** (LAMMPS-shaped, on `cpu::md::MD` /
  `cpu::minimizer::Minimize` / `cpu::mc::MC`): `add_potential, set_bond, set_angle,
  set_dihedral, set_neighbor, set_observer, set_record, set_thermo` (shared);
  `set_integrator, set_thermostat, set_barostat` (MD); `set_optimizer`
  (Minimize); `set_mover` (MC). NeighborList is user-set; halo exchange auto-injects under
  `ATV_BUILD_MPI` when `add_potential` is called; `VirialStress` auto-injects when an
  NPT/Berendsen barostat requires_system_stress(). `add()` /
  `on(phase).add()` remain public for hooks the categorical API lacks.

```cpp
// Single-point RHF (CPU) — target shape: concrete method, Frame only
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-10;
rhf.max_iter = 100;
rhf.run(frame);                                       // energy on frame.meta

// Classical MD (Dynamics) — categorical setters
cpu::md::MD sim;
sim.add_potential(cpu::PairLJCut{0.0104, 3.4, 6.8});
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{0.01});
sim.set_thermo(100);
sim.build();  sim.run(frame, 1000);

// AIMD — outer MD Driver; force slot binds a concrete method, not SCFSolver
cpu::md::MD aimd;
aimd.set_integrator(cpu::md::VelocityVerlet{0.5});
aimd.on(md::Phase::force).add(cpu::scf::RHF{/* members */});
aimd.build();  aimd.run(frame, 100);

// Geometry optimization
cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});   // memory, alpha0, max_ls
opt.add_handler(cpu::ForceConverged{0.05, 0.03, 0.1, 0.05, 1e-3});  // the tolerances
opt.on(minimizer::Phase::force).add(cpu::scf::SCF{cfg});
opt.build();  opt.run(frame, 100);

// Monte Carlo sampling (NVT Metropolis)
cpu::mc::MC mc{/*T [K]=*/300.0, /*seed=*/12345u};
mc.add_potential(cpu::PairLJCut{eps, sigma, cutoff});
mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, true});
mc.set_mover(cpu::mc::TranslateMove{/*delta [Å]=*/0.2});
mc.set_thermo(100);
mc.build();  mc.run(frame, 100000);   // 100k trial moves
double e = mc.current_energy();       // [eV]; mc.acceptance_ratio() in [0,1]

// Plane-wave DFT single point
cpu::pw::PWEvaluator pwe{cpu::pw::PWConfig{.ecut_ry = 30.0, .k_mesh = 4}};
cpu::single_point(frame, pwe);   // frame.meta.get(kEnergy) [eV]
                                 // pwe.last_result() [Ry/Ha]

// Termination is a policy — compose handlers, or write your own (LAMMPS `fix halt`)
class StopWhenEnergyAbove : public cpu::LoopHandler {
  public:
    explicit StopWhenEnergyAbove(double e_max) : e_max_(e_max) {}
    const char* name() const override { return "stop_when_energy_above"; }
    ATV::Verdict after_iteration(const cpu::LoopState& s) override {
        return (s.frame.meta.get(ATV::kEnergy) > e_max_)
                   ? ATV::Verdict::Stop
                   : ATV::Verdict::Continue;
    }
  private:
    double e_max_;
};
sim.add_handler(cpu::StopOnNaN{});            // plural — Stop > Converged > Continue
sim.add_handler(StopWhenEnergyAbove{10.0});   // written OUTSIDE the library
sim.run(frame, 1000);
if (sim.converged()) { /* ... */ }            // the LOOP knows; the algorithm does not
sim.stop_reason();                            // which handler ended it

// A Driver is a node: nest one in another, on ONE shared Runtime
cpu::minimizer::Minimize inner;                     // its own phases + handlers
inner.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
inner.on(minimizer::Phase::force).add(forces);
cpu::md::MD outer;                                  // a DIFFERENT phase enum
outer.set_integrator(cpu::md::VelocityVerlet{0.5});
outer.on(md::Phase::end_of_step).add(cpu::SubDriver{inner, /*budget=*/50});
outer.build();                                      // builds BOTH; inner allocates once
outer.run(frame, 1000);                             // inner's convergence CANNOT stop outer
```

## Implemented components

### Drivers — all **presets** over one loop node; there is no category

| Preset | Backend | Phase enum | Default policy |
|--------|---------|------------|----------------|
| `cpu::Evaluate` / `cuda::Evaluate` | CPU/CUDA | `evaluate::Phase` | — (one pass) |
| `cpu::md::MD` / `cuda::md::MD` | CPU/CUDA | `md::Phase` | — (the step budget) |
| `cpu::minimizer::Minimize` / `cuda::minimizer::Minimize` | CPU/CUDA | `minimizer::Phase` | handler `ForceConverged` |
| `cpu::vib::Hessian` | CPU | `vib::Phase` | — (fixed 6N + finalize) |
| `cpu::mc::MC` | CPU | `sampling::Phase` | **outcome** `Metropolis` |
| `cpu::PathDriver` | CPU | `path::Phase` | handler `AllReplicasConverged` |

Each derives the loop node (`cpu::CPUDriver<P>` / `cuda::CUDADriver<P>`) **directly** and
configures its defaults in its **constructor**. `cpu::SubDriver<P>` wraps any of them as a
node in another driver's Plan (nesting, on one shared Runtime).

**Target:** SCF / PW / GPW / NDDO self-consistency loops are **Drivers** (or
Driver-shaped presets on concrete methods `RHF`, `UHF`, …). **Debt:** the tree
still has bag-shaped `SCFSolver` / `PWSolver` / `GPWSolver` / `NDDOSolver` —
do not grow that pattern.

### MD components
VelocityVerlet, NVELimit, InitVelocity, NoseHoover (Thermostat), NoseHooverNPT
(Integrator+thermostat+barostat), FixLangevin (Thermostat + Force), Barostat
(Berendsen), Deform, NeighborList, PairLJCut (Energy|Force|Stress),
BondHarmonic / AngleHarmonic / DihedralMultiHarmonic (Energy|Force),
VirialStress. **LAMMPS-style names are canonical** — the old
`LJ`/`HarmonicBond`/`HarmonicAngle`/`MultiHarmonicDihedral` CPU aliases are
deleted (the torch backend keeps its own `LJ`/`HarmonicBond`).

### MC components (CPU)
Three presets over one shared loop (`mc::McBase`): `mc::MC` (NVT), `mc::NptMc`
(NPT), `mc::Gcmc` (μVT) — each attaches its `OutcomePolicy` (`Metropolis`,
`NptMetropolis`, `GcmcMetropolis`) in its constructor. Movers: `TranslateMove`
(single-particle displacement δ [Å]), `VolumeMove` (isotropic, uniform-in-V),
`InsertDeleteMove` (single species; tags from `Atom::acquire_tag`), composed
one-trial-per-sweep by `MoveSelector{}.add(mover, weight)…` (ref-qualified
chaining — a temporary chain stays an rvalue so the driver owns the copy). The
trial changeset is `mc::Transaction`, a **Frame patch** (touched atoms'
original rows + box + pre-trial `nlocal`; rows travel via
`comm::pack/unpack_exchange`), published on `ctx.mc.proposal`: the Mover
records before mutating, the policy decides, `Transaction::rollback` splices
the patch back bitwise (topology included). Per-verb attempt/accept tallies on
`McBase::attempts/accepts(verb)`. Reuses the MD force pipeline for trial
energies (`add_potential`/`set_neighbor`); `NeighborList` runs in always-rebuild
mode; trial energy is a deliberate O(N) full recompute. Single-rank v1.

### Path components (CPU)
`cpu::path::NebCoupling` (`PathCoupling`, bound at `couple`) — Henkelman improved
tangent + spring + climbing image; overwrites each replica's true force with the NEB
force. `cpu::path::BandLBFGS` (`Optimizer`) — L-BFGS over the concatenated band
(`n_images x 3N`), reusing `minimizer::LbfgsHistory`; trust-radius step, no line search
(NEB forces are not the gradient of anything, so there is no phi(alpha) to search).
Expansions: `cpu::LinearInterpolation` (NEB; owns the product state as its own config)
and `cpu::IdentityReplication` (the PIMD/REMD shape; no in-tree user yet).

### Minimizer components
LBFGS, CG (Polak-Ribière+), OptLog / TrajectoryWriter / FinalGeometry
(Recorders). Optimizer is hot-swappable on the same `Minimize` driver. MPI:
`install_defaults()` folds Migrate/Halo/Reduce into the `force` phase; LBFGS/CG
are DD-aware; recorders gather owners to rank 0.

### Gaussian SCF (`-DATV_BUILD_HF=ON`)
**Target:** concrete methods `RHF` / `UHF` / `RKS` / `UKS`; SCF iteration is a
Driver loop; parameters are members; Frame in/out. **Debt (shipping):** still
`cpu::scf::SCFSolver` + `SCFConfig` + public `IntegralPack`, plus component
`cpu::scf::SCF` (`HF` kind; forces with `with_gradient(true)`). CUDA path is
energy/density only (no CUDA SCF gradient). Gradients and utilities exist under
`gradient.h`, `basis_eval.h`, `grid.h`, `xc_functional.h`, `aspc_predictor.h`.
New API work lands on the target shape, not the debt.

### GPW data (in-tree static `.h` tables, no parsing)
GTH pseudopotentials, MOLOPT/PADE basis sets and the DFT-D3 reference tables
are **static C++ data headers** (`cpu/gpw/basis_data/gpw_tables.h`,
`basis_data/dftd3_tables.h`), transcribed once from the published tables
(Krack 2005; VandeVondele-Hutter 2007; Grimme 2010; generator preserved at
`tools/gen_gpw_tables.cpp`). Access is **OOP lookup constructors only**:
`GthPotential(molrs::Element, name)` / `GaussianBasis(molrs::Element, name)`
(`cpu/gpw/gth_potential.h`, `gaussian_basis.h`) and `D3Params{}`
(`internal/dftd3_params.h`). `Element` is molrs's periodic-table enum
(`atomiverse/molrs.hpp` includes its generated CXX definition directly) — the single
element-identity type everywhere; a GPW `Kind{element, basis, pot}`
(CP2K's `&KIND`) pairs it with its data. **There are no parsers, no
`builtin_*`/`make_*` factory functions, and nothing reads any data file at
build, test or run time.** Unsupported element/name throws (no fallback).

`GpwConfig.local_pp_on_grid` (default true) selects CP2K's route of collocating
the GTH local short-range PP on the grid. **Energy and analytic force both read
it** — the gradient must be taken of the functional the SCF converged, or the two
disagree by the grid discretization error of V_loc^SR (this was a real bug: ZnO
analytic-vs-FD 0.53 Ha/Bohr). See `cpu/gpw/internal/local_pp.h`.

### Plane-wave DFT (`-DATV_BUILD_PW=ON`, FFTW3)
Inner solver `cpu::pw::PWSolver` (Poisson → XC → Veff → per-k Davidson →
density + Pulay mix → convergence; `build()` parses UPF, geometry setup rebuilt
per `run()`). Evaluator `cpu::pw::PWEvaluator` (Energy; writes
`frame.meta.get(kEnergy)` [eV], Ry/Ha on `last_result()`). There is no CPU
solve-function wrapper: callers construct, build, and run `PWSolver`. MPI:
k-point round-robin over `MPI_COMM_WORLD`. CUDA migration is deferred; its
current `cuda::pw::pw_scf` surface is not a CPU compatibility API.

### Vibrational (CPU)
`cpu::vib::Hessian` (Response): finite-difference Hessian + thermochemistry.

### Output (cross-driver)
Thermo, Recorder (XYZ), RDF, VirialStress, Mulliken, density.

## Skills & agents

General workflow skills (impl / fix / debug / refactor / review / docs / test /
spec / litrev / note) come from the **molcrafts `mol` plugin** (`/mol:impl`,
`/mol:fix`, …), auto-adapted via the `mol_project:` frontmatter above.

Atomiverse-specific skills layer on top for domain audits: `/av-arch`,
`/av-cuda`, `/av-hpc`, `/av-perf` — each delegating to a paired agent
(`av-architect`, `av-cuda`, `av-hpc`, `av-optimizer`) that encodes
Atomiverse-only knowledge (CUDA execution-model classification, xsimd 11.2.0 /
xtensor 0.25.0 constraints, ERI shell-block dispatch + `-march=native`, Frame
purity, dual-backend hot-path anti-patterns).

Knowledge flow: `.claude/notes/notes.md` (evolving) → AGENTS.md (stable) ← agents
(read-only). Specs live under `.claude/specs/` while in flight and are deleted
at `done` (completed-spec history is the git log).
