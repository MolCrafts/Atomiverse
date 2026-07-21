> **Canonical notes: `.claude/notes/notes.md`.** This file is a legacy archive.
> New decisions go there (e.g. 2026-07-16 Role + kinds hard cut).

# Architecture Notes (legacy copy)

Accumulated architectural decisions and corrections from development iterations.
Agents read this alongside CLAUDE.md. When a note stabilizes, promote it into CLAUDE.md.

---

## [2026-06-06] Async analysis: freeze-then-dispatch invariant + device-ring (async-analysis-snapshot-ring)

**Invariant (freeze-then-dispatch).** Any snapshot delivered to an off-hot-path
consumer (recorder/observer) must be frozen into an immutable, step-stamped copy
**on the stream that mutates the frame**, before the next step can overwrite it;
consumers only ever read the frozen copy, never the live frame. Accuracy
(per-frame consistency + correct step stamp) is unconditional for every
component; completeness (may-drop-frames) is the orthogonal axis, governed by a
single `drop_oldest` policy. The MD hot loop never blocks.

**CPU (link 01).** `cpu::Publish` freezes the frame into a K-deep
`cpu::AnalysisExecutor` host ring with a synchronous OMP memcpy on the hot-path
thread (naturally ordered before the next step). Single uniform path:
`drop_oldest` only (block/grow + the per-component `backpressure()` override
removed), build-time static `mask_union_all`, global ring depth K
(`Config.ring_size`), shared worker pool.

**CUDA (link 02).** The fix lives in `cuda::AsyncBridge`, NOT a new
`DevicePublish` and NOT `cuda::Context.executor`: Thermo/RDF/Mulliken/Density
capture scalars / pair-distances / density-matrix, none of which fit the CPU
`FrameSnapshot` ring, so a shared-host-ring unification was the wrong shape.
Each `AsyncBridge` subclass owns a K-deep device ring + K pinned host slots
(K = compile-time `AsyncBridge::kRingDepth` = 3). `launch()` order:
`freeze_slot` (D2D live→device_ring[slot] on the **compute** stream) →
`cudaEventRecord(pack_events_[slot], main)` → `cudaStreamWaitEvent(defer)` →
`d2h_slot` (device_ring[slot]→pinned, defer stream) →
`cudaLaunchHostFunc(on_data_ready(slot))`. The compute-stream freeze is what
closes the prior cross-stream read-after-write tearing (the old path D2H'd from
live device pointers on the defer stream while the next step raced ahead).
`AsyncRecordBase` gains an internal `device_ring_[K]`+`host_ring_[K]` with
**slot-busy drop_oldest** (its HostWorker queue is unbounded, so it must drop a
capture whose slot is still in flight rather than overwrite it; `AsyncBridge`
needs no explicit drop because `cudaLaunchHostFunc` back-pressures the host
enqueue thread, keeping compute within K steps of defer).

**CUDA execution-model classification — add the device-ring-freeze row.**
Alongside *compute* (GPU-pure launch), *CPU bridge* (D2H + host callback), and
*host-dependent compute*: an `AsyncBridge`/`AsyncRecordBase` recorder/observer
is a **deferred device-ring-freeze CPU bridge** — `launch()` enqueues a
compute-stream D2D freeze + defer-stream D2H + host callback, zero
`cuda::stream_synchronize` on the step loop (guarded by
`tests/cuda/async/test_hot_path_no_sync.cu` and `test_async_bridge_tearing.cu`).

**Memory (link 03).** Ring footprint ≈ K·N·B (device ring) + equal pinned host
ring ⇒ ≈ 2·K·N·B; verified by `benchmarks/bench_async_snapshot_mem`.

---

## [2026-05-22] CUDA driver taxonomy migration — CUDA matches the three-category shape

Landed `.claude/specs/cuda-driver-taxonomy-*.md` (the 5-link
`cuda-driver-taxonomy` chain). Brings the CUDA backend onto the same
category-driven taxonomy CPU adopted in the 2026-05-06 redesign,
replacing the per-algorithm CUDA driver shape.

**The 5-link CUDA migration:**

  - **01** — namespace-normalized the CUDA backend (`ATV::cuda::*`) and
    added the `cuda::CUDADriver<PhaseEnum>` layer + the three category
    bases (`cuda::EvaluateDriver` / `DynamicsDriver` / `OptimizeDriver`),
    parallel to `cpu::CPUDriver<PhaseEnum>`.
  - **02** — `cuda::md::MD` migrated to `cuda::DynamicsDriver`.
  - **03** — `cuda::minimizer::Minimize` (`OptimizeDriver`) with
    `cuda::minimizer::SteepestDescent` as a hot-swappable Optimizer
    component, replacing the `cuda::minimizer::SD`/`CG`
    algorithm-as-driver classes.
  - **04** — CUDA SCF demoted from a `Driver` to the
    `cuda::scf::SCFSolver` inner solver behind the `cuda::scf::SCFEvaluator`
    Evaluator component; the SCF step helpers relocated to
    `cuda::scf::internal::` (`include/atomiverse/cuda/scf/internal/`),
    `cuda::scf::scf.cuh` deleted. `cuda::Evaluate` is the one-pass driver;
    single-point CUDA energy is `cuda::Evaluate + cuda::scf::SCFEvaluator`
    (or the `cuda::single_point` helper) — no `cuda::scf::SinglePoint`
    class.
  - **05** (this link) — scrubbed the now-dangling CUDA references to
    `ATV::scf::Phase`: deleted the dead
    `Driver<scf::Phase, cuda::device_tag>` explicit instantiation in
    `src/cuda/driver/driver.cu` and re-pointed the surviving `@see` /
    usage-snippet doc-comments to the `cuda::scf::SCFSolver` API. A
    `tests/cuda/scf_phase_scrub.sh` grep gate (CTest
    `test_cuda_scf_phase_scrub`) now asserts no `scf::Phase` token
    survives under `include/atomiverse/cuda/` or `src/cuda/`. CLAUDE.md
    and this log updated to the shipped shape.

**Two latent bugs surfaced and fixed in link 01.** The namespace
normalization in 01 replaced the explicit `<float>`/`<double>` CUDA
kernel-launcher instantiations with `<work_t>`-only forced
instantiations. Under the default build config
(`ATV_CUDA_REAL=float`, `ATV_CUDA_WORK=double`) this exposed two
pre-existing precision-tier defects:

  1. **`real_t`/`work_t`/`prec_t` signature drift** (build regression,
     ~25 errors across 6 files). Neighbor-list/SCF kernels were already
     correct, but their launch wrappers and the shared
     `nlist_impl.cuh` declarations had drifted: per-atom SoA/AABB/
     displacement buffers are `real_t` (not `work_t`), cell/inverse-cell
     matrices are `real_t` (consumed by `triclinic.cuh`), and
     `frame.box` is `prec_t`. Launchers and header decls were
     corrected to flow each precision tier uniformly.
  2. **D2H copies sized by the wrong element type.** The RDF observer
     and the XYZ/density recorders copied per-atom / per-pair `real_t`
     device buffers (`frame.pairs.dist`, `frame.atoms.x/y/z`) sized by
     `sizeof(double)`, reading 2x past the end and leaving a sticky
     CUDA "invalid argument" error. Fixed by staging through
     `real_t` pinned buffers copied as `sizeof(real_t)`, then widening
     to `double` for the CPU consumers.

**`ATV::scf::Phase` kept — owned by the CPU plane-wave SCF.** The chain
plan assumed `scf::Phase` (`facade/stage.h`) would be unreferenced once
CUDA SCF became an inner solver and could be deleted. A tree-wide grep
during link 05 drafting showed that is false: the CPU **plane-wave**
SCF (`cpu::pw`) runs as a `Driver<scf::Phase, cpu::device_tag>`
(`src/cpu/driver/driver.cpp`) and seven `cpu::pw::PW_*` components return
`DefaultBinding` values keyed on `scf::Phase::*`. The enum and the CPU
instantiation are kept; only the enum's doc-comment was corrected from
the now-false "Legacy CUDA-only … will retire" to name `cpu::pw` as
its owner.

**Follow-up — migrate `cpu::pw` off `scf::Phase`.** The plane-wave SCF
is the last `Driver<scf::Phase>` outer loop. Moving it onto its own
phase enum / category driver is a genuine architectural refactor with
behavior risk, unrelated to the CUDA migration, and was left out of
this chain. Recorded here as the open follow-up.

## [2026-05-06] CPU driver taxonomy redesign — three categories + SCFSolver inner solver

Landed `.claude/specs/cpu-driver-taxonomy.md` (phases 1-6). Replaces the
single `Driver<PhaseEnum, DeviceTag>` template-with-algorithm-subclasses
scheme with a category-driven layer.

**Three CPU outer driver categories** (decided by termination semantic,
not by algorithm):

  - `cpu::EvaluateDriver` — one pass; concrete: `cpu::Evaluate`
  - `cpu::DynamicsDriver` — full step budget; concrete: `cpu::md::MD`
  - `cpu::OptimizeDriver` — early stop on convergence; concrete:
    `cpu::minimizer::Minimize` (renamed from `cpu::minimizer::LBFGS`)

The underlying `Driver<PhaseEnum, DeviceTag>` template is now a private
plan/dispatch primitive — user code subclasses one of the three
category bases via `cpu::CPUDriver<PhaseEnum>` (the CPU backend
layer, in `include/atomiverse/cpu/driver/cpu_driver.h`). Subclassing
`Driver<>` directly is forbidden in CPU code. CUDA migration will
add a parallel `cuda::CUDADriver<PhaseEnum>` sibling.

**SCF is not an outer category.** `cpu::scf::SCFSolver` is a public
low-level inner-solver class (NOT a Driver subclass), used internally
by `cpu::scf::SCFEvaluator`. Its 5 step helpers
(`HamiltonianBuilder`, `EigenSolver`, `DensityBuilder`, `DIISMixer`,
`DualConvergenceCheck`) live in `cpu::scf::internal::` and never
inherit `cpu::ComponentBase`. Tests may include the internal headers
directly; production facade headers must not. There is no
`SinglePoint` class — single-point energy is `cpu::Evaluate +
cpu::scf::SCFEvaluator(.with_gradient(false))`, plus the optional
`cpu::single_point(frame, evaluator)` helper.

**Algorithm-as-component.** LBFGS / CG / FIRE are Optimizer
components plugged into `Minimize`. NoseHoover / Langevin /
Berendsen are Thermostat / Barostat components plugged into `MD`.
The new `cpu::minimizer::CG` (Polak-Ribière+) is hot-swappable with
`cpu::minimizer::LBFGS` inside the same `Minimize` driver.

**Component tagging — two orthogonal axes**:
  - `role()` — structural slot (`Integrator`, `Optimizer`,
    `Evaluator`, `Thermostat`, `Barostat`, `ConvergenceCheck`,
    `Recorder`, `Observer`, `None`)
  - `capabilities()` — bitset of frame fields written
    (`Energy | Force | Stress | Charge | Dipole | DensityMatrix |
    Orbitals`)

`DynamicsDriver` validates "≥ 1 component with `Capability::Force`"
regardless of role — `FixLangevin` (Thermostat role, Force-capable)
satisfies a pure-Langevin run.

**Outer iteration status channel**:
`cpu::Context::outer_status` is a `{converged, stop_requested,
reason}` struct, owned by the category driver, reset before each
iteration. ConvergenceCheck-tagged components write it; SCF
internals never touch it.

**Refactor was full no-backwards-compat**: `cpu::scf::DIIS` →
`cpu::scf::internal::DIISMixer` (renamed in place; one test caller
migrated). `cpu::minimizer::LBFGSStep` → `cpu::minimizer::LBFGS`.
`cpu::minimizer::LBFGS` driver → `cpu::minimizer::Minimize`. All 6
in-tree call sites updated in one commit per phase. No
`[[deprecated]]` shims.

**CUDA migration deferred**: `cuda::md::MD`, `cuda::minimizer::CG`/
`SD`, `cuda::scf::SCF` keep the per-algorithm shape until
`.claude/specs/cuda-driver-taxonomy.md` lands.

**`scf_solve()`** survives as a ≤30-line shim that constructs a
transient `SCFSolver` and runs it. Decision (vs deletion): retained
because `rhf_solve` / `uhf_solve` / `uks_solve` and the gradient
finite-difference path still call it; revisit during CUDA SCF
migration.

**Phase 5b deferred**: IntegralPack/ShellList split (drop `shells` +
`shell_data` from `IntegralPack` into a separate `ShellList` view)
and the 5 large file splits (`native_integrals.cpp` 1710 lines etc.)
remain TODO. Mechanical work; better as dedicated commits.

**Rule promoted to CLAUDE.md**: the three-category taxonomy, the
`role()`/`capabilities()` axes, and the "SCF is an inner solver, not
an outer category" decision.

## [2026-03-25] DeviceSystem removed — metadata lives in cuda::Frame

DeviceSystem class was deleted. Host-cached metadata (n_atoms, n_bonds, n_angles, n_dihedrals) now lives directly in `cuda::Frame` as plain Int fields with accessor methods. The device-side atom count pointer is `d_n_atoms` (not `n_atoms`). (Note: `DeviceFrame` was later renamed to `cuda::Frame`.)

**Rule**: Contexts (BuildContext, LaunchContext, DestroyContext) hold `cuda::Frame& frame` directly. Components access `ctx.frame.n_atoms()` for host-cached count and `ctx.frame.fx` etc. for device pointers. No wrapper class around `cuda::Frame`.

## [2026-04-06] CPU SCF force evaluators → Component refactoring

RHFForce, UHFForce, KSForce bypass the Driver/Component model — each owns a velocity Verlet loop (170 lines duplicated 3x), ASPC predictor logic (140 lines duplicated), and gradient skeleton (90 lines duplicated). Total ~590 lines of unnecessary duplication.

**Completed refactoring:**
1. ✓ Deduplicated gradient.cpp: extracted `accumulate_W`, `accumulate_common_gradient`, `make_result` helpers; unified FD gradient into single callback-based function
2. ✓ Unified `scf_gradient()` replacing separate rhf_gradient/uhf_gradient/ks_gradient — dispatches HF vs KS based on SCFConfig. Key subtlety: restricted HF stores P_alpha = P_total (factor 2 included), unrestricted stores per-spin densities
3. ✓ Created `SCFEvaluator` (cpu::ComponentBase at md::Phase::force) — single component for all methods, with optional ASPC predictor
4. ✓ Deleted legacy `RHFForce`, `UHFForce`, `KSForce` standalone classes
5. ✓ ASPC predictor lives inside SCFEvaluator (no separate wrapper needed)

**Rule promoted to CLAUDE.md:** "Driver-owns-the-loop" and "No duplication across methods" rules added to Component model section.

**Systematic bugs found during L-BFGS implementation:**
1. CUDA Driver classes use `Driver<Phase>` (1 arg) but template requires 2 `<PhaseEnum, DeviceTag>`. Latent bug — compiles only because CUDA sources not compiled in CPU-only builds. Fix deferred to CUDA session.
2. Infrastructure components (`md::ZeroForces`) hardcode `md::Phase::pre_force` in default_bindings — can't be reused by minimizer. Fix: create `cpu::minimizer::ZeroForces`/`ZeroEnergy` with bindings to `minimizer::Phase::force`.
3. CPU Driver subclass headers MUST include `cpu/backend_traits.h` for the `backend_traits<cpu::device_tag>` specialization. Without it, `this->add()` / `this->on()` fail at template instantiation.

## [2026-04-04] molrs integration complete

> **Superseded [2026-07-02]:** `cpu::MolRec` (and `io/molrec.{h,cpp}`) is
> deleted — the bridge is now raw-array FFI wrappers (`cpu/io/frame_writer.h`
> per-frame writers; `cpu::compute::*` batch computes + streaming accumulators
> in `cpu/observer/trajectory_metrics.h`, PIMPL). See the 2026-07-02 entry.

molrs CXX FFI integration is done. Key decisions that diverged from original spec:
- No `MolFrame` type in C++ -- all frame data lives in `molrs::MolRec` on the Rust side
- `cpu::MolRec` uses PIMPL (not direct `rust::Box`), keeping `rust/cxx.h` out of public headers
- Bridge is precision-templated (`bridge.rs` generated from template via `build.rs`)
- `ZarrWriter` delegates to `molrs::MolRec::write_zarr` (no separate `zarrs_ffi` dependency)
- Mulliken analysis stays in C++ (xtensor-based), not migrated to molrs
- `ATV_HAS_ZARR` / `ATV_BUILD_ZARR` flags removed -- Zarr is always available via molrs

## [2026-04-06] Test directory mirrors src/ layout

Tests are structured as `tests/{cpu,cuda,torch,facade}/...` mirroring the `src/` directory layout. Each test file uses GoogleTest fixtures (`TEST_F`) to organize tests by component/driver.

**Directory mapping:**
- `tests/cpu/md/` → CPU MD components (VelocityVerlet, LJ, CellList, bonded)
- `tests/cpu/scf/` → CPU SCF (integrals, solvers, gradient, grid, XC, evaluator)
- `tests/cpu/minimizer/` → CPU minimizer (L-BFGS)
- `tests/cuda/{md,potentials,minimizer,scf}/` → CUDA backend tests
- `tests/facade/` → Plan builder tests
- `tests/torch/` → LibTorch backend tests

**CMake naming convention:** `test_{backend}_{domain}_{component}`, e.g. `test_cpu_scf_gradient`, `test_cuda_md_thermostat`. Helper functions `atv_add_test(name, source_path)` and `atv_add_cpp_test(name, source_path)` take explicit source paths.

**Rule**: New test files go in the directory matching their source module. Use `TEST_F` fixtures to group tests by component. Each fixture class = one component or driver.

---

## Probe 1 verdict — many-body potentials (EAM + Stillinger-Weber), 2026-07-12

Breadth-first architecture campaign, probe #1. Spec `manybody-potentials`. The
deliverable is this verdict as much as the two potentials.

### The headline: `potentials/` was never the obstacle

The hypothesis was "the pair abstraction will break". It did not — **it simply does
not apply**, and that is a different finding:

- `facade/potentials/pair_kernel.h` (`PairKernel`/`SumKernel`) models `fpair(r^2)`
  and carries **no per-atom state**. A many-body style is not an extension of it;
  it is a sibling. Do not try to grow it.
- `cpu::Pair` marks `launch()`/`role()` `final` and `set_pair` is a singular slot,
  so an EAM cannot be a `Pair` even cosmetically. The blessed route already existed:
  `CoulLong`'s pattern — a plain `ComponentBase + NeighborConsumer` registered with
  `.on(md::Phase::force).add(...)`. **Both EAM and SW took it, and needed nothing new
  from the Component layer.**
- The `force` phase's `order` mechanism was never needed either: EAM's two passes
  (accumulate `rho_i`, then use `F'(rho_i)`) live inside one `launch()`, because they
  share a per-atom array and nothing else consumes `rho`.

**What actually broke is `neighbor/` and `comm/`.**

### `neighbor/`: the half list, and dead code that was already right

`CellList::refresh_full`/`build_full` had been written, documented and compiled since
the neighbor rewrite — **and called by nothing**. `NeighborList` only ever asked for
the half list. Both many-body styles need the full one (EAM's `rho_i` is a sum over
*all* neighbours of `i`; SW's angular term enumerates `(j,k)` from `i`'s complete
neighbour set), so the fix was to switch the flag on, not to write an algorithm:
`cpu::DynamicsServices::full_list_requested` (b0b6d9b).

Two things worth remembering:

1. **The flag must be resolved in `launch()`, not `build()`** — components build in
   Plan order and the neighbor phase precedes force, so the *requester's* `build()`
   always lands **after** the *provider's*. The CUDA neighbor list carries a comment
   saying exactly this about its own `flat_pairs_requested` export. Same trap,
   discovered independently on two backends.
2. CUDA was **ahead** of CPU here: its `PairList` was already full, and it already had
   the request-flag pattern. When a capability exists on one backend only, check
   whether the *other* one is the one that is behind.

### `comm/`: the real wall, and it is shared with probe 2

The step runs **exactly one forward halo** (`pre_neighbor`) and **one reverse halo**
(`post_force`), and the reverse payload is audited to `fx/fy/fz` — the header even
predicts this exact situation ("Any future kernel that writes to additional ghost-row
fields must either extend this payload list … or operate on owners only").

EAM pass 2 needs `F'(rho_j)` for a **ghost** `j`, and a ghost's `rho` cannot be
completed locally unless the halo reaches `2*cutoff` (ghost-of-ghost). **There is no
mid-`force` communication point.** Three routes, none free:

- (a) two extra halo passes inside `force` — breaks the one-forward/one-reverse
  invariant; `HaloSend::launch` is not re-entrant (it mutates `ago_`, rebuild
  generations, send lists).
- (b) full list + halo widened to `2*r_c` — no mid-force comm at all, at the cost of a
  fatter halo. **This is the recommended route.**
- (c) declare `rho` as a forward `CommField` — **wrong**: the forward pass fires at
  `pre_neighbor`, so it would ship the *previous step's* `rho`, silently stale by one
  timestep.

**Current state: EAM and SW THROW under MPI**, with the reason in the exception text.
Computing silently-wrong forces at rank boundaries was never an option. Closing this is
deliberately deferred to its own spec, because **probe 2 (constraints) hits the same
wall from the other side** — it needs *ghost velocities*, which the forward payload
(positions only) also does not carry. One comm redesign serves both; doing it twice
inside two probes would be the duplication the campaign exists to avoid.

`HaloParticipant::CommField` (the declarative extra-payload channel) therefore **still
has no production user** — its first one will be that shared comm spec.

### Force accounting under a full list (a real bug the FD oracle caught)

The full list visits each **pair** twice. Energy and virial are therefore halved. **The
force is not** — each visit writes only atom `i`, so the `i`-visit must deposit the whole
force that the pair exerts on `i`. My first EAM draft halved `fpair` by analogy with the
energy; the energy stayed *correct*, so nothing but the FD oracle could have seen it.

A **triplet** is different: it is enumerated once per vertex, so the three-body term is
not halved at all — a triplet has a distinguished atom, a pair does not.

**Standing rule reaffirmed: every force gets a finite-difference oracle against the
energy the component actually reports.** This is the gate GPW lacked when it shipped a
force that was the gradient of a *different* functional and hid for 17 commits behind
energy-only tests.

### Data: no parser, and no generator either

Static C++ tables + `EAMPotential(Element, name)` / `SWPotential(Element, name)` lookup
constructors that throw on an unknown pair (GPW precedent). Deliberately **no generator
under `tools/`**: unlike the GPW tables, these are five-to-ten numbers per element read
off a printed table, so a generator would be a program with no second caller.

### API gap found by an out-of-library consumer

`enable_timing()` existed on `cpu::md::MD` alone; `Evaluate` had none, so a regression
case could not quiet a chain of single points. Hoisted to `CPUDriver` — the timing
registry is Backend infrastructure every category owns. This is precisely what the
`regressions/` "public API only" rule is for.

---

## Probe 2 verdict — MD constraints (RATTLE rigid water), 2026-07-12

Breadth-first campaign, probe #2. Spec `md-constraints`. Serialized behind probe 1
because both touch `comm/`. They hit the **same wall from opposite sides**, which is the
single most useful thing this probe produced.

### The write-set was a lie, and nothing noticed

`Capability` is documented as "what frame fields may `launch()` write". Before this,
`VelocityVerlet` rewrote every position and velocity in the system and declared
`Capability::None`. So did the entire integrator/thermostat family. Nothing broke —
because **nothing read the bits**. That is the failure mode of a declaration nobody
consumes: it rots silently, and the first feature to trust it is the one that gets hurt.

Decision: **`Capability` IS the write-set**, stated in the header, and the mutators now
declare `Position`/`Velocity` (new bits 8, 9). `tests/cpu/md/test_write_sets.cpp` keeps
it honest, and pins the *other* direction too — `FixLangevin` must **not** claim
`Velocity`: it writes friction+noise into the **force** and lets the integrator move the
atoms. An over-broad write-set is as wrong as an empty one, just less obviously so.

Corollary: a constraint is **not** an Integrator. `ComponentRole::Constraint` exists, and
`DynamicsDriver` still demands ≥ 1 real Integrator — otherwise a Plan of "constraint, no
integrator" validates fine, nothing moves, every constraint is trivially satisfied, and
the tests are green.

### `post_integrate` was already there

The LAMMPS-shaped phase enum already reserved `post_integrate` — exactly where SHAKE runs
in LAMMPS. **No new phase was needed.** The phase design was right before we had a user
for it, which is worth recording precisely because the *rest* of this probe is a list of
places where it wasn't.

### Ndof: one owner, and a build-order trap that bit twice

`Ndof = 3N − 3` was hardcoded at **five** sites. It is now published once
(`src/cpu/driver/dof.cpp` → `ctx.md.n_dof_global`) in `setup_pass()`, between
`set_frame()` and `build_components()` — i.e. **before any thermostat caches it**, since
thermostats read Ndof in their own `build()`.

Two bugs, both from build **order**, both invisible to the unit tests:

1. **The owner count is `ctx.md.dd->n_local`, not `frame.atoms.size()`.** After
   `partition_atoms`, the frame's `nlocal` is still the *full* allocation on every rank
   (the tail is ghost slack). Summing it across 8 ranks gave an 8× Ndof, and NPT
   equilibrated at **2404 K** against a 300 K setpoint.
2. **Constraint providers must be captured in `build()`**, not scanned from `pending_` in
   `setup_pass()` — `Base::build()` has already consumed `pending_` by then. Ndof got
   published with **zero** constraints: geometry perfectly rigid, thermostat driving the
   box to 447 K. The unit test couldn't see it (it called `n_constraints()` directly);
   only the **regression** could. Then `for_each_pending` turned out to walk *bindings*,
   not components, so `FixRattle`'s 3 bindings counted 3× → "no degrees of freedom left
   (−3)". Dedup by pointer.

A thermostat with a wrong Ndof does not crash. It *drives the system to the wrong
temperature and reports success.*

### The kick+drift/wrap trap

`VelocityVerlet` fuses the half-kick and the drift and then **wraps** into the box. A
constraint needs the *unconstrained* post-drift positions and the pre-wrap ones are gone,
so `FixRattle` keeps its own shadow of `x_unc`. This is the one place the Frame-purity
rule bends: the shadow is **component-owned**, not a Frame field.

### The finding that spans both probes: nobody wants `CommField`

Probe 1 (EAM) needed ghost **densities**; probe 2 (RATTLE) needed ghost **velocities**.
The engine ran exactly one forward halo (`pre_neighbor`, positions only) and one reverse
(`post_force`, payload audited to forces). Both probes needed data that neither message
carried. `HaloParticipant::CommField` — the protocol *designed* for exactly this — was
used by **neither**, and still has **no production user** outside a test.

- **EAM** widened the shell instead: `NeighborConsumer::halo_cutoff() = 2·cutoff`, so
  every ghost's ρ completes **locally** from its own ghost-of-ghost shell and no F′(ρ) is
  ever communicated. Comm traded for halo width — the standard trade. One-forward /
  one-reverse invariant intact.
- **RATTLE** couldn't widen anything: its velocity stage runs at `final_integrate`, where
  **no message exists to ride**. It needed a genuinely new exchange —
  `comm::HaloRefresh`, a forward halo bindable at an *arbitrary* phase, auto-injected by
  the driver on sight of a `ComponentRole::Constraint`. Honest cost: **+2 forward halos
  per step**. Paid knowingly.

Two independent probes, two different escapes, zero uses of the protocol built for this.
`CommField` should be read as **on probation**: the next probe that needs ghost data is
the third vote on whether it earns its keep, or gets deleted.

### Bugs the multi-rank path surfaced in `comm/`

All three were latent, and none was a constraint bug:

- A **collective guarded by `if (n_ghost == 0) return;`** — a rank with no ghosts is still
  somebody's *neighbour*. Deadlock.
- **Two sequential `MPI_Sendrecv`**, one per face of an axis. On 2 ranks the lo and hi
  neighbour are the **same rank**, so the tags cross. Deadlock. Fixed with concurrent
  Isend/Irecv + Waitall.
- **A halo shell wider than the subdomain was silently truncated** — which is how EAM's
  multi-rank forces came out 0.67 eV/Å wrong. `HaloSend::build` now **throws**. A halo
  that cannot be filled is not a smaller halo; it is a wrong answer.

Also: under MPI, `frame.atoms.map` still held the *pre-partition* identity mapping, so
looking a tag up there read the wrong row entirely. `partition_atoms` now calls
`map_release()`. (This first presented as a "failing" test that was off by 9.5 Å while the
physics was fine — the test was reading the wrong atoms.)

### Verdict

Same as probe 1: **the domain directory was never the problem.** `md/` took RATTLE
without a fight; `potentials/` took EAM/SW without a fight. Both probes broke in `comm/`
and `neighbor/` — the *infrastructure* layers, which had been built against a single
implicit consumer (pair forces, half list, positions forward, forces back) and quietly
assumed it was the only one.

---

## Probe 3 verdict — Path driver + CI-NEB, 2026-07-12

Breadth-first campaign, probe #3, and the one aimed straight at the Driver taxonomy.
CLAUDE.md predicted this category ("Path for NEB — terminate when every image of a
coupled chain has converged"). **The prediction was half right, and the half it got wrong
is the finding.**

### The taxonomy is two-dimensional

CLAUDE.md: *"Category is decided by **termination semantics**, never by algorithm or
method."* True, and incomplete. "N coupled replicas" is a **second axis, orthogonal to
the first**:

|                 | terminate on convergence | terminate on a budget          |
|-----------------|--------------------------|--------------------------------|
| **one Frame**   | `OptimizeDriver`         | `DynamicsDriver` / `Sampling`  |
| **N replicas**  | **`PathDriver`** (NEB)   | REMD / PIMD / umbrella (later) |

Read one-dimensionally — the way the doc reads today — REMD and PIMD each eventually
demand *their own category*, and each re-implements ~90 % of the replica machinery NEB
already needed. The bottom-right cell is not one category; it is a whole row that does
not exist yet.

So the replica machinery is **infrastructure** (`cpu/replica/`), not driver-private, and
`PathDriver` contributes exactly one thing on top of it: the AND over per-replica
convergence. A budget-terminating replica category reuses `ReplicaSet` unchanged and
contributes *its* one thing (the exchange attempt). Whether that split was drawn in the
right place is testable, and REMD is the test.

### Replicas are generated, never supplied

User directive, and it corrected the spec's original design: nobody hands REMD sixteen
starting structures — they hand it one, plus a temperature ladder. So `ReplicaSet` is
built from `(Frame, ReplicaExpansion&)`, never from a `vector<Frame>`, and the
per-replica parameter (temperature / bias centre / bead index / path coordinate) belongs
to the **expansion**, which is exactly the thing that would otherwise weld the category
to NEB.

`run()` still takes ONE Frame and writes ONE Frame back — for CI-NEB, the climbing image,
i.e. the saddle configuration. **Frame in, Frame out** survives a category that internally
runs N of them.

Two expansions ship deliberately: `LinearInterpolation` (which owns the *product state*
as its own config — so NEB's "two endpoints" never become a driver signature) and
`IdentityReplication` (the PIMD/REMD shape, **with no in-tree user**). The second one is
the evidence: an abstraction that only ever admits one implementation has not earned its
indirection.

### **Components cannot be cloned** — and that killed the spec's design

The spec called for a Backend per replica (hence a Context, hence an `outer_status` per
replica), which would have made collective termination free. **It is not implementable.**
`Driver::add()` *moves* a component into `owned_components_`, and there is no `clone()`
anywhere in the tree. N backends would need N independently-constructed component sets,
reachable only by:

- a user-supplied component **factory callback** — against the OOP style rule, and a
  miserable API; or
- a `clone()` virtual on `ComponentBase` — a dead interface on every component that will
  never be in a band.

So: **one** backend, one plan, and the band reached through a `ctx.path` slice (the same
shape as `ctx.md.pair_list` / `ctx.mc.proposal`). The driver replays the `force` phase
once per replica by re-pointing the backend's frame; the band-wide phases run once and
see everything. Per-replica convergence lives on `ReplicaSet`, and the driver ANDs it.

This is the same lesson as probe 2's `ac-008`: **a criterion that names a mechanism can
only be satisfied by building the thing you already assumed.** Two probes running, two
specs whose prescribed mechanism was wrong, both times because the requirement (rigidity
across ranks; all-replicas-converged) was achievable another way. Write criteria as
requirements.

Known ceiling of the one-backend route: a force-phase component that caches per-frame
state (a neighbor list's bins) is invalidated on every replica switch and rebuilds N times
per iteration. Correct, but O(N) rebuilds. Müller-Brown has no neighbor list; a band of a
real solid would want a per-replica neighbor cache — which is the point at which a
component genuinely *would* need to be replica-aware, and the point at which the
no-clone finding becomes a real design decision rather than a workaround.

### The phase enum stays generic, on purpose

`path::Phase` = `force, couple, search_direction, line_search, convergence_check,
end_of_step`. **No `tangent`, no `spring`, no `climb`.** Those are components bound at
`couple` — the same slot PIMD's ring-polymer spring will use (mathematically the identical
k±1 harmonic coupling) and REMD's exchange operator. One NEB-shaped phase name would have
welded the category to NEB and left the next method with nowhere to bind.

### The L-BFGS Frame-welding debt, found and paid

The two-loop recursion sat inline in `LBFGS::launch`, reading `ctx.frame` directly, with
`global_sum(ctx, ...)` scattered through **every inner product** — that last part is what
made it unextractable, not the frame access. A band optimizer could only have copy-pasted
the algorithm.

Now `minimizer::LbfgsHistory`: curvature history + recursion over a plain vector, with the
all-reduce as an **injected `Reducer`**. `cpu::minimizer::LBFGS` passes its MPI
all-reduce; `path::BandLBFGS`, whose vector is the whole band and already rank-complete,
passes the identity. Behavior-preserving — every minimizer test, including the multi-rank
ones, passes unchanged.

This is the "extract at the second caller" rule's **positive case**. It was right to be
inline while `Minimize` was the only caller.

### Also worth knowing

- Adding a 6th category touches: `core/binding.h` (`PhaseKind` + its name switch),
  `facade/stage.h` (the phase namespace), `CategoryTraits<>` specialization in
  `cpu_driver.h` (**without it the inherited categorical setters do not instantiate** —
  a link error, not a compile error, and a confusing one), the explicit
  `template class Driver<path::Phase, cpu::device_tag>;` in `driver.cpp`, the explicit
  source list in `CMakeLists.txt` (cpu/ is not globbed), and the five-driver claims in
  `tests/cpu/test_driver_binding_kind.cpp`.
- **No concrete `PathDriver` subclass ships.** The other categories have one (`MD`, `MC`,
  `Minimize`) because they auto-inject infrastructure — neighbor lists, halo exchange.
  Path injects nothing, so a subclass would be pure boilerplate for the sake of symmetry.
- The async snapshot ring is **not** reusable as image storage: it is a lossy POD subset
  (no mass / charge / topology / scalars), write-only, and its slots are recycled.
- MPI v1: the band is replicated on every rank, with spatial DD *inside* each image. A
  LAMMPS-style one-partition-per-replica layout would **deadlock** today —
  `minimizer/global_reduce.h` reduces over the spatial `cart_comm` and no replica
  communicator exists anywhere in the tree.

---

## Driver-handler verdict — the Driver category is deleted, 2026-07-12

Three probes walked into the same wall. The wall was called **category**.

- **probe 1** (many-body potentials) — fine; no driver pressure.
- **probe 2** (RATTLE constraints) — `LBFGS` wrote `ctx.outer_status.converged`
  **unconditionally**. A convergence criterion (a *policy*) was buried inside an
  optimizer (an *algorithm*).
- **probe 3** (CI-NEB) — so the first image to converge would have stopped the whole
  band. To dodge it, `ReplicaSet` grew a second, private per-replica convergence table:
  a workaround for a status it could not trust.
- **probe 3, again** — expressing "N coupled replicas" needed a *second axis* on a
  one-dimensional taxonomy. Read one-dimensionally, REMD and PIMD would each have
  demanded a category of their own and each re-implemented ~90 % of what NEB needed.

**One disease.** Termination semantics was welded into a class hierarchy, so it could not
be reused, could not be composed, and grew a new class per method.

### What a Driver actually is

Strip it to the bone and there are three things:

1. **Stages** — a typed phase sequence (an enum); the Plan binds nodes to it.
2. **Iteration** — walk the stages, repeatedly.
3. **Termination** — when to stop.

Only (3) was in the class hierarchy, and (3) is the *only* thing the six category classes
disagreed about. Take it out and hand it to a policy object and there is nothing left for
a category to be. Eight classes deleted (`EvaluateDriver`, `DynamicsDriver`,
`OptimizeDriver`, `ResponseDriver`, `SamplingDriver` on CPU; the three CUDA mirrors),
four inheritance levels collapsed to two.

### The replacement triple

| | what it is | how many |
|---|---|---|
| **Runtime** (`cpu::Backend`) | frame binding, node storage, timing, MPI, DD | **one per run**, shared by the whole driver tree |
| **Driver** (`cpu::CPUDriver<P>`) | **one loop node**: stages + plan + handlers | any number, **nestable** |
| **Handler** (`cpu::LoopHandler`) | asked once per iteration | any number, composable, **user-definable** |

Two structural decisions carry the whole design:

- **A Driver does not own a Runtime — it is handed to one to execute** (`ctx.exec.runtime`,
  `run_nested`). Nesting therefore does not double the Backend, rebuild the domain
  decomposition, re-bind the frame, or split the communicator.
- **A Driver is itself an executable node** (build/launch/destroy — a component's
  lifecycle), so a Plan slot holds a component *or* a sub-driver (`cpu::SubDriver`). "Can
  a driver call a driver?" stops being a question: a Driver is a node in a node tree.

A second policy slot falls out, orthogonal to termination: **`OutcomePolicy`** —
*does this iteration count?* (`Commit`/`Rollback`). MC's accept/reject was never a
termination; rejecting a trial move discards one iteration's work and keeps going.
`Metropolis` is one implementation, and `SamplingDriver` — a class whose entire
contribution was that hardcoded `if` — had nothing left to be.

### Type erasure belongs at the NODE boundary, never at the stage boundary

This is the load-bearing rule and it is worth stating on its own.

The six Phase enums all start at 0, so an `int` phase cannot be told apart across
workflows: `stage == 1` is simultaneously `md::pre_force`, `minimizer::line_search` and
`evaluate::evaluate`. `Backend::run_phase(int)` took a raw int, and both of its callers
reached it by writing `run_phase(static_cast<int>(some::Phase::force), s)`. There was
nothing to catch a value from the wrong workflow.

So stages stay **typed enums all the way down**. `ATV::slot_of` is the one conversion in
the codebase and `test_stage_typing_scrub` fails the build on any other. The typed entry
points are `slot_of` / `slot_count<P>()` / `ctx.at(phase)` / `run_stage(phase, step)` /
`make_binding(phase, order)`.

Erasure happens exactly twice, and in both places it is the *point*, not a leak:

- at the **policy** boundary — `LoopHandler` is deliberately **not** a template. It sees a
  `LoopState` and *never a phase*, which is precisely what lets ONE handler serve MD,
  Minimize, MC and Path alike. A handler that knew `md::Phase` could only ever serve MD.
- at the **node** boundary — a sub-driver enters its parent's Plan as a `ComponentBase*`.
  The child keeps its own Phase enum; parent and child never have to agree about stages.

`DefaultBinding` still carries `{PhaseKind, int phase, int order}`. That pair IS the
erasure: `ComponentBase` is not a template, so `default_bindings()` must return one
concrete type for every workflow. `PhaseKind` travelling beside the int is what makes the
int safe, `make_binding` is the only way to build the pair, and no caller ever types it.

### The bug that shipped twice is now unwritable

Each loop node **owns** its `IterationStatus` + `LoopMetrics`; the Context only *borrows*
the active loop's (`ctx.loop`, a `LoopServices` slice). `run_nested` re-points them at the
child for the duration of the inner loop and restores the parent's on exit.

A pointer that points at the current loop cannot reach the loop above it. `LBFGS` and `CG`
no longer decide anything: they **publish metrics** (`max_force`, `rms_force`,
displacement, ΔE), and a **handler judges** them. `LBFGS::converged()` is deleted —
`Driver::converged()` replaces it, because the *loop* knows and the algorithm does not.

`tests/cpu/driver/test_sub_driver.cpp` pins it, and it **bites**: with the `ctx.loop`
re-point removed it fails (the child's zero-force metrics land on the parent's, the
parent's `ForceConverged` fires, and the outer loop stops after one iteration having moved
nothing). Verified by deliberately sabotaging the fix and watching the test go red.

The CUDA port surfaced the same class of landmine live: `SteepestDescent` published
**nothing**, so once `Minimize` attached a default `ForceConverged`, an all-zero
`LoopMetrics` satisfies every criterion — SD would have declared an untouched system
converged at iteration 0. Fixed by making SD *measure*, not by weakening the handler.
**A handler that lies in order to pass is worse than no handler.**

### The replica axis was never an axis

Probe 3's "two-dimensional taxonomy" folds away. **Replica count is not an axis of the
category — it is a layer of nested loop.** REMD and PIMD therefore need no new category
at all: they are "a replica loop + a budget handler + a different `couple` component".

### What this does NOT solve — and makes sharper

**Components cannot be cloned.** Running one inner Driver body over N replicas needs N
independent component states (N L-BFGS curvature histories), and there is no `clone()` on
`ComponentBase`. Nesting makes this **sharper**, not softer: today a `SubDriver` holds ONE
inner driver with ONE component set. Three routes remain unvalidated:

  (i) one sub-driver instance per replica — needs `clone()`;
  (ii) component state keyed by loop instance — dirty;
  (iii) band-level components (what NEB does today) — works, but does not use nesting.

This spec deliberately does not decide it. It only promises not to have made it worse.

### Honest deviations from the spec's own criteria

- **`cpu::PathDriver` keeps its name.** ac-007 listed it for deletion and called the
  replacement `NEB`; but `regressions/path/neb_muller_brown` constructs `cpu::PathDriver`
  directly, and ac-008 forbids editing a regression. ac-008 is the stronger signal (the
  spec says so itself). PathDriver was already *both* base and concrete driver ("used
  directly"), so it simply **became** the preset. Renaming it would have meant editing the
  acceptance gate in order to pass the acceptance gate.
- **No `StepBudget` / `TrialBudget` / `FixedBudget` / `SinglePass` handlers.** The spec's
  table proposed them. The budget is the `run(frame, n)` argument — the loop bound — and a
  `TrialBudget{n}` handler whose `n` must agree with `run(frame, n)` is a second source of
  truth for one number. Handlers exist for policies the loop bound *cannot* express
  (`ForceConverged`, `AllReplicasConverged`, `StopOnNaN`, and whatever the user writes).
- **Nested loops are not timed.** `Backend::op_handle_` is dense over the *bound* Plan's
  slots; a nested Plan's ops have no entry. `step_plan` takes the untimed path
  deliberately — a per-plan handle table is a follow-up, not a silent wrong answer.

## Probe 4 verdict — GCMC + NPT-MC, 2026-07-14

### The headline: the Frame's "fixed-N assumption" was a myth

The probe's founding hypothesis — "Frame assumes fixed N, GCMC will break it" — was
**false**, and proving it false was the probe's real payoff. Variable N already existed
end to end; it was just walled off inside `comm/`:

- `Atom::grow` / `set_capacity` were always dynamic-capacity (LAMMPS DELTA growth).
  **N belongs to `Atom` (`nlocal`), not to `Frame`** — the "fixed N" lived only in
  `Frame`'s constructor.
- `comm::pack_exchange` / `unpack_exchange` / `max_exchange_doubles` already serialized a
  complete atom row (SoA fields + bonded topology + special lists), public and MPI-free.
- MPI migration already did insertion (grow → write at `nlocal` → `++nlocal`) and
  deletion (compact-survivors-forward) every exchange.
- `NeighborList`'s `size_stale` trigger (added for migration) already forced a full
  rebuild on any N change; `CellList::bin_atoms` already grew its arrays on demand.

What GCMC needed was **lifting, not building**: `move_row` (static, `#ifdef ATV_HAS_MPI`,
inside `atom_migration.cpp`) became the public `Atom::copy_row`; the exchange serializers
became the Frame-patch payload. `mc/` contains no row serializer and no array-growth code
of its own.

### What was actually missing — four small, sharp things

1. **A tag allocator.** `grow()` seeds identity tags; after delete-compaction + insert,
   the new row inherits a survivor's tag. `Atom::acquire_tag()` (LAMMPS `tag_extend`
   shape: lazy max-scan seed + monotonic counter, never reused) is the whole fix.
2. **Planck's constant.** `units.h` had no `h`, so Λ (needed by every μVT prefactor) was
   uncomputable. `h_planck_ev_fs` + `lambda_thermal(m, T)`.
3. **`CellList::grid_stale` compared only the x edge** (`box_at_last_build_` was one
   scalar). An anisotropic volume change silently kept a stale grid and dropped pairs —
   the exact NPT blow-up mode documented in the file. Now all three edges.
4. **`mc::Transaction` was seven parallel coordinate arrays** (`moved` + `old_*`/`new_*`)
   — a position diff with no verbs, structurally unable to record an insert, a delete, or
   a box change. Rewritten as a **Frame patch** per the standing directive (Frame is the
   only I/O currency): the touched atoms' original rows in a small `cpu::Frame` + the
   original `Box` + the pre-trial `nlocal`. All four verbs collapse into ONE rollback:
   restore `nlocal`, splice the rows back (via `pack_exchange`/`unpack_exchange`),
   restore the box. Delete resurrects topology a bare xyz array could never carry;
   volume-move rollback is bitwise (s·x/s ≠ x in floating point — only a stored row
   passes).

### GCMC needed no new driver category — confirmed twice over

The spec predicted "subclass `SamplingDriver`"; by implementation time the
driver-handler refactor had **deleted the category layer entirely**, and the prediction
got stronger, not weaker: the μVT/NPT acceptance rules are `OutcomePolicy`
implementations (`GcmcMetropolis`, `NptMetropolis` — the seam's first real users beyond
`Metropolis`), and `Gcmc`/`NptMc` are presets whose constructors attach them. The shared
trial loop lives once, in `mc::McBase` (extracted when the third preset appeared —
second-caller rule): bootstrap energy pass → propose → energy → policy → rollback-via-
patch, plus per-verb attempt/accept tallies.

The `Verb` enum on the proposal is how a policy learns what it is judging **without
touching the frame** — the policy decides, the Transaction that recorded undoes. An
empty proposal (`Verb::None` — deletion drawn at N = 0, volume draw clamped at V ≤ 0)
is "simply rejected": no Boltzmann draw, no undo.

### One trial per sweep is an API decision, not a convention

All Movers bound at `propose` fire every sweep — correct for MD-style pipelines, wrong
for MC's one-trial-per-iteration contract. `MoveSelector` (a Mover of Movers: weighted
draw, fires exactly one child, forwards build/destroy) is the composition point.
Children are held by `shared_ptr` only so the selector survives the driver's forwarding
setters as a copy — configured once, never mutated after registration.

### Deliberate ceilings (v1)

- **O(N) full-recompute per trial** stands (the only energy form valid for all
  potentials, incl. kspace). `PairEvaluator::energy_of_rows()` is the follow-up
  performance spec; GCMC at N ~ 10³ is fine, N ~ 10⁵ is not.
- **Single rank.** Multi-rank GCMC needs a rank-safe tag allocator + DD-aware
  insert/delete placement.
- **Volume-move patch is O(N)** (every row snapshotted for bitwise rollback) — the cost
  of exactness, invisible next to the O(N) energy recompute.
- **Single species** (`InsertDeleteMove{type, mass}`, one Λ in the policy).

### Reference-data lesson for the regression tier

The engine's `PairLJCut` is cut-AND-shifted; NIST SRSW's μVT tables are rc = 3σ **with
long-range corrections** (or linear-force-shifted) — a different Hamiltonian. Rather
than compare across conventions, both regressions anchor to **analytic statistical
mechanics evaluated for exactly the sampled potential**: B₂ by quadrature (McQuarrie
eq. 12-30) feeding the Mayer activity expansion ⟨N⟩(z) = V(z − 2B₂z²) for μVT (two
activities — the spec's own detailed-balance criterion) and the exact finite-N
⟨V⟩ = (N+1)k_BT/P + NB₂ for NPT (F&S §5.4 — the N+1 is the fingerprint of the V^N
weight). Convention-exact, and sharper than a table lookup.

### A real bug the probe caught: `Driver::add`'s lvalue branch is a dangling trap

`facade::Driver::add(Comp&&)` stores an **lvalue** argument as a NON-OWNING raw pointer
("caller retains lifetime") and only move-owns rvalues. A builder-style component whose
chained setter returns `*this` — `MoveSelector{}.add(…).add(…)` — turns the temporary
into an *lvalue* reference, so the driver silently kept a pointer into an object that
died at the end of the full expression. Debug: `role()` read a trashed vtable →
"Mover missing". Release: virtual call through the dead vtable → jump to address 0.

Fix pattern (applied to `MoveSelector`, worth repeating on any future chained builder):
**ref-qualified overloads** — `add(…) &` returns `MoveSelector&`, `add(…) &&` returns
`MoveSelector&&` — so a temporary chain stays an rvalue into `set_mover` and takes the
owning branch. Pinned by `MoveSelector.TemporaryChainIsOwnedByTheDriver`.

The deeper lesson: the lvalue/non-owning convenience is a foot-gun whose failure mode
(release-mode jump to 0, no diagnostic) is far from its cause. Candidate follow-up:
require an explicit `add_ref(comp&)` for non-owning registration and make `add` always
own. Not done in this probe — it touches every driver call site.

### The heaviest find: `Atom::grow` silently destroyed every live atom

`grow()` trusted `xt::xtensor::resize` to preserve the array prefix. **It does not.**
The first chunked reallocation with live data below `nlocal` — GCMC's first insertion
past the initial capacity (64 → 4160) — zeroed every per-atom array: all atoms collapsed
onto the origin, the LJ energy went NaN (inf − inf), and the Markov chain froze solid
(every trial rejected forever, `<N>` pinned at wherever the corruption struck).

Why three probes and years of MPI runs never saw it: `materialize` pre-allocates the
GLOBAL atom count and then *shrinks* via `set_capacity` (which always did an explicit
copy-into-fresh-storage), and the halo/migration overflow path also calls
`set_capacity`. `grow()`'s chunked path had **never run with live data** — it only ever
initialized fresh frames (where zeroing is invisible) — until variable-N MC became its
first real consumer. A latent data-loss bug sat on the hottest allocation path in the
engine, reachable the whole time, and only a new *method* walked into it.

Fix: `grow()` keeps its chunk policy and delegates the reallocation to `set_capacity`.
Pinned by `AtomGrow.ChunkedGrowPreservesEveryLiveRow`. The regression harness also
gained frozen-chain guards (a chain whose acceptance ratio is ~0 FAILS instead of
coasting through on its start configuration — the NPT case initially "passed" with a
dead chain sitting on its initial volume, which is why both cases now start deliberately
off-prediction).

## frame-block-02 — 迁移把两颗雷踩了出来（2026-07-14）

CPU 消费面（59 文件、~570 访问点）从 `atoms.x[i]` 迁到 launch 边界的 `span()` 惯用法。
六个并行 agent 分片改，主 agent 统一构建 + 全套门。**迁移本身是机械的；价值在于它把两个
潜伏隐患变成了硬失败。**

### 雷一：SCF evaluator 在 force 阶段偷换 Frame 的列（heap-use-after-free）

`scf_evaluator.cpp` 的 `restore_wrapped()` 做的是 `frame.atoms.x = std::move(snap.x)`
—— **整个数组换掉**，旧缓冲区释放。这违反 Frame 的既定契约：per-atom 列只有
`grow`/`set_capacity` 可以重分配，且只在 `pre_force`（`cpu/frame/atom.h` 白纸黑字）。

迁移前没人看得见：所有消费者每次都经 xtensor 成员访问，换了缓冲区也照样读到新的。迁移后
LBFGS 在 launch 顶部提升裸指针、line-search 重放 force 阶段（→ SCFEvaluator →
`restore_wrapped`），指针当场悬垂。ASAN 一枪命中 `lbfgs.cpp:310` heap-use-after-free。

修法：快照存 `std::vector`，恢复走**逐元素拷回**，列存储身份不变。
**教训："换整个数组"和"写数组内容"在 xtensor 成员访问下长得一模一样，在裸指针下是天壤之别。
mapping 化的副作用是它让这条契约变得可执行。**

### 雷二：解析计数窗口不可嵌套（我自己的仪表化 bug）

01 步的解析计数守卫按 launch 开窗。但 `Backend::run_slot_`（line-search 的 force 重放）
在**调用方的窗口还开着**的时候启动内层组件——ZeroForces/Pair/Halo/Reduce 各自的解析全被
记在了 CG 头上，10 次试探后 65 > 64 预算，守卫抛异常。

守卫没错，归因错了。修法：`run_slot_` 与 `step_plan`（子驱动）为每个内层 launch **存/恢复
外层窗口**，内层组件的解析记在它自己名下。
**教训：任何"按调用计费"的仪表，遇到重入/嵌套就必须显式栈化——否则它会指控无辜者。**

### 并行分片的意外收益

`Frame::span()` 缺 const 重载这件事，被两个独立 agent 在几分钟内**分别撞到**（一个放宽了
函数签名，一个干脆留下 5 处没迁，还有一个自造了 `const_cast` 权宜函数）。顺序开发时我自己
只会撞一次、随手补上、不留痕迹；并行时同一个 API 缺口被三种不同的绕行方案同时暴露，反而
让"这是个真缺口"变得无可辩驳。

### 性能：mapping 化不但没慢，反而快了 ~4%

LJ ⚡ 门：1.107→1.066 ms/step（50k 原子）。原因不是 mapping 本身，而是迁移**强制**把散布的
`atoms.x[i]`（每次经 xtensor `operator[]`）提升成了循环外的裸指针——这是 mapping 纪律的
副产品。**"解析停在 launch 边界"这条纪律，本身就是一次全库范围的手工提升。**

## torch 后端的测试是假绿——CMake 静默丢弃 .cu（2026-07-14）

frame-block-03 的调研（构建并测试 cuda/torch 后端）第一件事就撞到这个：

`ATV_BUILD_TORCH=ON` + `ATV_BUILD_CUDA=OFF` 时，`enable_language(CUDA)` 从未执行，CMake
于是把**每一个 `.cu` 源文件静默排除出构建**——不警告、不报错。结果：

- `libatomiverse.a` 里 **0 个** torch 对象文件
- 6 个 `test_torch_*` 二进制各自链接一个空的 `gtest_main`
- 运行输出 `Running 0 tests from 0 test suites` / `[ PASSED ] 0 tests`，退出码 0
- **ctest 报告 "100% tests passed, 0 tests failed out of 6"**

也就是说：torch 后端的 6 个测试从**未真正运行过一次**，而 CI 会给它盖满绿章。

叠加事实：torch 后端本身是 **CUDA-only** 的（每个源文件硬编码 `::torch::kCUDA`，
`test_transfer.cu` 直接调 `cudaStreamCreate`），CPU-only 的 libtorch 根本跑不了它。所以
"torch 在 CPU 节点上全绿"这件事，从第一天起就是纯粹的幻觉。

**修法（已落地）**：
1. `ATV_BUILD_TORCH=ON` 且 `ATV_BUILD_CUDA=OFF` → **configure 阶段 FATAL_ERROR**，附解释。
   一个不可能工作的配置必须当场失败，而不是回赠一份"什么都没测"的绿色报告。
2. 测试宏加同类守卫（`atv_guard_cuda_sources`）：任何 `.cu` 测试在 CUDA 关闭时注册 → 报错。
   这条更通用——防的是"往 CPU target 里塞 .cu"这个同型陷阱再次发生。

**教训：绿色的 ctest 不等于代码被执行过。** 一个测试二进制里有几个 test case，是需要
独立验证的事实。凡是"某个开关关掉后某类源文件被跳过"的构建系统，都有产出空测试二进制的
能力——而空二进制永远是 PASS。

## frame-block-03 侦察结论：三后端的转换边界都没坏，坏的是测试基础设施（2026-07-14）

问题："frame-block-01/02 之后，cuda/torch 的 Frame 和各后端 ⇄ molrs.Frame 的转换还对吗？"

### 答案：转换全对 —— 因为它们都是**逐字段具名拷贝**

| 后端 | 构建 | 测试 | 转换机制 | 结论 |
|---|---|---|---|---|
| cpu | ✅ | 153/153 | `materialize`/`write_back`（注册表驱动） | 金标契约未改一字 |
| cuda | ✅ GH200 | **175/175**（含 test_cuda_frame / materialize / multi_gpu） | `to_cuda`/`to_cpu`：逐列 `cudaMemcpy` | **没坏**（首次真验证） |
| torch | ❌ 不可构建 | ❌ **从未运行过** | 逐字段具名拷贝 | 静态安全，但从未被验证 |

关键原因：frame-block-02 **保留了 `cpu::Atom` 的全部命名成员**（删除是终态步骤），而
cuda/torch 的转换从不做 `sizeof(Frame)` / 结构体 memcpy —— 它们逐个成员名拷贝。所以
frame-block-01 新增的 `extras` / `owned_meta_` / 解析计数器成员对它们完全透明。
`cuda::to_cpu` 的写回集合（x/v/f + tag + charge + image）与 CPU 金标契约**逐字段一致**。

### 真正的发现：torch 的测试是假绿（见上一节），CUDA 的测试很少被跑

- torch：6 个测试从未执行过一次（CMake 静默丢 `.cu`）。已修为 configure 阶段 FATAL。
- cuda：需要 GPU 节点排队（本次队列 2778 个作业），所以**平时根本没人跑**。本次是
  frame-block 之后的第一次。8 分钟跑完，全绿。

**教训：一个后端"没坏"这件事，必须被证明，不能被推定。** 而证明的代价（GPU 排队、
LibTorch 环境）恰恰是它长期无人验证的原因——这两者互为因果。CUDA 的 GPU 作业配方已
沉淀在 `~/.claude/jobs/atv-cuda-frame-block/job.sh`，下次直接复用。

### molrs 边界的新契约（已钉测试）

扩展列（`insert("rho")`）与用户 meta **不跨 FFI**：它们是引擎本地 scratch，交换 schema
就是交换 schema。`write_back` 不导出它们、不被它们干扰；`materialize` 不凭空造它们。
以前这是 `write_back` 循环的一个无人知晓的副作用，现在是 `test_materialize.cpp` 里的契约。

## Component Role (2026-07-16)

See `.claude/notes/notes.md` § Component Role = RoleFlag set.
`Component` (not ComponentBase); kinds in `cpu/component/component.h`; no Capability.
