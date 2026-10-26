# Semiempirical execution and extension review

Scope: source review of CPU NDDO, DFTB0/2/3, GFN0/1/2 and IPEA; CUDA
AM1/RM1/MNDO/PM3 fronts and their shared workspace. No numerical tests or
residency measurements were run for this review. Existing evidence in the
SE-backend spec is not a fresh run by this review.

Rules: docs/developer/project-contract.md's Frame/Component/Driver boundary, one owner per real loop,
composition, unique physical parameter data and transactional history; user's
clarification that a small physical method does not require its own Driver.
Design target: [es-extensibility.md](../specs/es-extensibility.md).
Method/backend capability claims remain in the semiempirical roadmap.

## Findings

### 1. CUDA NDDO repeats execution policy for four model identities

High impact; high confidence. The problem is copied loop/lifecycle code, not
the presence of a Driver for a genuine SCF solve.

- Each front constructs the same four-stage nddo plan and owns its own
  `solve_`, host entry, borrowed mount, failure cleanup and history transaction
  implementation. `solve_` has the same iteration/criterion loop. AM1/RM1
  source diffs are predominantly method names/identities; MNDO/PM3 retain the
  same execution structure while selecting their own model.
- Scientific stages, criterion and buffers already live in the shared
  `internal::NddoWorkspace`. Separate calculation instances need separate
  workspaces, not four copies of how a loop is executed.
- A future correction to budget handling, retry, stream teardown or parent
  transaction must currently be made in all four fronts. This conflicts with
  extending a physical model without reproducing scheduling.
- Locations: `include/atomiverse/cuda/semiempirical/{am1,rm1,mndo,pm3}_solver.cuh:18`
  (class), `:64` (private execution state);
  `src/cuda/semiempirical/{am1,rm1,mndo,pm3}_solver.cu:15` (plan construction),
  `:189` (solve), `:222` (iteration), `:240` (host entry), and `:364`
  (AM1/RM1/PM3 nested entry; MNDO `:366`). Shared implementation candidate:
  `include/atomiverse/cuda/semiempirical/internal/nddo_workspace.cuh:60` and
  existing `cuda::Driver` borrowed execution primitives.
- Could be justified if the models had different loop topology or ownership
  contracts. The reviewed fronts do not show that distinction. Their model
  physics must remain distinct where needed; their execution policy matches.
- Direction / fix: consolidate one backend-owned NDDO execution implementation
  with constructor-selected immutable model data and independent state per
  method instance. Keep public concrete names if useful; no forwarding-only
  evaluator or shared mutable global Driver.
- Unverified: no observed numerical divergence between the copies is claimed;
  CUDA behavior needs GH200 regression/transaction evidence after migration.

### 2. External model admission is closed by enums and table lookup

High impact for the requested extensibility; high confidence. This is a gap
against the new extension goal, not a claim that the built-in models are wrong.

- NDDO identities are a fixed enum; `nddo_parameter` and method validation
  switch/whitelist those identities. Public AM1 construction chooses the
  internal engine with an enum, not a third-party model provider. Subclassing
  its loop base does not open its physical parameter/integral boundary.
- DFTB's public `DftbSkSet` similarly selects fixed tables. `MioSkTable`
  resolves an element pair against static data. Supplying a new compatible
  SK set requires core edits rather than an external data object.
- Locations: `include/atomiverse/semiempirical/nddo_result.h:11`,
  `src/semiempirical/nddo_parameters.cpp:9`,
  `src/cpu/semiempirical/nddo_solver.cpp:393`,
  `src/cpu/semiempirical/am1_solver.cpp:32` and its RM1/MNDO/PM3 equivalents;
  `include/atomiverse/cpu/semiempirical/dftb2.h:34`,
  `include/atomiverse/cpu/semiempirical/internal/mio_sk_table.h:35`.
  Physical payload to reuse, rather than copy:
  `include/atomiverse/semiempirical/am1_parameters.h:17`.
- Could be justified for a deliberately built-ins-only product. That no longer
  satisfies the explicitly requested out-of-tree extension goal.
- Direction / fix: constructor-injected NDDO model/parameter and SK providers;
  retain one source for physical tables, matched energy/derivatives and
  explicit model identity/provenance. Existing built-in selectors may choose
  those providers but cannot remain the extension registry. Consolidate
  admission and harden unsupported-element/model checks.
- Unverified: arbitrary NDDO or xTB physics cannot be expressed as parameter
  changes alone. Additional integral/multipole terms require their own proven
  physical contracts; no automatic cross-backend support is claimed.

### 3. SCC orchestration has two implementations

Medium impact; high confidence.

- DFTB2 now owns a persistent Driver plan and calls its `run_nested`.
  DFTB3/GFN1/GFN2/IPEA instantiate `SccDriver` inside their solve body. The
  helper constructs its stages each solve and manually drives the iteration,
  swaps loop services, polls criteria and tears down the plan.
- Both use the same `SccStep` / `SccResidual` primitives and
  shift/diagonalize/charges/accept-or-mix/check recipe. Sharing primitives is
  correct; lifecycle/iteration orchestration still has parallel owners.
- Locations: `src/cpu/semiempirical/dftb2.cpp:1881` (persistent workspace),
  `:1957` (solve), `:2005` (nested execution);
  `src/cpu/semiempirical/scc_method.cpp:24` and `:106` (helper iteration);
  consumers `src/cpu/semiempirical/dftb3.cpp:357`, `gfn1.cpp:558`,
  `gfn2.cpp:1123`, `ipea.cpp:605`; shared primitives
  `include/atomiverse/cpu/semiempirical/internal/scc_stages.h:20` / `:44`.
- Could be justified while the DFTB2 retention migration is in progress; that
  migration is recorded in `se-backend-alignment.md`. Retention is a real
  difference, so do not blindly replace DFTB2 with the transient helper.
- Direction / fix: consolidate the matching stage/iteration recipe around
  existing Driver execution, retaining method-owned buffers and residuals.
  A method can own that private loop while remaining a Component. Different
  SCC variables/physics stay in model-specific stages. Do not merge NDDO's
  different stage order into this recipe just to reduce class count.
- Unverified: this review did not measure allocation costs or prove that every
  GFN update metric can use the same persistent workspace shape.

### 4. The SCC helper exposes execution plumbing in a public header

Medium impact; high confidence.

- `SccDriver::solve(Runtime*, ...)` is public in the installed-facing
  `cpu/semiempirical/scc_method.h`. Its reviewed callers are implementation
  files. It exposes parent execution plumbing and untyped physics callbacks,
  rather than a stable physical extension boundary.
- Locations: `include/atomiverse/cpu/semiempirical/scc_method.h:40` / `:57`,
  `src/cpu/semiempirical/scc_method.cpp:24`; consumers are the four files
  listed in finding 3. The DFTB2 persistent execution is already private.
- Could be justified as a private helper, not as a supported external API.
- Direction / fix: move the canonical SCC orchestration behind the method's
  internal boundary when consolidating finding 3; export physical model/data
  contracts instead. Third-party users must not construct or supply a Runtime.
- Unverified: consumers outside this repository were not inspected. The
  repository forbids compatibility shims; document the boundary change.

### 5. Older SE fronts retain a second force-request control

Medium impact; high confidence.

- DFTB3/GFN0/GFN1/GFN2/IPEA gate computation on both `with_forces` and the
  parent's force request. GFN fronts also lack the same standalone Frame
  run/forces surface now present on DFTB0/2 and NDDO. This is public/request
  migration debt, not evidence that those methods need Driver inheritance.
- Locations: `include/atomiverse/cpu/semiempirical/dftb3.h:77`,
  `gfn0.h:37`, `gfn1.h:55`, `gfn2.h:49`, `ipea.h:61`;
  `src/cpu/semiempirical/dftb3.cpp:725`, `gfn0.cpp:1044`,
  `gfn1.cpp:947`, `gfn2.cpp:1438`, `ipea.cpp:979`.
- Could be justified as a transitional API; the current Frame/request contract
  requires one authoritative request and an explicit unsupported-operation
  error, not silent suppression of requested forces.
- Direction / fix: harden/migrate Frame entries and parent request handling;
  remove duplicate switches with their C++/Python/QM/MM callers. Preserve
  model-specific force/response validation and history semantics.
- Unverified: no claim that a current assembled workflow silently lost forces
  is made; plan validation may reject some configurations beforehand.

## Minor

`NddoMethod` comments still name only AM1/RM1 although MNDO/PM3 reuse the CPU
base (`include/atomiverse/cpu/semiempirical/nddo_method.h:2` / `:19`). Update
those comments in the NDDO migration; do not infer a missing second CPU loop.

## Not problems

- DFTB0 is a single-pass Component with retained scratch and Frame entries
  (`include/atomiverse/cpu/semiempirical/dftb0.h:39`). It must not gain an
  artificial Driver. GFN0's EEQ/direct electronic work also does not justify
  inventing an SCC loop (`gfn0.h:32`, `src/cpu/semiempirical/gfn0.cpp:1029`).
- CPU NDDO already reuses `NddoMethod` for a genuine SCF loop, with stage
  bodies in an internal solver and a separate Criterion:
  `nddo_method.h:29`, `src/cpu/semiempirical/nddo_method.cpp:195`.
- DFTB2's Driver corresponds to a real SCC loop, not a small formula:
  `dftb2.h:155`, `src/cpu/semiempirical/dftb2.cpp:1881`.
- A Component owning a genuine private SCC Driver is an acceptable composition
  shape. DFTB3/GFN/IPEA need shared execution/Frame contracts, not mandatory
  promotion to public Driver subclasses.
- NDDO parameter tables and CUDA workspace/stage kernels are already shared;
  preserve them. A provider does not justify per-backend table duplication.
- Criterion-owned convergence, parent Runtime borrowing and staged history in
  CPU NDDO are correct boundaries worth retaining (`nddo_method.cpp:258`,
  `:311`). These source observations are not fresh runtime acceptance.

## Coverage

Read the listed public declarations, CPU NDDO solve/publication, both SCC
orchestrations and their four callers, model lookup, older SE request paths,
and CUDA NDDO front/loop/workspace code. Compared CUDA AM1/RM1/MNDO sources
and checked PM3's matching entry/loop definitions. Did not rederive every
integral, inspect every GPU kernel, audit all parameter values, rerun Python,
run oracle binaries or measure runtime/residency. Existing SE alignment work
and unrelated working-tree edits remain intact.

## Next

1. Implement the provider/loop-boundary gates in
   `.claude/specs/es-extensibility.md` step 4, coordinated with the active
   `.claude/specs/se-backend-alignment.md` CPU milestone; preserve DFTB0's shape.
2. Consolidate `src/cuda/semiempirical/{am1,rm1,mndo,pm3}_solver.cu:189` under
   one backend-owned execution implementation, then run affected GH200
   numerical, retry and transaction gates.
3. Migrate the five force-request paths in finding 5 using their existing
   method cases; verify standalone/nested and Python/QM/MM consumers.

Findings: 5 (high 2).
