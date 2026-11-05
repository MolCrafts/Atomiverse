# Scientific benchmark framework

Status: the framework provides strict method declarations, bounded workers,
independent numerical comparisons, complete sample records, hardware-annotated
matrices, roadmap evidence and the molci 0.3 observation protocol. Implemented
methods are molecular HF and KS-DFT ([`benchmarks/scf`](../../benchmarks/scf/README.md))
and GAFF parameterization ([`benchmarks/forcefield`](../../benchmarks/forcefield/README.md));
`python -m benchmarks list` prints their rows. Further domains (MD, post-HF),
historical regression gates, shard merging and automated trace qualification
remain separate delivery items; this document specifies their target contracts.
[`benchmarks/README.md`](../../benchmarks/README.md) is the operational entry point.

## Purpose and terminology

Call the whole system the **scientific benchmark suite**. A benchmark evaluates
accuracy, speed and resource use against a specified problem and reference.
It is not limited to speed measurement. A regression check compares a new
version with an accepted historical baseline to detect deterioration. Thus
oracle comparisons and cross-backend tables are benchmarks; historical
regression checks are an optional capability within this benchmark framework.

Each method benchmark executes Atomiverse and independent references on the
same scientific problem. Its verdicts separately describe numerical acceptance,
measured performance and, when a historical baseline exists, regression.

One execution produces one evidence bundle. Numerical reports, performance
reports and roadmap status are projections of that bundle, not independently
maintained results. A faster incorrect calculation never receives a speedup
verdict. A correct calculation can be slow and must remain visible as such.

## Repository layout and ownership

```text
benchmarks/
  README.md  __init__.py  __main__.py  roadmap.toml  requirements.txt
  framework/
    specification.py          # method, system, participant and check contracts
    execution.py              # isolated workers, timeouts, timing and budgets
    comparison.py             # observable checks and comparison eligibility
    reporting.py              # JSON, matrix, JUnit and molci projections
    provenance.py             # actual hardware and verified source snapshots
    build.py                  # CMake/source/native-library build receipts
  systems/                    # immutable inputs shared by ATV and references
    hf.json  dft.json  gaff.json
  scf/                        # framework methods: bench_{rhf,uhf,rohf,ghf,rks,uks}.py
  forcefield/                 # framework method: bench_gaff.py
  perf/                       # native C++/CUDA timing harnesses (CMake, <build>/bin)
  compare/                    # scripts against external engines (LAMMPS, OpenMM, SQM)
```

`__main__.py` discovers methods by globbing `benchmarks/*/bench_*.py`, so a
method file sits exactly one level below `benchmarks/` and must export a
`Method` named `SPEC`. A new domain (for example `md/bench_nve.py` or
`scf/bench_mp3.py`) is added the same way; neither exists yet. Native harnesses
belong only under `perf/`, and cross-code scripts only under `compare/`.

**No results in git.** Campaign bundles, receipts and logs are written under
`build/` or outside the source tree; `.gitignore` excludes
`benchmarks/**/results/`. Documentation may quote a number with its hardware
and date but does not link to a result bundle.

Do not revive the removed C++ registry, CMake option
or CTest end-to-end runner. Unit tests for the new framework itself live
under `tests/benchmarks/`; scientific solves remain outside the unit suite.

Each `bench_*.py` owns a scientific method or explicitly named model combination.
It defines systems, supported participants, calculations, checks, tolerances
and roadmap links. CPU and CUDA belong to the same method file. The framework
owns execution, measurement and reporting; it must not contain a switch over
MP3, DFT, NVE, or an alternative implementation of their scientific algorithms.

Use existing engine Python APIs and command-line interfaces. Reuse generic
engine adapters only when several methods need the same launch/parsing code.
Do not create a second simulation API inside the benchmark framework.

### Clean replacement, without backward compatibility

The new suite has one CLI, one method contract and one result schema. Do not
support the removed C++ registry, old `reg_*.py` names, legacy command aliases,
old result readers, schema conversion layers or dual runner paths. Adapt
molcrafts-ci to the new benchmark contract directly, including snapshot identity
and rendering. Do not distort the domain schema to fit an old generic view.

Old source/reference files can be consulted for scientific provenance. They are
not imported as current results or maintained as a second supported execution
path. Reused inputs and external fixtures must be qualified under the new
contract. Incompatible schema versions are rejected explicitly; upgrade the
producer and consumer together instead of adding compatibility branches.

## Identity: a method is not a system

Use the following stable keys; display names are not identifiers.

| Key | Meaning | Example |
| --- | --- | --- |
| `method_id` | Scientific method family | `scf.rmp3`, `md.nve` |
| `row_id` | Method + model variant + requested task | `dft.rks.pbe.def2-svp.energy`, `scf.rmp3.sto-3g.energy` |
| `system_id` | Input system, independent of engine | `water-chain-16`, `lj-fcc-10648` |
| `participant_id` | Engine + actual backend + execution variant | `atv.cpu.direct`, `atv.cuda.fp64`, `lammps.kokkos.cuda` |
| `comparison_id` | Named ATV/reference pairing with a shared comparison contract | `atv-cuda-vs-pyscf-cpu` |
| `capability_id` | Stable roadmap capability, independent of Markdown position | `post-hf.rmp3.energy` |
| `campaign_id` | One immutable set of executions and its expected coverage | UUID or CI run identity |

MP3 can enumerate several basis variants in `bench_mp3.py`; each variant gets a
separate report row. A DFT XC/basis combination has its own file and row IDs.
Energy and energy+forces are different tasks/rows, even when defined together
in one file. NVE variants identify the integrator, potential and boundary model;
one LJ case does not establish NVE for all force fields.

A method file contains a discoverable `SPEC` plus ordinary Python calculation
functions. Discovery reads declarations without importing ATV, CUDA, PySCF,
LAMMPS or another optional engine. Imports happen inside participant workers.
Every method module is independently runnable through the shared runner, using
`python -m benchmarks.<domain>.bench_<method>` from the repository root.

The implementation should start with a small typed contract, not a plugin DSL:

```python
# Proposed API shape; not an executable example against the current checkout.
SPEC = MethodSpec(
    method_id="scf.rmp3",
    rows=[...],               # basis / task variants and required coverage
    systems=[...],            # shared inputs and size labels
    participants=[...],       # explicit ATV and reference callables
    comparisons=[...],        # reference identity and eligible pairings
    checks=[...],             # observable, metric, units, limit, justification
    roadmap=[...],            # capability IDs and the claims covered
)

if __name__ == "__main__":
    raise SystemExit(run_method(SPEC))
```

Each participant callable follows the same lifecycle: construct engine inputs
from the canonical system, construct the public calculation object, execute,
synchronize, materialize the requested outputs, release per-call resources.
The framework times that lifecycle and validates its returned record. Engines
may use different internal algorithms that satisfy the declared task.

### Public API coverage

The primary suite is a Python consumer of the installed ATV package and its
public bindings. It must record the imported package and native-library paths
and hashes. It must never silently import another installed build.

The HF delivery exposes CPU RHF/UHF/ROHF/GHF and CUDA CudaRHF/CudaUHF through
public Python bindings. Other domains, including MP3, need their own public
binding qualification. A missing binding is `binding_missing`, not CPU fallback
or a fabricated Python call. Archived C++ evidence does not prove Python
reachability. The native bridge and Python runtime must pass molpy 0.20's ABI
checks; matching the version string alone is insufficient.

## Shared scientific input and reference qualification

Both engines receive the same canonical input. Persist it, its units and its
content hash. Do not generate two independently randomized systems or have a
reference consume an ATV-generated density, orbital set or trajectory when the
claim is an independent end-to-end solve.

The comparison contract must resolve, as applicable:

- Geometry, atom ordering, isotopes/masses, cell, boundary conditions, charge,
  spin, electronic state and occupation conventions.
- Basis functions including contractions and spherical/Cartesian convention,
  pseudopotential content, frozen core, XC expression and coefficients, grids,
  cutoffs, dispersion, k mesh and smearing.
- MD parameters, topology, exclusions and special pairs, potential shift/tail
  conventions, timestep, initial velocities, constraints, seed and step count.
- Requested outputs, precision policy, accuracy targets, convergence checks,
  initial-state policy and resource budgets.

Record resolved settings, not only engine aliases such as `PBE`, `level=3` or
`double`. Different algorithms are allowed. Different numerical approximations
require an explicit comparison class: `matched_model` for matched settings or
`matched_accuracy` for separately qualified approximations meeting the same
physical target and observable limits. Never mix these classes in one trend or
silently call different Hamiltonians equivalent. Any required grid/basis/cutoff
convergence evidence belongs to the comparison contract.

| Reference kind | Numerical role | Production speed comparison |
| --- | --- | --- |
| `external_native` | Independent engine computes the requested method | Eligible after matching and acceptance |
| `external_fixture` | Archived output from an independent engine, with complete provenance | No current timing |
| `analytic` / `literature` | Qualified formula, identity or published result | No engine timing |
| `derived_reference` | A custom script combines engine components or implements the method | Diagnostic timing only |
| `internal_consistency` | ATV CPU/CUDA twin, finite difference, invariance | Supplementary check |

Record engine/version/source revision, exact public entry point, build features,
input/output artifacts and hashes. A custom MP3 implementation using PySCF
orbitals is named `custom-mp3-on-pyscf`, not `PySCF MP3`. It cannot be the sole
independent acceptance evidence or support a production-oracle speedup. Native
MP3 support must be checked in the chosen, pinned reference version.

Choose a qualified external MP3 engine during migration. Until then, report
`reference_missing`; do not turn the old custom-script comparison into a ✓.
Recorded fixtures can support numerical evidence but do not satisfy the normal
live ATV/reference execution requirement or establish present performance.
Default paired rows require a live qualified external calculation. Offline
fixture runs are labeled as such and cannot satisfy that live requirement.
Analytic/literature-only capabilities must declare that reference policy
explicitly in the catalog, rather than inherit a generic external-engine claim.

## Numerical acceptance

Each check declares the observable, unit, comparison metric, absolute/relative
limit, justification, required/optional role and reference. Serialize the
observed value, reference value, error, limit and verdict. Normalize units and
atom ordering explicitly. Preserve raw outputs and precision before formatting.
Reject nonfinite values, missing outputs, unconverged solves and wrong shapes
before computing errors. Required checks cannot disappear when data is missing.

The default scalar comparison is
`abs(atv - ref) <= atol + rtol * abs(ref)`. Vector/tensor checks declare their
norm explicitly, for example max absolute component and RMS. Energy, force,
stress and density checks have separate units and limits. A per-atom energy
limit cannot replace a required total-energy limit without an explicit policy.
Preserve all existing accepted correctness limits until a separately justified
scientific change replaces them; do not loosen them to obtain a faster result.

| Family | Required evidence to declare in its method file |
| --- | --- |
| HF/DFT energy | Total energy, convergence, correct state; residual/electron count where exposed; a nontrivial system |
| Post-HF energy | Reference HF energy, correlation energy and total energy; reference/correlation convergence where applicable; frozen-core/spin-channel checks when claimed |
| Forces/stress | Nonzero external vector/tensor comparisons, with explicit sign and units; symmetry-zero cases are supplementary |
| NVE | Initial energy/forces, one-step position/velocity/energy agreement, and a declared conservation envelope over physical time |

For NVE, record both drift slope and maximum excursion, with an explicit
normalization such as energy per atom per unit physical time. Long chaotic
trajectories must not require pointwise coordinate agreement. Long-run physical
or statistical checks use declared sampling, uncertainty and reference windows;
neither an arbitrary late-time trajectory difference nor energy conservation
alone establishes the correctness of the force field.

Required systems include a small fast acceptance case and a nontrivial case
where the claimed method contributes. Performance systems span declared sizes.
Do not extrapolate an H2/STO-3G result to production-scale MP3 or mark all XC,
basis, spin, force and boundary combinations from one passing DFT calculation.

## Performance contract

The main table reports **warm end-to-end wall time**, seconds, at a declared
resource profile. A canonical input is already loaded; the clock starts before
engine-specific input conversion and object construction. It includes setup,
allocations, integrals/neighbors/grids, all scientific iterations, dependencies,
required transfers, result materialization and explicit per-call cleanup.
The clock stops after completion and final synchronization. Shared JSON/HTML
serialization and engine-neutral error analysis are outside the clock.

For MP3, the default task includes HF, transformations and MP3. For DFT it
includes the requested complete SCF and derivatives. An isolated correlation
or derivative stage is a separately named diagnostic scope. No engine may get
precomputed scientific intermediates outside the default timer.

Record these scopes separately:

| Scope | Meaning |
| --- | --- |
| `process_wall` | Parent-observed complete worker process, including imports and runtime/context initialization |
| `warm_e2e` | Initialized runtime; fresh scientific state on every repeat, with the full lifecycle above |
| `steady_loop` | MD-only integration window after setup/equilibration; report steps/s or ns/day alongside E2E |
| named component scopes | Profiling diagnostics; never substituted for E2E |

An initialized CUDA context/library/JIT cache is permitted for `warm_e2e` on
both sides. Reusing converged densities, integrals, neighbors or amplitudes
between independent solves is not. MD steady loops intentionally retain their
state and use a separate contract. If a binding cannot provide a deterministic
cleanup/completion boundary, expose the limitation and use `process_wall` for
the valid comparison until the lifecycle is fixed.

Measurement policy:

1. Run participants in isolated workers under one allocation. Performance
   participants run serially on shared CPU/GPU resources. Declare affinity,
   MPI ranks, OpenMP and BLAS thread counts, GPU count/model and memory limits.
2. Keep first-process timing, then perform declared untimed warm-up and at least
   five fresh-state E2E samples per participant in the standard performance tier.
   Use a recorded interleaved A/B schedule. Never benchmark a shared GPU while
   another campaign is using it without recording the run as contaminated.
3. Keep every sample and its result checks; report median, range and dispersion.
   Do not silently discard outliers, failures or unfavorable systems. Capture
   GPU/CPU contention and initialization anomalies as observations, not guesses.
4. Compute speedup as `S = median(reference) / median(atv)`; `S > 1` means ATV
   is faster. Every ratio names its reference participant. Do not auto-select
   whichever oracle/backend makes ATV look best.
5. A ratio requires passing numerical checks, matching tasks/timing scopes,
   declared resource policies and qualified reference evidence. Otherwise keep
   raw times but set the ratio to null with a reason.
6. Timeouts, memory admission limits and unsupported sizes are explicit cells.
   Predicted budget rejection is `resource_limit`; an unexpected OOM is an
   execution error. Neither counts as a numerical pass.

CPU/GPU comparisons within one allocation can show product choice, while CPU
scaling tests also require matched CPU resources. A reference wheel lacking
OpenMP must not be described as an eight-thread reference. Cross-machine raw
times are visible but do not receive a controlled speedup or regression verdict.

Distinguish two performance conclusions: comparison with an oracle, and change
against a previous ATV baseline. Historical regression gates require the same
hardware/resource profile, task, input, protocol and compatible engine/build
environment. Prefer a same-allocation baseline binary when possible. Store the
baseline bundle identity. With no suitable baseline, report `unbaselined`.

Declare the material slowdown threshold, absolute noise floor and uncertainty
rule before measuring. A suitable gate requires both a material increase and
an uncertainty interval excluding no change; noisy measurements are
`inconclusive`, not automatically passed. Tiny correctness cases normally have
no performance failure threshold. A global geometric mean cannot hide a
required system's regression.

### CUDA-only qualification

Numerical acceptance and CUDA-only qualification are separate checks. CUDA
cells must record the actual execution path and cannot inherit CPU results.
For a CUDA-only claim, the profiled science interval begins after setup uploads
and reservations and ends before final requested-result download. It includes
all scientific stages, iteration/convergence decisions and post-HF work.

Require source/path qualification tied to the measured build plus complete
trace evidence for that interval: no host scientific computation or fallback,
scientific H2D/D2H traffic, Unified Memory migration, or dynamic allocation/free.
Host kernel launch orchestration is allowed. Trace completeness and boundary
coverage must be checked; absence of a trace table is not proof of zero events.
Profiler and sanitizer runs are separate from uninstrumented performance runs.
Their source, binary and protocol identities must match.

Peak device allocation, host allocation, memory reservations and per-block
shared memory are distinct metrics. Preserve their definitions and budgets;
do not call allocator-reserved VRAM or a free-memory snapshot a measured peak.

## Result records and artifacts

The canonical result is a **long-form** dataset: one execution record per
`row_id × system_id × participant_id × repeat`. Comparisons are separate edges
between records; a participant can be compared with several independent engines
without recomputing its result. The report is a deterministic pivot of these
records and the expected coverage declared before execution.

| Section | Required contents |
| --- | --- |
| Bundle | Domain schema version, campaign ID, start/end, expected matrix, completeness, suite/policy hash |
| Source/build | ATV commit and dirty diff hash, resolved partner commits, native binary/package hashes, compiler and build flags |
| Environment | OS, CPU/GPU, runtime/driver, BLAS/MPI/OpenMP features, actual resource allocation and budgets |
| Identity/input | Stable IDs, model/input hash, resolved settings, requested task and comparison class |
| Execution | Actual backend/algorithm, status/reason, convergence/iterations, resource observations and artifact links |
| Observables/checks | Raw and normalized observables, units, reference IDs, error metrics, limits and individual verdicts |
| Timing | Scope, warm-up/reset policy, order, all samples, summaries, instrumentation flag and comparison eligibility |
| Qualification | Public-client path, external-reference provenance, CUDA residency evidence and supported limits |

Execution states include `completed`, `failed`, `error`, `timeout`,
`not_implemented`, `binding_missing`, `reference_missing`, `dependency_missing`,
`unsupported`, `resource_limit` and `not_run`. Keep reasons structured. Numerical
status is separately `pass`, `fail` or `unverified`; performance status is
`measured`, `regressed`, `inconclusive`, `unbaselined` or `ineligible`.
An expected unsupported capability is coverage information, not a passing test.
A selected required cell that cannot run makes acceptance incomplete or failed.

Always write a partial terminal result after failure or timeout, including the
expected cells that never ran. Exit nonzero for failed required checks, execution
errors or incomplete required coverage. Optional unavailable engines remain
visible without failing an otherwise complete required selection.

Produce these artifacts from the same bundle:

```text
results.json                  # authoritative structured evidence
results.long.csv              # method/system/participant observations
matrix.html, matrix.md        # requested wide table
roadmap-status.json           # derived capability/backend status + reasons
junit.xml                     # operational gate summary, not scientific storage
snapshots/benchmark.json      # molci accuracy/performance/resource snapshot
artifacts/                    # canonical inputs, stdout/stderr, vectors, traces
```

Large artifacts have content hashes and durable links. JSON numbers must be
finite; unavailable values are null with reasons, never zero or a string NaN.
Summary formatting must not reduce the precision stored in scientific records.

## The method table

The default view has **one row per `row_id`**, a systems column and dynamic
columns for every declared engine/backend. Within each cell, system IDs align
with the systems column; nothing is averaged across different systems.

| Method / model / task | Systems | ATV CPU | ATV CUDA | Oracle A CPU | Oracle A CUDA | Oracle B CPU | Notes: hardware / environment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RHF / STO-3G / energy | water-1; water-16 | per-system result | per-system result | per-system result | explicit status | explicit status | Profile H1; actual resources per participant |
| RMP3 / STO-3G / energy | required small + nontrivial systems | per-system result | explicit status | qualified engine result | explicit status | explicit status | Profile H1; actual resources per participant |
| RKS / PBE / def2-SVP / E+F | required molecules | per-system result | per-system result | per-system result | per-system result | explicit status | Profile H1; actual resources per participant |
| NVE / Verlet / LJ-cut | required atom counts | per-system result | per-system result | per-system result | per-system result | explicit status | Profile H2 if a different allocation is used |

This is a layout example, not measured data. A result cell contains:

```text
system-id: acceptance; warm E2E median [dispersion]; error/limit; speedup vs ID
```

Oracle cells identify the engine/backend and show their raw time and reference
qualification. The HTML view expands all observables, samples and provenance.
Markdown uses aligned per-system entries or linked detail tables. A selected
single-system view stays one row per method; the CSV detail view may have many
rows. Every overview shows hardware profile, source commit, task/timing scope,
measurement date and completeness. Never silently stitch latest passing cells
from different commits into an apparently current table.

### Hardware in table notes

The final **Notes** column must identify the measured hardware and execution
environment, not just describe the algorithm. Shared hardware details can be
expanded once in a visible table footnote (H1, H2, ...), with each row linking
to the applicable profile. If participants in a row use different hardware,
map each participant to its profile explicitly. A heading such as `CPU` or
`CUDA`, or a cluster name alone, is insufficient.

Each profile's human-readable annotation and structured record must include:

- CPU vendor/model, architecture, installed sockets/cores, allocated physical
  cores/logical threads, MPI ranks and actual OpenMP/BLAS thread counts.
- GPU model/count, device memory capacity, GPU allocation or partition,
  host-memory capacity and allocated host/device memory budgets.
- OS, compiler/build mode, GPU driver/CUDA versions, BLAS/MPI versions, ATV
  revision and oracle versions/build features, including precision mode.
- Timing scope, warm-up/repeat policy and measurement date. Fixed clock/power
  policies are recorded when configured; unknown values are labeled unknown.

Generate these annotations from the actual run metadata. Method scripts declare
resource requirements rather than hard-code a machine description that becomes
stale when the same script runs elsewhere. For example, the note format is:

```text
H1: CPU=<detected model>, <allocated cores>, MPI=<ranks>, OMP=<threads>,
BLAS=<threads>; GPU=<detected model> x <count>, VRAM=<capacity>;
host RAM=<capacity>, budget=<allocation>; <runtime/build/engine versions>;
precision=<mode>; warm E2E, warm-up=<count>, repeats=<count>; <date>.
```

## Roadmap as a generated evidence view

`roadmap.toml` is the single declarative mapping from stable capability IDs to
the existing roadmap chapters. Each capability declares its scope, backend
implementation state, public client, required method rows/systems/checks,
designated hardware profiles and supported limits. It must cover every roadmap
row, including unimplemented rows. Unmapped rows, duplicate IDs, orphan method
specifications and missing required coverage fail catalog validation.

The roadmap continues to own the human explanation and planned gaps. Generated
status columns and evidence links are projections of the catalog plus results.
Do not copy measured status into a second manually edited registry. Composite
rows, such as energy+gradient, declare all requirements; one passing subtask
cannot complete the composite capability.

Proposed evidence-based status rules:

| Mark | Evidence required for that backend and declared scope |
| --- | --- |
| ○ | Catalog explicitly declares no implementation |
| ◐ | Reachable/partial implementation, but required numerical, client or qualification evidence is incomplete |
| ✓ | All required public-consumer numerical checks pass against qualified independent evidence, with complete coverage; CUDA-only qualification also passes when claimed |
| ⚡ | ✓ plus complete, current, reproducible E2E performance evidence against a qualified production reference for the required performance systems |

**⚡ means benchmark-qualified, not faster.** The table separately states the
speedup or slowdown. If a capability declares an explicit performance target,
that target must also pass for ⚡. Merely having a harness is insufficient.
The current projector awards ⚡ only from a performance-tier bundle with
complete acceptance and performance coverage for the capability. Repeating
acceptance samples cannot replace required performance systems, and a
performance-only subset cannot inherit unmeasured acceptance cases. Aggregation
of separate campaigns remains future work.

A current required failure displays **✗** with its reason; it must not be hidden
by an older green run or relabeled as unimplemented. Missing/stale evidence
displays an explicit freshness/coverage warning and cannot award a new ✓/⚡.
Keep the last accepted result available as history, with its actual commit.
Skipping unavailable CUDA hardware cannot promote CUDA status.

Official current status requires a clean source/build matching the target
commit and resolved dependencies. Local dirty runs are reviewable evidence but
are not published as clean-commit acceptance. A release view requires evidence
for that release. An older result can be carried forward only by an explicit,
audited equivalence policy with dependency/build/input hashes; never infer
freshness from the last successful run alone.

Each tier declares its own runtime and resource budget. Fast smoke cases do not
replace larger nontrivial qualification/performance cases. No timing budget or
qualification result is inherited from the removed framework.

## molcrafts-ci integration

Use `import molci as mci` and its `Source`, `Manifest`, `Tracking` and `Snapshot`
models, updating their contract where required below. Atomiverse owns the
scientific payload/schema validation; molcrafts-ci owns the envelope, immutable
storage, index, history and publishing. It must not learn how to compute MP3
errors or MD conservation.

Publish one `record="benchmark"` snapshot under project `atomiverse` per
campaign/profile. Its payload contains numerical results, performance and
resource measurements, qualification, coverage, roadmap status and artifact
links. Historical regression verdicts, when available, are fields within this
record, not a second independently published record.

A failed campaign still produces a benchmark snapshot. Its summary distinguishes
numerical pass/fail/unverified, measured/ineligible performance, historical
regressions and required/optional missing cells. The dedicated benchmark view
reads this summary directly. Do not flatten it into old generic test counts or
emit a dimensionally meaningless maximum combining Ha and force errors. JUnit
remains a separate operational CI gate.

The local molci 0.3 implementation requires a **mandatory `observation_id` for
benchmark records**. It has no commit-key fallback for that record type:

```python
snapshot = mci.Snapshot(
    manifest=mci.Manifest(
        record="benchmark",
        observation_id=campaign_id,
        source=mci.Source(repository="MolCrafts/Atomiverse", commit=verified_sha),
        producer="atomiverse-benchmarks",
        profile=resource_and_suite_profile,
        tracking=mci.Tracking(enabled=publish_official, generation=1),
        provenance={"campaign_id": campaign_id, "bundle_sha256": bundle_hash},
    ),
    payload=validated_domain_payload,
)
```

`publish_official` is false for dirty/local/partial review runs. Record actual
run timestamps and scheduler/CI identities in the source/payload as applicable.

Profiles separate hardware, resource/precision policy and suite selection,
for example `gh200-cpu8-gpu1-fp64-acceptance-v1`. Keep all detailed identities in
the payload. Domain payload versions and molci envelope versions are separate;
`tracking.generation` changes for incompatible record semantics, not a rerun.

Implemented integration boundaries:

1. `.github/ci.toml` checks can run commands and collect JUnit summaries, but its
   `dashboard` shortcut builds test-count snapshots. It is not a custom-domain
   payload ingestion API. Use the producer's snapshots and `actions/submit`
   after domain validation and `molci validate-snapshot`.
2. Benchmark records use
   `record + profile + generation + observation_id`; keep the commit as source
   metadata. `observation_id` is required in the manifest, snapshot identity and
   index. Every new campaign gets a new ID, including same-commit remeasurement.
   A transport retry uses the same ID and identical content. Conflicting content
   for an existing ID is an error, not a silent skip. Do not retain an old-key
   fallback or an observation-ID-absent compatibility path.
3. Aggregate all method/shard outputs for a profile/campaign before publishing
   its single benchmark snapshot. Preserve the declared expected coverage and
   all terminal failures. A rerun creates a new observation; neither it nor the
   UI may overwrite or hide the previous observation.
4. The dedicated Benchmark view reads this result schema: method/backend
   matrix, hardware notes, numerical verdicts, speedups and roadmap links.
   Producer, envelope/index validation and renderer change together. The old
   generic Regression/Benchmark rendering is not an interim acceptance target.

CPU smoke/acceptance can be invoked from standard checks. GPU and large MPI
campaigns execute on declared resources, using molq for HPC submission, and
return the same result bundle. The publication job verifies source/build
identities and completeness. It runs even after scientific failure so that
failures remain visible, while preserving the failing CI result.

Do not hide a GPU campaign in an ordinary GitHub-hosted CPU job. Automatic
publication uses the repository's existing credential policy; local design or
measurement does not imply permission to publish externally.

Implemented commands:

```bash
python -m benchmarks list
python -m benchmarks catalog
python -m benchmarks run --method rhf --tier acceptance --out build/hf-run
python -m benchmarks.scf.bench_rhf --tier smoke --out build/rhf-smoke
python -m benchmarks source --out build/source-manifest.json
python -m benchmarks attest-build --build-dir build/hf-native --jobs 2 --out build/hf-build-receipt.json
python -m benchmarks report --input build/campaign/results.json --out build/report
python -m benchmarks validate --input build/campaign/results.json
python -m benchmarks roadmap --input build/campaign/results.json --chapter hartree-fock.md --check
molci validate-snapshot build/report/snapshots/*.json
```

Implemented tiers select predeclared coverage and budgets: `smoke`, `acceptance`
and `performance`. A future `qualification` tier will own profiling/sanitizers.
A smoke run cannot
claim full acceptance by omitting the other required cases. Merging requires
compatible campaign/source/protocol identities, detects duplicate conflicting
cells, and preserves failures; the merge command is not implemented in the HF
slice. Selective developer runs are marked partial.

The manual `ci / HF scientific benchmark` job builds the public extension with
molpy 0.20, runs the framework/binding tests and the selected CPU tier, validates
the snapshot through molci, and preserves numerical failures and hardware
artifacts. It currently submits an **untracked** observation. Source/build and
CUDA-only qualification must be complete before official roadmap promotion;
running a local benchmark does not publish a global acceptance claim.

`attest-build` invokes the existing CMake target and captures source identity
before and after the successful build, its native binary hash, cache settings
and the molpy source/lock identity. `run --build-receipt` requires matching
current source and verifies every loaded ATV binary in its worker. Only such
matched receipts can satisfy the build qualification gate; a source commit or
manually labeled library path alone is insufficient.

## Delivery gates

1. Inventory every roadmap capability and its scientific evidence requirements;
   create stable IDs and the declared coverage catalog. Track reference/binding
   gaps. Start the new result dataset empty; qualify reused scientific inputs
   and obtain fresh results instead of importing the old output format.
2. Implement and test the small contract, worker lifecycle, comparisons, merge
   rules and deterministic reports. Check invalid outputs, missing cells,
   timeouts, unit conversion, async completion and snapshot collisions with
   synthetic framework tests before running large scientific campaigns.
3. Deliver vertical slices for RHF, NVE/LJ, one RKS PBE/basis combination and
   RMP3. Add missing public bindings and a qualified native MP3 reference as
   needed. Each slice must produce the same table and traceable verdicts.
4. Deliver the new benchmark snapshot contract, observation identity, hardware
   annotations, matrix rendering and failure publication in molcrafts-ci as one
   integration. No legacy snapshot/rendering compatibility layer is required.
5. Generate roadmap status and evidence links; compare with every current mark.
   Review any downgrade as a missing-evidence or actual-failure finding. Migrate
   remaining methods by chapter and replace overlapping historical harnesses.
   Maintain one supported benchmark path rather than parallel legacy/new paths.

The framework is delivered when a fresh checkout can reproduce the selected
method matrix; each cell explains its accuracy, time and missing capabilities;
CI stores failed as well as passing observations; and every roadmap mark resolves
to the exact required evidence for its backend and declared scientific scope.
