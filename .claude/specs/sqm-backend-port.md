# SQM method coverage with modern native CPU/CUDA implementations

Status: in progress; this takes priority over the unfinished DFTB-first slices
in `se-backend-alignment.md`. Existing DFTB work and scientific coverage stay.

## Contract

The target is SQM's scientific methods and capabilities, not compatibility
with its historical program implementation. Preserve each method's defining
Hamiltonian, published fitted parameters and correction terms. Modernize the
numerical algorithms, storage, scheduling and derivatives independently.
Do not replace a method's fitted model merely because a newer model exists.
There is no requested legacy-SQM numerical mode or Amber-specific runtime.


Atomiverse owns the Hamiltonian implementation, data flow, SCF termination,
gradient assembly, workspace lifetime and parent Driver integration. SQM is
an offline oracle and source reference, never a production subprocess or a
second runtime. Frame is the public input/output token on both backends.

Use BLAS/LAPACK/cuBLAS/cuSOLVER and existing Atomiverse primitives when their
semantics, layouts, workspace and synchronization costs fit the complete
solve. Measure energy-only and energy-plus-force cold/warm calls, not only a
kernel. Record peak memory, allocations, transfers, iterations and thread
counts. Do not add conversions or stored intermediates just to call a library;
do not replace a suitable primitive just to own its implementation.

No new performance library is needed for the initial slice. Numerical work is
FP64. Start within the existing one-GH200/eight-CPU allocation envelope and
one-hour bounded jobs; measure host and device memory before increasing size.
No new automation, push or multi-GPU work is implied.

## Source and scientific identities

- Official source reference: `Amber-MD/AmberClassic`, commit
  `76b00b2e9d7dc41ab1ce7c7aa2fe52b3608ec002`, `src/sqm/`.
- Source checkout: `/tmp/atv-sqm-reference`. Root license: GPL-2.0-or-later;
  preserve the file-specific notices and attribution on translated code.
- Executed oracle: local conda AmberTools **26.0**, SQM banner version 19.
  This is distinct from the older repository's AmberTools 26.1 BCC pins and
  from the AmberClassic source revision. Binary hash and input/output are in
  `tools/sqm_oracle/`.
- Important mismatch: current AM1/RM1 use MOPAC-shaped analytic STO overlap
  and CODATA constants; SQM's s/p path uses STO-6G expansion, `AU_TO_EV=27.21`,
  `BOHRS_TO_A=0.529177249`, and `EV_TO_KCAL=23.061`, with correspondingly
  derived parameters. Retain modern unit constants and evaluate approximation
  choices on accuracy and full-solve cost; do not introduce a legacy numerical
  mode merely to match old printed results. Attribute differences using
  controlled comparisons of constants, parameter precision and overlap.
  Raw SQM output remains external evidence, but is not an exact-equality gate
  for a different numerical convention. Do not hide unexplained differences
  in looser tolerances or alter fitted parameters to cancel them.
- SQM's output headed “Forces” is its gradient in kcal/mol/Angstrom. Verify
  sign and conversion before pinning Atomiverse forces.

## Ordered slices and acceptance

1. **Reference and baseline.** Census built-in theories, parameter coverage,
   corrections and supported observables from the pinned source. Execute SQM
   fixtures for energy, charges and full gradients. Preserve inputs, output,
   constants, settings and version. Profile current native NDDO end to end.
2. **CPU s/p foundation.** Complete AM1/RM1 method definitions, pair integrals,
   core repulsion and full analytic derivatives using modern constants.
   Reuse existing NDDO matrix solvers, density formation and contractions where
   their contracts fit. The concrete method owns persistent stages/workspace;
   the parent Runtime executes nested solves and Criteria judge convergence.
   Establish numerical accuracy separately from differences in oracle conventions.
   Implement analytic pair derivatives; retain pair finite differences as an
   independent diagnostic, not the production endpoint. Reuse integral data
   across SCF iterations only when measured recompute-versus-storage cost
   justifies it. Fix invalid-input, identity-rebuild, failed-result and retry
   boundaries before adding new Hamiltonians.
3. **CUDA s/p foundation.** Own device parameter tables, pair work, Fock,
   density, mixing, diagonalization and derivatives on the parent's stream.
   No host scientific-array round trips during SCF or nested dynamics.
   Verify against the same external fixtures and CPU; then measure resources
   and end-to-end times. Unimplemented combinations reject explicitly.
4. **Hamiltonian coverage on both backends.** MNDO and PM3; PDDG/MNDO,
   PDDG/PM3 and PDDG/PM3-08; PM3-CARB1/ZNB/MAIS; PM6 and its non-planarity
   term; AM1/d and MNDO/d with required d-orbital integrals. Each method needs
   its own parameter and external energy/gradient evidence. A selector in
   upstream source is not proof that every element/combination works.
5. **Corrections and other built-ins.** Dispersion/DH+, peptide correction,
   external point-charge interactions and supported solvent corrections;
   DFTB2/DFTB3, parameter sets and charge models through the existing DFTB
   implementation. External-engine switches (QUICK, XTB, DFTBPLUS, EXTERN,
   TCPB/TERACHEM, SEBOMD) are not native SQM algorithms to copy.
6. **Composition and acceptance.** Evaluate, Minimize and short BOMD use the
   native methods directly; verify accumulation, initial force, trial history,
   chunk input invalidation, failure/retry and Python parity. Preserve SE-only
   builds. Performance results must include correctness and resource data;
   small molecules establish correctness, not a GPU speedup claim.

Each slice finishes implementation, local tests, external numerical gates,
affected regressions and a status update before its coverage is promoted.
CUDA compilation and execution remain GH200-only. Do not mark a CPU-only
slice or a numerical-force path as CUDA/analytic/full-SQM coverage.

## Current evidence and remaining acceptance

The capability matrix lives in `docs/roadmap/semiempirical.md`; this spec
records execution evidence and the work needed to close the current slice.
A method is not complete merely because its CPU implementation or CUDA
energy-only path passes. AM1 and RM1 must each satisfy every applicable gate.

### CPU foundation

- AM1/RM1 production forces use three-direction analytic differentiation of
  the existing overlap, two-center ERI and core-repulsion expressions, followed
  by pair-local fixed-density contraction. No production finite-difference
  step, SCF differentiation or global derivative tensor is used.
- Method-owned stages, Criterion, derivative scratch, accepted density and
  LAPACK syevd workspace persist across solves. Nested solves reuse the parent
  Runtime. Normal success commits density; initial-force, trial and failed
  evaluations do not. Same-object retry, identity rebuild, live controls,
  overwrite/additive publication and idle/mounted move contracts are covered.
- Six executed OpenMOPAC 23.2.5 fixtures cover AM1/RM1 distorted water, tilted
  CO and ammonia. Max CPU errors: 2.10e-7 eV energy, 1.25e-8 e charge,
  7.86e-7 eV/Angstrom force. External gates remain 1e-6 eV, 1e-7 e and
  1e-5 eV/Angstrom; HOF has its independent 1e-6 kcal/mol gate.
  Source/input/output and independent gradient qualification are archived in
  `tools/tb_oracle/inputs/mopac/live-23.2.5/`.
- Latest local closeout: **711/711 CTest checks and 26/26 Python checks pass**
  in `/tmp/atv-se-alignment-build`. Logs are
  `/tmp/atv-sqm-evidence/cpu-closeout-{build,tests,python}.log`.
  This includes all enabled SE method cases; it does not upgrade weak external
  evidence. DFTB3 Hamiltonian closure and GFN2 full-vector/tolerance closure
  remain separate scientific gaps documented in the roadmap.
- CPU QM/MM prepares its private Frame before borrowing the parent Runtime.
  AM1/RM1 history, retry and forces pass; affected HF integration also passed
  on GH200. Python currently exposes the CPU AM1 path, not complete RM1/CUDA
  parity.
- Standing allocation measurements in `benchmarks/semiempirical/nddo_forces/`
  retain identical scientific values and SCF counts. Persistent stages save
  66 C++ allocations/call; Fock scratch saves `(iterations+1)*atoms²`; retained
  LAPACK workspace saves nine allocations/iteration. At 48 atoms the latest
  count is 287–1135 calls and 0.50–3.10 MB requested bytes per warm force call.
  These are allocation traffic, not peak memory or an isolated speedup claim.
  The broader analytic-force revision's measured 48-atom warm median is
  7.052 ms versus 21.568 ms for the earlier finite-difference revision; that
  comparison includes changed convergence behavior and is not a kernel claim.

### CUDA molecular energy/charge/force foundation

- AM1/RM1 own native FP64 energy/charge SCF using the unique shared parameter
  source, analytic-STO Hcore, core repulsion, direct two-center Fock, cuSOLVER
  Dsyevd, cuBLAS Dgemm and CUB reductions. Six matrices and library scratch are
  persistent. No global ERI cache is stored. The envelope is neutral,
  closed-shell CHNO with no periodic cell; stress requests are unsupported.
- CUDA forces analytically differentiate the same overlap, ERI and core terms.
  Pair-local directional derivatives contract the converged density, with
  equal/opposite FP64 force accumulation and no global derivative tensor.
  Persistent force scratch is allocated at build/rebuild. Public `run` preserves
  Frame forces; `forces` overwrites E/q/F; normal mounted Evaluate adds E/F.
  Output publication and accepted-density commit follow force/error checks.
- Parent Backend/stream/device Frame are borrowed, including component
  lifecycle, mutation locks, request flags, trial/initialization intent and
  energy addition. Six external energy/charge/HOF fixtures and CPU comparisons
  pass, along with rotations/translations, edits, budgets and failure/retry.
- GH200 **4352923**: 211 CUDA checks passed, 13 multi-GPU skips; 25 NDDO checks
  pass memcheck, 12 two-center checks pass racecheck, complete energy regression
  passes memcheck; 52 CPU SE/QM/MM checks and affected MD/RHF/UHF sections pass.
- GH200 **4353290**, incarnation `2026-10-24T09:49:31Z`: final logs reviewed,
  **36/36** NDDO/nested-request/borrowed-execution checks pass in the complete
  SCF SE-only build (HF/PW/GPW/MD/Minimize off). All three available AM1 case
  sections pass with `--device=both`; only the energy section has a CUDA path.
  The combined-module LJ-dimer section also passes on CPU/CUDA: 6/30 iterations,
  distances 3.816412907/3.819859755 Angstrom versus analytic 3.816370964.
  This validates the minimizer infrastructure, not NDDO CUDA force composition.
- Evidence is `/tmp/atv-sqm-evidence/gpu-se-only-minimize-final-{stdout,stderr}.log`
  and the adjacent job spec/reference. SSH access has recovered. Do not rerun
  this completed slice just because its former pending note existed.

### Current closeout evidence

- Evaluate cleans a failed Backend without retrieving partial results. Cleanup
  attempts all resources even if synchronization or component destroy fails,
  preserving the original exception. Timer failure cannot replace it or replay
  stale samples. Partial stream initialization is tracked. AM1/RM1 detach their
  parent and drain/release owned resources even when a user stage destroy
  throws; explicit shutdown reports the original failure, while destructors
  finish cleanup without terminating. Same-method standalone/new-parent retry
  is verified. Poisoned hardware contexts can still require process restart.
- GH200 **4368690** passed 31 NDDO units and memcheck, plus 18 two-center
  racecheck tests with zero hazards/errors. Pair derivatives are additionally
  exercised by the complete public force gate below.
- GH200 **4369345**, incarnation `2026-10-24T19:28:55Z`, passed:
  - **106** SE-only CUDA units, with three unavailable-capability skips;
  - the three AM1 case sections on CPU/CUDA, including all six external
    AM1/RM1 E/q/HOF/full-force pins and CPU force comparisons;
  - memcheck of the full public E/q/F section, **zero errors**, 9.9 seconds;
  - **221** combined-build CUDA checks, with 13 multi-GPU skips (234 listed);
  - both methods' initial forces, 2+2 versus 4 VelocityVerlet steps and CPU
    four-step agreement at 0.01 fs; initial minimization forces, three SD
    descent steps and a standalone E/F check at the final geometry;
  - affected MD initial-force/LJ, RHF Driver/constant-basis/gradient, UHF and
    LJ-minimizer regression sections. All sections remain below 60 seconds.
  The complete molecular force/composition section ran in 1.2 seconds. These
  short composition checks do not establish long-run NVE drift, time-step
  convergence, line-search acceptance or complete optimization convergence.
- Two intermediate builds mixed old NDDOSolver callers with its new persistent
  diagonalizer member. This changed object size and residual offsets; stale
  dependency files predated the new header include. A full source/header/test
  refresh rebuilt all consumers and removed the link/nonconvergence failures
  without changing scientific code or tolerances. Source transport checksums
  now match for the numerical and execution paths. The harness requires
  consistent dependency invalidation after object-layout changes.
- Evidence root: `/tmp/atv-sqm-evidence/`. Current job spec/reference are
  `gpu-public-forces-r4-{spec,ref}.json`; that job is terminal-success and its
  observer has stopped. Source snapshot hashes are in
  `public-forces-final-source.json`, with its patch/untracked archive and a
  separate molrs toolchain patch/head. Earlier failed-job logs are retained;
  they are not numerical acceptance evidence.
- The final energy/charge benchmark completed all eight method/size groups
  (3/12/24/48 atoms). Its executable SHA256 is
  `3c15b5fce61a0071680c7d156f151f4585819abff60ca1050ad7a3616edf8eb7`.
  Maximum final CPU/CUDA differences are 4.730e-11 eV and 4.730e-14 e. The
  report, raw JSON, source hashes, cache and source-archive provenance live in
  `benchmarks/semiempirical/nddo_energy/`; rerunning its read-only analysis on
  the raw data reproduces the saved aggregate exactly.
- Both full-process traces have no warning/error diagnostics. The 48-atom,
  twenty-call resident energy windows contain 57/56 AM1/RM1 SCF iterations,
  only 8/56-byte D2H blocks, no H2D, no visible CUDA allocation/free calls,
  and all correlated GPU activity completed before the final synchronization
  returns. This is bounded energy-path evidence; forces, moving geometries,
  host allocation ownership and actual peak memory remain unmeasured.
- Module-off closure **4369657**, incarnation `2026-10-24T19:38:39Z`, succeeded
  with semiempirical disabled and HF/MD/CUDA enabled: build/link, 13 shared
  recovery/request/borrowed-lifecycle checks, RHF Driver and MD initial-force
  regression sections all passed. Final logs are `gpu-se-off-{stdout,stderr}.log`
  in the evidence root, and the cache is in `sqm-force-closeout-results/`.
  Both GPU jobs are terminal and their observers have stopped. Remote
  `build-gpu-make` is now SE-off; reconfigure explicitly before future SE work.

## Remaining ordered work

The current numerical, short-composition, module-build and bounded
energy-residency/report gates are complete. Remaining acceptance is:

1. Extend the measured energy-only residency interval to energy-plus-force and
   moving-geometry calls. Add sustained NVE energy-drift and time-step-halving
   acceptance; four-step CPU/CUDA agreement is not a long-run stability claim.
2. Close trial-acceptance transactions and host chunk invalidation. CUDA
   Minimize currently delays acceptance to the next loop. Backend input edits
   must be compared with the last successful host write-back, uploaded without
   overwriting edits, and invalidate neighbors/electronic guesses and relevant
   PME/Ewald caches. Coordinate/velocity/cell edits must preserve thermostat,
   deformation and RNG history; topology/identity changes rebuild mappings.
   Never retrieve an old device Frame into a host Frame with a changed count.
3. Complete Python parity and sustained allocation/transfer/memory measurements.
   Do not claim zero steady-state allocations from source inspection or a small
   matrix-storage estimate; CPU full SCF still allocates.
4. Resolve broader element coverage and the SQM numerical-convention difference:
   distorted-water SQM AM1 is -348.56087243 eV versus native MOPAC-shaped
   -348.5620074163 eV. Preserve modern units and fitted parameters while
   attributing the difference; no arbitrary tolerance loosening.
5. Proceed through the remaining Hamiltonians/corrections in the ordered slices
   above. MNDO/PM/PDDG/d-orbital implementations are not completed by the
   AM1/RM1 foundation. Preserve DFTB/GFN independent scientific blockers.


## Active iteration: AM1/RM1 transactions, continuation and Python (2026-10-24)

The latest user order is to close these AM1/RM1 boundaries on CPU and CUDA,
then implement MNDO/PM3, with separate method acceptance. Do not promote the
later methods while the foundation gate is still open.

- Starting HEAD is `9de0abfb00d13d9d4931503c91a36ca3019a5cc8`, already on
  `origin/dev`. Existing notes/results and unrelated local changes remain
  excluded from commits. No new commit or push in this iteration yet.
- CPU history transactions now stage a persistent candidate until the physical
  parent succeeds. Four ledger tests, actual AM1/RM1 Armijo accept/reject/throw,
  final-step publication, and later-parent-component failure/retry pass.
  Candidate storage is two packed FP64 arrays: `8*Norb*(Norb+1)` bytes;
  74,496 bytes at 96 orbitals. No additional SCF is introduced by that storage.
- Tightening line-search rejection exposed an existing distinction:
  StrongWolfe can return a genuinely evaluated Armijo-decreasing fallback with
  `ok=false`. `LineSearchResult::sufficient_decrease` records that evidence;
  CG/LBFGS accept it, but reject untested/uphill candidates. No numerical
  tolerance was relaxed. After the fix, 18 targeted tests, both
  `minimizer/geometry` sections and all four `semiempirical/am1` sections pass.
  The earlier full unit pass had 1283/1285 pass; its one real CG failure is
  resolved by this targeted rerun and its unbuilt app target has now been built.
  Repeat the full affected suite after input invalidation is integrated.
- CPU Python AM1/RM1 uses typed borrowed mounts retaining the original method.
  Nanobind cannot adjust the nonzero DriverNode base offset in these multiple-
  inheritance classes; the invalid Python inheritance was removed in favor of
  concrete-type setters. 35 interface tests and the existing-case
  `python_nddo.py --device cpu` pass all six external E/q/HOF/full-force pins,
  run-preserve-force, Evaluate, retry, 2+2 versus 4 MD and three SD steps.
- CUDA transaction job **4373046**, incarnation `2026-10-24T20:34:51Z`, is the
  sole active SQM job. Its SE-only gate passes 107 tests, all three AM1 sections
  and public-force memcheck with zero errors. Combined units/affected workflows
  are still running. It predates the CPU StrongWolfe fallback fix; a related
  final CPU minimizer failure would require an updated targeted gate, not a
  tolerance change. Watch session 59828 observes it. Do not alter remote source
  or resubmit until terminal. Specs/refs/log chunks live under
  `/tmp/atv-sqm-evidence/gpu-transactions-*`.
- The second slice adds exact input snapshots at run boundaries, shared author
  input-change hooks, explicit identity-support preflight, and numeric refresh
  preserving component/stream lifetime. CPU and CUDA code are being completed;
  these gates are NOT yet passed. Unknown custom-component identity edits and
  CUDA PME/Ewald charge edits reject instead of using stale caches.
  This is electronic-history/input-boundary recovery, not a promise of complete
  thermostat/RNG rollback after arbitrary failure in a multi-step MD chunk.
- CUDA Python awaits an ARM build/runtime gate. Existing ARM CPython/uv/Rust
  are available. The previous OpenBLAS is OpenMP; a same-version LP64 pthread
  library is available for an isolated Python build. Preparation script is
  `/tmp/atv-sqm-evidence/prepare-gh200-python.sh`; root controls job submission.
- MNDO/PM3 external preparation only: six OpenMOPAC23.2.5 H2O/CO/NH3 fixtures
  plus 324 displaced re-SCF calculations are in
  `/tmp/atv-sqm-evidence/mndo-pm3-oracle`. Maximum pinned-gradient versus
  extrapolated re-SCF difference is `4.685851e-7 eV/Angstrom`. Binary hash
  matches the existing official MOPAC executable. No MNDO/PM3 implementation
  or capability promotion has happened yet.

The earlier six-matrix energy-residency/performance report is historical:
transaction staging adds a seventh persistent CUDA density matrix and input
snapshots add boundary host storage. Re-measure before claiming current-revision
allocation or residency results. Raw notes/results remain local.

### Input/Python gate handoff (supersedes the running-job status above)

- CPU second-slice acceptance is green: 25 targeted checks, 1,297 full unit
  checks, 14 selected semiempirical/MD/minimizer cases, 35 Python tests and
  `python_nddo.py --device cpu` including same-owner input edits. Logs use
  `/tmp/atv-sqm-evidence/cpu-inputs-*`. Formatting-only rebuild also completed.
- Transaction job **4373046** ended **failed/134** at its final *CPU*
  StrongWolfe LJ minimizer check, using the older snapshot before the fallback
  fix. Its preceding SE-only 107 passing tests, three AM1 sections, memcheck
  zero errors, combined 223 passing CUDA tests (13 multi-GPU skips), MD initial
  force and RHF Driver gates passed. Full stdout/stderr are saved; its watcher
  was stopped. Do not rerun or treat that job as wholly successful.
- Sole new SQM job: **4374526**, incarnation `2026-10-24T20:48:14Z`, currently
  running. `gpu-input-python-{spec,ref,submit}.json` and watch session **74093**
  track it. This snapshot includes all input hooks, the CPU line-search fix,
  concrete-type Python mounting fix and new composed tests. It runs SE-only,
  combined input/line-search/force gates + memcheck + affected workflows, then
  the isolated ARM Python build and CPU/CUDA Python acceptance. Resource limit
  remains one GPU, eight CPUs, one hour. No remote mutation until it is terminal.
- Reproducible source checkpoint:
  `input-python-source.patch`, `input-python-source.json`, and
  `input-python-source-untracked.tar.gz` in the evidence root. Sources were
  checksum-synchronized without preserving old mtimes; notes/results were not
  transferred. Python helper is `prepare-gh200-python.sh` at the remote source
  root; its new build is `build-gpu-python-sqm`, environment `sqm-python-env`.
- CUDA RHF/UHF/KRHF input-cache audit found no persistent accepted guess to
  invalidate: each solve reconstructs its cycle/engine and clears density/DIIS.
  No new SCF hook or storage was added. QM/MM's separate QmEngine input-history
  forwarding and arbitrary mid-chunk MD state rollback remain outside this gate.

### MPI boundary correction after the frozen CUDA submission

A read-only review found that rank-local change detection could diverge force
preparation and deadlock MD communication. `Runtime::refresh_external_inputs`
now collectively ORs change/error flags over the existing spatial cartesian
communicator in three local phases (snapshot, support preflight, invalidation).
The empty-communicator serial replica path stays local. No distributed
semiempirical solve was added. The isolated MOLRS/SE/HF/CUDA-off MPI build
`/tmp/atv-sqm-input-mpi-build` passes the six new `test_mpi_input_updates_n2`
checks and existing `test_mpi_criterion_n2`; logs are `cpu-inputs-mpi-*`.
This CPU-only source change occurred after job 4374526 was frozen; the remote
source is intentionally untouched until that job finishes. The corresponding
serial build/targeted/Python rerun uses `cpu-inputs-mpi-serial-*` logs.

### Native input/transaction acceptance and Python packaging retry

Job **4374526** ended at `2026-10-24T21:03:12Z`, exit 128. Native gates
passed: 112 SE-only CUDA unit tests; 241 combined CTest registrations (228
passes, 13 multi-GPU skips); AM1/RM1 external/composed gates; memcheck zero
errors; affected MD, RHF and minimizer regressions. Full stdout/stderr are
`/tmp/atv-sqm-evidence/gpu-input-python-{stdout,stderr}.log`.
Python dependencies and same-source ARM molrs built successfully, but the
provenance step incorrectly required Git metadata in the deliberately Git-free
remote source transport. Python CMake/tests did not run, so Python CUDA remains
unverified. Repair the packaging step using the exported source hash manifest
and run only the pending Python gate. Native gates need no repeat for this
packaging-only change. The previous watch was stopped after terminal status.

Pending Python-only retry: **4375722**, incarnation `2026-10-24T21:04:56Z`.
Script verifies every file of exported molrs source manifest before the build;
no remote Git metadata is required. Job reuses the installed ARM dependencies
and isolated build, with unchanged native source snapshot. Spec/ref/submit/watch
files use `gpu-python-retry-*` under the evidence root. Do not duplicate.

Python retry 4375722 correctly rejected a mismatched provenance manifest before
build: remote molrs uses its committed rust-toolchain.toml, whereas the local
checkout has an unrelated unstaged stable-channel edit intentionally not synced.
Manifest corrected to the actual committed source (no source file changes); all
986 remote file hashes verified read-only before the next submit. The job selects
the same ARM Rust toolchain explicitly via environment. Sole continuation job
4375742 uses `gpu-python-verified-*`; observer remains
existing session 82274 (`gpu-python-retry-watch.log`).

Job 4375742 built and loaded the CUDA Python extension with verified LP64
pthread BLAS and passed 35 interface tests. It ended at 21:12:29Z because the
source transport omitted the existing AM1/RM1 oracle JSON required by the
Python E2E script. Transported the complete existing fixed fixture archive,
without touching native/Python code; sole follow-up job 4375988
runs only that remaining E2E script on the same binary. Ref/spec use
`gpu-python-e2e-*`; existing watcher continues. Next source manifest must include
`tools/tb_oracle/inputs` fixed fixtures as runtime test inputs.

Job4375988 ran CPU Python E2E, reached CUDA boundary checks, and exposed an
invalid-control diagnostic mismatch: max_iter=0 was grouped under generic
positive finite controls, while the gate expects an actionable iteration-budget
error. Separate max_iter validation now names its iteration budget; numerical
checks/tolerances are unchanged. Only AM1/RM1 .cu were synchronized into the
frozen remote tree; new-method changes remain local. Sole retry 4376085
(`gpu-python-budget-*`) rebuilds _atomiverse and runs the existing E2E script.

Job4376085 passed all six external Python E/q/fullF fixtures on CPU and CUDA,
then the input-edit gate rejected the correct exception text because its test
accepted only “box”/“periodic”, not the implementation's explicit “cell”.
The test now recognizes those three equivalent terms; it still requires an
exception and unchanged numerical checks/retry. Only the frozen self-contained
Python test was updated remotely. Sole E2E-only continuation 4376256
is tracked by `gpu-python-cell-*`; new local method slice remains separate.

AM1/RM1 targeted foundation is CLOSED: job **4376256** succeeded at
`2026-10-24T21:15:53Z`, exit0. CPU+CUDA Python six external fixtures and both
methods' same-owner MD numeric/cell/count/type edits, failed-refresh retry,
initial forces, split continuation and short Minimize all pass. Together with
4375742's35 interface checks and4374526's native gates this closes requested
transactions/input invalidation/Python slice. No NVE/performance/broad-elements
claim added. Logs `gpu-python-cell-{stdout,stderr}.log`.
Parallel local MNDO/PM3 implementation now integrated; root local aggregate
build session23867, log `cpu-mndo-pm3-build.log`. CPU owns sharedtables/selector
andCPU methods/regressions; CUDA owns concrete methods/workspace lookup and
CUDA sections; Python owns same-instance bindings/sharedregressionhelper.
No new-method CUDA job has been submitted yet.

MNDO/PM3 local acceptance: all six independent external E/q/HOF/fullF pins,
Frame semantics and Python workflows pass. New C++ normal-acceptance LS test
initially exhausted its artificially limited3 probes; its normal branch now
uses the existing default10, with rejection/error branches unchanged at3 and
no numerical tolerance/production algorithm change. Full local gate now passes
1299 units,16 affected regressions,49 Python interface tests (4 CUDA-only skips),
and all AM1/RM1/MNDO/PM3 CPU Python E2E. Logs `cpu-mndo-pm3-*`, individual
`cpu-{mndo,pm3}-{regression,python}*`. Final native case evidence is the16-case
ctest log, superseding earlier failed limited-budget test runs.
Sole new GPU job 4376370 (`gpu-mndo-pm3-*`) runs SE-only
and combined native gates, both newmethods' memcheck, then allfourPython paths.
OneGPU/8CPU/3600seconds; unchanged watcher82274. Source manifest/patch/untracked
archive `mndo-pm3-source*`,2561files including fixedoraclefixtures. No remote
mutation while running. Source sync uses checksum/no-times; excludesnotes/results.

Job4376370 ended21:21:39Z exit2: production library/unit executable compiled,
but the new regression-only CUDA contribution kernel called host-only n_atoms().
It now receives the count as a host launch scalar; no production/numerical change.
Only nd​do_mopac_cuda.cuh was synced (hash mndo-pm3-kernel-fix.sha256). Sole
continuation 4376567 repeats the unfinished gate using cached
builds; ref/spec `gpu-mndo-pm3-kernel-*`, existing watcher82274.

Full MNDO/PM3 gate **4376567 SUCCEEDED**21:29:33Z:113SE-onlyCUDA units;
242combined CTests(229passed,13multiGPUskips); bothmethod external/composed
sections inSE-only andcombined; eachfullnewmethodmemcheck0errors;53Python
interfacechecks; allfourmethods' externalPython andsame-ownerMD/Minimize pass
onCPU+CUDA. No numericaltolerancechanges.
Final APIreview removedonlynewMNDO/PM3 publicsystem()/state() andmovedinternal
solverincludeinto .cpp (no layout/math/state change). Local rebuiltalltargets,
13targetedunits/methodcases+49Pythoninterfaces(4skips)+bothnewPythonE2Epass.
These4fileswereupdatedremotelyonlyafter4376567terminal; sole finalbuild/API
validation 4376818 (`gpu-mndo-pm3-public-api-*`) recompiles
thethreeconsumerbuildsandrerunsnewmethodnative/Python. No repeatmemcheckneeded
forremovalofunusednonvirtualaccessors. Watch82274stillactive.

### Final closeout of the requested four-method slice

**4376818 succeeded at2026-10-24T21:33:31Z, exit0.** The final private-workspace
API rebuild passed SE-only and combined MNDO/PM3 native cases on CPU/CUDA, all
53 Python interface checks and both newmethods' CPU/CUDA Python workflows.
Final logs: `gpu-mndo-pm3-public-api-{stdout,stderr}.log`; full scientific and
memcheck gates: `gpu-mndo-pm3-kernel-{stdout,stderr}.log` (4376567).
Final source provenance: `sqm-final-source.{json,patch}` and untracked archive.
No SQM job remains running; observer is stopped at closeout. Current checkout
changes remain local; no new commit/push, no notes/results upload.

Requested AM1/RM1 transaction, input-invalidation and Python slice is accepted.
MNDO/PM3 now have native CPU/CUDA neutral closed-shell CHNO energy/charge/HOF/
analytic-force methods and validated Frame/Evaluate/shortMD/Minimize/Python
composition. Shared parameters/selectors are unique, no new physical kernels
or alternate runtime; internal workspace accessors are absent from newmethods.

Remaining work is separately gated: longer NVE and time-step-halving, current
force-path residency/allocation/performance measurements; broader elements and
charged/open-shell/periodic/solvent variants; PDDG/PM6/PM7/d-orbital features per
spec; existing DFTB3/GFN2 oracle closure. No claim of allSQM completeness or
GPU speedup. Existing automation was found PAUSED and remains unchanged.


### 2026-10-25 NVE refinement closeout

- Prior native MNDO/PM3 + transactional composition slice committed/pushed as
  `9d2a9c9d4`; notes/results and unrelated local modifications excluded.
- Added each AM1/RM1/MNDO/PM3 case's `nddo-nve`: public MD+VV, 10 fs at
  0.1/0.05 fs, same 0.5 fs samples, nonzero zero-COM initial velocities,
  fixed 1 meV sampled absolute E+K bound and fine/coarse <=0.4.
  No self-golden or tolerance weakening. Existing external water E pins
  anchor initial energies; existing full-force sections remain separate gates.
- Fixed helper Frame/Driver lifetime ordering and reject zero-error degenerate
  trajectories. No production physics changes. Updated harness input-refresh
  debt wording and explicit PDDG/special PM3/d-orbital gaps in roadmap.
- CPU local 16/16 affected regressions pass after final source freeze.
  Logs/metrics/source hash in `/tmp/atv-sqm-evidence/*nddo-nve*`.
- GH200 job `4377844` (2026-10-24T22:13:55Z) passed all four full method cases,
  including external pins and NVE CPU/CUDA comparisons; NVE sections take
  0.7–1.5 seconds. Ratios 0.24969–0.25000 on both backends.
  The job as a whole FAILED (exit8) only because the requested entire md/nve
  case included a 2-GPU section in this authorized 1-GPU allocation. This is
  not a successful full job or a multi-GPU pass. Both minimizer cases passed.
- Corrected bounded followup `4377905` (2026-10-24T22:16:34Z) succeeded,
  running only applicable CPU/CUDA initial-force, LJ argon and bonded NVE
  sections, all passed. No SQM rerun and no multi-GPU resource request.
  Logs reviewed; observer stopped after terminal. No job remains to resume.
- Current source freeze `nddo-nve-source.json` matches uploaded 2402 files;
  subsequent differences only REFERENCE documentation. No new force-path
  residency/allocation or performance evidence, longer-time stability,
  PDDG/PM6/d methods, broadened elements/spin/PBC acceptance implied.
- Existing sqm-cpu-cuda automation remains PAUSED; no new/resumed automation.

- NVE closeout committed and pushed as `8e9b2daad`; both commit/push hooks
  passed. Notes, results and unrelated edits remain local and untouched.

### 2026-10-25 four-method force resource slice (in progress)

Evidence `/tmp/atv-sqm-evidence/force-perf-20261025/`. Baseline HEAD/origin
8e9b2daad. Existing unrelated notes/spec/molrs edits protected; results excluded.
New schema2 native/Python/allocation harness: all four methods, isolated
3/12/24/48 atom distorted-water clusters, cold5, fresh mode owner+warmup5base,
20 fixed/moving calls; controls1000/1e-10density/1e-12eVenergy unchanged.
CPU density buffers reused without changing contraction order/criteria.
Local720 completeforce outputs/iterations exactly equal,52primitive rows
unchanged/zero new. Every call saves2I allocations,16packedI requestedbytes.
1301 localunits,16affectedregressions,49Pythoninterfaces+4CUDAonlyskips passed.
LocalPython fourmethod benchmark completed; its RSS includes entireprocess.

GH200 baseline4379153 succeeded after4379140 failed beforebuild(newtarget
needed explicitCMakeconfigure). Source/executable hashes and fullprocess
Nsight2025.3.2 traces retained. Initial movingwarm benchmark omitted geometry
kernel atstep0, exposing272B firstuse device-static trigconstants inresident
window; corrected warmup toexecute samekernel beforemeasure, unalteredcoords.
Baseline fullcapture verified no runtimeCUDAallocatorAPIs inwarmwindow;
that alone didnot establish zeromemoryevents. cuSOLVER dominates; force_pairs
only0.66% ofAM1 moving48 summedkernel time.

Jacobi n<=128 candidate4379202 passed115SECUDAunits inclresidual/orthogonality
andforcednonconvergence/retry, but FAILED existing completeAM1 CUDAcase
`CUDA AM1 SCF failed or exhausted its iteration budget`. No candidate timing
or performance claim; rejected, workspace sources restored exactly toHEAD,
newcandidateunit removed. Diagnostic patch/tests retained locally in
`rejected-jacobi/`. No science tolerance changed. StablecuSOLVER retained.

Job4379227 is current soleGPUjob: stableGPU+CPUdensityreuse; allfour external
cases,6alternatingbefore/aftertimings withfinalsameharness,allfour moving48
fullprocess traces/memcheck,combinedworkflow+Python rebuild. OneGPU/8CPU/1h;
CPUthreads1. Existing watcher59779 reused, automation remainsPAUSED.
Do not duplicate this job; resume itsref `paired-stable-ref.json`.

### Force resource/performance closeout

4379227 completed all six ABBAAB runs and AM1/RM1 moving48 captures;
AM1case memcheck includesRM1 and reports0errors. Job thenfailedexit255 on
attempting memcheck forlegacy RM1cpu-onlycase (noCUDAcalls), NOT sciencefailure.
Followup4387715 succeeded2026-10-25T09:21:43Z: onlymissingMNDO/PM3 captures
andmemchecks(0errors),32combinedCUDAunits,allfour fullcombinedexternal/
transaction/MD/Minimize/NVEcases,53Pythoninterfaces+allfour external/composed
Pythonworkflows. No repeatedtimings. 113SECUDAunitspassed in4379227.
Watcher25192/session59779 stopped afterterminal. AutomationstillPAUSED.

Six-roundpaired analysis passed exactCPUscience/iterations; CUDAfullvectors
withinunchangedtolerances. GH200720allocationcalls exactbefore/after plus
52primitive rowsunchanged. Savings2Inew/16packedIbytes (43.5–44.8% moving48
requestedbytes), still1232–2688newcalls/force. CPUmoving48 pooledratios
AM1.9967 RM11.0003 MNDO.9939 PM3.9989; no reliableendtoendspeedupclaim.
Finalfourtrace analysis: zerowarmallocation/freeevents/runtimeallocationAPIs,
noH2D, only4/8/24/56B D2Hstatus, matricesD2D. Dynamicdevicepeak/live169924475B;
238alloc/freepairs; pinned56B7pairs; static3044B retainedcaptureend, notsummed
intoanunprovenownership/reservedVRAMclaim. Alllifetimespairingcomplete.
AM1/RM1tracesn481, MNDO/PM3n179; alltiminingsn481. Profiletimesexcluded.
Finalnumericalsourcemanifestmatchescheckoutexactly. Finalsourcearchive/
patch/hash, binaries and rawreports retainedlocal. Roadmap/harnessupdated;
commit/pushincludesexplicit12filesonly, excludesnotes/results/spec/molrs/tools.

Force-resource closeout committed/pushed toorigin/dev as7eb9f5630. Bothcommitandpushhookspassed; twelveexplicitfilesonly. Notes/results/specs/molrs/unrelatededitsremainedlocal. NoGPUjobsleftfromthisslice; automationstillPAUSED.
