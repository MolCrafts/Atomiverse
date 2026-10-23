# Semiempirical CPU/CUDA alignment

Approved first milestone: CPU/CUDA DFTB0 and DFTB2 with Frame entries,
device-resident CHNO/mio neutral closed-shell molecular energy, charges and
analytic forces, mounted directly in Evaluate, BOMD and Minimize. No speedup
claim is required; reproducible GH200 measurements are required.

## Contract

- DFTB2 owns its `Driver<scc::Phase>` stages and workspace. DFTB0 remains a
  single-pass Component. Reuse existing SK data, derivatives and charge mixing.
- `run(Frame&)` overwrites energy/charges without touching forces;
  `run(const Frame&)` writes no Frame outputs; `forces(Frame&)` overwrites
  energy/charges/forces. Nested execution accumulates energy/forces and obeys
  parent force/stress, initialization and trial requests.
- Nested methods use the parent Runtime/Backend, stream and Frame. Criteria
  judge convergence. Failed solves never publish successful physical outputs.
- Workspace is retained across compatible geometries; identity/topology changes
  rebuild it. Initialization/trials never advance predictor history. Failed or
  rejected trials do not contaminate subsequent calculations.
- CUDA scientific arrays stay resident throughout SCC and forces. Only status
  and convergence scalars may cross during a solve. Use cuBLAS/cuSOLVER/CUB
  directly; no CPU fallback or duplicate parameter tables.
- CPU capabilities remain available. Unsupported CUDA configurations throw.
  Public configuration bags and forwarding compatibility APIs are not added.

## Ordered acceptance gates

1. W0: isolated SE build; record actual unit/regression outcomes and oracle
   coverage. DFTB0/2 and their dependencies must be green before migration.
2. W1: CPU Frame entries, DFTB2 as method Driver, retained workspace; migrate
   C++, Python and QM/MM consumers. Retire duplicate force/stress switches.
3. W2: CUDA nested request propagation, initial/trial handling, exception
   recovery and host-edit invalidation; verify affected existing workflows.
4. W3: CUDA DFTB0 matrices, generalized eigensolve, density, energy and analytic
   forces; existing method case gains CPU/CUDA sections.
5. W4: CUDA DFTB2 charge kernel, device mixer, staged SCC and analytic forces.
6. W5: Evaluate/MD/Minimize and Python integration, additive output semantics,
   zero-step force preparation, continuation and failed-solve retry.
7. W6: GH200 residency/allocation inspection, size-dependent cold/steady timing,
   memory and SCC counts; update roadmap only for verified quantities.

Small-system CPU/CUDA bounds: energy 1e-6 eV, max force component 1e-5 eV/A,
charge 1e-7 e, SCC residual 1e-10 e. External tolerances retain independently
documented provenance. FD is supplemental, not the sole oracle.

Unit tests belong in tests; full solves in existing method regression cases;
each section <=60 s; timing only in benchmarks. Verify SE/CUDA module on/off
and SE-only independence from HF/PW/GPW. Composition builds add only their
required modules. GPU validation requires GH200, never inferred from CPU.

## Current execution evidence

- Starting HEAD: f67127af88bcc1c548600138df9a702053328b75.
- Existing uncommitted SCC/NDDO work is preserved and is part of the baseline.
- Local build: `/tmp/atv-se-alignment-build`; logs and original diff:
  `/tmp/atv-se-alignment-evidence`.
- Arrhenius ControlMaster is available. Independent source snapshot:
  `/nobackup/proj/disk/teoroo/personal/jicli594/work/atv-se-alignment.S4C49E`.
  GPU baseline job 4321103 failed before compilation because Ninja is absent.
  The installed molq runtime rejected Slurm's successful empty-queue warning;
  it is repaired with the repository's existing `slurmWarningsPartial` fix.
  Live status/log queries now work (all Go packages and 11 Python checks pass).
  Repair receipt: `/tmp/atv-se-alignment-evidence/molq-repair.md`.
  Job 4322439 used Unix Makefiles and built the CUDA library, then exposed
  unconditional RDF test registration with ANALYSIS off. Registration now
  requires ANALYSIS and MD. Job 4323349 exposed another missing VIB guard
  in the RHF frequency regression; that requirement is corrected too.
  Job 4323510 passed the W0 snapshot: all 11 SE cases and 170 CUDA unit
  checks; 13 multi-GPU checks skipped on the single GPU. This validates
  existing code, not W1 changes or new CUDA SE physics.
- SE-only initially failed to link because tests registered disabled MD,
  potentials and neighbor implementations. Test registration now follows the
  existing product flags (including mixed-module consumers). The VIB-only
  build also exposed the missing MD/Minimize/Evaluate requirements of
  `response/vibrations/harmonic_driver_matrix`; its registration is corrected.
- W0 CPU baseline: 618/618 checks pass with VIB off; 649/649 with VIB on,
  including all 11 SE cases. OMP/OpenBLAS use one thread; HF/PW/GPW/CUDA off.
  Logs: `baseline-all.log`, `baseline-vib.log`, `gfn2-baseline-detail.log` in
  the evidence directory. GFN2's six sections pass: the historical red is not
  reproduced in this checkout/configuration. Energy error is 1.14e-5 eV;
  Hessian frequency errors are 0.046, 5.415 and 0.835 cm^-1. Existing loose
  force/Hessian tolerances and incomplete vector pins still need closure
  before GFN2 migration; passing this run does not close those evidence gaps.
- W1 DFTB0 slice: retained matrices/eigensolver/force scratch; Frame run/forces,
  Mulliken charge output, additive nesting, failure/retry, and removal of
  `with_forces`. Reuses MioSketchPair/MioSkTable derivatives and LAPACK/BLAS.
  The existing DFTB+ section now also checks its archived water charges and
  output semantics. 651/651 checks pass (`w1-dftb0-tests.log`). Allocation
  traces and CUDA residency have not been measured.
- W1 DFTB2 SCC node slice: DFTB2 owns a persistent five-stage Plan and charge
  Criterion. All five existing SCC variants execute on that concrete node,
  reusing `cpu::Driver::run_nested` and the parent Runtime. The shared SCC
  callbacks/criterion remain available to other methods via SccDriver.
  The water oracle section verifies actual parent/child Runtime identity,
  separate loop status, stage execution and retry after exhausted SCC budget.
  Invalid/unpublished residuals now fail before accept/mix; zero-budget reuse
  cannot retain a converged status. 654/654 checks pass, including all 13 DFTB2
  sections (`w1-dftb2-lifecycle-tests.log`).
- W1 DFTB2 Frame and request migration: direct run/forces/stress, transactional
  physical publication, private +U displaced coordinates, predictor commit
  only after success, identity-aware predictor invalidation. C++ consumers
  use the Frame entries or parent requests; duplicate force/stress switches
  are removed. Existing system-stress capability reports unsupported +U;
  direct stress rejects that combination. 656/656 CPU checks pass after
  request migration (`w1-request-migration-tests.log`). Direct/nested stress
  units, overwrite and accumulation also pass in the existing water case.
- Python bridges call the same Frame methods, including DFTB0. The existing
  molrs exchange and force-array conversion are reused; no parameter copy.
  Python-enabled build passes. Actual molpy.Frame run/forces round trips for
  DFTB0/2 pass against the existing H2 pin, including force preservation and
  charge publication; DFTB2 stress returns finite Voigt values. Surface and
  namespace tests: 72 pass, 133 omitted-domain skips. Isolated Python env:
  `/tmp/atv-se-python`; evidence `python-dftb-frame-check.log` and
  `python-se-surface-tests.log`.
- W1 ChargeMixer slice: retain the existing Anderson/Pulay algorithm, with
  fixed-capacity history and Gram/solution/result buffers. DFTB2 owns and
  resets the mixer for every geometry; other methods keep using the same
  implementation. Reset/reuse and aliased-output tests pass. Including IO
  for the Python bridge exposed a missing MD/potentials/neighbor test guard;
  corrected. Full current CPU suite: 692/692 pass (`w1-mixer-python-tests.log`).
  Allocation tracing has not yet been run.
- W0/W1 API/SCC/mixer work is committed as `1705d834d`; unrelated NDDO and
  planning changes remain outside that commit.
- W1 Gamma workspace slice: restricted mio, collinear spin and 3ob retain
  H0/S/H/P and the generalized LAPACK solver across geometry evaluations.
  Internal force/TD consumers borrow the converged orbitals from that owner.
  Open-shell density and Mulliken populations reuse the existing BLAS density
  builder and quadratic diagonal contraction. Species/atom-count changes
  rebuild the workspace; unsupported-species failure followed by retry passes.
  Full CPU suite: 692/692 pass (`w1-gamma-reuse-tests.log`), including periodic,
  spin, multi-k, 3ob, additions and TD-DFTB regressions. Allocation tracing has
  not yet been run; analytic-force scratch and multi-k buffers still allocate.
- W1 predictor boundary validation: the existing scripted trajectory now
  interleaves initial-force requests, trial evaluations and post-SCC failures.
  Subsequent normal SCC iteration counts match the unperturbed history exactly,
  with final energy within 1e-10 eV. DFTB2's full case passes in 2.60 seconds
  (`w1-history-tests.log`). This exercises request handling, not real MD or
  Minimize orchestration; those composition checks remain pending.
- W1 is not complete: finish scientific workspace retention and allocation at
  the component build/rebuild boundary, then validate MD/Minimize composition.
  The independent build uses `/tmp/atv-se-rust-1.99.1` after a concurrent
  global toolchain replacement interrupted compilation. A later external
  change to molrs/rust-toolchain.toml is preserved.
- W2-W6 pending. molq repair is complete; live GH200 submission, observation,
  completed-job status and logs have all been exercised successfully.

### W0 evidence census

All section counts below are from the VIB-enabled executable actually run.
External evidence is limited to the documented pinned systems/quantities.

| Method | Passed sections | Evidence and remaining gap |
| --- | ---: | --- |
| AM1-BCC | 1 | Archived AmberTools charges |
| AM1 | 4 | MOPAC energy/formation energy; structural Hessian only |
| RM1 | 1 | MOPAC water formation energy |
| DFTB0 | 2 | DFTB+ energy and full forces; CH4/NH3 pins still needed |
| DFTB2 | 13 | DFTB+ molecular/spin energy and forces; several stress/addition checks are structural/formula checks |
| DFTB3 | 3 | Mixed mio SK + 3ob Hubbard derivative; reduction/FD/snapshot, no standard full-Hamiltonian external pin |
| GFN0 | 3 | Archived xTB energy/forces/Hessian |
| GFN1 | 2 | Archived xTB energy/forces; GFN1 Hessian also runs in GFN2 case |
| GFN2 | 6 | Archived xTB/tblite; current run green, tolerance/vector coverage remains incomplete |
| IPEA | 4 | Archived xTB energy/forces/IP/EA/Hessian; charged occupation model still restricted |
| TD-DFTB | 1 | Analytic structural limits and snapshots; no independent absolute excitation pin |

Later milestones: DFTB2 periodic/stress, spin, multi-k and additions; then
NDDO/GFN0/GFN1/IPEA; GFN2 only after its baseline is trustworthy; DFTB3 only
after standard Hamiltonian closure. New physics and multi-GPU are out of this
first milestone. No push or long-running automation is authorized.
