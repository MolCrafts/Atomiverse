# MD unification — active implementation

Base: upstream 3519ea6; retain all existing NVIDIA primitive and initial-force
fixes in this worktree. The user authorized the complete five-part plan.

Implemented and checked (the complete approved plan is still unfinished):
- Shared unique-instance lifecycle ledger, reverse release, nested references,
  failed-build rollback; shared unmount removes every binding of an instance.
- Shared Plan initial-force preparation and CUDA opt-in hooks; initial-force
  Langevin samples differ from the first ordinary step; NPT seeds measured KE.
- BigInt MD budgets/results/dispatch, ramp clocks and snapshot step propagation;
  negative MD budgets rejected. Remaining per-domain counter census is needed.
- Shared scalar NHC half-step, pressure/Berendsen laws, ramps and thermal DOF.
  MTK denominators use actual atom count independently of constraint DOF.
- CUDA NPT chains/scalars entirely device-resident, Thrust history storage;
  CUDA initial temperature rescaling uses existing CUB KE and cuBLAS scaling.
  CPU MPI initialization removes global COM, with distinct rank streams;
  one-owned-atom/rank temperature and momentum units pass at 2/4 ranks.
- Backend device-box publication after box writers; device geometry consumed by
  atom/pair/neighbor kernels; device neighbor reference cell and frozen XYZ cell.
- NPT 1/2-step trajectories migrated from unit tests into the existing
  reg_npt_lj_mtk case; unit launch capture test proves no host readback.

Still required (do not report the approved plan complete):
1. Frame::edit/commit and enforced external writable-view boundary, revisions,
   execution exclusion, targeted cache/history invalidation.
2. Migrate all classical model parameters and evolving biases into typed Frame
   blocks/meta, including C++/Python/tasks/molrs/MPI/device transfer call sites.
3. Driver-lifetime trajectories independent of Frame object address; fix molrs
   transient-frame lifecycle; unify setup/segment epilogue/defaults/protocols.
4. CUDA conditional WHILE execution and device criteria/cumulative clock;
   neighbor list overflow service and entirely device-controlled rebuilds;
   graph-compatible long-range/packed-force consumers.
5. Remove separate MultiGpu production MD loops, use shared Plan and bounded
   multi-device graphs, device count/communication/termination protocols.
6. Full CPU-only domain/AIMD, 1/2/4 GPU, precision, performance/profiling coverage;
   final primitive census and docs/capability updates.

Validation:
- `/tmp/atv-md-unified-host-build.log`: MD-only complete build, including all
  enabled regression binaries. Four minimizer test gates and the GAFF protocol
  regression gate corrected; a molrs-only CUDA fixture is gated on MOLRS.
- `/tmp/atv-md-unified-host-tests.log`: all 135 MD-only CPU tests passed.
- `/tmp/atv-md-unified-last-host-tests.log`: latest preparation/MTK units pass.
- `/tmp/atv-md-unified-initial-force.log`: CPU initial-force/chunk regression;
  first error 3.47e-18, continued error 6.94e-18, chunk difference zero.
- `/tmp/atv-architecture-pw`: MPI/HF/PW/full CPU library and Python extension
  build; nesting/force-request/SCF wiring units pass. Latest MPI initialization
  units pass at 2/4 ranks (`/tmp/atv-md-unified-global-com-tests.log`).
- molq Arrhenius job 4173953 succeeded. GH200 double/mixed each passed 10
  selected targets; 80-step plus 1/2-step NPT regressions and CUDA initial-force
  regression passed. Mixed 80-step volume relative difference 9.2962e-8.
  Primitive memcheck and NPT launch-capture racecheck reported zero errors.
  Local copy: `/tmp/atv-md-cuda-review/validation-unified-final2.log`.
- GPU snapshot contains the device/lifecycle/MTK changes. The subsequent CPU-only
  global-COM/rank-stream fix is independently built/tested locally under MPI.
- Remote work is confined to the own snapshot
  `/home/jicli594/work/molcrafts/Atomiverse-md-nvidia`.

No commits, merges, or deployment have been made. Keep the existing working-tree
NVIDIA primitive changes and ignore unrelated .DS_Store files.
