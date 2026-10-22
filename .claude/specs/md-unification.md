# MD unification — active implementation

Goal: one MD execution model for CPU and CUDA. The user authorized the complete
plan below; do not report it complete until every item is done.

## In the tree

- BigInt MD budgets/results/dispatch, ramp clocks and snapshot step
  propagation; negative MD budgets rejected (residual `int` counters: CLAUDE.md
  › Known debt).
- Shared scalar laws in `core/dynamics_math.h` (NHC half-step, pressure /
  Berendsen, ramps, thermal DOF); MTK denominators use the actual atom count
  independently of constraint DOF.
- CUDA NPT chains/scalars device-resident (Thrust history); CUDA initial
  temperature rescaling via CUB KE + cuBLAS. CPU MPI velocity initialization
  removes the global COM with distinct rank streams.
- CUDA backend publishes a device Box after box writers; atom/pair/neighbor
  kernels consume the device cell; neighbor reference cell and XYZ snapshot cell
  are frozen copies.
- CUDA lifecycle and initial forces run through core
  `detail::ComponentLifecycle` + `Plan::prepare_forces` (with CUDA
  `initialize_forces` opt-in). **CPU does not use them** — it runs dev's
  `cpu::Runtime` ledger and `prepare_run`. The two are not shared.

## Still required

0. Re-unify lifecycle and initial-force preparation: port CPU onto the core
   helpers (extending the ledger with the alive token and DriverNode identity
   counting CPU needs) or move the helpers into `cuda/driver/` and keep two
   documented implementations. `core/` must not claim "shared" until then.
1. `Frame::edit/commit` with an enforced external writable-view boundary,
   revisions, execution exclusion, targeted cache/history invalidation (also
   closes CUDA's missing host-edit invalidation between chunks).
2. Move all classical model parameters and evolving biases into typed Frame
   blocks/meta, including C++/Python/tasks/molrs/MPI/device transfer call sites.
3. Driver-lifetime trajectories independent of Frame object address; fix the
   molrs transient-frame lifecycle; unify setup / segment epilogue / defaults /
   protocols.
4. CUDA conditional WHILE execution, device criteria and cumulative clock;
   neighbor-list overflow service and fully device-controlled rebuilds;
   graph-compatible long-range / packed-force consumers.
5. Remove the separate `MultiGpu` production MD loops; shared Plan, bounded
   multi-device graphs, device count / communication / termination protocols.
6. Full CPU-only domain/AIMD, 1/2/4 GPU, precision and performance/profiling
   coverage; final primitive census; docs/roadmap updates.

## Validation expectations

- MD-only CPU build with all enabled regressions; CPU initial-force/chunk
  regression; MPI initialization units at 2/4 ranks.
- GH200 double and mixed: MD primitive targets, NPT 80-step and 1/2-step
  regressions, CUDA initial-force regression, memcheck on primitives and
  racecheck on NPT launch capture.
- Remote work stays in its own snapshot on Arrhenius
  (`/home/jicli594/work/molcrafts/Atomiverse-md-nvidia`).
