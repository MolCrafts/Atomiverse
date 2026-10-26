# Minimizer backend alignment

CPU and CUDA minimization are independent of MD. Shared infrastructure lives
in the component/driver layers, with each preset retaining its own phase order.

Implemented:
- Shared CUDA accumulator initializers in `cuda/component/zero.cuh` and
  `src/cuda/component/zero.cu`, compiled independently of MD.
- Backend initial force preparation for minimization, including zero-budget runs;
  no optimizer or end-of-step output in preparation. Negative budgets reject
  before setup.
- CUDA criteria are Components, supporting owned/borrowed registration, scheduled
  measurement, lifecycle-only registration, shared unmounting, and dispatch mutation
  exclusion. Declared-convergence requirements use conjunction, with stop guards neutral
  and stop requests taking priority. Guard-only minimizers install ForceConverged.
- CPU optimizer slot clears after unfix, permitting replacement.
- First energy measurement cannot satisfy delta-energy convergence.
- Public minimizer budgets, optimizer counters and records are 64-bit, including
  Python and task serialization/materialization. OptRecord lives in core.
- CUDA LBFGS captures actual initial origin energy for its first Armijo trial.
- CUDA minimizer exception cleanup uses the backend lifecycle ledger; teardown
  drains execution streams before destroying component buffers.

Still open before full architecture alignment:
- CPU and CUDA presets still own separate host iteration loops. CUDA ConvCheck
  reads metrics back each iteration; whole-loop device execution is not implemented.
- Nested physical CUDA workflows and trial-force services do not yet match CPU Runtime.
- CUDA LBFGS line-search trials and CPU exact line-search execution differ.
- Generic nested DriverNode budgets still use int; their migration includes
  electronic and other workflow overrides, beyond the public minimizer entry.
- Both minimizers retain fresh-optimization semantics per run, rather than MD
  trajectory continuation. No history-continuation API change is made here.
