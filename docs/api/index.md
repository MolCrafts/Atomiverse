# API Reference

This page is the front door to the generated API reference. It is not meant to
duplicate the class and file pages that Doxygen already generates. Its job is to
help you find the right part of the reference quickly.

All public code examples in Atomiverse assume **C++17 or later**.

## How To Use The Generated Reference

Use the **Classes**, **Namespaces**, and **Files** tabs when you already know the
name of the type you want. Those pages are the most authoritative source for
signatures, inheritance, public members, and documented headers.

Use this page when you know the task you want to perform, but not yet the exact
class or function name.

## Entry Points By Task

### Describe a system

Start with these types:

- [`cpu::Frame`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/frame.h)
- [`Mat3`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/mat3.h)
- [`StorageReal` / `WorkReal` / `AccReal`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/precision.h)
- [`LocalInt` / `TagInt` / `ImageInt` / `BigInt`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/precision.h)

### Build staged workflows

Start with these facade headers:

- [`Driver`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/facade/driver.h)
- [`Plan`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/facade/plan.h)
- [`Phase` enums](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/facade/stage.h)
- [`Verdict` / `ConvergenceMetrics`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/facade/loop.h) — the vocabulary termination is decided in

### Decide when a run stops

The loop node (`cpu::CPUDriver<P>`, which every CPU driver derives from) has two
policy slots. Both are user-definable from outside the library:

- [`cpu::Criterion`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/component/criterion.h) —
  *should the loop stop?* A Component the driver polls once per iteration:
  `driver.add_criterion(c)`, plural. Ships `ForceConverged`,
  `AllReplicasConverged`, `StopOnNaN`
  ([`cpu/driver/criteria.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/criteria.h)).
- [`cpu::OutcomePolicy`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/outcome_policy.h) —
  *does this iteration count?* `driver.set_outcome(p)`, singular. Ships `Metropolis`.
- [`cpu::SubDriver`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/sub_driver.h) —
  put one driver in another's plan; the child runs on the parent's runtime.

### Run molecular dynamics and minimization

Start with these CPU driver and component headers (the ones the tutorials
use; each has a CUDA counterpart under `cuda/` with the same name):

- [`cpu::md::MD`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/md/md.h)
- [`cpu::md::VelocityVerlet`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/md/velocity_verlet.h)
- [`cpu::md::NeighborList`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/neighbor/neighbor_list.h)
- [`cpu::PairLJCut`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/potentials/pair_lj_cut.h)
- [`cpu::minimizer::Minimize`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/minimizer/minimize.h) + `LBFGS` / `CG`

The CUDA equivalents are `cuda::md::MD` (`cuda/md/md.cuh`),
`cuda::PairLJCut`, `cuda::minimizer::Minimize`, etc. — same API, GPU-resident.

### Run electronic-structure calculations on the CPU

Start with these CPU headers:

- [`methods.h` — RHF / UHF / RKS / UKS (`run(frame)`, `energy()`, `converged()`, `determinant()`)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/methods.h)
- [`slater_determinant.h` — SlaterDeterminant (per-spin orbitals, `homo()`/`lumo()`, densities, `spin_contamination()`)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/slater_determinant.h)
- [`scf_enums.h` — SpinType / MeanFieldType / InitGuess](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/scf_enums.h)
- [`scf_guess.h` — SCFInitialGuess](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/scf_guess.h)
- [`gradient_result.h` — GradientResult (the `forces()` return)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/gradient_result.h)
- [`XCType`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/xc_functional.h)

### Run SCF workflows on the CUDA backend

CUDA SCF evaluator (device path; host public API is still `cpu::scf::RHF` etc.)
component, not a staged driver. Start with these headers:

- [`cuda::scf::SCFEvaluator`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/scf/scf_evaluator.cuh)
- [`cuda::scf::SCFSolver`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/scf/scf_solver.cuh)
- [`cuda::Evaluate`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/driver/evaluate.cuh)
- [`cuda::single_point`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/driver/single_point.cuh)

### Record output and analyze results

Start with these types:

- [`cpu::Thermo`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/recorder/thermo.h)
- [`cpu::write_frame_xyz` / `cpu::write_frame_zarr` (molrs-backed frame writers)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/io/frame_writer.h)
- [`ATV::RDF`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observer/rdf.h)
- [trajectory analysis (`cpu::compute::*` batch computes + streaming accumulators)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observer/trajectory_metrics.h)
- [`cpu::XYZ`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/recorder/xyz.h)
- [`mulliken_charge` and related helpers](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observer/mulliken.h)
- [`DensityGrid`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observer/density.h)

## When To Leave The API Reference

If you need a conceptual explanation of what the software does, go back to the
[User Guide](../user-guide/index.md).

If you need to understand why the code is structured this way or where to place
a new feature, switch to the [Developer Manual](../developer/index.md).
