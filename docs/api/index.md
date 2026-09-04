# API Reference

This page is the front door to the generated API reference. It is not meant to
duplicate the class and file pages that Doxygen already generates. Its job is to
help you find the right part of the reference quickly.

All public code examples in Atomiverse assume **C++20 or later**.

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

Start with these headers (`core/` holds the device-independent contracts; the
CPU loop node lives under `cpu/driver/`):

- [`cpu::Driver<PhaseEnum>` — the loop node every CPU preset derives from](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/driver.h)
- [`Driver<PhaseEnum, DeviceTag>` — the backend-independent dispatch template](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/driver_template.h)
- [`Plan`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/plan.h)
- [`Phase` enums + `slot_of` / `slot_count`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/stage.h)
- [`Verdict` / `ConvergenceMetrics`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/verdict.h) — the vocabulary termination is decided in
- [`IterationStatus`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/iteration_status.h) — the per-loop status a driver owns

### Decide when a run stops

The loop node (`cpu::Driver<P>`, which every CPU driver derives from) has two
policy slots. Both are user-definable from outside the library:

- [`cpu::Criterion`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/component/criterion.h) —
  *should the loop stop?* A Component the driver polls once per iteration:
  `driver.add_criterion(c)`, plural. Ships `ForceConverged`,
  `AllReplicasConverged`, `StopOnNaN`
  ([`cpu/driver/criteria.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/criteria.h)).
- [`cpu::OutcomePolicy`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/outcome_policy.h) —
  *does this iteration count?* `driver.set_outcome(p)`, singular. Ships `Metropolis`.
- [`cpu::DriverNode`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/driver/driver_node.h) —
  the erased node interface that lets a Plan slot hold a nested driver instead
  of a component; the child runs on the parent's runtime. There is no separate
  `SubDriver` wrapper: a Driver *is* an executable node.

### Run molecular dynamics and minimization

Start with these CPU driver and component headers (the ones the tutorials
use; each has a CUDA counterpart under `cuda/` with the same name):

- [`cpu::md::MD`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/dynamics/md.h)
- [`cpu::md::VelocityVerlet`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/dynamics/velocity_verlet.h)
- [`cpu::md::NeighborList`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/neighbor/neighbor_list.h)
- [`cpu::PairLJCut`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/potentials/pair_lj_cut.h)
- [`cpu::minimizer::Minimize`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/minimize/minimize.h) + `LBFGS` / `CG`

The CUDA equivalents are `cuda::md::MD` (`cuda/dynamics/md.cuh`),
`cuda::PairLJCut`, `cuda::minimizer::Minimize`, etc. — same API, GPU-resident.

### Run electronic-structure calculations on the CPU

Start with these CPU headers:

- [`methods.h` — RHF / UHF / RKS / UKS (`run(frame)`, `energy()`, `converged()`, `determinant()`)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/methods.h)
- [`slater_determinant.h` — SlaterDeterminant (per-spin orbitals, `homo()`/`lumo()`, densities, `spin_contamination()`)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/slater_determinant.h)
- [`scf_enums.h` — SpinType / MeanFieldType / InitGuess](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/scf_enums.h)
- [`scf_guess.h` — SCFInitialGuess](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/scf_guess.h)
- [`gradient_result.h` — GradientResult (the `forces()` return)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/gradient_result.h)
- [`XCType`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/xc_functional.h)
- [`cpu::semiempirical::GFN1`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/semiempirical/gfn1.h) — GFN1-xTB organic energy and analytic forces (H/C/N/O; Evaluate)

### Run SCF workflows on the CUDA backend

CUDA SCF presents the same public shape as CPU —
`cuda::scf::RHF{basis}.run(frame)` / `.forces(frame)`, `Frame` in and out.
Start with these headers:

- [`cuda::scf::RHF`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/scf/rhf.cuh)
- [`cuda::scf::SCFEvaluator`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/scf/scf_evaluator.cuh)
- [`cuda::Evaluate`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/evaluate/evaluate.cuh)
- [`cuda::single_point`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cuda/evaluate/single_point.cuh)

### Record output and analyze results

Start with these types:

- [`cpu::Thermo`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/record/thermo.h)
- [`cpu::write_frame_xyz` / `cpu::write_frame_zarr` (molrs-backed single-frame writers; the latter writes a `*.mrec` checkpoint)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/io/frame_writer.h)
- [`cpu::MrecRecorder` (streaming `*.mrec` MD trajectory Recorder — `set_record`, step stride, `time` axis from the integrator; needs the molrs TrajectoryWriter bridge)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/record/mrec_recorder.h)
- [`cpu::MrecWriter` (bare streaming writer beneath the Recorder)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/record/mrec.h)
- [`ATV::RDF`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observe/rdf.h)
- [trajectory analysis (`cpu::compute::*` batch computes + streaming accumulators)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observe/trajectory_metrics.h)
- [`cpu::XyzRecorder` (MD trajectory Recorder — `set_record`, step stride)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/record/xyz_recorder.h)
- [`cpu::XYZ` (bare XYZ writer beneath the Recorder)](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/record/xyz.h)
- [`mulliken_charge` and related helpers](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observe/mulliken.h)
- [`DensityGrid`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/observe/density.h)

## When To Leave The API Reference

If you need a conceptual explanation of what the software does, go back to the
[User Guide](../user-guide/index.md).

If you need to understand why the code is structured this way or where to place
a new feature, switch to the [Developer Manual](../developer/index.md).
