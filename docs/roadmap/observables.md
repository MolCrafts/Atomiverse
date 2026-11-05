# Observables and Analysis

Tutorial: [Trajectory analysis](../tutorial/ch13-observables.md).

Census of `src/cpu/observe/`. Observers **compute**; Recorders write I/O —
never both. Trajectory math lives in **molpy**; C++ streams frames in.
Legend: [Roadmap index](index.md).

CUDA counterparts: Mulliken, RDF, stress only.

| Observable | Where | Anchoring |
| ---------- | ----- | --------- |
| RDF g(r) | `RDF` (+ CUDA); math in molpy | **No ATV e2e regression** — molpy-side only |
| MSD / VACF | Asynchronous observers (`cpu::md::AsyncObserver`) | Unit tests; lossless under backpressure (curves bit-identical behind a one-frame budget and a slow consumer); numerics molpy-side |
| Trajectory-metric accumulators (MSD / VACF / RDF, streaming only) | `trajectory_metrics.h` PIMPL | Unit tests; numerics molpy-side |
| Electron density grid | `DensityGrid` | Via cube Recorder unit test |
| Species density profile | `SpeciesDensityProfile` | Unit + EDL box regression |
| Multipole / dipole | `MultipoleObserver` | PySCF-pinned |
| Virial / kinetic stress | `Virial` | Indirect via NPT/NPH regressions |
| Mulliken | CPU + CUDA | MOPAC / GPW DFT+U |
| Löwdin / MK / CHELPG | CPU | PySCF `h2o_charges_pyscf` |
| Widom μ_ex | `Widom` | Analytic HS free volume |
| Voronoi volume | `VoronoiVolume` | MetalWalls |
| Ψ(z) profile | `PotentialProfile` | Analytic planar Poisson |
| Electrode Q(t) | `ElectrodeChargeObserver` | LAMMPS |
| Capacitance ⟨δQ²⟩ | `CapacitanceChargeFluctuation` | Limmer closed form |
| C_QQ(τ) / elastance | electrode observers | **No standing regression** — owe external pin |
| Classical polarization P | `ClassicalPolarization` | MetalWalls finite-field |

## Recorders

Recorders write outside the process at `end_of_step`. Marks: ✓ public API
+ unit tests, ◐ public API without its own unit test, ○ absent.

All MD recorders on both backends follow the LAMMPS output step convention
(src/verlet.cpp `Verlet::run`, src/output.cpp `Output::setup` /
`Output::write`): a record's `step` is the number of completed MD steps and
holds the state after exactly that many steps (`time = step · dt`). Every
`run()` first writes the current state when its step is on cadence (thermo:
always), the loop writes after every k-th completed step (thermo also on the
last step), and a dump never writes a step twice — so a fresh run of n steps
at cadence k records 0, k, …, n, and continued runs extend the axis without
repeating the boundary step. Thermo cadence 0 prints the first and last step
of each run only (LAMMPS `thermo 0`). In a minimization (src/min.cpp
`Min::setup` / `Min::run`) a step is an optimizer iteration: the initial
state is written as step 0, a criterion stop before the iteration budget
writes the final state off cadence, and no step is written twice — for
`Thermo`, `XyzRecorder`, `MrecRecorder` and the minimizer writers
`TrajectoryWriter`, `OptLog`, `DensityCubeWriter`, `OrbitalCubeWriter`
(unit tests against independent minimizations; cube writers in the HF
build). A single-pass `Evaluate` (LAMMPS `run 0`) writes its one state as
step 0 whatever the cadence.

| Recorder | CPU | CUDA | Python | Anchoring |
| -------- | --- | ---- | ------ | --------- |
| `Thermo` (`set_thermo`) | ✓ | ✓ async (capture buffer) | both | `ThermoStyle` unit tests; step convention (setup line, completed-step labels, last step off cadence, continued runs) against independent runs (CPU, CUDA); Python CUDA print test |
| `XyzRecorder` (extxyz via molpy) | ✓ | ✓ async (capture buffer) | CPU only (`XYZ`) | CPU unit tests; step convention against independent runs (CPU, CUDA) |
| `MrecRecorder` (`*.mrec`, molpy `MrecWriter`) | ✓ (MPI: rank-0 gather) | ✓ | both (`MrecRecorder`), MPI launcher (wire ID 39) | Unit tests: cadence, tag-ordered rows, frame-0 content, identity columns type-only / species-only / both round-tripped through `cpu::from_molpy`, `status` section (CPU); count / cell / image flags on a free flight, CPU ⇄ CUDA column parity incl. `energy_pair` of a two-step LJ run, identity columns (three cases), a held writer never stalls the loop while the buffer has room, a slow writer behind a two-frame and a one-frame capture budget blocks the run yet loses no frame (each equal to an independent run; blocking recorded in `status` and `capture_stats()`), a failing writer stops the run promptly with a failed `status`, a normal run drains without blocking, a budget below one frame is refused when the run starts, a closed recorder refuses a later run (CUDA); step convention — steps 0…n, frame `s` equal to an independent `s`-step run, time = s·dt, boundary step once on continued runs (CPU, CUDA, pytest); task codec + materializer; MPI `DispatchedMdTaskRecordsTheWholeSystemOnRankZero` at 1/2/4 ranks; pytest read-back through `molpy.io.mrec.MrecReader` / `MrecFile.status()` incl. CPU ⇄ CUDA frames |
| `PressureSeries` (SLLOD −P*) | ○ | ✓ | CUDA (`set_pressure_series`) | Step convention: rows 0, k, …, n, each the −P* of an independent CPU run of that many steps |
| Density cube | ✓ | ○ | — | CPU unit test |

`MrecRecorder` writes the same frames on both backends: x/v/f, charge,
int32 images, tag, mass, cell, energies (total and per term) and virial,
`step` and `time = step·dt`, rows in tag order, plus each identity column the
Frame carries — force-field type as `type_id`, species as `atomic_number`; an
absent column stays absent (never zero-filled, never a type written as Z).
`close()` ends the store with the molrec `status` section (`state` —
`"failed"` with an `error` when an append or commit failed — `stage`,
`frames_written`, `finished_at`, and on CUDA a `buffer` object:
`budget_bytes`, `frame_bytes`, `peak_bytes`, `blocked_episodes`,
`blocked_seconds`). Under the MPI launcher every rank gathers its owned rows
to rank 0 in tag order (`md::mpi::gather_for_root_io`) and rank 0 alone
writes. CUDA path — asynchronous but lossless (iron law 10 and its one
sanctioned exception), through the driver's shared capture buffer (below).
Open: no multi-GPU path; molpy's own chunk buffer (`flush_every`) sits
outside the capture budget. The `status` group is written by the engine in
the spec's reference binding because molcxx has no status door yet.

## Asynchronous capture (recorders and online analysis)

Every asynchronous consumer of a driver — the CUDA recorders (`MrecRecorder`,
`XyzRecorder`, `Thermo`, `md::PressureSeries`, `DensityRecorder`), the CUDA
observers (`RDF`, `MullikenObs`) and the CPU online-analysis observers (`RDF`,
`MSD`, `VACF`) — derives from one base per backend (`cuda::AsyncConsumer`,
`cpu::AsyncConsumer`) and reads its steps from ONE shared frame buffer per
driver (`cuda::CaptureBuffer`, `cpu::CaptureBuffer`; both run the host-side
`ATV::CaptureLedger`, core/capture_ledger.h, so admission, fan-out, release
and failure semantics are identical). The CPU trajectory / thermo recorders
write synchronously on the stepping thread and do not use it.

- **Capture once.** A step any consumer needs is copied once into a buffer
  slot (CUDA: frozen on the compute stream into pinned host-mapped memory by
  graph-capturable kernels; CPU: copied on the stepping thread) as an
  independent FP64 copy of the union of the consumers' columns (+ per-consumer
  payload blocks, e.g. the SCF density). Every consumer due on that step reads
  the same immutable copy on its own worker thread, in step order, and
  releases it; the slot frees when the last one has.
- **Byte budget.** One budget per driver bounds every frame in flight across
  all consumers: `Driver::set_capture_buffer_bytes` (C++), Python
  `MD(capture_buffer_bytes=…)`, MPI launcher `MDTask::capture_buffer_bytes`;
  default the CMake cache variable `ATV_CAPTURE_BUFFER_BYTES` (256 MiB).
  Allocated once when a run starts (re-laid out only at a run boundary after
  the consumer set or the population changed), never in the loop; a budget
  below one captured frame is refused then, naming both sizes.
- **Lossless backpressure.** When the budget is full the stepping thread
  waits on a condition variable (CUDA: in `before_launch`, before the step is
  enqueued, never inside a graph) while the consumers keep draining;
  consumers never wait on the producer, and nothing is dropped. With space,
  admission is host counters only (iron law 10's zero-sync fast path). The
  first wait of a run is logged (naming the consumers behind), the run's waits
  are summarised at its end, and the cumulative blocked time, blocking
  episodes and peak occupancy are returned by `Driver::capture_stats()`
  (Python `MD.capture_stats()`) and written to each `*.mrec` store's
  `status.buffer`.
- **Failures.** A consumer that throws stops the run: the next step (a host
  flag test) or the end-of-run drain raises `std::runtime_error` /
  `RuntimeError` naming the consumer, the step and the cause; the failed
  consumer's later captures are released unread, so the producer never waits
  on space that will not free; `*.mrec` stores record `state = "failed"` with
  the error.
- **Completion.** Normal completion drains every consumer before `run()`
  reports success (CUDA: in the end-of-run retrieve; CPU: at the end of
  `MD::run` / the driver teardown).
- **Step labels** follow the LAMMPS output convention above for every
  consumer (initial state when on cadence, every k-th completed step, never a
  step twice; LAMMPS e8c4c278a8). An analysis observer at cadence k therefore
  samples steps 0, k, …, n.

Tests: `tests/cpu/asynchronous/test_capture_buffer.cpp`,
`tests/cpu/observe/test_async_observers_lossless.cpp`,
`tests/cuda/asynchronous/test_capture_buffer.cu`,
`tests/cuda/observe/test_rdf_molpy.cu` (`CurveIdenticalUnderBackpressure`),
`tests/cuda/record/test_mrec_recorder.cu`, `python/tests/test_mrec_recorder.py`.

## Claim gap

Every observable owes **one analytic pin** of its own. Worst offenders: RDF
(ideal-gas g(r)=1 or FCC peak), MSD (free-flight / Einstein), VACF
(Langevin exponential). molpy’s tests validate math, not this engine’s
streaming path (cadence, unwrap, masks).
