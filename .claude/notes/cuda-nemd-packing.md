# CUDA NEMD (molten NaCl, GearPC) and packing runs on one GH200

Harness: `benchmarks/perf/md/nacl_pack.py` + `nacl_pack.sh` (512 ions, the
`bench_nacl_ewald` paper setting, through the Python drivers). Tests:
`tests/cuda/dynamics/test_nacl_terms.cu`, `test_gear_pc.cu` (CPU as oracle).
Numbers are not kept in git; rerun it.

## What the CUDA twins are

- `cuda::PairBorn`, `cuda::md::GearPC` (the CPU `GaussThermostat` /
  `ConfigurationalThermostat` parameter objects), `cuda::ConfigurationalTemperature`;
  `CoulLong`, `KSpaceEwald` are `CurvatureSource`s and `FixEfield` an
  `ExternalDrive` (`cuda/component/curvature_protocol.cuh`, negotiated in
  `Driver::build` as on the CPU). The curvature of a pair term is one more
  cluster-tile walk (`ClusterTileEngine::curvature`), not a second engine.
- Python selects the device the way AM1 / SteepestDescent do: the factory
  takes `device=` (`GearPC(dt, law, device="cuda")`,
  `ConfigurationalTemperature(every, device="cuda")`) and `MD(device=...)`
  refuses an object of the other device. `Born`, `KSpaceEwald`, `CoulLong`,
  `FixEfield`, `NeighborList` are parameter mirrors and need nothing.
- Agreement with the CPU in FP64 on the 64-ion jittered melt: forces,
  energy, virial and the curvature rows to ~1e-13 relative per term; a
  40-step driven trajectory to ~5e-13; each Gauss constraint to 1e-15.
  Chunked and graphed runs are bit-identical to the eager one.

## Where the two devices differ (by design or inherited)

- (Fixed: both devices now continue.) A `run()` on the molpy.Frame the
  previous one returned, unedited, continues the Gear history, ξ, the step
  count and the recorders' step axis (LAMMPS `run N pre no`); an edited
  frame restarts what the edit invalidates. The CPU handle keeps its engine
  frame across calls (python/src/bindings/bind_md.cpp `run_local_`). Another
  molpy.Frame starts an independent simulation on both devices.
- (Fixed.) `InitVelocity` makes one draw on both devices
  (`core/velocity_create.h`: mt19937 by tag, host side), so a seed gives
  the same velocities; it draws once per trajectory, at build — neither a
  continued `run()` nor an edited frame redraws.
- The flow thermostat's species table is a fixed device array
  (`kMaxSpecies = 8`).
- A list radius above L/2 (the paper's cutoff L/2 plus a 2 Å skin) is
  correct on CUDA as long as the force cutoff is ≤ L/2: the tile kernels
  take the minimum image per pair, there are no ghost rows.
- Thermostat faults (zero constrained kinetic energy, Laplacian ≤ 0) are
  device status bits, thrown when the run returns.
- (Fixed.) A second `set_integrator` of the same kind replaces the first
  on both devices (LAMMPS `fix` with an existing ID).

## Cost and packing (GH200, FP64, 512 ions)

- One process: ≈0.43 ms/step Gauss, ≈0.87 ms/step configurational, the same
  in the liquid and in the fluid (cutoff L/2: every pair is in the list at
  any density; 447 k-vectors at any L). A step this small is launch latency,
  not arithmetic: alone, the card is slower than 16 EPYC threads on the
  Gauss step (0.28 ms) and about level on the configurational one (0.60 ms).
- Several processes on one card without MPS are time-sliced: the total rate
  stays that of one process (K = 16: 7.4 ms/step each, 2150 steps/s in all
  against 2340 alone).
- Under NVIDIA MPS the per-process step time grows slowly (Gauss 0.43 → 0.56
  → 0.69 → 0.83 ms at K = 1, 16, 32, 48; configurational 0.87 → 1.09 → 1.26 →
  1.48 ms), so one card gives ≈58 000 Gauss or ≈32 000 configurational
  steps/s at K = 48: what 80 and 120 single-thread EPYC cores give.
- MPS admits 48 client processes per card here; the 49th fails at context
  creation with "CUDA-capable device(s) is/are busy or unavailable". More
  simulations per card would need several in one process.
- Each process holds one CPU core. A GH200 node has 72 cores per card, so
  ask for `-c K` with `--gpus 1`.
- Python on CUDA always replays step graphs (`MD.enable_timing` refuses);
  a per-component profile needs a C++ harness.

## Running a pack

- `nacl_pack.sh OUT PYTHON` is the whole recipe: a private MPS daemon
  (`CUDA_MPS_PIPE_DIRECTORY` in a temp dir, `nvidia-cuda-mps-control -d`,
  `echo quit | nvidia-cuda-mps-control` on exit), one process per simulation
  bound to its own core, `OMP_NUM_THREADS=1`, `wait`.
- The library logs on stdout: a launcher that reads a worker's stdout must
  skip to its own marker line.
- The rock-salt start of the fluid has ΣΔU < 0; melt with a Gauss
  thermostat first (the bench does 1000–2000 steps), then start the
  configurational thermostat from that frame.
- A run-only job needs the build job's environment
  (`tools/gpu_job.sh`: module, CUDA 13.2 `lib64`, the pthread OpenBLAS in
  `LD_LIBRARY_PATH`), or the extension fails on `libfftw3.so.3`.

## Traps met while porting

- `cudaGetLastError` is sticky: a `cudaFree` of a stale pointer anywhere
  surfaces as "invalid argument at driver.cu (publish_box_)" in the *next*
  backend that is built. In a test, a borrowed component
  (`set_observer_ref`, `set_integrator_ref`) declared after the driver is
  destroyed before it and `Backend` then calls `destroy` on the dead object.
  Declare the borrowed object first.
- A conservation gate does not notice a wrong reference velocity in the
  projection: the projection is repeated every step and converges
  quadratically. The constraint definition is tested through the measured
  Σ m w² (and through the trajectory against the CPU), and only with a
  start that has net momentum, since the barycentric mean is otherwise 0.
- The CUDA sources can be syntax-checked without nvcc: strip `<<<…>>>`
  launch configurations and compile with host clang against stub
  `cuda_runtime.h` / CUB headers. It caught every compile error of this port
  before the first GH200 build; it says nothing about device semantics.
