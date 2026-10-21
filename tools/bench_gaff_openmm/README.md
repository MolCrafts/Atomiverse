# GAFF PEO: Atomiverse CUDA versus OpenMM CUDA

`bench_gaff` and `compare.py` consume the same LAMMPS `units real` data and
coefficients. The harness checks the initial potential energy and every force
component before accepting timing. OpenMM evaluates its own `System`; the
reference numbers are never generated from Atomiverse's formulas.

Build `bench_gaff` with CUDA, MD, potentials, neighbor lists, KSPACE and
benchmarks enabled. No molrs installation or Python binding is required for
the C++ consumer. Run all experiments in an Arrhenius GPU batch allocation:

```sh
export DATA=/path/to/peo/system.data FF=/path/to/peo/system.ff
export RUN_SCRIPT=tools/bench_gaff_openmm/run.sh
sbatch -A <gpu-account> tools/bench_gaff_gpu_job.sh
```

The job uses an isolated `.openmm-env` and records the installed package
versions. `OPENMM_ENV` can select an existing environment. Compilation and
package installation happen before timing. Each engine process has a
10-second deadline; three timed samples use their median. `STEPS` defaults to
100 and `REPLICAS` to `1 3` (replication
along each axis). `PRECISION=mixed BUILD=build-gaff-mixed` builds a separate
mixed-precision binary. Keep both engines on the same GPU.

Matched defaults: unconstrained NVE, 0.25 fs, zero initial velocities, 10 warmup
steps, unshifted Lorentz–Berthelot LJ without tail correction, cutoff 9 Å,
Ewald alpha 0.35 Å⁻¹, order-5 PME, mesh dimensions rounded from 1 Å spacing to
2/3/5/7-smooth sizes. Amber graph-distance exceptions use LJ `(0,0,1/2)` and
Coulomb `(0,0,5/6)`, minimum-image geometry, and no exception cutoff.
Bond, angle and torsion forces use periodic geometry in both engines.
The benchmark reserves a neighbor skin of 8% of the cutoff (0.72 Å), matching
OpenMM CUDA's default padding. This does not change the force cutoff.

Replication follows minimum-image bonded paths and maps boundary-crossing
endpoints to adjacent replicas, preserving bond/angle/torsion geometry.

Supported input: contiguous atom/type IDs, orthorhombic PEO data, harmonic
bonds/angles, Fourier torsions with multiplicities 1–4 and phases 0 or pi.
Those torsions map exactly to the existing `DihedralMultiHarmonic` polynomial.
Unsupported sections, impropers, coefficients or phases raise errors. This
harness does not establish support for every GAFF molecule or NPT workflow.

OpenMM's leapfrog velocity starts one half-kick behind the zero full-step
velocity so the positions follow the same velocity-Verlet trajectory as ATV.
Both timed calls synchronize and retrieve final state. Initial forces/energy
and final periodic positions are checked (final position limits: 1e-4 Å double,
1e-2 Å mixed over the short trajectory). Neighbor capacity uses the normal
build-time density/initial-count headroom; overflow throws before fill,
with no allocation in timed steps. The separate profile
run enables GPU event timing; its wall time is not used in the comparison.
CUDA PME uses real FFTs with a half spectrum. Its fixed-cell influence
coefficients are prepared in build, with conjugate averaging at Nyquist modes
to preserve the full-grid real potential and energy. The separate
`cuda-pme-half-spectrum` regression checks even/odd and skew meshes against
an independent full Fourier sum and analytic force gradient.
Spline/grid/FFT precision follows the selected work precision;
energy and self-energy reductions remain double in both tested modes.
Screening and exception corrections retain the library special functions.
System-level acceptance thresholds are identical across optimization experiments.
Outputs under `logs/gaff-openmm/` include raw forces, energies, actual OpenMM
mesh, input/binary SHA-256 hashes, process durations and ms/step ratios.
`benchmark_seconds` records the whole comparison case, including both engines.
A timing is accepted only when all four initial/final energy, initial force
and final position checks pass (`accepted: true` in the JSON).
Double-mode acceptance uses maximum force error divided by maximum reference
force below `3e-5`, and energy error below `max(1e-3 eV, 3e-5 |E|)`.
Mixed-mode inputs (coordinates and cell) use a shared binary lattice that is
exactly representable in float32 in both Å and OpenMM's nm. Merely rounding
Å values to float32 introduces a second, different rounding in nm. The lattice
is `5*k*2^e` Å (`k*2^(e-1)` nm), with `|5*k| < 2^24`; its spacing is recorded
in the JSON and checked for exact representability before running.
Thus initial parity compares the same positions. Mixed-mode force tolerance
remains `1e-4`; no acceptance threshold is relaxed for a speed experiment.
Both engines retain a float coordinate plus a residual and use double
velocities/integration in mixed mode. ATV also retains double masses and a
canonical double cell. Its force kernels and FFT remain FP32; energies are
accumulated in FP64. This aligns motion-state semantics, not atomic reduction
order or bitwise trajectories: ATV force atomics remain floating point,
whereas OpenMM uses fixed-point force accumulation. The binary reports
`motion_bytes=8 position_residual=1`, which the harness verifies.
Non-quantized sub-FP32 increments, boundary wrapping, import/export, optimizer
history/rollback and asynchronous coordinate snapshots have separate focused
unit checks; the shared initial lattice is only for the external force comparison.

Architecture reference: OpenMM source commit
`3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e`, CUDA cluster AABB hierarchy,
nonbonded parameter fusion and sparse exception handling. The implementation
uses ATV's own CSR, masks, Frame and Component lifecycle. The fused force path
packs nonempty J columns across cluster boundaries only when neighbor masks
are rebuilt; timed force calls reuse the build-reserved packed buffers.

## Speed and memory scaling

Submit `RUN_SCRIPT=tools/bench_gaff_openmm/run_scaling.sh TARGETS=bench_gaff`
with the same `DATA`, `FF`, `BUILD` and `PRECISION` settings as above.
Set `MEASURE_MEMORY=1` to collect the separate memory probes.
The default cubic replications `1 2 3 4` give 1,840, 14,720, 49,680 and
117,760 atoms at fixed density. Each point uses 100 steps, 10 warmup steps,
and three samples. Initial/final scientific acceptance is unchanged.

`--measure-memory` runs an external
[`nvidia-smi`/NVML monitor](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
with a 100-ms
period. Measurements are filtered by engine PID and include the CUDA context
and temporary allocations during setup. The observed GPU peak can miss brief
transients; it is not an allocator-exact high-water mark. Host RAM is the
maximum observed `/proc/PID/status` `VmHWM`, including the C++ or Python
engine front-end; the common comparison parent is excluded. The last
sampling interval can miss a peak just before process exit.
Throughput runs have no concurrent memory monitor. With `--measure-memory`,
each engine is run a second time with the same physics and step counts for
memory measurement; its output goes to a different destination. Probe times
never replace the unmonitored throughput result. Each of the four engine
processes retains its own ten-second deadline, so the combined comparison
including replication and memory probes can take longer than ten seconds.
Step timing excludes setup. The harness rejects missing measurements or different
physical GPUs between the compared engines. Batch shells without Slurm's
CUDA visibility mask use the allocated `SLURM_JOB_GPUS` explicitly.
The parent validates replication once and writes the OpenMM child's input as
JSON, avoiding duplicate topology preparation within the short deadline.
Each engine still constructs and evaluates its own forces independently.

The accepted GH200 measurements, numerical summaries and current scaling
figure are checked in under [results/gh200](results/gh200/README.md).

After both precisions finish, render the accepted measurements with:

```sh
python tools/bench_gaff_openmm/plot_scaling.py logs/gaff-scaling --out plots/gaff-scaling
```

This produces a six-panel PNG/SVG (step time, GPU memory and host RAM for each
precision) and the numerical CSV. Matplotlib is required only for rendering.
Connecting lines are guides between measured points; no scaling model is fitted.

The plotter requires compatible precision-state, throughput-monitor and
resource-probe metadata. It rejects older measurements taken with concurrent
memory monitoring or the former float-only mixed integrator; those cannot be
combined with this protocol.

For IO/analysis precision checks on Arrhenius accounts without rustup, set
`RUST_MODULES="GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 GCCcore/14.3.0 Rust/1.91.1"`
and append `-DATV_BUILD_MINIMIZER=ON -DATV_BUILD_ANALYSIS=ON -DATV_BUILD_IO=ON`
through `ATV_CMAKE_FLAGS`. These dependencies are built before the benchmark.
`run_precision_checks.sh` runs the focused checks and short scientific pins
before the scaling series. `profile_scaling.sh` diagnoses prepared cases with
Nsight; its timings are excluded from throughput results.
