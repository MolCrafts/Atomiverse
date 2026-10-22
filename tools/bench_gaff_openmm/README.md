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
along each axis). `SKEW_REPLICAS=3` adds the same replicated system in a
tilted supercell (`compare.py --skew`: A = n·a, B = a + n·b, C = b + n·c).
Any sublattice of the base lattice tiles the identical infinite periodic
system, so atom count, density and every bonded geometry equal the
orthorhombic case while both engines run triclinic neighbor, pair, PME and
bonded paths. `PRECISION=mixed BUILD=build-gaff-mixed` builds a separate
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

Supported input: contiguous atom/type IDs, orthorhombic input data (the
replicated cell may be tilted with `--skew`), harmonic
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
CUDA PME uses real FFTs with a half spectrum. Its influence coefficients
live in device tables keyed on the published cell revision, with conjugate
averaging at Nyquist modes to preserve the full-grid real potential and energy. The separate
`cuda-pme-half-spectrum` regression checks even/odd and skew meshes against
an independent full Fourier sum and analytic force gradient.
Spline/grid/FFT precision follows `force_t`; energy, virial and
self-energy reductions are FP64 in every mode.
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
Thus initial parity compares the same positions. Mixed-mode force
tolerance is `1e-4`. Both engines now use the same mixed strategy: FP32 force
math on FP32 coordinates, an FP32 position correction reconstructed in FP64
for integration, FP64 velocities and 64-bit fixed-point force accumulation.
The ATV binary reports `real_bytes` (stored coordinate), `work_bytes` (force
math), `accum_bytes` (reductions) and `mixed_bytes` (integration state).

Architecture reference: OpenMM source commit
`3a2326d0dd091229eaa4c0d7b5ad1001d5fe376e`, CUDA cluster AABB hierarchy,
nonbonded parameter fusion and sparse exception handling. The implementation
uses ATV's own CSR, masks, Frame and Component lifecycle. The fused force path
packs nonempty J columns across cluster boundaries only when neighbor masks
are rebuilt; timed force calls reuse the build-reserved packed buffers.

## NVE energy drift

`RUN_SCRIPT=tools/bench_gaff_openmm/run_drift.sh` (with `DATA`, `FF`,
`PRECISION`, `BUILD`) runs `compare.py --drift`: for each fixed seed
(`SEEDS`, default six) a 300 K Maxwell–Boltzmann velocity set with zero total
momentum is generated once and fed to both engines (`bench_gaff` reads the
optional velocity rows of its input; OpenMM starts its leapfrog half a kick
behind the same full-step velocity). Each engine integrates `DRIFT_TIME` fs
(default 4000) of unconstrained NVE at `DT` and `DT/2`, sampling the total
energy every 10 fs. Drift slopes over a few picoseconds are noise dominated
(they do not shrink with the time step in either engine, because the
unshifted cutoff makes the energy jump as pairs cross it), so the seeds define
the reference distribution: the ATV mean slope must agree with OpenMM's within
two pooled standard errors and the mean fluctuation RMS within 20 %
(`drift.json`, `accepted`). Energy series of both engines are kept per seed.

## Speed and memory scaling

Submit `RUN_SCRIPT=tools/bench_gaff_openmm/run_scaling.sh TARGETS=bench_gaff`
with the same `DATA`, `FF`, `BUILD` and `PRECISION` settings as above.
The default cubic replications `1 2 3 4` give 1,840, 14,720, 49,680 and
117,760 atoms at fixed density. Each point uses 30 steps, 10 warmup steps,
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
Monitoring runs concurrently for both engines; step timing
excludes setup. The harness rejects missing measurements or different
physical GPUs between the compared engines. Batch shells without Slurm's
CUDA visibility mask use the allocated `SLURM_JOB_GPUS` explicitly.
The parent validates replication once and writes the OpenMM child's input as
JSON, avoiding duplicate topology preparation within the short deadline.
Each engine still constructs and evaluates its own forces independently.

After both precisions finish, render the accepted measurements with:

```sh
python tools/bench_gaff_openmm/plot_scaling.py logs/gaff-scaling --out plots/gaff-scaling
```

This produces a six-panel PNG/SVG (step time, GPU memory and host RAM for each
precision) and the numerical CSV. Matplotlib is required only for rendering.
Connecting lines are guides between measured points; no scaling model is fitted.

Accepted GH200 measurements with OpenMM 8.6.1 (time in ms/step, memory in
MiB; memory columns show Atomiverse / OpenMM):

| Precision | Atoms | ATV time | OpenMM time | Observed GPU peak | Observed host peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| double | 1,840 | 0.13425 | 0.12034 | 562 / 1002 | 438 / 430 |
| double | 14,720 | 0.24422 | 0.27595 | 568 / 628 | 433 / 492 |
| double | 49,680 | 0.55698 | 0.66757 | 718 / 1042 | 438 / 574 |
| double | 117,760 | 1.13624 | 1.46810 | 982 / 768 | 527 / 829 |
| mixed | 1,840 | 0.12144 | 0.10916 | 562 / 616 | 445 / 427 |
| mixed | 14,720 | 0.18866 | 0.15647 | 574 / 626 | 405 / 463 |
| mixed | 49,680 | 0.34325 | 0.34051 | 708 / 1160 | 436 / 590 |
| mixed | 117,760 | 0.66605 | 0.69327 | 960 / 1154 | 496 / 802 |

All eight initial/final comparisons pass. Each engine process finishes within
the ten-second deadline; the longest whole comparison, including input
preparation and both engines, is 11.52 s. Memory includes initialization and
context costs; the sampled GPU peaks are not monotone and do not establish
an allocator complexity bound.
