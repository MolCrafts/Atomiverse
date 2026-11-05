# GAFF PEO: Atomiverse CUDA vs OpenMM CUDA

`bench_gaff` (Atomiverse CUDA MD) and `compare.py` (which builds an OpenMM
`System`) read the same LAMMPS `units real` data and coefficients. OpenMM
computes its own energies and forces; reference numbers never come from
Atomiverse formulas. A timing counts only after the initial energy, every
initial force component, the final positions and the final energy all agree.

## Prerequisites

- One CUDA GPU, shared by both engines.
- `OPENMM_ENV`: a Python venv with `openmm` (CUDA platform) and `numpy`.
  `gpu_job.sh` creates it with `pip install 'openmm[cuda13]' numpy` if missing.
- `DATA`, `FF`: the GAFF PEO system as LAMMPS data + coefficient files.
  The dataset is external and never committed. `fragment.data` /
  `fragment.ff` is the small in-tree fixture that `run.sh` checks first.
- `matplotlib`, only for `plot_scaling.py`.

## Build the Atomiverse side

```sh
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_BENCHMARKS=ON \
      -DATV_BUILD_CUDA=ON -DATV_BUILD_KSPACE=ON -DATV_BUILD_EVALUATE=ON \
      -DATV_CUDA_PREC_MODE=double          # or mixed; one build tree per mode
cmake --build "$BUILD" --target bench_gaff   # -> $BUILD/bin/bench_gaff
```

## Run

All scripts take the build directory as their argument and write under
`logs/` in the current directory.

| Script | Purpose | Extra knobs |
|---|---|---|
| `run.sh` | fragment check, then timed comparison per replication | `PRECISION`, `STEPS` [100], `REPLICAS` ["1 3"], `SKEW_REPLICAS`, `REFERENCE_CHECK_DATA` |
| `run_drift.sh` | NVE energy-drift comparison | `DT` [0.25], `DRIFT_TIME` [4000], `SEEDS`, `REPLICAS`, `BUDGET` |
| `run_scaling.sh` | step time + memory, replications 1–4 | `REPLICAS`, `PRECISION` |
| `diagnose.sh` | per-term force comparison (nonbonded/bond/angle/dihedral) | — |

```sh
OPENMM_ENV=... DATA=... FF=... PRECISION=double bash run.sh "$BUILD"
```

`gpu_job.sh` is the cluster-specific Slurm wrapper. It loads the GPU toolchain
and Python modules (`ATV_GPU_MODULE`, `ATV_PYTHON_MODULE`; the defaults name
the Arrhenius GH200 stack, so override them on other clusters), builds
`bench_gaff`, prepares `OPENMM_ENV`, hashes the relevant sources, and runs
`RUN_SCRIPT` (default `run.sh`):

```sh
OPENMM_ENV=... DATA=... FF=... PRECISION=mixed BUILD=build-gaff-mixed \
  RUN_SCRIPT=benchmarks/compare/gaff-peo-openmm/run_drift.sh \
  sbatch -A <gpu-account> benchmarks/compare/gaff-peo-openmm/gpu_job.sh
```

## Matched protocol

- Unconstrained NVE, dt 0.25 fs, zero initial velocities, 10 warmup steps,
  3 timed samples (median). Each engine process has a 10 s deadline
  (`--budget`), and setup counts against it.
- LJ: unshifted Lorentz–Berthelot, no tail correction, cutoff 9 Å. Neighbor
  skin is 8 % of the cutoff (OpenMM CUDA's default padding).
- Coulomb: Ewald α 0.35 Å⁻¹, order-5 PME. The mesh is rounded from 1 Å
  spacing up to 2/3/5/7-smooth sizes.
- Amber exceptions: LJ (0, 0, 1/2) and Coulomb (0, 0, 5/6) by graph
  distance, minimum image, no exception cutoff. Bonded terms use periodic
  geometry in both engines.
- Replication (`--replicate n`) follows minimum-image bonded paths.
  `--skew` tiles the same system in a tilted supercell (A = n·a,
  B = a + n·b, C = b + n·c), so both engines run their triclinic paths on
  identical physics.
- OpenMM's leapfrog velocity starts half a kick behind the full-step
  velocity, so both engines follow the same velocity-Verlet trajectory.
- Mixed mode: positions and cell are snapped to a binary lattice
  (`5·k·2^e` Å) that is exactly representable in float32 in both Å and nm.
  Initial parity therefore compares identical geometry.
- Supported input: contiguous IDs, an orthorhombic input cell, harmonic
  bonds/angles, and Fourier torsions with n = 1–4 and phase 0 or π.
  Anything else raises an error.

## Acceptance (`summary.json` → `accepted`)

| Check | double | mixed |
|---|---|---|
| initial energy | \|ΔE\| ≤ max(1e-3 eV, 3e-5 \|E\|) | same |
| initial forces: max \|ΔF\| / max(1, max \|F_ref\|) | ≤ 3e-5 | ≤ 1e-4 |
| final positions (minimum image) | ≤ 1e-4 Å | ≤ 1e-2 Å |
| final energy | \|ΔE\| ≤ max(1e-3 eV, 3e-5 \|E\|) | same |

`compare.py` also rejects non-finite values. It rejects a binary whose
reported `real_bytes`/`work_bytes`/`accum_bytes` do not match `--precision`
(8/8/8 double, 4/4/8 mixed). It rejects engines that ran on different GPUs.
A failed acceptance exits nonzero.

**Drift** (`--drift`, `drift.json` → `accepted`). Each seed generates one
300 K Maxwell–Boltzmann velocity set with zero momentum and feeds it to both
engines. Each engine integrates `DRIFT_TIME` fs at `DT` and `DT/2`, sampling
E_total every `DRIFT_EVERY` fs. Pass requires two things at both time steps:
the ATV mean drift slope matches OpenMM's within 2 pooled standard errors,
and the mean fluctuation RMS matches within 20 %. Short-run slopes are noise
dominated because the unshifted cutoff makes the energy jump, so the seeds
define the distribution.

**Memory** (`--measure-memory`, used by `run_scaling.sh`):
- GPU peak comes from PID-filtered `nvidia-smi` samples every 100 ms. It
  includes the CUDA context and can miss short transients.
- Host peak is the largest observed `VmHWM` of the engine process.
- Missing samples, or the two engines on different GPUs, fail the case.

`plot_scaling.py logs/gaff-scaling --out <dir>` reads only accepted
`summary.json` files. It needs both precisions with at least three matching
atom counts and one settings/input/binary cohort. It writes a CSV and a
six-panel PNG/SVG.
